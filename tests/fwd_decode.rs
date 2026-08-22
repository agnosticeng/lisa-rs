// Full Qwen3.8 decoder block (layer 17, GDN) through the NAX tensor-core on
// REAL 27B weights, chaining in one pass:
//   x   bf16[M,H]
//   xn  = input_layernorm(x)                    (rms_norm_rows)
//   r   = linear_attn(xn): in_proj_qkv/z/a/b GEMMs + conv1d + gdn recurrence
//   h   = residual(x, r)                        (residual_add -> bf16)
//   hn  = post_attention_layernorm(h)           (rms_norm_rows)
//   o   = mlp(hn): silu(gate)*up then down_proj (NAX)
//   y   = h + o                                 (residual_add -> bf16)
// Anchored vs scripts/mlx_fwd_truth.py which runs the ACTUAL unmodified
// mlx-vlm Qwen3_5DecoderLayer(17) on the same weights/h (/tmp/mlx_fwd_truth.raw).
use std::fs::read_dir;
use std::path::{Path, PathBuf};

use lisa_rs::device::metal::{ComputeEncoder, ComputePipeline, MTLSize, MetalBuffer, MetalDevice};
use lisa_rs::format::dtype::{bf16_to_f32, f32_to_bf16};
use lisa_rs::format::safetensor as st;
use lisa_rs::kernels::linear::NORM_SHADER;
use lisa_rs::kernels::linear::{GDN_SHADER, MLP_SHADER, NAX_AFFINE_U4_SHADER};

const M: usize = 16;
const H: usize = 5120;
const KH: usize = 16; // key heads
const VH: usize = 48; // value heads
const HD: usize = 128;
const KEY_DIM: usize = KH * HD; // 2048
const VALUE_DIM: usize = VH * HD; // 6144
const CONV_DIM: usize = 2 * KEY_DIM + VALUE_DIM; // 10240
const INTER: usize = 17408;
const EPS: f32 = 1e-6;
const LAYER: &str = "language_model.model.layers.17.";

fn shard_paths(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in read_dir(dir).expect("rd") {
        let p = entry.expect("e").path();
        if p.is_dir() {
            shard_paths(&p, out);
        } else if p
            .file_name()
            .expect("fn")
            .to_string_lossy()
            .ends_with(".safetensors")
        {
            out.push(PathBuf::from(&p));
        }
    }
}

fn find<'a>(shards: &'a [st::Shard], full: &str) -> (usize, &'a st::Tensor) {
    for (i, s) in shards.iter().enumerate() {
        if let Some(t) = s.tensors.iter().find(|t| t.name == full) {
            return (i, t);
        }
    }
    panic!("tensor not found: {full}");
}

fn u(o: usize) -> u64 {
    u64::try_from(o).unwrap()
}

fn copy_bytes(buf: &MetalBuffer, dest: usize, shards: &[st::Shard], name: &str) {
    let (i, t) = find(shards, name);
    let db = st::data_bytes(&shards[i]);
    let start = usize::try_from(t.off - shards[i].data_start).unwrap();
    let len = usize::try_from(t.len).unwrap();
    buf.write_bytes(u64::try_from(dest).unwrap(), &db[start..start + len]);
}

fn copy_bf16_as_f32(buf: &MetalBuffer, dest: usize, shards: &[st::Shard], name: &str) {
    let (i, t) = find(shards, name);
    let db = st::data_bytes(&shards[i]);
    let start = usize::try_from(t.off - shards[i].data_start).unwrap();
    let len = usize::try_from(t.len).unwrap();
    let n = len / 2;
    let mut f = Vec::<u8>::new();
    for j in 0..n {
        let b0 = db[start + j * 2];
        let b1 = db[start + j * 2 + 1];
        let bits: u16 = (b1 as u16) << 8 | (b0 as u16);
        f.extend_from_slice(&bf16_to_f32(bits).to_le_bytes());
    }
    buf.write_bytes(u64::try_from(dest).unwrap(), &f);
}

fn affine_padded_bytes(
    shards: &[st::Shard],
    wname: &str,
    sname: &str,
    bname: &str,
    real: usize,
    pad: usize,
    in_dim: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (wi, w) = find(shards, wname);
    let (si, s) = find(shards, sname);
    let (bi, b) = find(shards, bname);
    if wi != si || wi != bi {
        eprintln!("WARN: cross-shard affine: w=shard{wi} s=shard{si} b=shard{bi} for {wname}");
    }
    let db_w = st::data_bytes(&shards[wi]);
    let db_s = st::data_bytes(&shards[si]);
    let db_b = st::data_bytes(&shards[bi]);
    let rw = in_dim / 8;
    let rg = in_dim / 64;
    let wrow = rw * 4;
    let srow = rg * 2;
    let ws0 = usize::try_from(w.off - shards[wi].data_start).unwrap();
    let ss0 = usize::try_from(s.off - shards[si].data_start).unwrap();
    let bs0 = usize::try_from(b.off - shards[bi].data_start).unwrap();
    let mut wv = vec![0u8; pad * wrow];
    let mut sv = vec![0u8; pad * srow];
    let mut bv = vec![0u8; pad * srow];
    for r in 0..real {
        let w0 = r * wrow;
        let s0 = r * srow;
        wv[w0..w0 + wrow].copy_from_slice(&db_w[ws0 + w0..ws0 + w0 + wrow]);
        sv[s0..s0 + srow].copy_from_slice(&db_s[ss0 + s0..ss0 + s0 + srow]);
        bv[s0..s0 + srow].copy_from_slice(&db_b[bs0 + s0..bs0 + s0 + srow]);
    }
    (wv, sv, bv)
}

#[allow(clippy::too_many_arguments)]
fn gemm(
    enc: &ComputeEncoder,
    pipe: &ComputePipeline,
    a: &MetalBuffer,
    a_off: usize,
    w: &MetalBuffer,
    w_off: usize,
    s: &MetalBuffer,
    s_off: usize,
    b: &MetalBuffer,
    b_off: usize,
    c: &MetalBuffer,
    c_off: usize,
    n: u32,
    k: u32,
) {
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(a, u64::try_from(a_off).unwrap(), 0);
    enc.set_buffer(w, u64::try_from(w_off).unwrap(), 1);
    enc.set_buffer(s, u64::try_from(s_off).unwrap(), 2);
    enc.set_buffer(b, u64::try_from(b_off).unwrap(), 3);
    enc.set_buffer(c, u64::try_from(c_off).unwrap(), 4);
    let mut nk = Vec::new();
    nk.extend_from_slice(&n.to_le_bytes());
    nk.extend_from_slice(&k.to_le_bytes());
    // Single 16-row tile: report its height so the tail-guard fills it.
    nk.extend_from_slice(&16u32.to_le_bytes());
    enc.set_bytes(&nk, 5);
    let tn = u64::from(n).div_ceil(32);
    enc.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: tn,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
}

fn f32_le(vals: &[f32]) -> Vec<u8> {
    let mut o = Vec::new();
    for v in vals {
        o.extend_from_slice(&v.to_le_bytes());
    }
    o
}

fn u32_le(vals: &[u32]) -> Vec<u8> {
    let mut o = Vec::new();
    for v in vals {
        o.extend_from_slice(&v.to_le_bytes());
    }
    o
}

#[allow(clippy::too_many_arguments)]
fn rms_norm_rows(
    enc: &ComputeEncoder,
    pipe: &ComputePipeline,
    ab: &MetalBuffer,
    x: usize,
    w: usize,
    y: usize,
    cols: u32,
    eps: f32,
) {
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(ab, u(x), 0);
    enc.set_buffer(ab, u(w), 1);
    enc.set_buffer(ab, u(y), 2);
    enc.set_bytes(&cols.to_le_bytes(), 3);
    enc.set_bytes(&eps.to_le_bytes(), 4);
    enc.dispatch_thread_groups(
        MTLSize {
            width: M as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
}

fn residual_add_bytes(
    enc: &ComputeEncoder,
    pipe: &ComputePipeline,
    ab: &MetalBuffer,
    x: usize,
    o: usize,
    y: usize,
    n: usize,
) {
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(ab, u(x), 0);
    enc.set_buffer(ab, u(o), 1);
    enc.set_buffer(ab, u(y), 2);
    enc.set_bytes(&(n as u32).to_le_bytes(), 3);
    enc.dispatch_threads(
        MTLSize {
            width: n as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    );
}

#[test]
fn fwd_block17() {
    let home = std::env::home_dir().expect("home");
    let root =
        home.join(".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots");
    let mut paths = Vec::new();
    shard_paths(&root, &mut paths);
    if paths.is_empty() {
        eprintln!("skip: model not cached");
        return;
    }
    let target = LAYER.to_owned() + "linear_attn.in_proj_qkv.weight";
    let mut shards: Vec<st::Shard> = Vec::new();
    for p in &paths {
        let s = st::open(Path::new(&p));
        if s.tensors.iter().any(|t| t.name == target)
            || s.tensors
                .iter()
                .any(|t| t.name.starts_with(&(LAYER.to_owned() + "mlp.")))
        {
            shards.push(s);
        }
    }
    assert!(!shards.is_empty(), "layer17 shards not found");

    let dev = MetalDevice::default();
    let nax_pipe = dev.new_compute_pipeline(
        &dev.new_library(NAX_AFFINE_U4_SHADER)
            .function_named("q4_gemm_nax_coop"),
    );
    let gdn = dev.new_library(GDN_SHADER);
    let conv_p = dev.new_compute_pipeline(&gdn.function_named("gdn_conv_norm_gates"));
    let rec_p = dev.new_compute_pipeline(&gdn.function_named("gdn_recurrence"));
    let rg_p = dev.new_compute_pipeline(&gdn.function_named("gdn_rms_gate"));
    let silu_pipe =
        dev.new_compute_pipeline(&dev.new_library(MLP_SHADER).function_named("silu_mul_up"));
    let norm = dev.new_library(NORM_SHADER);
    let rms_pipe = dev.new_compute_pipeline(&norm.function_named("rms_norm_rows"));
    let res_pipe = dev.new_compute_pipeline(&norm.function_named("residual_add"));
    let q = dev.new_command_queue();

    // ---- per-projection weight heaps ----
    let qkv_w: usize = CONV_DIM * 640 * 4;
    let qkv_s: usize = CONV_DIM * 80 * 2;
    let qkheap = dev.new_heap(qkv_w + 2 * qkv_s + 4096);
    let qkbuf = qkheap.new_buffer(qkv_w + 2 * qkv_s);
    copy_bytes(
        &qkbuf,
        0,
        &shards,
        &(LAYER.to_owned() + "linear_attn.in_proj_qkv.weight"),
    );
    copy_bytes(
        &qkbuf,
        qkv_w,
        &shards,
        &(LAYER.to_owned() + "linear_attn.in_proj_qkv.scales"),
    );
    copy_bytes(
        &qkbuf,
        qkv_w + qkv_s,
        &shards,
        &(LAYER.to_owned() + "linear_attn.in_proj_qkv.biases"),
    );

    let zw: usize = VALUE_DIM * 640 * 4;
    let zs: usize = VALUE_DIM * 80 * 2;
    let zheap = dev.new_heap(zw + 2 * zs + 4096);
    let zbuf = zheap.new_buffer(zw + 2 * zs);
    copy_bytes(
        &zbuf,
        0,
        &shards,
        &(LAYER.to_owned() + "linear_attn.in_proj_z.weight"),
    );
    copy_bytes(
        &zbuf,
        zw,
        &shards,
        &(LAYER.to_owned() + "linear_attn.in_proj_z.scales"),
    );
    copy_bytes(
        &zbuf,
        zw + zs,
        &shards,
        &(LAYER.to_owned() + "linear_attn.in_proj_z.biases"),
    );

    let (awv, asv, abv) = affine_padded_bytes(
        &shards,
        &(LAYER.to_owned() + "linear_attn.in_proj_a.weight"),
        &(LAYER.to_owned() + "linear_attn.in_proj_a.scales"),
        &(LAYER.to_owned() + "linear_attn.in_proj_a.biases"),
        VH,
        64,
        H,
    );
    let (bwv, bsv, bbv) = affine_padded_bytes(
        &shards,
        &(LAYER.to_owned() + "linear_attn.in_proj_b.weight"),
        &(LAYER.to_owned() + "linear_attn.in_proj_b.scales"),
        &(LAYER.to_owned() + "linear_attn.in_proj_b.biases"),
        VH,
        64,
        H,
    );
    let ba4 = awv.len() + asv.len() + abv.len();
    let aheap = dev.new_heap(ba4 + 4096);
    let a_buf = aheap.new_buffer(ba4);
    a_buf.write_bytes(u(0), &awv);
    a_buf.write_bytes(u(awv.len()), &asv);
    a_buf.write_bytes(u(awv.len() + asv.len()), &abv);
    let bheap = dev.new_heap(ba4 + 4096);
    let b_buf = bheap.new_buffer(ba4);
    b_buf.write_bytes(u(0), &bwv);
    b_buf.write_bytes(u(bwv.len()), &bsv);
    b_buf.write_bytes(u(bwv.len() + bsv.len()), &bbv);

    let ow: usize = H * 768 * 4;
    let os: usize = H * 96 * 2;
    let oheap = dev.new_heap(ow + 2 * os + 4096);
    let obuf = oheap.new_buffer(ow + 2 * os);
    copy_bytes(
        &obuf,
        0,
        &shards,
        &(LAYER.to_owned() + "linear_attn.out_proj.weight"),
    );
    copy_bytes(
        &obuf,
        ow,
        &shards,
        &(LAYER.to_owned() + "linear_attn.out_proj.scales"),
    );
    copy_bytes(
        &obuf,
        ow + os,
        &shards,
        &(LAYER.to_owned() + "linear_attn.out_proj.biases"),
    );

    // ---- MLP weight heaps (one per projection, each <~50 MiB for the driver) ----
    let mi: usize = INTER * 640 * 4;
    let ms: usize = INTER * 80 * 2;
    let mproj: usize = mi + 2 * ms;
    let gheap = dev.new_heap(mproj + 4096);
    let gbuf = gheap.new_buffer(mproj);
    let uheap = dev.new_heap(mproj + 4096);
    let ubuf = uheap.new_buffer(mproj);
    let dheap = dev.new_heap(mproj + 4096);
    let dbuf = dheap.new_buffer(mproj);
    copy_bytes(
        &gbuf,
        0,
        &shards,
        &(LAYER.to_owned() + "mlp.gate_proj.weight"),
    );
    copy_bytes(
        &gbuf,
        mi,
        &shards,
        &(LAYER.to_owned() + "mlp.gate_proj.scales"),
    );
    copy_bytes(
        &gbuf,
        mi + ms,
        &shards,
        &(LAYER.to_owned() + "mlp.gate_proj.biases"),
    );
    copy_bytes(
        &ubuf,
        0,
        &shards,
        &(LAYER.to_owned() + "mlp.up_proj.weight"),
    );
    copy_bytes(
        &ubuf,
        mi,
        &shards,
        &(LAYER.to_owned() + "mlp.up_proj.scales"),
    );
    copy_bytes(
        &ubuf,
        mi + ms,
        &shards,
        &(LAYER.to_owned() + "mlp.up_proj.biases"),
    );
    copy_bytes(
        &dbuf,
        0,
        &shards,
        &(LAYER.to_owned() + "mlp.down_proj.weight"),
    );
    copy_bytes(
        &dbuf,
        mi,
        &shards,
        &(LAYER.to_owned() + "mlp.down_proj.scales"),
    );
    copy_bytes(
        &dbuf,
        mi + ms,
        &shards,
        &(LAYER.to_owned() + "mlp.down_proj.biases"),
    );

    // ---- activation heap ----
    let hb = M * H * 2;
    let nf = M * H * 2;
    let qf = M * CONV_DIM * 4;
    let zf = M * VALUE_DIM * 4;
    let vf = M * 64 * 4;
    let be48 = M * VH * 4;
    let qn = M * KEY_DIM * 4;
    let kn = M * KEY_DIM * 4;
    let state = VH * HD * HD * 4;
    let yf = M * VALUE_DIM * 4;
    let gated = M * VALUE_DIM * 2;
    let oo = M * H * 4;
    let h1 = M * H * 2;
    let hi = M * H * 2;
    let gato = M * INTER * 4;
    let upo = M * INTER * 4;
    let si = M * INTER * 2;
    let cw = CONV_DIM * 4 * 4;
    let alog = VH * 4;
    let dtb = VH * 4;
    let nwg = HD * 4;
    let cst = 3 * CONV_DIM * 4;
    let iw = H * 2;
    let pw = H * 2;
    let sz = hb
        + nf
        + qf
        + zf
        + 2 * vf
        + qf
        + qn
        + kn
        + 2 * be48
        + state
        + yf
        + gated
        + oo
        + h1
        + hi
        + gato
        + upo
        + si
        + cw
        + alog
        + dtb
        + nwg
        + cst
        + iw
        + pw
        + 4096;
    let ah = dev.new_heap(sz);
    let ab = ah.new_buffer(sz);

    let mut off = 0usize;
    let h0 = off;
    off += hb;
    let n0 = off;
    off += nf;
    let q0 = off;
    off += qf;
    let z0 = off;
    off += zf;
    let a0 = off;
    off += vf;
    let b0 = off;
    off += vf;
    let co0 = off;
    off += qf;
    let qn0 = off;
    off += qn;
    let kn0 = off;
    off += kn;
    let be0 = off;
    off += be48;
    let de0 = off;
    off += be48;
    let st0 = off;
    off += state;
    let y0 = off;
    off += yf;
    let ga0 = off;
    off += gated;
    let oo0 = off;
    off += oo;
    let h10 = off;
    off += h1;
    let hi0 = off;
    off += hi;
    let gr0 = off;
    off += gato;
    let ur0 = off;
    off += upo;
    let si0 = off;
    off += si;
    let cw0 = off;
    off += cw;
    let al0 = off;
    off += alog;
    let dt0 = off;
    off += dtb;
    let nw0 = off;
    off += nwg;
    let cs0 = off;
    off += cst;
    let iw0 = off;
    off += iw;
    let pw0 = off;
    off += pw;
    assert!(off <= sz);

    // deterministic hidden
    let mut hv = Vec::new();
    for i in 0..(M * H) {
        let v = (i % 97) as f32 * 0.001;
        hv.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
    }
    ab.write_bytes(u(h0), &hv);
    copy_bf16_as_f32(
        &ab,
        cw0,
        &shards,
        &(LAYER.to_owned() + "linear_attn.conv1d.weight"),
    );
    copy_bf16_as_f32(&ab, al0, &shards, &(LAYER.to_owned() + "linear_attn.A_log"));
    copy_bf16_as_f32(
        &ab,
        dt0,
        &shards,
        &(LAYER.to_owned() + "linear_attn.dt_bias"),
    );
    copy_bf16_as_f32(
        &ab,
        nw0,
        &shards,
        &(LAYER.to_owned() + "linear_attn.norm.weight"),
    );
    copy_bytes(
        &ab,
        iw0,
        &shards,
        &(LAYER.to_owned() + "input_layernorm.weight"),
    );
    copy_bytes(
        &ab,
        pw0,
        &shards,
        &(LAYER.to_owned() + "post_attention_layernorm.weight"),
    );
    let zeros = [0u8; 4096];
    for offz in [cs0, st0] {
        let mut left = if offz == cs0 { cst } else { state };
        let mut at = offz;
        while left > 0 {
            let w = left.min(4096);
            ab.write_bytes(u(at), &zeros[..w]);
            at += w;
            left -= w;
        }
    }

    let cb = q.new_command_buffer();
    let enc = cb.compute_compute_encoder();

    // 0. input_layernorm(x)
    rms_norm_rows(&enc, &rms_pipe, &ab, h0, iw0, n0, H as u32, EPS);

    // 1. linear_attn projections on n0
    gemm(
        &enc,
        &nax_pipe,
        &ab,
        n0,
        &qkbuf,
        0,
        &qkbuf,
        qkv_w,
        &qkbuf,
        qkv_w + qkv_s,
        &ab,
        q0,
        CONV_DIM as u32,
        H as u32,
    );
    gemm(
        &enc,
        &nax_pipe,
        &ab,
        n0,
        &zbuf,
        0,
        &zbuf,
        zw,
        &zbuf,
        zw + zs,
        &ab,
        z0,
        VALUE_DIM as u32,
        H as u32,
    );
    gemm(
        &enc,
        &nax_pipe,
        &ab,
        n0,
        &a_buf,
        0,
        &a_buf,
        awv.len(),
        &a_buf,
        awv.len() + asv.len(),
        &ab,
        a0,
        64,
        H as u32,
    );
    gemm(
        &enc,
        &nax_pipe,
        &ab,
        n0,
        &b_buf,
        0,
        &b_buf,
        bwv.len(),
        &b_buf,
        bwv.len() + bsv.len(),
        &ab,
        b0,
        64,
        H as u32,
    );

    let inv = 0.088_388_346_f32;
    enc.set_compute_pipeline_state(&conv_p);
    enc.set_buffer(&ab, u(q0), 0);
    enc.set_buffer(&ab, u(b0), 1);
    enc.set_buffer(&ab, u(a0), 2);
    enc.set_buffer(&ab, u(cw0), 3);
    enc.set_buffer(&ab, u(cs0), 4);
    enc.set_buffer(&ab, u(al0), 5);
    enc.set_buffer(&ab, u(dt0), 6);
    enc.set_buffer(&ab, u(co0), 7);
    enc.set_buffer(&ab, u(qn0), 8);
    enc.set_buffer(&ab, u(kn0), 9);
    enc.set_buffer(&ab, u(be0), 10);
    enc.set_buffer(&ab, u(de0), 11);
    enc.set_bytes(&u32_le(&[KH as u32, VH as u32, HD as u32, HD as u32]), 12);
    enc.set_bytes(&f32_le(&[inv * inv, inv]), 13);
    enc.set_bytes(&(M as u32).to_le_bytes(), 14);
    enc.dispatch_thread_groups(
        MTLSize {
            width: 48,
            height: M as u64,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    enc.set_compute_pipeline_state(&rec_p);
    enc.set_buffer(&ab, u(co0), 0);
    enc.set_buffer(&ab, u(qn0), 1);
    enc.set_buffer(&ab, u(kn0), 2);
    enc.set_buffer(&ab, u(be0), 3);
    enc.set_buffer(&ab, u(de0), 4);
    enc.set_buffer(&ab, u(st0), 5);
    enc.set_buffer(&ab, u(y0), 6);
    enc.set_buffer(&ab, u(st0), 9);
    enc.set_bytes(
        &u32_le(&[VH as u32, HD as u32, HD as u32, (VH / KH) as u32]),
        7,
    );
    enc.set_bytes(&(M as u32).to_le_bytes(), 8);
    enc.set_bytes(&0u32.to_le_bytes(), 10);
    enc.set_buffer(&ab, u(st0), 9);
    enc.set_bytes(&0u32.to_le_bytes(), 10);
    enc.dispatch_threads(
        MTLSize {
            width: 32,
            height: 128,
            depth: 48,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    enc.set_compute_pipeline_state(&rg_p);
    enc.set_buffer(&ab, u(y0), 0);
    enc.set_buffer(&ab, u(z0), 1);
    enc.set_buffer(&ab, u(nw0), 2);
    enc.set_buffer(&ab, u(ga0), 3);
    enc.set_bytes(&u32_le(&[VH as u32, HD as u32, M as u32]), 4);
    enc.set_bytes(&EPS.to_le_bytes(), 5);
    enc.dispatch_thread_groups(
        MTLSize {
            width: 48,
            height: M as u64,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    // 2. BF16 gated output -> out_proj, no CPU synchronization.
    gemm(
        &enc,
        &nax_pipe,
        &ab,
        ga0,
        &obuf,
        0,
        &obuf,
        ow,
        &obuf,
        ow + os,
        &ab,
        oo0,
        H as u32,
        VALUE_DIM as u32,
    );
    // 3. h = x + r  (bf16)
    residual_add_bytes(&enc, &res_pipe, &ab, h0, oo0, h10, M * H);

    // 4. post_attention_layernorm(h) -> hi0
    rms_norm_rows(&enc, &rms_pipe, &ab, h10, pw0, hi0, H as u32, EPS);

    // 5. MLP: gate/up on hi0
    gemm(
        &enc,
        &nax_pipe,
        &ab,
        hi0,
        &gbuf,
        0,
        &gbuf,
        mi,
        &gbuf,
        mi + ms,
        &ab,
        gr0,
        INTER as u32,
        H as u32,
    );
    gemm(
        &enc,
        &nax_pipe,
        &ab,
        hi0,
        &ubuf,
        0,
        &ubuf,
        mi,
        &ubuf,
        mi + ms,
        &ab,
        ur0,
        INTER as u32,
        H as u32,
    );

    enc.set_compute_pipeline_state(&silu_pipe);
    enc.set_buffer(&ab, u(gr0), 0);
    enc.set_buffer(&ab, u(ur0), 1);
    enc.set_buffer(&ab, u(si0), 2);
    enc.set_bytes(&((M * INTER) as u32).to_le_bytes(), 3);
    enc.dispatch_threads(
        MTLSize {
            width: (M * INTER) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    );

    // 6. down_proj -> oo0 (reuse; MLP reads si0)
    gemm(
        &enc,
        &nax_pipe,
        &ab,
        si0,
        &dbuf,
        0,
        &dbuf,
        mi,
        &dbuf,
        mi + ms,
        &ab,
        oo0,
        H as u32,
        INTER as u32,
    );

    // 7. final residual h + o -> y (separate bf16 out region)
    residual_add_bytes(&enc, &res_pipe, &ab, h10, oo0, y0, M * H);

    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let y = ab.read_bytes(u(y0), M * H * 2);
    let mut yf32 = Vec::with_capacity(M * H);
    for i in 0..(M * H) {
        let b0 = y[i * 2];
        let b1 = y[i * 2 + 1];
        yf32.push(bf16_to_f32(((b1 as u16) << 8) | (b0 as u16)));
    }
    let mut maxabs = 0.0f32;
    for &v in &yf32 {
        maxabs = maxabs.max(v.abs());
    }
    eprintln!(
        "block17 y[0,:4] = {:5} {:5} {:5} {:5}  max|y|={maxabs}",
        if yf32.len() > 3 { yf32[0] } else { 0.0 },
        if yf32.len() > 3 { yf32[1] } else { 0.0 },
        if yf32.len() > 3 { yf32[2] } else { 0.0 },
        if yf32.len() > 3 { yf32[3] } else { 0.0 }
    );

    // anchored correctness vs mlx-vlm ground truth (scripts/mlx_fwd_truth.py)
    if let Ok(raw) = std::fs::read("/tmp/mlx_fwd_truth.raw") {
        let n = M * H;
        assert_eq!(raw.len(), n * 4, "truth size");
        let mut md = 0.0f32;
        let mut mi = 0;
        for i in 0..n {
            let mine = yf32[i];
            let refv =
                f32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
            let dd = (mine - refv).abs();
            if dd > md {
                md = dd;
                mi = i;
            }
        }
        eprintln!(
            "fwd block17 vs mlx-vlm truth max abs diff = {md} at row={} col={} mine={} ref={}",
            mi / H,
            mi % H,
            yf32[mi],
            f32::from_le_bytes([
                raw[mi * 4],
                raw[mi * 4 + 1],
                raw[mi * 4 + 2],
                raw[mi * 4 + 3]
            ])
        );
        assert!(
            md <= 1.25e-1,
            "full block drifted from mlx-vlm truth by {md}"
        );
    } else {
        eprintln!("skip MLX comparison (run scripts/mlx_fwd_truth.py first)");
    }
}
