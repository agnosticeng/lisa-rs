// Full Gated Delta Net (linear_attn) decoder layer (layer 17) through the NAX
// tensor-core on REAL 27B weights, porting saragossa's exact GDN kernels:
//   qkv = in_proj_qkv(h)  z=in_proj_z(h)  a_=in_proj_a(h)  b_=in_proj_b(h)  (NAX)
//   conv+norm+gates: causal K4 conv+SiLU over fused qkv, RMS-norm+scale q/k
//                    (q*1/128, k*1/sqrt(128)), beta=sigmoid(b), decay=
//                    exp(-exp(A_log)*softplus(a+dt_bias))
//   recurrence:      GDN state f32 [48,128,128], GQA vh->kh=vh/3
//   y = state.q ; gated = RMSNorm(norm.weight)(y) * silu(z)
//   out = out_proj(gated) (NAX)
// Anchor vs MLX in scripts/mlx_gdn_ref.py (writes /tmp/mlx_gdn_out.raw).
use std::fs::read_dir;
use std::path::{Path, PathBuf};

use lisa_rs::device::metal::{ComputeEncoder, ComputePipeline, MTLSize, MetalBuffer, MetalDevice};
use lisa_rs::format::dtype::{bf16_to_f32, f32_to_bf16};
use lisa_rs::format::safetensor as st;
use lisa_rs::kernels::linear::GDN_SHADER;
use lisa_rs::kernels::linear::NAX_AFFINE_U4_SHADER;

const M: usize = 16;
const H: usize = 5120;
const KH: usize = 16; // key heads
const VH: usize = 48; // value heads
const HD: usize = 128;
const KEY_DIM: usize = KH * HD; // 2048
const VALUE_DIM: usize = VH * HD; // 6144
const CONV_DIM: usize = 2 * KEY_DIM + VALUE_DIM; // 10240
const EPS: f32 = 1e-6;
const LAYER: &str = "language_model.model.layers.17.linear_attn.";

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

fn find<'a>(shard: &'a st::Shard, full: &str) -> &'a st::Tensor {
    shard
        .tensors
        .iter()
        .find(|t| t.name == full)
        .expect("tensor not found")
}

fn u(o: usize) -> u64 {
    u64::try_from(o).unwrap()
}

fn copy_bytes(buf: &MetalBuffer, dest: usize, shard: &st::Shard, t: &st::Tensor) {
    let db = st::data_bytes(shard);
    let start = usize::try_from(t.off - shard.data_start).unwrap();
    let len = usize::try_from(t.len).unwrap();
    buf.write_bytes(u64::try_from(dest).unwrap(), &db[start..start + len]);
}

fn copy_bf16_as_f32(buf: &MetalBuffer, dest: usize, shard: &st::Shard, t: &st::Tensor) {
    let db = st::data_bytes(shard);
    let start = usize::try_from(t.off - shard.data_start).unwrap();
    let len = usize::try_from(t.len).unwrap();
    let n = len / 2;
    let mut f = Vec::<u8>::new();
    for i in 0..n {
        let b0 = db[start + i * 2];
        let b1 = db[start + i * 2 + 1];
        let bits: u16 = (b1 as u16) << 8 | (b0 as u16);
        f.extend_from_slice(&bf16_to_f32(bits).to_le_bytes());
    }
    buf.write_bytes(u64::try_from(dest).unwrap(), &f);
}

// Pad an affine-q4 linear to `pad` output rows (real <= pad) with zeroed rows,
// so the NAX gemm (32-wide output tiles) never writes/reads past the buffers.
fn affine_padded_bytes(
    shard: &st::Shard,
    w: &st::Tensor,
    s: &st::Tensor,
    b: &st::Tensor,
    real: usize,
    pad: usize,
    in_dim: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let db = st::data_bytes(shard);
    let rw = in_dim / 8; // u32 words per row
    let rg = in_dim / 64;
    let wrow = rw * 4;
    let srow = rg * 2;
    let ws0 = usize::try_from(w.off - shard.data_start).unwrap();
    let ss0 = usize::try_from(s.off - shard.data_start).unwrap();
    let bs0 = usize::try_from(b.off - shard.data_start).unwrap();
    let mut wv = vec![0u8; pad * wrow];
    let mut sv = vec![0u8; pad * srow];
    let mut bv = vec![0u8; pad * srow];
    for r in 0..real {
        let w0 = r * wrow;
        let s0 = r * srow;
        wv[w0..w0 + wrow].copy_from_slice(&db[ws0 + w0..ws0 + w0 + wrow]);
        sv[s0..s0 + srow].copy_from_slice(&db[ss0 + s0..ss0 + s0 + srow]);
        bv[s0..s0 + srow].copy_from_slice(&db[bs0 + s0..bs0 + s0 + srow]);
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
#[test]
fn gdn_decode_layer17() {
    let home = std::env::home_dir().expect("home");
    let root =
        home.join(".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots");
    let mut paths = Vec::new();
    shard_paths(&root, &mut paths);
    if paths.is_empty() {
        eprintln!("skip: model not cached");
        return;
    }
    let target = LAYER.to_owned() + "in_proj_qkv.weight";
    let mut shard: Option<st::Shard> = None;
    for p in &paths {
        let s = st::open(Path::new(&p));
        if s.tensors.iter().any(|t| t.name == target) {
            shard = Some(s);
            break;
        }
    }
    let shard = shard.expect("layer17 not found");

    let dev = MetalDevice::default();
    let nax = dev.new_library(NAX_AFFINE_U4_SHADER);
    let nax_pipe = dev.new_compute_pipeline(&nax.function_named("q4_gemm_nax_coop"));
    let gdn = dev.new_library(GDN_SHADER);
    let conv_p = dev.new_compute_pipeline(&gdn.function_named("gdn_conv_norm_gates"));
    let rec_p = dev.new_compute_pipeline(&gdn.function_named("gdn_recurrence"));
    let rg_p = dev.new_compute_pipeline(&gdn.function_named("gdn_rms_gate"));
    let q = dev.new_command_queue();

    // ---- per-projection heaps ----
    let qkv_w: usize = CONV_DIM * 640 * 4;
    let qkv_s: usize = CONV_DIM * 80 * 2;
    let qkheap = dev.new_heap(qkv_w + 2 * qkv_s + 4096);
    let qkbuf = qkheap.new_buffer(qkv_w + 2 * qkv_s);
    copy_bytes(
        &qkbuf,
        0,
        &shard,
        find(&shard, &(LAYER.to_owned() + "in_proj_qkv.weight")),
    );
    copy_bytes(
        &qkbuf,
        qkv_w,
        &shard,
        find(&shard, &(LAYER.to_owned() + "in_proj_qkv.scales")),
    );
    copy_bytes(
        &qkbuf,
        qkv_w + qkv_s,
        &shard,
        find(&shard, &(LAYER.to_owned() + "in_proj_qkv.biases")),
    );

    let zw: usize = VALUE_DIM * 640 * 4;
    let zs: usize = VALUE_DIM * 80 * 2;
    let zheap = dev.new_heap(zw + 2 * zs + 4096);
    let zbuf = zheap.new_buffer(zw + 2 * zs);
    copy_bytes(
        &zbuf,
        0,
        &shard,
        find(&shard, &(LAYER.to_owned() + "in_proj_z.weight")),
    );
    copy_bytes(
        &zbuf,
        zw,
        &shard,
        find(&shard, &(LAYER.to_owned() + "in_proj_z.scales")),
    );
    copy_bytes(
        &zbuf,
        zw + zs,
        &shard,
        find(&shard, &(LAYER.to_owned() + "in_proj_z.biases")),
    );

    // in_proj_a/b are 48-output (not a multiple of 32): pad weights/scales to 64
    // rows so the NAX gemm's 32-wide output tiles stay in-bounds.
    let (awv, asv, abv) = affine_padded_bytes(
        &shard,
        find(&shard, &(LAYER.to_owned() + "in_proj_a.weight")),
        find(&shard, &(LAYER.to_owned() + "in_proj_a.scales")),
        find(&shard, &(LAYER.to_owned() + "in_proj_a.biases")),
        VH,
        64,
        H,
    );
    let (bwv, bsv, bbv) = affine_padded_bytes(
        &shard,
        find(&shard, &(LAYER.to_owned() + "in_proj_b.weight")),
        find(&shard, &(LAYER.to_owned() + "in_proj_b.scales")),
        find(&shard, &(LAYER.to_owned() + "in_proj_b.biases")),
        VH,
        64,
        H,
    );
    let (aw, as_, ab_) = (awv.len(), asv.len(), abv.len());
    let ba4 = aw + as_ + ab_;
    let aheap = dev.new_heap(ba4 + 4096);
    let a_buf = aheap.new_buffer(ba4);
    a_buf.write_bytes(u(0), &awv);
    a_buf.write_bytes(u(aw), &asv);
    a_buf.write_bytes(u(aw + as_), &abv);
    let bheap = dev.new_heap(ba4 + 4096);
    let b_buf = bheap.new_buffer(ba4);
    b_buf.write_bytes(u(0), &bwv);
    b_buf.write_bytes(u(aw), &bsv);
    b_buf.write_bytes(u(aw + as_), &bbv);

    let ow: usize = H * 768 * 4;
    let os: usize = H * 96 * 2;
    let oheap = dev.new_heap(ow + 2 * os + 4096);
    let obuf = oheap.new_buffer(ow + 2 * os);
    copy_bytes(
        &obuf,
        0,
        &shard,
        find(&shard, &(LAYER.to_owned() + "out_proj.weight")),
    );
    copy_bytes(
        &obuf,
        ow,
        &shard,
        find(&shard, &(LAYER.to_owned() + "out_proj.scales")),
    );
    copy_bytes(
        &obuf,
        ow + os,
        &shard,
        find(&shard, &(LAYER.to_owned() + "out_proj.biases")),
    );

    // ---- activation heap ----
    let hb = M * H * 2;
    let qf = M * CONV_DIM * 4;
    let zf = M * VALUE_DIM * 4;
    let vf = M * 64 * 4; // a/b outputs padded to 64 cols (gemm 32-wide tiles)
    let be48 = M * VH * 4; // beta/decay, 48 rows
    let qn = M * KEY_DIM * 4;
    let kn = M * KEY_DIM * 4;
    let state = VH * HD * HD * 4;
    let yf = M * VALUE_DIM * 4;
    let gated = M * VALUE_DIM * 2;
    let oo = M * H * 4;
    let cw = CONV_DIM * 4 * 4;
    let alog = VH * 4;
    let dtb = VH * 4;
    let nw = HD * 4;
    let cst = 3 * CONV_DIM * 4;
    let sz = hb
        + qf
        + zf
        + vf
        + vf
        + qf
        + qn
        + kn
        + be48
        + be48
        + state
        + yf
        + gated
        + oo
        + cw
        + alog
        + dtb
        + nw
        + cst
        + 4096;
    let ah = dev.new_heap(sz);
    let ab = ah.new_buffer(sz);

    let mut off = 0usize;
    let h0 = off;
    off += hb;
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
    let cw0 = off;
    off += cw;
    let al0 = off;
    off += alog;
    let dt0 = off;
    off += dtb;
    let nw0 = off;
    off += nw;
    let cs0 = off;
    off += cst;
    assert!(off <= sz);

    // deterministic hidden
    let mut hv = Vec::new();
    for i in 0..(M * H) {
        let v = (i % 97) as f32 * 0.001;
        hv.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
    }
    ab.write_bytes(u64::try_from(h0).unwrap(), &hv);
    copy_bf16_as_f32(
        &ab,
        cw0,
        &shard,
        find(&shard, &(LAYER.to_owned() + "conv1d.weight")),
    );
    copy_bf16_as_f32(
        &ab,
        al0,
        &shard,
        find(&shard, &(LAYER.to_owned() + "A_log")),
    );
    copy_bf16_as_f32(
        &ab,
        dt0,
        &shard,
        find(&shard, &(LAYER.to_owned() + "dt_bias")),
    );
    copy_bf16_as_f32(
        &ab,
        nw0,
        &shard,
        find(&shard, &(LAYER.to_owned() + "norm.weight")),
    );
    // conv state zero (fresh) + state zero
    let zeros = [0u8; 4096];
    for offz in [cs0, st0] {
        let mut left = if offz == cs0 { cst } else { state };
        let mut at = offz;
        while left > 0 {
            let w = left.min(4096);
            ab.write_bytes(u64::try_from(at).unwrap(), &zeros[..w]);
            at += w;
            left -= w;
        }
    }

    let cb = q.new_command_buffer();
    let enc = cb.compute_compute_encoder();
    gemm(
        &enc,
        &nax_pipe,
        &ab,
        h0,
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
        h0,
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
        h0,
        &a_buf,
        0,
        &a_buf,
        aw,
        &a_buf,
        aw + as_,
        &ab,
        a0,
        64,
        H as u32,
    );
    gemm(
        &enc,
        &nax_pipe,
        &ab,
        h0,
        &b_buf,
        0,
        &b_buf,
        aw,
        &b_buf,
        aw + as_,
        &ab,
        b0,
        64,
        H as u32,
    );

    // conv + norm + gates
    let inv = 0.088_388_346_f32; // 1/sqrt(128)
    enc.set_compute_pipeline_state(&conv_p);
    enc.set_buffer(&ab, u64::try_from(q0).unwrap(), 0);
    enc.set_buffer(&ab, u64::try_from(b0).unwrap(), 1);
    enc.set_buffer(&ab, u64::try_from(a0).unwrap(), 2);
    enc.set_buffer(&ab, u64::try_from(cw0).unwrap(), 3);
    enc.set_buffer(&ab, u64::try_from(cs0).unwrap(), 4);
    enc.set_buffer(&ab, u64::try_from(al0).unwrap(), 5);
    enc.set_buffer(&ab, u64::try_from(dt0).unwrap(), 6);
    enc.set_buffer(&ab, u64::try_from(co0).unwrap(), 7);
    enc.set_buffer(&ab, u64::try_from(qn0).unwrap(), 8);
    enc.set_buffer(&ab, u64::try_from(kn0).unwrap(), 9);
    enc.set_buffer(&ab, u64::try_from(be0).unwrap(), 10);
    enc.set_buffer(&ab, u64::try_from(de0).unwrap(), 11);
    let mut dims = Vec::new();
    dims.extend_from_slice(&u32_le(&[KH as u32, VH as u32, HD as u32, HD as u32]));
    enc.set_bytes(&dims, 12);
    let mut scales = Vec::new();
    scales.extend_from_slice(&f32_le(&[inv * inv, inv]));
    enc.set_bytes(&scales, 13);
    let mut batch = Vec::new();
    batch.extend_from_slice(&(M as u32).to_le_bytes());
    enc.set_bytes(&batch, 14);
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

    // recurrence
    enc.set_compute_pipeline_state(&rec_p);
    enc.set_buffer(&ab, u64::try_from(co0).unwrap(), 0);
    enc.set_buffer(&ab, u64::try_from(qn0).unwrap(), 1);
    enc.set_buffer(&ab, u64::try_from(kn0).unwrap(), 2);
    enc.set_buffer(&ab, u64::try_from(be0).unwrap(), 3);
    enc.set_buffer(&ab, u64::try_from(de0).unwrap(), 4);
    enc.set_buffer(&ab, u64::try_from(st0).unwrap(), 5);
    enc.set_buffer(&ab, u64::try_from(y0).unwrap(), 6);
    enc.set_buffer(&ab, u64::try_from(st0).unwrap(), 9);
    let mut rdims = Vec::new();
    rdims.extend_from_slice(&u32_le(&[
        VH as u32,
        HD as u32,
        HD as u32,
        (VH / KH) as u32,
    ]));
    enc.set_bytes(&rdims, 7);
    let mut steps = Vec::new();
    steps.extend_from_slice(&(M as u32).to_le_bytes());
    enc.set_bytes(&steps, 8);
    enc.set_bytes(&0u32.to_le_bytes(), 10);
    enc.set_buffer(&ab, u64::try_from(st0).unwrap(), 9);
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

    // rms_gate
    enc.set_compute_pipeline_state(&rg_p);
    enc.set_buffer(&ab, u64::try_from(y0).unwrap(), 0);
    enc.set_buffer(&ab, u64::try_from(z0).unwrap(), 1);
    enc.set_buffer(&ab, u64::try_from(nw0).unwrap(), 2);
    enc.set_buffer(&ab, u64::try_from(ga0).unwrap(), 3);
    let mut rd = Vec::new();
    rd.extend_from_slice(&u32_le(&[VH as u32, HD as u32, M as u32]));
    enc.set_bytes(&rd, 4);
    let mut e = Vec::new();
    e.extend_from_slice(&EPS.to_le_bytes());
    enc.set_bytes(&e, 5);
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

    // BF16 gated output feeds the quantized projection without a CPU readback.
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
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let out = ab.read_bytes(u64::try_from(oo0).unwrap(), M * H * 4);
    let mut finite = true;
    let mut maxabs = 0.0f32;
    for i in 0..(M * H) {
        let v = f32::from_le_bytes([out[i * 4], out[i * 4 + 1], out[i * 4 + 2], out[i * 4 + 3]]);
        if !v.is_finite() {
            finite = false;
        }
        maxabs = maxabs.max(v.abs());
    }
    eprintln!("gdn layer17 o_out max|.| = {maxabs}, all_finite={finite}");
    assert!(finite, "gdn output must be finite");

    if let Ok(raw) = std::fs::read("/tmp/mlx_gdn_out.raw") {
        let n = M * H;
        assert_eq!(raw.len(), n * 4, "ref size");
        let mut md = 0.0f32;
        for i in 0..n {
            let mine =
                f32::from_le_bytes([out[i * 4], out[i * 4 + 1], out[i * 4 + 2], out[i * 4 + 3]]);
            let refv =
                f32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
            md = md.max((mine - refv).abs());
        }
        eprintln!("gdn vs MLX max abs diff = {md}");
        assert!(md < 2.5e-2, "gdn output drifted from MLX by {md}");
    } else {
        eprintln!("skip MLX comparison (run scripts/mlx_gdn_ref.py first)");
    }
}
