// Full-attention decoder layer (layer 3, 0-indexed) through the NAX tensor-core
// on the REAL 27B weights, mirroring the mlp_decode wiring pattern (per-
// projection heaps < ~50 MiB, shared activation heap):
//   q = h·q_proj^T  (M,12288)   k = h·k_proj^T  (M,1024)   v = h·v_proj^T (M,1024)
//   qn,gate = qk_norm_gate_rope(q)   ok = k_norm_rope(k)     (proportional RoPE)
//   attn = sdpa_scalar(qn, ok, v)                            (causal, scalar)
//   gated = gate_out_bf16(attn, gate) = attn * sigmoid(gate)
//   o = gated · o_proj^T  (M,5120)
// Correctness-first: scalar kernels, single thread per (row,head). Not yet
// anchored against the actual mlx-vlm attention module.
use std::fs::read_dir;
use std::path::{Path, PathBuf};

use lisa_rs::device::metal::{ComputeEncoder, ComputePipeline, MTLSize, MetalBuffer, MetalDevice};
use lisa_rs::format::dtype::f32_to_bf16;
use lisa_rs::format::safetensor as st;
use lisa_rs::kernels::linear::ATTENTION_SHADER;
use lisa_rs::kernels::linear::NAX_AFFINE_U4_SHADER;

const M: usize = 16;
const H: usize = 5120;
const LAYER: &str = "language_model.model.layers.3.self_attn.";
const EPS: f32 = 1e-6;

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

fn copy_triple(buf: &MetalBuffer, db: &[u8], shard: &st::Shard, t: &st::Tensor, dest: u64) {
    let start = usize::try_from(t.off - shard.data_start).unwrap();
    let len = usize::try_from(t.len).unwrap();
    buf.write_bytes(dest, &db[start..start + len]);
}

#[allow(clippy::too_many_arguments)]
fn gemm(
    enc: &ComputeEncoder,
    pipe: &ComputePipeline,
    a: &MetalBuffer,
    a_off: u64,
    w: &MetalBuffer,
    w_off: u64,
    s: &MetalBuffer,
    s_off: u64,
    b: &MetalBuffer,
    b_off: u64,
    c: &MetalBuffer,
    c_off: u64,
    n: u32,
    k: u32,
) {
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(a, a_off, 0);
    enc.set_buffer(w, w_off, 1);
    enc.set_buffer(s, s_off, 2);
    enc.set_buffer(b, b_off, 3);
    enc.set_buffer(c, c_off, 4);
    let mut nk = Vec::new();
    nk.extend_from_slice(&n.to_le_bytes());
    nk.extend_from_slice(&k.to_le_bytes());
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

fn uq16le(b: &[u8]) -> u16 {
    (b[0] as u16) | ((b[1] as u16) << 8)
}

#[test]
fn attn_decode_layer3() {
    let home = std::env::home_dir().expect("home");
    let root =
        home.join(".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots");
    let mut paths = Vec::new();
    shard_paths(&root, &mut paths);
    if paths.is_empty() {
        eprintln!("skip: model not cached");
        return;
    }
    let target = LAYER.to_owned() + "q_proj.weight";
    let mut shard: Option<st::Shard> = None;
    for p in &paths {
        let s = st::open(Path::new(&p));
        if s.tensors.iter().any(|t| t.name == target) {
            shard = Some(s);
            break;
        }
    }
    let shard = shard.expect("layer3 not found");
    let dbg = st::data_bytes(&shard);

    let dev = MetalDevice::default();
    let nax = dev.new_library(NAX_AFFINE_U4_SHADER);
    let nax_pipe = dev.new_compute_pipeline(&nax.function_named("q4_gemm_nax_coop"));
    let atn = dev.new_library(ATTENTION_SHADER);
    let qk_pipe = dev.new_compute_pipeline(&atn.function_named("qk_norm_gate_rope"));
    let krop_pipe = dev.new_compute_pipeline(&atn.function_named("k_norm_rope"));
    let sdpa_pipe = dev.new_compute_pipeline(&atn.function_named("sdpa_scalar"));
    let gate_pipe = dev.new_compute_pipeline(&atn.function_named("gate_out_bf16"));
    let qq = dev.new_command_queue();

    // ---- per-projection heaps ----
    let qw = LAYER.to_owned() + "q_proj.weight";
    let qs = LAYER.to_owned() + "q_proj.scales";
    let qb = LAYER.to_owned() + "q_proj.biases";
    let kw = LAYER.to_owned() + "k_proj.weight";
    let ks = LAYER.to_owned() + "k_proj.scales";
    let kb = LAYER.to_owned() + "k_proj.biases";
    let vw = LAYER.to_owned() + "v_proj.weight";
    let vs = LAYER.to_owned() + "v_proj.scales";
    let vb = LAYER.to_owned() + "v_proj.biases";
    let ow = LAYER.to_owned() + "o_proj.weight";
    let os = LAYER.to_owned() + "o_proj.scales";
    let ob = LAYER.to_owned() + "o_proj.biases";

    let qqw = find(&shard, &qw);
    let qqs = find(&shard, &qs);
    let qqb = find(&shard, &qb);
    const QOUT: usize = 12288;
    const KVOUT: usize = 1024;
    const OOUT: usize = 5120;
    let q_wbytes: usize = QOUT * (usize::try_from(qqw.shape[1]).unwrap()) * 4;
    let q_sbytes: usize = QOUT * (usize::try_from(qqs.shape[1]).unwrap()) * 2;

    let qheap = dev.new_heap(q_wbytes + 2 * q_sbytes + 4096);
    let qbuf = qheap.new_buffer(q_wbytes + 2 * q_sbytes);
    copy_triple(&qbuf, dbg, &shard, qqw, 0);
    copy_triple(&qbuf, dbg, &shard, qqs, u64::try_from(q_wbytes).unwrap());
    copy_triple(
        &qbuf,
        dbg,
        &shard,
        qqb,
        u64::try_from(q_wbytes + q_sbytes).unwrap(),
    );

    let kw_ = find(&shard, &kw);
    let kv_wbytes: usize = KVOUT * (usize::try_from(kw_.shape[1]).unwrap()) * 4;
    let kv_sbytes: usize = KVOUT * (usize::try_from(find(&shard, &ks).shape[1]).unwrap()) * 2;
    let kheap = dev.new_heap(kv_wbytes + 2 * kv_sbytes + 4096);
    let kbuf = kheap.new_buffer(kv_wbytes + 2 * kv_sbytes);
    copy_triple(&kbuf, dbg, &shard, kw_, 0);
    copy_triple(
        &kbuf,
        dbg,
        &shard,
        find(&shard, &ks),
        u64::try_from(kv_wbytes).unwrap(),
    );
    copy_triple(
        &kbuf,
        dbg,
        &shard,
        find(&shard, &kb),
        u64::try_from(kv_wbytes + kv_sbytes).unwrap(),
    );

    let vheap = dev.new_heap(kv_wbytes + 2 * kv_sbytes + 4096);
    let vbuf = vheap.new_buffer(kv_wbytes + 2 * kv_sbytes);
    copy_triple(&vbuf, dbg, &shard, find(&shard, &vw), 0);
    copy_triple(
        &vbuf,
        dbg,
        &shard,
        find(&shard, &vs),
        u64::try_from(kv_wbytes).unwrap(),
    );
    copy_triple(
        &vbuf,
        dbg,
        &shard,
        find(&shard, &vb),
        u64::try_from(kv_wbytes + kv_sbytes).unwrap(),
    );

    let ow_ = find(&shard, &ow);
    let o_wbytes: usize = OOUT * (usize::try_from(ow_.shape[1]).unwrap()) * 4;
    let o_sbytes: usize = OOUT * (usize::try_from(find(&shard, &os).shape[1]).unwrap()) * 2;
    let oheap = dev.new_heap(o_wbytes + 2 * o_sbytes + 4096);
    let obuf = oheap.new_buffer(o_wbytes + 2 * o_sbytes);
    copy_triple(&obuf, dbg, &shard, ow_, 0);
    copy_triple(
        &obuf,
        dbg,
        &shard,
        find(&shard, &os),
        u64::try_from(o_wbytes).unwrap(),
    );
    copy_triple(
        &obuf,
        dbg,
        &shard,
        find(&shard, &ob),
        u64::try_from(o_wbytes + o_sbytes).unwrap(),
    );

    // q_norm / k_norm (bf16 [256])
    let qn = find(&shard, &(LAYER.to_owned() + "q_norm.weight"));
    let kn = find(&shard, &(LAYER.to_owned() + "k_norm.weight"));

    // ---- activation heap ----
    let hb = M * H * 2;
    let qf = M * QOUT * 4;
    let kvf = M * KVOUT * 4;
    let oqb = M * QOUT * 2; // query/gate are per-head-interleaved, oq is [M,6144]
    let ogf = M * 6144 * 4;
    let okb = M * 2048 * 2;
    let aof = M * 6144 * 4;
    let gatedb = M * 6144 * 2;
    let oof = M * OOUT * 4;
    let nn = 256 * 2;
    let asz = hb + qf + 2 * kvf + oqb + ogf + okb + aof + gatedb + oof + nn + nn + 4096;
    let ah = dev.new_heap(asz);
    let ab = ah.new_buffer(asz);

    let mut off = 0usize;
    let h0 = off;
    off += hb;
    let q0 = off;
    off += qf;
    let k0 = off;
    off += kvf;
    let v0 = off;
    off += kvf;
    let oq0 = off;
    off += oqb;
    let og0 = off;
    off += ogf;
    let ok0 = off;
    off += okb;
    let ao0 = off;
    off += aof;
    let g0 = off;
    off += gatedb;
    let oo0 = off;
    off += oof;
    let qn0 = off;
    off += nn;
    let kn0 = off;
    assert!(kn0 + nn <= asz);

    // deterministic hidden m*5120
    let mut hv = Vec::new();
    for i in 0..(M * H) {
        let v = (i % 97) as f32 * 0.001;
        hv.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
    }
    ab.write_bytes(u64::try_from(h0).unwrap(), &hv);
    copy_triple(&ab, dbg, &shard, qn, u64::try_from(qn0).unwrap());
    copy_triple(&ab, dbg, &shard, kn, u64::try_from(kn0).unwrap());

    // ---- encode ----
    let cb = qq.new_command_buffer();
    let enc = cb.compute_compute_encoder();
    // Q/K/V projections (NAX)
    gemm(
        &enc,
        &nax_pipe,
        &ab,
        u64::try_from(h0).unwrap(),
        &qbuf,
        0,
        &qbuf,
        u64::try_from(q_wbytes).unwrap(),
        &qbuf,
        u64::try_from(q_wbytes + q_sbytes).unwrap(),
        &ab,
        u64::try_from(q0).unwrap(),
        QOUT as u32,
        H as u32,
    );
    gemm(
        &enc,
        &nax_pipe,
        &ab,
        u64::try_from(h0).unwrap(),
        &kbuf,
        0,
        &kbuf,
        u64::try_from(kv_wbytes).unwrap(),
        &kbuf,
        u64::try_from(kv_wbytes + kv_sbytes).unwrap(),
        &ab,
        u64::try_from(k0).unwrap(),
        KVOUT as u32,
        H as u32,
    );
    gemm(
        &enc,
        &nax_pipe,
        &ab,
        u64::try_from(h0).unwrap(),
        &vbuf,
        0,
        &vbuf,
        u64::try_from(kv_wbytes).unwrap(),
        &vbuf,
        u64::try_from(kv_wbytes + kv_sbytes).unwrap(),
        &ab,
        u64::try_from(v0).unwrap(),
        KVOUT as u32,
        H as u32,
    );

    // qk norm+split+rope ; k norm+rope
    enc.set_compute_pipeline_state(&qk_pipe);
    enc.set_buffer(&ab, u64::try_from(q0).unwrap(), 0);
    enc.set_buffer(&ab, u64::try_from(qn0).unwrap(), 1);
    enc.set_buffer(&ab, u64::try_from(oq0).unwrap(), 2);
    enc.set_buffer(&ab, u64::try_from(og0).unwrap(), 3);
    let mut lbytes = Vec::new();
    lbytes.extend_from_slice(&(M as u32).to_le_bytes());
    enc.set_bytes(&lbytes, 4);
    let mut ebytes = Vec::new();
    ebytes.extend_from_slice(&EPS.to_le_bytes());
    enc.set_bytes(&ebytes, 5);
    enc.dispatch_thread_groups(
        MTLSize {
            width: M as u64,
            height: 24,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );

    enc.set_compute_pipeline_state(&krop_pipe);
    enc.set_buffer(&ab, u64::try_from(k0).unwrap(), 0);
    enc.set_buffer(&ab, u64::try_from(kn0).unwrap(), 1);
    enc.set_buffer(&ab, u64::try_from(ok0).unwrap(), 2);
    enc.set_bytes(&lbytes, 3);
    enc.set_bytes(&ebytes, 4);
    enc.dispatch_thread_groups(
        MTLSize {
            width: M as u64,
            height: 4,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );

    // scalar sdpa
    enc.set_compute_pipeline_state(&sdpa_pipe);
    enc.set_buffer(&ab, u64::try_from(oq0).unwrap(), 0);
    enc.set_buffer(&ab, u64::try_from(ok0).unwrap(), 1);
    enc.set_buffer(&ab, u64::try_from(v0).unwrap(), 2);
    enc.set_buffer(&ab, u64::try_from(ao0).unwrap(), 3);
    enc.set_bytes(&lbytes, 4);
    enc.dispatch_thread_groups(
        MTLSize {
            width: M as u64,
            height: 24,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );

    // gate * sigmoid -> bf16
    enc.set_compute_pipeline_state(&gate_pipe);
    enc.set_buffer(&ab, u64::try_from(ao0).unwrap(), 0);
    enc.set_buffer(&ab, u64::try_from(og0).unwrap(), 1);
    enc.set_buffer(&ab, u64::try_from(g0).unwrap(), 2);
    let ng = (M * 6144) as u32;
    let mut ngbytes = Vec::new();
    ngbytes.extend_from_slice(&ng.to_le_bytes());
    enc.set_bytes(&ngbytes, 3);
    enc.dispatch_threads(
        MTLSize {
            width: ng as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );

    // o_proj on gated context
    gemm(
        &enc,
        &nax_pipe,
        &ab,
        u64::try_from(g0).unwrap(),
        &obuf,
        0,
        &obuf,
        u64::try_from(o_wbytes).unwrap(),
        &obuf,
        u64::try_from(o_wbytes + o_sbytes).unwrap(),
        &ab,
        u64::try_from(oo0).unwrap(),
        OOUT as u32,
        6144,
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let out = ab.read_bytes(u64::try_from(oo0).unwrap(), M * OOUT * 4);
    let mut finite = true;
    let mut maxabs = 0.0f32;
    for k in 0..(M * OOUT) {
        let b: [u8; 4] = [out[k * 4], out[k * 4 + 1], out[k * 4 + 2], out[k * 4 + 3]];
        let v = f32::from_le_bytes(b);
        if !v.is_finite() {
            finite = false;
        }
        maxabs = maxabs.max(v.abs());
    }
    eprintln!("attn layer3 o_out max|.| = {maxabs}, all_finite={finite}");
    assert!(finite, "attention output must be finite");

    // Tightest signal: the pre-o_proj gated context must match the mlx-vlm truth
    // (scripts/mlx_attn_truth.py -> /tmp/mlx_attn_gated_truth.raw). Compare in f32;
    // expect ~1 ulp bf16 quantization of our stored gated (a real divergence shows
    // clearly here, before the o_proj GEMM can amplify it back).
    let mut gated_md = 0.0f32;
    if let Ok(gref) = std::fs::read("/tmp/mlx_attn_gated_truth.raw") {
        let g = ab.read_bytes(u64::try_from(g0).unwrap(), M * 6144 * 2);
        let n = M * 6144;
        assert_eq!(gref.len(), n * 4, "gated truth size");
        for i in 0..n {
            let rv = f32::from_le_bytes([
                gref[i * 4],
                gref[i * 4 + 1],
                gref[i * 4 + 2],
                gref[i * 4 + 3],
            ]);
            let mv = f32::from_bits((uq16le(&g[i * 2..i * 2 + 2]) as u32) << 16);
            gated_md = gated_md.max((mv - rv).abs());
        }
    }
    eprintln!("gated vs mlx-vlm truth max abs diff = {gated_md}");
    assert!(
        gated_md < 2e-3,
        "attention gated context drifted from mlx-vlm truth by {gated_md}"
    );

    // Anchored correctness vs mlx-vlm ground truth (scripts/mlx_attn_truth.py -> /tmp/mlx_attn_truth.raw)
    if let Ok(raw) = std::fs::read("/tmp/mlx_attn_truth.raw") {
        let n = M * OOUT;
        assert_eq!(raw.len(), n * 4, "truth size");
        let mut max_diff = 0.0f32;
        for i in 0..n {
            let mine =
                f32::from_le_bytes([out[i * 4], out[i * 4 + 1], out[i * 4 + 2], out[i * 4 + 3]]);
            let refv =
                f32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
            let d = (mine - refv).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        eprintln!("attention vs mlx-vlm truth max abs diff = {max_diff}");
        assert!(
            max_diff < 5e-2,
            "attention output drifted from mlx-vlm truth by {max_diff}"
        );
    } else {
        eprintln!("skip mlx-vlm comparison (run scripts/mlx_attn_truth.py first)");
    }
}
