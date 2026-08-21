// One full MLP decoder layer (layer 17) through the NAX tensor-core:
//   gate = h·gate_proj^T, up = h·up_proj^T (affine-u4 GEMMs)
//   si = silu(gate) * up  (bf16)
//   out = si·down_proj^T
// Allocations are split into per-projection heaps (<~50 MiB each, the size the
// driver sub-allocator accepts) — real 27B weights; skip if not cached.
use std::fs::read_dir;
use std::path::{Path, PathBuf};

use lisa_rs::device::metal::{ComputeEncoder, ComputePipeline, MTLSize, MetalBuffer, MetalDevice};
use lisa_rs::format::dtype::f32_to_bf16;
use lisa_rs::format::safetensor as st;
use lisa_rs::kernels::linear::{MLP_SHADER, NAX_AFFINE_U4_SHADER};

const M: usize = 16;
const H: usize = 5120;
const INTER: usize = 17408;

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

fn u64v(v: usize) -> u64 {
    u64::try_from(v).expect("u64")
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
fn run_gemm(
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

#[test]
fn mlp_decode_layer17() {
    let home = std::env::home_dir().expect("home");
    let root =
        home.join(".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots");
    let mut paths = Vec::new();
    shard_paths(&root, &mut paths);
    if paths.is_empty() {
        eprintln!("skip: model not cached");
        return;
    }
    let prefix = "language_model.model.layers.17.mlp.";
    let target = prefix.to_owned() + "gate_proj.weight";
    let mut shard: Option<st::Shard> = None;
    for p in &paths {
        let s = st::open(Path::new(&p));
        if s.tensors.iter().any(|t| t.name == target) {
            shard = Some(s);
            break;
        }
    }
    let shard = shard.expect("layer17 not found");

    let gw = find(&shard, &(prefix.to_owned() + "gate_proj.weight"));
    let gs = find(&shard, &(prefix.to_owned() + "gate_proj.scales"));
    let gb = find(&shard, &(prefix.to_owned() + "gate_proj.biases"));
    let uw = find(&shard, &(prefix.to_owned() + "up_proj.weight"));
    let us = find(&shard, &(prefix.to_owned() + "up_proj.scales"));
    let ub = find(&shard, &(prefix.to_owned() + "up_proj.biases"));
    let dw = find(&shard, &(prefix.to_owned() + "down_proj.weight"));
    let ds = find(&shard, &(prefix.to_owned() + "down_proj.scales"));
    let db = find(&shard, &(prefix.to_owned() + "down_proj.biases"));

    let dev = MetalDevice::default();
    let nax = dev.new_library(NAX_AFFINE_U4_SHADER);
    let nax_pipe = dev.new_compute_pipeline(&nax.function_named("q4_gemm_nax_coop"));
    let mlp = dev.new_library(MLP_SHADER);
    let silu_pipe = dev.new_compute_pipeline(&mlp.function_named("silu_mul_up"));
    let q = dev.new_command_queue();

    // Dedicated shared buffers avoid heap subresource option mismatches between
    // the large weight allocations when integration tests run back-to-back.
    let wgi = INTER * (H / 8) * 4; // gate weight bytes
    let sgi = INTER * (H / 64) * 2; // gate scales/one bias block
    let gbuf = dev.new_buffer(wgi + 2 * sgi);
    let ubuf = dev.new_buffer(wgi + 2 * sgi);
    let wdi = H * (INTER / 8) * 4;
    let sdi = H * (INTER / 64) * 2;
    let dbuf = dev.new_buffer(wdi + 2 * sdi);
    // activations / outputs
    let ah = M * H * 2;
    let ai = M * INTER;
    let abuf = dev.new_buffer(ah + 2 * (ai * 4) + ai * 2 + M * H * 4);

    let dbg = st::data_bytes(&shard);
    copy_triple(&gbuf, dbg, &shard, gw, 0);
    copy_triple(&gbuf, dbg, &shard, gs, u64v(wgi));
    copy_triple(&gbuf, dbg, &shard, gb, u64v(wgi + sgi));
    copy_triple(&ubuf, dbg, &shard, uw, 0);
    copy_triple(&ubuf, dbg, &shard, us, u64v(wgi));
    copy_triple(&ubuf, dbg, &shard, ub, u64v(wgi + sgi));
    copy_triple(&dbuf, dbg, &shard, dw, 0);
    copy_triple(&dbuf, dbg, &shard, ds, u64v(wdi));
    copy_triple(&dbuf, dbg, &shard, db, u64v(wdi + sdi));

    // layout inside abuf: h, cg, cu, si, cd
    let h0 = 0usize;
    let cg0 = h0 + ah;
    let cu0 = cg0 + ai * 4;
    let si0 = cu0 + ai * 4;
    let cd0 = si0 + ai * 2;

    let mut hv = Vec::new();
    for i in 0..(M * H) {
        let v = (i % 97) as f32 * 0.001;
        hv.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
    }
    abuf.write_bytes(u64v(h0), &hv);

    let cb = q.new_command_buffer();
    let enc = cb.compute_compute_encoder();
    run_gemm(
        &enc,
        &nax_pipe,
        &abuf,
        u64v(h0),
        &gbuf,
        0,
        &gbuf,
        u64v(wgi),
        &gbuf,
        u64v(wgi + sgi),
        &abuf,
        u64v(cg0),
        INTER as u32,
        H as u32,
    );
    run_gemm(
        &enc,
        &nax_pipe,
        &abuf,
        u64v(h0),
        &ubuf,
        0,
        &ubuf,
        u64v(wgi),
        &ubuf,
        u64v(wgi + sgi),
        &abuf,
        u64v(cu0),
        INTER as u32,
        H as u32,
    );
    // silu(g)*u -> bf16
    enc.set_compute_pipeline_state(&silu_pipe);
    enc.set_buffer(&abuf, u64v(cg0), 0);
    enc.set_buffer(&abuf, u64v(cu0), 1);
    enc.set_buffer(&abuf, u64v(si0), 2);
    let mut nn = Vec::new();
    nn.extend_from_slice(&((M * INTER) as u32).to_le_bytes());
    enc.set_bytes(&nn, 3);
    enc.dispatch_threads(
        MTLSize {
            width: (M * INTER) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    run_gemm(
        &enc,
        &nax_pipe,
        &abuf,
        u64v(si0),
        &dbuf,
        0,
        &dbuf,
        u64v(wdi),
        &dbuf,
        u64v(wdi + sdi),
        &abuf,
        u64v(cd0),
        H as u32,
        INTER as u32,
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let out = abuf.read_bytes(u64v(cd0), M * H * 4);
    let mut v: [f32; 4] = [0.0; 4];
    for k in 0..4 {
        let b: [u8; 4] = [out[k * 4], out[k * 4 + 1], out[k * 4 + 2], out[k * 4 + 3]];
        v[k] = f32::from_le_bytes(b);
    }
    eprintln!(
        "mlp layer17 out[0,:4] = {} {} {} {}",
        v[0], v[1], v[2], v[3]
    );
    assert!(
        v[0].is_finite() && v[1].is_finite(),
        "MLP output must be finite"
    );

    // Anchored correctness vs MLX reference (scripts/mlx_mlp_ref.py -> /tmp/mlx_out.raw)
    if let Ok(raw) = std::fs::read("/tmp/mlx_out.raw") {
        let n = M * H;
        assert_eq!(raw.len(), n * 4, "ref size");
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
        eprintln!("MLP vs MLX max abs diff = {max_diff}");
        assert!(
            max_diff < 5e-4,
            "MLP output drifted from MLX reference by {max_diff}"
        );
    } else {
        eprintln!("skip MLX comparison (run scripts/mlx_mlp_ref.py first)");
    }
}
