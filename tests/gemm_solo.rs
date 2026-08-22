// Isolated tiled-GEMM micro-benchmark on real weights (layer-17 gate/down).
// Prints TFLOPS for q4_gemm_nax_tiled vs tiled_align vs coop at several M, and
// a correctness check (align vs coop bit-diff) on the gate projection.
// Run: cargo test --test gemm_solo --release -- --ignored --nocapture
use std::fs::read_dir;
use std::path::{Path, PathBuf};
use std::time::Instant;

use lisa_rs::device::metal::{ComputePipeline, MTLSize, MetalDevice};
use lisa_rs::format::dtype::f32_to_bf16;
use lisa_rs::format::safetensor as st;
use lisa_rs::kernels::linear::NAX_AFFINE_U4_SHADER;

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

struct Proj {
    w: Vec<u8>,
    s: Vec<u8>,
    b: Vec<u8>,
    n: usize,
    k: usize,
}

fn load_projection(shards: &[st::Shard], prefix: &str) -> Proj {
    let mut model: Option<&st::Shard> = None;
    for s in shards {
        if s.tensors.iter().any(|t| t.name == format!("{}weight", prefix)) {
            model = Some(s);
            break;
        }
    }
    let m = model.expect("proj");
    let wt = m.tensors.iter().find(|t| t.name == format!("{}weight", prefix)).unwrap();
    let sc = m.tensors.iter().find(|t| t.name == format!("{}scales", prefix)).unwrap();
    let bi = m.tensors.iter().find(|t| t.name == format!("{}biases", prefix)).unwrap();
    let n = wt.shape[0] as usize;
    let k = wt.shape[1] as usize * 8;
    let db = st::data_bytes(m);
    let cw = usize::try_from(wt.off - m.data_start).unwrap();
    let w = db[cw..cw + usize::try_from(wt.len).unwrap()].to_vec();
    let cs = usize::try_from(sc.off - m.data_start).unwrap();
    let s = db[cs..cs + usize::try_from(sc.len).unwrap()].to_vec();
    let cb = usize::try_from(bi.off - m.data_start).unwrap();
    let b = db[cb..cb + usize::try_from(bi.len).unwrap()].to_vec();
    Proj { w, s, b, n, k }
}

struct Harness {
    buf: lisa_rs::device::metal::MetalBuffer,
    a0: usize,
    w0: usize,
    s0: usize,
    b0: usize,
    c0: usize,
    mnk: Vec<u8>,
}

impl Harness {
    fn read_c(&self, clog: usize) -> Vec<f32> {
        self.buf
            .read_bytes(self.c0 as u64, clog)
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
    fn dispatch(
        &self,
        q: &lisa_rs::device::metal::CommandQueue,
        pipe: &ComputePipeline,
        w: u64,
        h: u64,
        tg: u64,
    ) {
        let cb = q.new_command_buffer();
        let enc = cb.compute_compute_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(&self.buf, u64v(self.a0), 0);
        enc.set_buffer(&self.buf, u64v(self.w0), 1);
        enc.set_buffer(&self.buf, u64v(self.s0), 2);
        enc.set_buffer(&self.buf, u64v(self.b0), 3);
        enc.set_buffer(&self.buf, u64v(self.c0), 4);
        enc.set_bytes(&self.mnk, 5);
        enc.dispatch_thread_groups(
            MTLSize { width: w, height: h, depth: 1 },
            MTLSize { width: tg, height: 1, depth: 1 },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    fn dispatch_loop(
        &self,
        q: &lisa_rs::device::metal::CommandQueue,
        pipe: &ComputePipeline,
        w: u64,
        h: u64,
        tg: u64,
        n: u32,
    ) -> f64 {
        let cb = q.new_command_buffer();
        let enc = cb.compute_compute_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(&self.buf, u64v(self.a0), 0);
        enc.set_buffer(&self.buf, u64v(self.w0), 1);
        enc.set_buffer(&self.buf, u64v(self.s0), 2);
        enc.set_buffer(&self.buf, u64v(self.b0), 3);
        enc.set_buffer(&self.buf, u64v(self.c0), 4);
        enc.set_bytes(&self.mnk, 5);
        for _ in 0..n {
            enc.dispatch_thread_groups(
                MTLSize { width: w, height: h, depth: 1 },
                MTLSize { width: tg, height: 1, depth: 1 },
            );
        }
        enc.end_encoding();
        let t0 = Instant::now();
        cb.commit();
        cb.wait_until_completed();
        t0.elapsed().as_secs_f64()
    }
}

fn make_harness(dev: &MetalDevice, m: usize, n: usize, k: usize, proj: &Proj) -> Harness {
    let ab = m * k * 2;
    let wb = n * (k / 8) * 4;
    let sb = n * (k / 64) * 2;
    let clog = m * n * 4;
    let total = ab + wb + sb + sb + clog + 8192;
    let heap = dev.new_heap(total);
    let buf = heap.new_buffer(total);
    let (a0, w0, s0, b0, c0) = (0, ab, ab + wb, ab + wb + sb, ab + wb + 2 * sb);
    buf.write_bytes(u64v(w0), &proj.w);
    buf.write_bytes(u64v(s0), &proj.s);
    buf.write_bytes(u64v(b0), &proj.b);
    let half = f32_to_bf16(0.5f32);
    let mut av = Vec::new();
    for _ in 0..(m * k) {
        av.extend_from_slice(&half.to_le_bytes());
    }
    buf.write_bytes(u64v(a0), &av);
    let mut mnk = Vec::with_capacity(12);
    for v in [m, n, k] {
        mnk.extend_from_slice(&(v as u32).to_le_bytes());
    }
    Harness { buf, a0, w0, s0, b0, c0, mnk }
}

#[test]
#[ignore = "loads real weights + benchmarks; diagnostic-only"]
fn gemm_tiled_vs_coop_profile() -> Result<(), String> {
    let home = std::env::home_dir().expect("home");
    let root =
        home.join(".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots");
    let mut paths = Vec::new();
    shard_paths(&root, &mut paths);
    if paths.is_empty() {
        eprintln!("skip: model not cached");
        return Ok(());
    }
    let shards: Vec<st::Shard> = paths.iter().map(|p| st::open(Path::new(p))).collect();
    let gate = load_projection(&shards, "language_model.model.layers.17.mlp.gate_proj.");
    let down = load_projection(&shards, "language_model.model.layers.17.mlp.down_proj.");

    let dev = MetalDevice::default();
    let lib = dev.new_library(NAX_AFFINE_U4_SHADER);
    let pipe_tiled = dev.new_compute_pipeline(&lib.function_named("q4_gemm_nax_tiled"));
    let pipe_tgual = dev.new_compute_pipeline(&lib.function_named("q4_gemm_nax_tiled_align"));
    let pipe_coop = dev.new_compute_pipeline(&lib.function_named("q4_gemm_nax_coop"));
    let q = dev.new_command_queue();

    // ---- correctness: align vs coop on gate M=512 ----
    let (gk, gout) = (gate.k, gate.n);
    let mm = 512usize;
    let hh = make_harness(&dev, mm, gout, gk, &gate);
    hh.dispatch(&q, &pipe_coop, (mm / 16) as u64, (gout / 32) as u64, 32);
    let coop_c = hh.read_c(mm * gout);
    hh.dispatch(&q, &pipe_tiled, (mm / 64) as u64, (gout / 64) as u64, 128);
    let tiled_c = hh.read_c(mm * gout);
    let maxd_t = tiled_c
        .iter()
        .zip(&coop_c)
        .fold(0.0f64, |a, (x, y)| a.max((*x as f64 - *y as f64).abs()));
    println!("tiled-vs-coop   gate M=512: max_abs_diff {maxd_t:.8}");
    hh.dispatch(&q, &pipe_tgual, (mm / 64) as u64, (gout / 64) as u64, 128);
    let algn_c = hh.read_c(mm * gout);
    let maxd = algn_c
        .iter()
        .zip(&coop_c)
        .fold(0.0f64, |a, (x, y)| a.max((*x as f64 - *y as f64).abs()));
    println!("align-vs-coop gate M=512: max_abs_diff {maxd:.8}");

    // ---- timing loop ----
    Ok(())
}