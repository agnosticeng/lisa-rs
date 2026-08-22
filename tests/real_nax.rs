// Dispatch the NAX tensor-core affine-u4 GEMM on a real MLX projection
// (down_proj, layer 17). Skips if the HF cache isn't present.
use std::fs::read_dir;
use std::path::{Path, PathBuf};

use lisa_rs::device::metal::{MTLSize, MetalDevice};
use lisa_rs::format::dtype::f32_to_bf16;
use lisa_rs::format::safetensor as st;
use lisa_rs::kernels::linear::NAX_AFFINE_U4_SHADER;

const ROWS: usize = 16;
const NO: usize = 5120;
const KI: usize = 17408;

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

#[test]
fn real_down_proj_nax_gemm() {
    let home = std::env::home_dir().expect("home");
    let root =
        home.join(".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots");
    let mut paths = Vec::new();
    shard_paths(&root, &mut paths);
    if paths.is_empty() {
        eprintln!("skip: model not cached");
        return;
    }
    let prefix = "language_model.model.layers.17.mlp.down_proj.";
    let target = format!("{}weight", prefix);
    let mut model: Option<st::Shard> = None;
    for p in &paths {
        let s = st::open(Path::new(&p));
        if s.tensors.iter().any(|t| t.name == target) {
            model = Some(s);
            break;
        }
    }
    let m = model.expect("down_proj.weight");

    let wt = m
        .tensors
        .iter()
        .find(|t| t.name == format!("{}weight", prefix))
        .unwrap();
    let sc = m
        .tensors
        .iter()
        .find(|t| t.name == format!("{}scales", prefix))
        .unwrap();
    let bi = m
        .tensors
        .iter()
        .find(|t| t.name == format!("{}biases", prefix))
        .unwrap();
    assert_eq!(wt.shape[0], NO as u64);
    assert_eq!(wt.shape[1], (KI / 8) as u64);

    let half = f32_to_bf16(0.5f32);
    let mut av = Vec::new();
    for _ in 0..(ROWS * KI) {
        av.extend_from_slice(&half.to_le_bytes());
    }

    let dev = MetalDevice::default();
    let lib = dev.new_library(NAX_AFFINE_U4_SHADER);
    let pipe = dev.new_compute_pipeline(&lib.function_named("q4_gemm_nax_coop"));
    let q = dev.new_command_queue();

    let ab = ROWS * KI * 2;
    let wb = NO * (KI / 8) * 4;
    let sb = NO * (KI / 64) * 2;
    let clog = ROWS * NO * 4;
    let total = ab + wb + sb + sb + clog + 8192;

    let heap = dev.new_heap(total);
    let buf = heap.new_buffer(total);
    let a0 = 0usize;
    let w0 = a0 + ab;
    let s0 = w0 + wb;
    let b0 = s0 + sb;
    let c0 = b0 + sb;

    let db = st::data_bytes(&m);
    let cw = usize::try_from(wt.off - m.data_start).unwrap();
    buf.write_bytes(u64v(w0), &db[cw..cw + usize::try_from(wt.len).unwrap()]);
    buf.write_bytes(
        u64v(s0),
        &db[usize::try_from(sc.off - m.data_start).unwrap()..][..usize::try_from(sc.len).unwrap()],
    );
    buf.write_bytes(
        u64v(b0),
        &db[usize::try_from(bi.off - m.data_start).unwrap()..][..usize::try_from(bi.len).unwrap()],
    );
    buf.write_bytes(u64v(a0), &av);

    let cb = q.new_command_buffer();
    let enc = cb.compute_compute_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(&buf, u64v(a0), 0);
    enc.set_buffer(&buf, u64v(w0), 1);
    enc.set_buffer(&buf, u64v(s0), 2);
    enc.set_buffer(&buf, u64v(b0), 3);
    enc.set_buffer(&buf, u64v(c0), 4);
    let mut nk = Vec::new();
    nk.extend_from_slice(&(NO as u32).to_le_bytes());
    nk.extend_from_slice(&(KI as u32).to_le_bytes());
    nk.extend_from_slice(&(ROWS as u32).to_le_bytes());
    enc.set_bytes(&nk, 5);
    let tm = (ROWS as u64).div_ceil(16);
    let tn = (NO as u64).div_ceil(32);
    enc.dispatch_thread_groups(
        MTLSize {
            width: tm,
            height: tn,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let c = buf.read_bytes(u64v(c0), clog);
    let mut v: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
    for k in 0..4 {
        let b: [u8; 4] = [c[k * 4], c[k * 4 + 1], c[k * 4 + 2], c[k * 4 + 3]];
        v[k] = f32::from_le_bytes(b);
    }
    eprintln!("down_proj C[0,:4] = {} {} {} {}", v[0], v[1], v[2], v[3]);
    assert!(v[0].is_finite(), "expected finite logits");
}
