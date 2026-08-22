use std::fs::read_dir;
use std::path::{Path, PathBuf};

use lisa_rs::device::metal::{MTLSize, MetalBuffer, MetalDevice};
use lisa_rs::format::dtype::bf16_to_f32;
use lisa_rs::format::safetensor as st;
use lisa_rs::kernels::linear::{EMBED_SHADER, NAX_AFFINE_U4_SHADER, NORM_SHADER};

const TOKEN: usize = 2005;
const HIDDEN: usize = 5120;
const HEAD_ROWS: usize = 248320;

fn shard_paths(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in read_dir(dir).expect("read model cache") {
        let path = entry.expect("cache entry").path();
        if path.is_dir() {
            shard_paths(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "safetensors") {
            out.push(path);
        }
    }
}

fn find<'a>(shards: &'a [st::Shard], name: &str) -> (usize, &'a st::Tensor) {
    shards
        .iter()
        .enumerate()
        .find_map(|(index, shard)| {
            shard
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .map(|tensor| (index, tensor))
        })
        .unwrap_or_else(|| panic!("missing tensor {name}"))
}

fn tensor_bytes<'a>(shards: &'a [st::Shard], name: &str) -> &'a [u8] {
    let (index, tensor) = find(shards, name);
    let shard = &shards[index];
    let start = usize::try_from(tensor.off - shard.data_start).expect("tensor offset");
    let len = usize::try_from(tensor.len).expect("tensor length");
    &st::data_bytes(shard)[start..start + len]
}

fn copy_rows(
    dst: &MetalBuffer,
    offset: usize,
    shards: &[st::Shard],
    name: &str,
    first: usize,
    rows: usize,
) {
    let (_, tensor) = find(shards, name);
    let total_rows = usize::try_from(tensor.shape[0]).expect("rows");
    let row_bytes = usize::try_from(tensor.len).expect("length") / total_rows;
    let bytes = tensor_bytes(shards, name);
    let start = first * row_bytes;
    dst.write_bytes(offset as u64, &bytes[start..start + rows * row_bytes]);
}

fn max_diff_bf16(actual: &[u8], reference: &[u8]) -> f32 {
    actual
        .chunks_exact(2)
        .zip(reference.chunks_exact(4))
        .map(|(a, r)| {
            let av = bf16_to_f32(u16::from_le_bytes([a[0], a[1]]));
            let rv = f32::from_le_bytes([r[0], r[1], r[2], r[3]]);
            (av - rv).abs()
        })
        .fold(0.0, f32::max)
}

#[test]
fn embedding_norm_and_full_head() {
    let root = std::env::home_dir()
        .expect("home")
        .join(".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots");
    let mut paths = Vec::new();
    shard_paths(&root, &mut paths);
    if paths.is_empty() {
        eprintln!("skip: model not cached");
        return;
    }
    let shards: Vec<_> = paths.iter().map(|path| st::open(path)).collect();

    let embed_prefix = "language_model.model.embed_tokens";
    let head_prefix = "language_model.lm_head";
    let embed_weight = HIDDEN / 8 * 4;
    let embed_group = HIDDEN / 64 * 2;
    let head_weight = HEAD_ROWS * embed_weight;
    let head_group = HEAD_ROWS * embed_group;

    let dev = MetalDevice::default();
    let embed_pipe = dev.new_compute_pipeline(
        &dev.new_library(EMBED_SHADER)
            .function_named("affine_u4_lookup"),
    );
    let norm_pipe =
        dev.new_compute_pipeline(&dev.new_library(NORM_SHADER).function_named("rms_norm_rows"));
    let q4_pipe = dev.new_compute_pipeline(
        &dev.new_library(NAX_AFFINE_U4_SHADER)
            .function_named("q4_qmv_fast"),
    );

    let embed_buf = dev.new_buffer(embed_weight + 2 * embed_group);
    copy_rows(
        &embed_buf,
        0,
        &shards,
        &(embed_prefix.to_owned() + ".weight"),
        TOKEN,
        1,
    );
    copy_rows(
        &embed_buf,
        embed_weight,
        &shards,
        &(embed_prefix.to_owned() + ".scales"),
        TOKEN,
        1,
    );
    copy_rows(
        &embed_buf,
        embed_weight + embed_group,
        &shards,
        &(embed_prefix.to_owned() + ".biases"),
        TOKEN,
        1,
    );

    let head_buf = dev.new_buffer(head_weight + 2 * head_group);
    copy_rows(
        &head_buf,
        0,
        &shards,
        &(head_prefix.to_owned() + ".weight"),
        0,
        HEAD_ROWS,
    );
    copy_rows(
        &head_buf,
        head_weight,
        &shards,
        &(head_prefix.to_owned() + ".scales"),
        0,
        HEAD_ROWS,
    );
    copy_rows(
        &head_buf,
        head_weight + head_group,
        &shards,
        &(head_prefix.to_owned() + ".biases"),
        0,
        HEAD_ROWS,
    );

    let rows = 1;
    let hidden_bytes = rows * HIDDEN * 2;
    let logits_bytes = rows * HEAD_ROWS * 4;
    let activations = dev.new_buffer(hidden_bytes * 2 + logits_bytes + HIDDEN * 2 + 4);
    activations.write_bytes(
        0,
        &vec![0; hidden_bytes * 2 + logits_bytes + HIDDEN * 2 + 4],
    );
    let ids = 0usize;
    let embedded = 4usize;
    let normalized = embedded + hidden_bytes;
    let logits = normalized + hidden_bytes;
    let norm_weight = logits + logits_bytes;
    activations.write_bytes(ids as u64, &0u32.to_le_bytes());
    activations.write_bytes(
        norm_weight as u64,
        tensor_bytes(&shards, "language_model.model.norm.weight"),
    );

    let queue = dev.new_command_queue();
    let command = queue.new_command_buffer();
    let enc = command.compute_compute_encoder();
    enc.set_compute_pipeline_state(&embed_pipe);
    enc.set_buffer(&activations, ids as u64, 0);
    enc.set_buffer(&embed_buf, 0, 1);
    enc.set_buffer(&embed_buf, embed_weight as u64, 2);
    enc.set_buffer(&embed_buf, (embed_weight + embed_group) as u64, 3);
    enc.set_buffer(&activations, embedded as u64, 4);
    enc.set_bytes(&(HIDDEN as u32).to_le_bytes(), 5);
    enc.dispatch_threads(
        MTLSize {
            width: HIDDEN as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );

    enc.set_compute_pipeline_state(&norm_pipe);
    enc.set_buffer(&activations, embedded as u64, 0);
    enc.set_buffer(&activations, norm_weight as u64, 1);
    enc.set_buffer(&activations, normalized as u64, 2);
    enc.set_bytes(&(HIDDEN as u32).to_le_bytes(), 3);
    enc.set_bytes(&1e-6f32.to_le_bytes(), 4);
    enc.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    enc.set_compute_pipeline_state(&q4_pipe);
    enc.set_buffer(&activations, normalized as u64, 0);
    enc.set_buffer(&head_buf, 0, 1);
    enc.set_buffer(&head_buf, head_weight as u64, 2);
    enc.set_buffer(&head_buf, (head_weight + head_group) as u64, 3);
    enc.set_buffer(&activations, logits as u64, 4);
    let mut nk = Vec::with_capacity(8);
    nk.extend_from_slice(&(HEAD_ROWS as u32).to_le_bytes());
    nk.extend_from_slice(&(HIDDEN as u32).to_le_bytes());
    enc.set_bytes(&nk, 5);
    enc.dispatch_thread_groups(
        MTLSize {
            width: HEAD_ROWS.div_ceil(16) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    command.commit();
    command.wait_until_completed();

    if let (Ok(embed_ref), Ok(norm_ref), Ok(logits_ref)) = (
        std::fs::read("/tmp/mlx_io_embed.raw"),
        std::fs::read("/tmp/mlx_io_norm.raw"),
        std::fs::read("/tmp/mlx_io_logits.raw"),
    ) {
        let embed_actual = activations.read_bytes(embedded as u64, HIDDEN * 2);
        let norm_actual = activations.read_bytes(normalized as u64, HIDDEN * 2);
        let logits_actual = activations.read_bytes(logits as u64, HEAD_ROWS * 4);
        let embed_diff = max_diff_bf16(&embed_actual, &embed_ref);
        let norm_diff = max_diff_bf16(&norm_actual, &norm_ref);
        let mut logits_diff = 0.0f32;
        let mut logits_index = 0usize;
        let mut logits_pair = (0.0f32, 0.0f32);
        let mut actual_best = (0usize, f32::NEG_INFINITY);
        let mut reference_best = (0usize, f32::NEG_INFINITY);
        for (index, (a, r)) in logits_actual
            .chunks_exact(4)
            .zip(logits_ref.chunks_exact(4))
            .enumerate()
        {
            let actual = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
            let reference = f32::from_le_bytes([r[0], r[1], r[2], r[3]]);
            let diff = (actual - reference).abs();
            if diff > logits_diff {
                logits_diff = diff;
                logits_index = index;
                logits_pair = (actual, reference);
            }
            if actual > actual_best.1 {
                actual_best = (index, actual);
            }
            if reference > reference_best.1 {
                reference_best = (index, reference);
            }
        }
        eprintln!(
            "io diffs: embed={embed_diff} norm={norm_diff} logits={logits_diff} at {logits_index} actual={} ref={}",
            logits_pair.0, logits_pair.1
        );
        assert_eq!(embed_diff, 0.0, "embedding must match BF16 exactly");
        assert!(norm_diff <= 1.25e-1, "final norm drifted by {norm_diff}");
        assert!(logits_diff <= 1.25e-1, "full head drifted by {logits_diff}");
        assert_eq!(actual_best.0, reference_best.0, "full head argmax differs");
    } else {
        eprintln!("skip MLX comparison (run scripts/mlx_io_truth.py first)");
    }
}
