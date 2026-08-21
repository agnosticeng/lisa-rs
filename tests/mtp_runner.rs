use std::path::{Path, PathBuf};

use lisa_rs::model::runner::{MtpRunner, QwenRunner};
use lisa_rs::model::{HIDDEN, VOCAB};

fn local_snapshot(model: &str) -> Option<PathBuf> {
    let root = std::env::home_dir()?.join(format!(
        ".cache/huggingface/hub/models--mlx-community--{model}/snapshots"
    ));
    std::fs::read_dir(root)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.is_dir()
                && std::fs::read_dir(path).is_ok_and(|entries| {
                    entries.filter_map(Result::ok).any(|entry| {
                        entry
                            .path()
                            .extension()
                            .is_some_and(|extension| extension == "safetensors")
                    })
                })
        })
}

fn reference(path: &Path, len: usize) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    assert_eq!(bytes.len(), len * 4, "MTP truth size");
    Some(
        bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]]))
            .collect(),
    )
}

fn max_diff(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f32::max)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index)
}

#[test]
#[ignore = "loads the 16 GB target and 239 MB MTP checkpoints onto Metal"]
fn deterministic_position_matches_mlx() {
    let Some(target_snapshot) = local_snapshot("Qwen3.8-27B-4bit") else {
        eprintln!("skip: target snapshot not cached");
        return;
    };
    let Some(mtp_snapshot) = local_snapshot("Qwen3.8-27B-MTP-4bit") else {
        eprintln!("skip: MTP snapshot not cached");
        return;
    };
    let target = QwenRunner::load(&target_snapshot, 4).expect("load target runner");
    let mut mtp = MtpRunner::load(&target, &mtp_snapshot, 4).expect("load MTP runner");
    let target_hidden: Vec<f32> = (0..HIDDEN)
        .map(|index| ((index % 257) as i32 - 128) as f32 / 64.0)
        .collect();

    let output = mtp
        .forward_position(2005, &target_hidden)
        .expect("MTP forward");
    assert_eq!(output.hidden.len(), HIDDEN);
    assert_eq!(output.logits.len(), VOCAB);
    assert!(output.hidden.iter().all(|value| value.is_finite()));
    assert!(output.logits.iter().all(|value| value.is_finite()));
    assert_eq!(output.argmax as usize, argmax(&output.logits));
    assert_eq!(mtp.position(), 1);

    if let (Some(expected_hidden), Some(expected_logits)) = (
        reference(Path::new("/tmp/mlx_mtp_hidden.raw"), HIDDEN),
        reference(Path::new("/tmp/mlx_mtp_logits.raw"), VOCAB),
    ) {
        let hidden_diff = max_diff(&output.hidden, &expected_hidden);
        let logits_diff = max_diff(&output.logits, &expected_logits);
        eprintln!(
            "MTP M=1: hidden diff={hidden_diff}, logits diff={logits_diff}, argmax native={} mlx={}",
            output.argmax,
            argmax(&expected_logits)
        );
        assert!(hidden_diff <= 0.5, "MTP hidden drifted by {hidden_diff}");
        assert!(logits_diff <= 0.5, "MTP logits drifted by {logits_diff}");
        assert_eq!(output.argmax as usize, argmax(&expected_logits));
    } else {
        eprintln!("skip comparison: run scripts/mlx_mtp_truth.py first");
    }

    mtp.forward_position(output.argmax, &output.hidden)
        .expect("second MTP position");
    assert_eq!(mtp.position(), 2);
    mtp.trim_state(1).expect("trim MTP cache");
    assert_eq!(mtp.position(), 1);
    assert!(mtp.trim_state(2).is_err());
    mtp.reset_state();
    assert_eq!(mtp.position(), 0);
}

#[test]
#[ignore = "loads the 16 GB target and 239 MB MTP checkpoints onto Metal"]
fn forward_position_decode_matches_full_logits_argmax() {
    let Some(target_snapshot) = local_snapshot("Qwen3.8-27B-4bit") else {
        eprintln!("skip: target snapshot not cached");
        return;
    };
    let Some(mtp_snapshot) = local_snapshot("Qwen3.8-27B-MTP-4bit") else {
        eprintln!("skip: MTP snapshot not cached");
        return;
    };
    let target = QwenRunner::load(&target_snapshot, 4).expect("load target runner");
    let mut mtp = MtpRunner::load(&target, &mtp_snapshot, 4).expect("load MTP runner");
    let target_hidden: Vec<f32> = (0..HIDDEN)
        .map(|index| ((index % 257) as i32 - 128) as f32 / 64.0)
        .collect();

    // Full-logits forward (CPU argmax over the readback) establishes the truth.
    let full = mtp
        .forward_position(2005, &target_hidden)
        .expect("MTP full forward");

    // Same position via the decode-only path (GPU argmax, no logits readback).
    mtp.reset_state();
    let (decode_argmax, decode_hidden) = mtp
        .forward_position_decode(2005, &target_hidden)
        .expect("MTP decode forward");

    assert_eq!(decode_argmax, full.argmax, "GPU argmax must match CPU argmax");
    assert_eq!(decode_hidden, full.hidden, "decode hidden must match full hidden");
    assert_eq!(mtp.position(), 1);
}
