// Batched NAX-GEMM prefill (forward_prefill) must reproduce the sequential
// M=1 decode argmaxes for a real prompt (argmax-consistent, like MLX's qmm vs
// qmv split). Also checks the recurrent state carries correctly across chunks.
use std::path::PathBuf;

use lisa_rs::model::runner::{MtpRunner, QwenRunner};

fn local_snapshot() -> Option<PathBuf> {
    let root = std::env::home_dir()?
        .join(".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots");
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

/// A realistic prompt > PREFILL_BATCH so at least two chunks are processed.
fn prompt() -> Vec<u32> {
    vec![
        6367, 264, 1026, 2154, 36420, 651, 295, 706, 3272, 4304, 733, 264, 10914, 4687, 13,
        1532, 21570, 572, 295, 706, 12, 478, 26941, 4304, 733, 264, 10914, 4687, 13, 1532,
        21570, 572, 2124, 78191, 383, 815, 60856, 449, 303, 40284, 60, 38, 553, 11, 476,
        12789, 353, 715, 21, 956, 3521, 263, 27420, 5666, 13, 368, 795, 363, 2279, 1192,
        5095, 21, 13, 17386, 4687, 515, 1765, 31347, 861, 574, 11, 476, 359, 1310, 632,
    ]
}

#[test]
#[ignore = "loads the 16 GB local checkpoint onto Metal"]
fn batched_prefill_matches_sequential_argmax() {
    let Some(snapshot) = local_snapshot() else {
        eprintln!("skip: Qwen3.8-27B-4bit snapshot not cached");
        return;
    };
    let prompt = prompt();

    // Sequential decode reference.
    let mut seq = QwenRunner::load(&snapshot, 4096).expect("load native runner");
    let mut seq_argmax = Vec::new();
    for &token in &prompt {
        seq_argmax.push(seq.forward_token_decode(token).expect("seq decode").0);
    }

    // Batched prefill.
    let mut batched = QwenRunner::load(&snapshot, 4096).expect("load native runner");
    let batched_out = batched.forward_prefill(&prompt).expect("batched prefill");
    assert_eq!(batched_out.len(), prompt.len());
    let mismatches: Vec<usize> = (0..prompt.len())
        .filter(|&i| batched_out[i].0 != seq_argmax[i])
        .collect();
    eprintln!(
        "positions={} argmax_mismatches={} ({:.2}%)",
        prompt.len(),
        mismatches.len(),
        100.0 * mismatches.len() as f64 / prompt.len() as f64
    );
    for &i in mismatches.iter().take(20) {
        eprintln!(
            "  pos {i}: batched={} seq={}",
            batched_out[i].0, seq_argmax[i]
        );
    }
    // Argmax-consistent (mirrors MLX qmm-vs-qmv): allow rare single-position
    // disagreements but require near-total agreement (<5%).
    assert!(
        mismatches.len() * 20 <= prompt.len(),
        "too many prefill argmax mismatches"
    );
    assert_eq!(batched.position(), prompt.len());
}

fn mtp_snapshot() -> Option<PathBuf> {
    local_snapshot_model("Qwen3.8-27B-MTP-4bit")
}

fn local_snapshot_model(model: &str) -> Option<PathBuf> {
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

#[test]
#[ignore = "loads the 16 GB local checkpoint onto Metal"]
fn batched_mtp_prefill_matches_sequential_argmax() {
    let Some(snapshot) = local_snapshot() else {
        eprintln!("skip: target snapshot not cached");
        return;
    };
    let Some(mtp_snapshot) = mtp_snapshot() else {
        eprintln!("skip: MTP snapshot not cached");
        return;
    };
    let prompt = prompt();

    // Sequential MTP prefill on the target's per-position hidden states.
    let mut target = QwenRunner::load(&snapshot, 4096).expect("load target");
    let mut target_hiddens = Vec::new();
    for &token in &prompt {
        let (_, hidden) = target.forward_token_decode(token).expect("target seq");
        target_hiddens.push(hidden);
    }
    let mut seq = MtpRunner::load(&target, &mtp_snapshot, 4096).expect("load mtp");
    let mut seq_argmax = Vec::new();
    for (t, hidden) in target_hiddens.iter().enumerate() {
        let token = if t + 1 < prompt.len() { prompt[t + 1] } else { 0 };
        seq_argmax.push(seq.forward_position_decode(token, hidden).expect("mtp seq").0);
    }

    // Batched MTP prefill.
    let mut batch_t = QwenRunner::load(&snapshot, 4096).expect("load target");
    let mut batch_hiddens = Vec::new();
    for &token in &prompt {
        let (_, hidden) = batch_t.forward_token_decode(token).expect("target batch");
        batch_hiddens.push(hidden);
    }
    let positions: Vec<(u32, &[f32])> = batch_hiddens
        .iter()
        .enumerate()
        .map(|(t, h)| {
            let token = if t + 1 < prompt.len() { prompt[t + 1] } else { 0 };
            (token, h.as_slice())
        })
        .collect();
    let mut m = MtpRunner::load(&batch_t, &mtp_snapshot, 4096).expect("load mtp");
    let batch_out = m.forward_position_batch(&positions).expect("mtp batch");
    // debug: sequential hidden too
    let mut seq2 = MtpRunner::load(&target, &mtp_snapshot, 4096).expect("load mtp");
    let mut seq_hidden = Vec::new();
    for (t, hidden) in target_hiddens.iter().enumerate() {
        let token = if t + 1 < prompt.len() { prompt[t + 1] } else { 0 };
        seq_hidden.push(seq2.forward_position_decode(token, hidden).expect("d").1);
    }
    for i in 0..prompt.len().min(4) {
        let d: f32 = batch_out[i].1.iter().zip(&seq_hidden[i]).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        let bf = batch_out[i].1.iter().filter(|v| v.is_finite()).count();
        eprintln!("pos {i}: batched_argmax={} seq_argmax={} hidden_max_abs_diff={d:.4} batched_finite={bf}/{}", batch_out[i].0, seq_argmax[i], batch_out[i].1.len());
    }
    let mismatches: Vec<usize> = (0..prompt.len())
        .filter(|&i| batch_out[i].0 != seq_argmax[i])
        .collect();
    eprintln!(
        "MTP positions={} argmax_mismatches={} ({:.2}%)",
        prompt.len(),
        mismatches.len(),
        100.0 * mismatches.len() as f64 / prompt.len() as f64
    );
    assert!(
        mismatches.len() * 20 <= prompt.len(),
        "too many MTP prefill argmax mismatches"
    );
    assert_eq!(m.position(), prompt.len());
}
