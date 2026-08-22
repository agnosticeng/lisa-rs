use std::path::PathBuf;

use lisa_rs::model::runner::QwenRunner;

fn local_snapshot() -> Option<PathBuf> {
    let root = std::env::home_dir()?
        .join(".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots");
    std::fs::read_dir(root)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(l, r)| (l - r).abs())
        .fold(0.0f32, f32::max)
}

#[test]
#[ignore = "loads the 16 GB local checkpoint onto Metal"]
fn gpu_checkpoint_restores_exact_decode_state() {
    let Some(snapshot) = local_snapshot() else {
        eprintln!("skip: target snapshot not cached");
        return;
    };
    let mut runner = QwenRunner::load(&snapshot, 4).expect("load target runner");
    let first = runner
        .forward_token_with_hidden(2005)
        .expect("first forward");
    let checkpoint = runner.checkpoint_state();
    assert_eq!(checkpoint.position(), 1);

    let expected = runner
        .forward_token_with_hidden(first.argmax)
        .expect("forward after checkpoint");
    runner
        .forward_token(expected.argmax)
        .expect("advance past checkpoint");
    runner
        .restore_state(&checkpoint)
        .expect("restore checkpoint");
    assert_eq!(runner.position(), 1);
    let restored = runner
        .forward_token_with_hidden(first.argmax)
        .expect("forward after restore");

    assert_eq!(restored.argmax, expected.argmax);
    assert_eq!(restored.hidden, expected.hidden);
    assert_eq!(restored.logits, expected.logits);
}

#[test]
#[ignore = "loads the 16 GB local checkpoint onto Metal"]
fn block3_matches_sequential_target_and_commits_first_row() {
    let Some(snapshot) = local_snapshot() else {
        eprintln!("skip: target snapshot not cached");
        return;
    };
    let mut runner = QwenRunner::load(&snapshot, 8).expect("load target runner");
    let initial = runner.checkpoint_state();
    let row0 = runner
        .forward_token_with_hidden(2005)
        .expect("sequential row 0");
    let row1 = runner
        .forward_token_with_hidden(row0.argmax)
        .expect("sequential row 1");
    let row2 = runner
        .forward_token_with_hidden(row1.argmax)
        .expect("sequential row 2");
    let next = runner
        .forward_token_with_hidden(row2.argmax)
        .expect("sequential next row");

    runner
        .restore_state(&initial)
        .expect("restore initial state");
    let batched = runner
        .verify_block3([2005, row0.argmax, row1.argmax])
        .expect("batched block-3");
    for (index, (batch, sequential)) in batched.iter().zip([&row0, &row1, &row2]).enumerate() {
        assert_eq!(batch.argmax, sequential.argmax, "argmax row {index}");
        // The M=3 verify uses a wider QMV reduction than the M=1 decode, so
        // hidden/logits are argmax-consistent but not bit-identical (same as
        // MLX, whose qmv_wide differs from qmv_fast). The token stream must
        // still match, so argmax is exact while hidden/logits stay within a
        // few bf16 quantization steps (observed ~2.0 in hidden, ~0.2 in logits).
        assert!(
            max_abs_diff(&batch.hidden, &sequential.hidden) < 4.0,
            "hidden row {index} diverged"
        );
        assert!(
            max_abs_diff(&batch.logits, &sequential.logits) < 2.0,
            "logits row {index} diverged"
        );
    }
    runner
        .commit_verified_prefix(3)
        .expect("commit complete block");
    let after_batch = runner
        .forward_token_with_hidden(row2.argmax)
        .expect("forward after batched commit");
    assert_eq!(after_batch.argmax, next.argmax);
    assert!(
        max_abs_diff(&after_batch.hidden, &next.hidden) < 4.0,
        "post-batch hidden diverged"
    );
    assert!(
        max_abs_diff(&after_batch.logits, &next.logits) < 2.0,
        "post-batch logits diverged"
    );

    runner
        .restore_state(&initial)
        .expect("restore initial state");
    runner
        .verify_block3([2005, row0.argmax, row1.argmax])
        .expect("second batched block-3");
    runner
        .commit_verified_prefix(1)
        .expect("commit first verified row");
    let after_prefix = runner
        .forward_token_with_hidden(row0.argmax)
        .expect("forward after prefix commit");
    assert_eq!(after_prefix.argmax, row1.argmax);
    assert!(
        max_abs_diff(&after_prefix.hidden, &row1.hidden) < 4.0,
        "post-prefix hidden diverged"
    );
    assert!(
        max_abs_diff(&after_prefix.logits, &row1.logits) < 2.0,
        "post-prefix logits diverged"
    );

    // rows=2 exercises the slot-1 rollback (commit reads snapshot 1, the state
    // after the first two verified tokens).
    runner
        .restore_state(&initial)
        .expect("restore initial state");
    runner
        .verify_block3([2005, row0.argmax, row1.argmax])
        .expect("third batched block-3");
    runner
        .commit_verified_prefix(2)
        .expect("commit two verified rows");
    let after_two = runner
        .forward_token_with_hidden(row1.argmax)
        .expect("forward after two-row commit");
    assert_eq!(after_two.argmax, row2.argmax);
    assert!(
        max_abs_diff(&after_two.hidden, &row2.hidden) < 4.0,
        "post-two-row hidden diverged"
    );
    assert!(
        max_abs_diff(&after_two.logits, &row2.logits) < 2.0,
        "post-two-row logits diverged"
    );
}
