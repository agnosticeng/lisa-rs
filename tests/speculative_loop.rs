// End-to-end speculative loop: prefill + block-3 generation must reproduce the
// sequential target-only greedy token stream (the same `exact_match` invariant
// the benchmark reports). Exercises `prefill_prompt`, `generate_greedy_block3_prefilled`,
// `verify_block3_decode`, `commit_verified_prefix` (all three rows), the MTP
// draft/replay path, and the GPU-argmax decode variants.
use std::path::PathBuf;

use lisa_rs::model::runner::{MtpRunner, QwenRunner};
use lisa_rs::speculative::{generate_greedy_block3_prefilled, prefill_prompt};

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

#[test]
#[ignore = "loads the 16 GB target and 239 MB MTP checkpoints onto Metal"]
fn block3_speculation_matches_sequential_target() {
    let Some(target_snapshot) = local_snapshot("Qwen3.8-27B-4bit") else {
        eprintln!("skip: target snapshot not cached");
        return;
    };
    let Some(mtp_snapshot) = local_snapshot("Qwen3.8-27B-MTP-4bit") else {
        eprintln!("skip: MTP snapshot not cached");
        return;
    };

    let prompt: Vec<u32> = vec![60856, 449, 303, 40284];
    let steps = 8usize;

    // ===== target-only sequential =====
    let mut target = QwenRunner::load(&target_snapshot, prompt.len() + steps + 3)
        .expect("load target runner");
    for &token in &prompt[..prompt.len() - 1] {
        target.forward_token_decode(token).expect("prefill token");
    }
    let mut token = *prompt.last().expect("non-empty prompt");
    let mut target_tokens = Vec::with_capacity(steps);
    for _ in 0..steps {
        let (next, _) = target.forward_token_decode(token).expect("target decode");
        token = next;
        target_tokens.push(token);
    }

    // ===== speculative block-3 =====
    target.reset_state();
    let mut mtp = MtpRunner::load(&target, &mtp_snapshot, prompt.len() + steps + 3)
        .expect("load MTP runner");
    let (bonus, target_hidden, mtp_seed) =
        prefill_prompt(&mut target, &mut mtp, &prompt).expect("prefill prompt");
    let speculative = generate_greedy_block3_prefilled(
        &mut target,
        &mut mtp,
        bonus,
        target_hidden,
        mtp_seed,
        steps,
    )
    .expect("speculative generation");

    assert_eq!(
        speculative.tokens, target_tokens,
        "speculative tokens must match sequential target"
    );
    assert!(speculative.rounds > 0, "speculation must run at least one round");
}
