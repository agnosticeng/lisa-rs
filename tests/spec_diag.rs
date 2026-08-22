// Diagnostic: reproduce `cli run`'s exact_match check (greedy M=1 target stream
// vs the MTP block-3 speculative stream) on the long 1641-token prompt, and
// when they diverge, report the position and the M=1 top-2 margin at that step
// to distinguish a near-tie artifact from a real bug.
use std::path::PathBuf;

use lisa_rs::model::runner::{MtpRunner, QwenRunner};
use lisa_rs::speculative::{generate_greedy_block3_prefilled, prefill_prompt};

fn snapshot(id: &str) -> PathBuf {
    let root = std::env::home_dir()
        .unwrap()
        .join(".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots");
    std::fs::read_dir(root)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .unwrap()
}

fn mtp_snapshot() -> PathBuf {
    let root = std::env::home_dir()
        .unwrap()
        .join(".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-MTP-4bit/snapshots");
    std::fs::read_dir(root)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .unwrap()
}

#[test]
#[ignore = "diagnostic: reproduces cli run exact-match on a /tmp prompt; logs near-tie margin"]
fn reproduce_cli_run_exact_match() {
    let csv = std::fs::read_to_string("/tmp/native_prompt.csv").unwrap();
    let prompt: Vec<u32> = csv.trim().split(',').map(|p| p.parse().unwrap()).collect();
    let steps = 64usize;
    let cap = prompt.len() + steps + 8;

    // ---- greedy M=1 target-only stream ----
    let mut target = QwenRunner::load(&snapshot("target"), cap).unwrap();
    let pre = target.forward_prefill(&prompt).unwrap();
    let mut token = pre.last().unwrap().0;
    let mut greedy = vec![token];
    for _ in 1..steps {
        let (next, _) = target.forward_token_decode(token).unwrap();
        token = next;
        greedy.push(token);
    }

    // ---- speculative MTP stream (fresh state, like the CLSP path) ----
    let mut t2 = QwenRunner::load(&snapshot("target"), cap).unwrap();
    let mut mtp = MtpRunner::load(&t2, &mtp_snapshot(), cap).unwrap();
    let (bonus, target_hidden, seed) = prefill_prompt(&mut t2, &mut mtp, &prompt).unwrap();
    let res = generate_greedy_block3_prefilled(
        &mut t2,
        &mut mtp,
        bonus,
        target_hidden,
        seed,
        steps,
    )
    .unwrap();

    eprintln!(
        "greedy[:5]={:?}  spec[:5]={:?}",
        &greedy[..5.min(greedy.len())],
        &res.tokens[..5.min(res.tokens.len())]
    );
    let div = greedy
        .iter()
        .zip(&res.tokens)
        .position(|(a, b)| a != b);
    match div {
        None => eprintln!("SAME: speculative == greedy for all {steps} tokens"),
        Some(i) => {
            // quantify the greedy margin at the first divergence (greedy[0] is
            // the prefill bonus = no forward; greedy[k] k>=1 is produced at
            // base position prompt.len()+k-1 by a forward on greedy[k-1]).
            let mut t3 = QwenRunner::load(&snapshot("target"), cap).unwrap();
            let pre3 = t3.forward_prefill(&prompt).unwrap();
            let mut t = pre3.last().unwrap().0;
            let mut logits_at = None;
            for k in 1..=i {
                let out = t3.forward_token_with_hidden(t).unwrap();
                if k == i {
                    let mut top = (0usize, f32::NEG_INFINITY);
                    let mut sec = (0usize, f32::NEG_INFINITY);
                    for (j, &v) in out.logits.iter().enumerate() {
                        if v > top.1 { sec = top; top = (j, v); }
                        else if v > sec.1 { sec = (j, v); }
                    }
                    logits_at = Some((top.1 - sec.1, top.0 as u32));
                }
                t = out.argmax;
            }
            eprintln!(
                "DIVERGE at token {i} (greedy={} spec={}): greedy argmax={} margin={:?}",
                greedy[i], res.tokens[i], greedy[i], logits_at
            );
            panic!("spec stream diverges from greedy at token {i}");
        }
    }
}
