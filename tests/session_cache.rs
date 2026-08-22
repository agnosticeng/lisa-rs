// Session prefix cache: a two-turn conversation served by the same `Model` must
// reproduce the exact token stream of a fresh full re-prefill. On the second
// turn the recurrent/KV state at the messages boundary is restored and only the
// delta (new messages + generation prompt) is prefilled, instead of re-prefilling
// the whole conversation.
use std::path::PathBuf;

use lisa_rs::serve::metrics::Metrics;
use lisa_rs::serve::Model;
use lisa_rs::tokenizer::ChatMessage;

const TARGET: &str = "models--mlx-community--Qwen3.8-27B-4bit";
const MTP: &str = "models--mlx-community--Qwen3.8-27B-MTP-4bit";

fn snapshot(cache: &str) -> Option<PathBuf> {
    let root = std::env::home_dir()?
        .join(".cache/huggingface/hub")
        .join(cache)
        .join("snapshots");
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

fn tokenizer_path() -> Option<PathBuf> {
    let root = std::env::home_dir()?
        .join(".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots");
    std::fs::read_dir(root)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.join("tokenizer.json").exists())
        .map(|dir| dir.join("tokenizer.json"))
}

#[test]
#[ignore = "loads the 16 GB target + MTP checkpoint"]
fn session_cache_matches_full_refill() {
    let Some(target) = snapshot(TARGET) else {
        eprintln!("target not cached; skipping");
        return;
    };
    let Some(mtp) = snapshot(MTP) else {
        eprintln!("mtp not cached; skipping");
        return;
    };
    let Some(tok) = tokenizer_path() else {
        eprintln!("tokenizer not cached; skipping");
        return;
    };
    let cap = 8192;

    // A long first user turn so the reused prefix is worth skipping.
    let long = "Please explain the memory bandwidth bottleneck in autoregressive \
                transformer decode, how four bit group wise affine quantization \
                reduces the bytes read per token, and how flash attention keeps \
                the attention cost from growing with the sequence length. Give a \
                detailed answer with several paragraphs covering the key ideas. "
        .repeat(30);

    let turn1 = vec![ChatMessage::new("user", long.trim())];

    // Serving Model (holds session state across turns).
    let mut model = Model::load_with_metrics(
        &target,
        &mtp,
        &tok,
        cap,
        "qwen3.8-27b".to_string(),
        Metrics::shared(),
    )
    .expect("load model");

    let mut out1 = Vec::new();
    model
        .complete(&turn1, 48, true, None, &[], |t| out1.push(t))
        .expect("turn 1");

    // Turn 2 extends the conversation with a follow-up user message. Its
    // rendered prefix is an exact extension of turn 1's (same long user turn
    // first), so the session cache restores the recurrent state and prefills
    // only the appended message + generation prompt.
    let turn2 = vec![
        ChatMessage::new("user", long.trim()),
        ChatMessage::new(
            "user",
            "Now compare the bandwidth bottleneck to the gated delta net layers \
             in a hybrid model: why do those layers cost constant compute \
             regardless of context length?",
        ),
    ];
    let mut out2 = Vec::new();
    let c2 = std::time::Instant::now();
    model
        .complete(&turn2, 48, true, None, &[], |t| out2.push(t))
        .expect("turn 2 (cached)");
    let cache_secs = c2.elapsed().as_secs_f64();

    // Fresh model, no cache: full re-prefill of turn 2's exact messages.
    let mut fresh = Model::load_with_metrics(
        &target,
        &mtp,
        &tok,
        cap,
        "qwen3.8-27b".to_string(),
        Metrics::shared(),
    )
    .expect("fresh model");
    let mut out_ref = Vec::new();
    let cf = std::time::Instant::now();
    fresh
        .complete(&turn2, 48, true, None, &[], |t| out_ref.push(t))
        .expect("turn 2 (full)");
    let full_secs = cf.elapsed().as_secs_f64();

    eprintln!("turn1: {} tokens", out1.len());
    eprintln!(
        "turn2(cache): {} tokens in {cache_secs:.2}s  turn2(full): {} tokens in {full_secs:.2}s",
        out2.len(),
        out_ref.len()
    );
    assert!(
        out1.len() > 4,
        "turn 1 unexpectedly produced almost no tokens: {out1:?}"
    );
    assert!(
        out2.len() > 4,
        "turn 2 (cache) unexpectedly produced almost no tokens: {out2:?}"
    );
    assert!(
        out2 == out_ref,
        "session-cache turn did not reproduce full re-prefill stream\n  cache: {out2:?}\n  full : {out_ref:?}"
    );
    // The cache path prefills only the delta (follow-up + gen prompt), so it
    // must be meaningfully faster than re-prefilling the whole 1600+ token
    // conversation. This guards against a silent fallback to the full path.
    assert!(
        cache_secs < full_secs * 0.5,
        "session cache did not speed up prefill (cache {cache_secs:.2}s vs full {full_secs:.2}s) — it likely fell back to full re-prefill"
    );
}
