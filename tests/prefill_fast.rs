// Fast prefill-only benchmark. Env: PREFILL_N (tokens, default 5120),
// PREFILL_PASSES (timed passes, default 3). One model load, warmup chunk first.
// Run: cargo test --test prefill_fast --release -- --ignored --nocapture
use std::time::Instant;

use lisa_rs::model::runner::QwenRunner;

fn tok_per_sec(tokens: usize, ms: f64) -> f64 {
    tokens as f64 / (ms / 1e3)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[test]
#[ignore = "loads real weights + benchmarks; diagnostic-only"]
fn prefill_fast() -> Result<(), String> {
    let n = env_usize("PREFILL_N", 5120);
    let passes = env_usize("PREFILL_PASSES", 3);

    let Ok(snapshot) = lisa_rs::cache::resolve_snapshot("mlx-community/Qwen3.8-27B-4bit") else {
        eprintln!("target model not in HF cache; skipping");
        return Ok(());
    };

    let prompt: Vec<u32> = (0..n).map(|i| (1000 + (i % 9000)) as u32).collect();
    let batch = 512usize; // PREFILL_BATCH constant in runner.rs
    let chunks = n.div_ceil(batch);

    let mut runner = QwenRunner::load(&snapshot, n + 8)?;
    // Warm up weights (cold first pass is ~2.5x slower).
    let _ = runner.forward_prefill(&prompt[..8])?;
    runner.reset_state();

    let mut times = Vec::new();
    for pass in 0..passes {
        runner.reset_state();
        let t0 = Instant::now();
        let out = runner.forward_prefill(&prompt)?;
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        times.push(dt);
        assert_eq!(out.len(), n);
        println!("pass {pass}: {n} tokens ({chunks} chunks x {batch}) = {dt:.1} ms -> {:.1} tok/s", tok_per_sec(n, dt));
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    println!("median: {:.1} ms -> {:.1} tok/s", med, tok_per_sec(n, med));
    Ok(())
}