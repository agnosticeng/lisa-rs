// Real-fused ablation: time the fused M=1 forward with each stage (GDN /
// attention / MLP / lm_head / readback) disabled and report per-stage ms/token.
// The authoritative throughput breakdown. Needs real cached weights.
use std::time::Instant;

use lisa_rs::model::runner::{
    ABLATE_ATTENTION, ABLATE_GDN, ABLATE_LM_HEAD, ABLATE_MLP, ABLATE_READBACK, QwenRunner,
};

#[test]
#[ignore = "loads real weights + benchmarks; diagnostic-only"]
fn fused_forward_ablation() -> Result<(), String> {
    let Ok(snapshot) = lisa_rs::cache::resolve_snapshot("mlx-community/Qwen3.8-27B-4bit") else {
        eprintln!("target model not in HF cache; skipping");
        return Ok(());
    };
    let tokens = 16u32;
    let measure = |mask: u32| -> Result<f64, String> {
        let mut runner = QwenRunner::load(&snapshot, (tokens + 8) as usize)?;
        Ok(time_forward(&mut runner, mask, tokens, 4))
    };

    let full: f64 = measure(0)?;
    let no_gdn: f64 = measure(ABLATE_GDN)?;
    let no_attn: f64 = measure(ABLATE_ATTENTION)?;
    let no_mlp: f64 = measure(ABLATE_MLP)?;
    let no_head: f64 = measure(ABLATE_LM_HEAD)?;
    let no_readback: f64 = measure(ABLATE_READBACK)?;

    let gdn = full - no_gdn;
    let attn = full - no_attn;
    let mlp = full - no_mlp;
    let head = full - no_head;
    let readback = full - no_readback;
    let embed_norms = full - (gdn + attn + mlp + head + readback);

    println!("full (ms/token)          : {full:.3}");
    println!("  gdn_48                 : {gdn:.3} ({:.1}%)", gdn / full * 100.0);
    println!("  attention_16           : {attn:.3} ({:.1}%)", attn / full * 100.0);
    println!("  mlp_64                 : {mlp:.3} ({:.1}%)", mlp / full * 100.0);
    println!("  lm_head                : {head:.3} ({:.1}%)", head / full * 100.0);
    println!(
        "  cpu_readback+argmax    : {readback:.3} ({:.1}%)",
        readback / full * 100.0
    );
    println!(
        "  embed+norms+residuals  : {embed_norms:.3} ({:.1}%)",
        embed_norms / full * 100.0
    );
    Ok(())
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1_000.0
}

fn time_forward(runner: &mut QwenRunner, mask: u32, tokens: u32, warmup: u32) -> f64 {
    for _ in 0..warmup {
        runner.forward_token_ablate(2005, mask).unwrap();
    }
    let start = Instant::now();
    for _ in 0..tokens {
        runner.forward_token_ablate(2005, mask).unwrap();
    }
    ms(start.elapsed()) / tokens as f64
}