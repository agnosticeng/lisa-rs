//! `run` subcommand: run the native target-only vs MTP block-3 benchmark.
//!
//! Prints a single-line JSON report. Supports both positional and named
//! arguments so the serving/reporting tooling can drive it either way:
//!   run <TARGET> <MTP> [STEPS] [PROMPT_CSV]
//!   run --target T --mtp M [--steps N] [--prompt CSV]
use std::process::ExitCode;
use std::time::Instant;

use crate::cache::resolve_snapshot;
use crate::cli::args as cli_args;
use crate::model::runner::{MtpRunner, QwenRunner};
use crate::speculative::{generate_greedy_block3_prefilled, prefill_prompt};

pub const USAGE: &str = "\
run <TARGET> <MTP> [STEPS] [PROMPT_CSV]
   or: run --target <TARGET> --mtp <MTP> [--steps N] [--prompt CSV]

Run the native target-only vs MTP block-3 benchmark and print a single-line
JSON report. <TARGET>/<MTP> are snapshot directories or Hugging Face ids.

options:
  --steps N            generated tokens (default 64)
  --prompt CSV         comma-separated prompt token ids (default 2005)
  --help               print this help";

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(2).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("run: {e}");
            ExitCode::FAILURE
        }
    }
}

pub fn run(args: &[String]) -> Result<(), String> {
    let config = run_config(args)?;

    let target_snapshot = resolve_snapshot(&config.target)?;
    let mtp_snapshot = resolve_snapshot(&config.mtp)?;

    let capacity = config.prompt.len() + config.steps + 3;
    let mut target = QwenRunner::load(&target_snapshot, capacity)?;

    // Warm up: the first GPU pass over the GBs of weight shards pays a
    // cold-start cost (~2.5x slower here) that would skew the raw numbers.
    // Touch both the batched GEMM and the decode QMV paths, then rewind.
    target.reset_state();
    let _ = target.forward_prefill(&config.prompt)?;
    target.reset_state();
    let _ = target.forward_token_decode(
        *config.prompt.last().expect("non-empty prompt"),
    )?;
    target.reset_state();

    // ===== target-only =====
    // Batched prefill of the full prompt (mirrors the speculative path and
    // MLX's batched prefill). The bonus is the first generated token; the
    // decode loop then emits `steps` tokens total (steps-1 forwards).
    let prefill_start = Instant::now();
    let prefill_outs = target.forward_prefill(&config.prompt)?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();
    let mut token = prefill_outs.last().expect("non-empty prompt").0;
    let mut target_tokens = Vec::with_capacity(config.steps);
    target_tokens.push(token);
    let target_start = Instant::now();
    for _ in 1..config.steps {
        let (next, _) = target.forward_token_decode(token)?;
        token = next;
        target_tokens.push(token);
    }
    let target_seconds = target_start.elapsed().as_secs_f64();

    // ===== speculative (MTP block-3) =====
    target.reset_state();
    let mut draft = MtpRunner::load(&target, &mtp_snapshot, capacity)?;
    // Warm the drafter prefill path the same way (fresh MTP weight shards).
    let _ = prefill_prompt(&mut target, &mut draft, &config.prompt)?;
    target.reset_state();
    draft.reset_state();
    let prefill_start = Instant::now();
    let (bonus, target_hidden, mtp_seed) = prefill_prompt(&mut target, &mut draft, &config.prompt)?;
    let speculative_prefill_seconds = prefill_start.elapsed().as_secs_f64();
    let speculative_start = Instant::now();
    let speculative = generate_greedy_block3_prefilled(
        &mut target,
        &mut draft,
        bonus,
        target_hidden,
        mtp_seed,
        config.steps,
    )?;
    let speculative_seconds = speculative_start.elapsed().as_secs_f64();

    if speculative.tokens != target_tokens {
        let token = speculative
            .tokens
            .iter()
            .zip(&target_tokens)
            .position(|(left, right)| left != right)
            .unwrap_or(speculative.tokens.len().min(target_tokens.len()));
        // M=1 vs M=3 (like MLX) are argmax-consistent, not bit-exact: on long
        // prompts a logit can sit within ~0.2 of a rival and the wider M=3
        // reduction flips it. This is the documented near-tie drift — never a
        // crash. Report it so the benchmark still produces numbers and the
        // report consumers can tell exact_match apart.
        eprintln!(
            "warning: speculative output differs from target-only at token {token} (near-tie drift, M=1 vs M=3)"
        );
    }

    let exact_match = speculative.tokens == target_tokens;

    let steps = config.steps;
    let target_tps = steps as f64 / target_seconds;
    let speculative_tps = steps as f64 / speculative_seconds;
    let target_e2e_tps = steps as f64 / (prefill_seconds + target_seconds);
    let speculative_e2e_tps =
        steps as f64 / (speculative_prefill_seconds + speculative_seconds);

    let prompt_tokens = config.prompt.len();
    let speedup = speculative_tps / target_tps;
    let target_decode_ms = target_seconds * 1_000.0;
    let speculative_decode_ms = speculative_seconds * 1_000.0;
    let target_prefill_tok_s = prompt_tokens as f64 / prefill_seconds;
    let speculative_prefill_tok_s = prompt_tokens as f64 / speculative_prefill_seconds;
    let target_prefill_ms = prefill_seconds * 1_000.0;
    let speculative_prefill_ms = speculative_prefill_seconds * 1_000.0;
    let acceptance = speculative.acceptance();

    eprintln!(
        "prefill : target {target_prefill_tok_s:.1} tok/s ({prompt_tokens} tok, {target_prefill_ms:.1} ms)  speculative {speculative_prefill_tok_s:.1} tok/s ({prompt_tokens} tok, {speculative_prefill_ms:.1} ms)"
    );
    eprintln!(
        "decode  : target {target_tps:.1} tok/s ({target_decode_ms:.1} ms)  speculative {speculative_tps:.1} tok/s ({speculative_decode_ms:.1} ms)"
    );

    println!(
        "{{\"tokens\":{steps},\"prompt_tokens\":{prompt_tokens},\"verify_mode\":\"batched_m3\",\"exact_match\":{exact_match},\"target_prefill_tok_s\":{target_prefill_tok_s:.3},\"target_prefill_ms\":{target_prefill_ms:.2},\"target_decode_tok_s\":{target_tps:.3},\"target_decode_ms\":{target_decode_ms:.2},\"target_tok_s\":{target_tps:.3},\"target_e2e_tok_s\":{target_e2e_tps:.3},\"speculative_prefill_tok_s\":{speculative_prefill_tok_s:.3},\"speculative_prefill_ms\":{speculative_prefill_ms:.2},\"speculative_decode_tok_s\":{speculative_tps:.3},\"speculative_decode_ms\":{speculative_decode_ms:.2},\"speculative_tok_s\":{speculative_tps:.3},\"speculative_e2e_tok_s\":{speculative_e2e_tps:.3},\"speedup\":{speedup:.3},\"acceptance\":{acceptance:.6},\"accepted_drafts\":{},\"drafted_tokens\":{},\"rounds\":{},\"target_seconds\":{:.6},\"speculative_seconds\":{:.6}}}",
        speculative.accepted_drafts,
        speculative.drafted_tokens,
        speculative.rounds,
        target_seconds,
        speculative_seconds,
    );
    Ok(())
}

struct RunConfig {
    target: String,
    mtp: String,
    steps: usize,
    prompt: Vec<u32>,
}

fn run_config(args: &[String]) -> Result<RunConfig, String> {
    let positionals = cli_args::positional(args);
    let target = cli_args::request(args, "target", &positionals, 0)?
        .ok_or("missing TARGET argument (path or Hugging Face id)")?;
    let mtp = cli_args::request(args, "mtp", &positionals, 1)?
        .ok_or("missing MTP argument (path or Hugging Face id)")?;
    let steps = cli_args::value(args, "steps")?
        .or_else(|| positionals.get(2).cloned())
        .map_or(Ok(64usize), |v| {
            v.parse::<usize>()
                .map_err(|e| format!("invalid STEPS: {e}"))
        })?;
    if steps == 0 {
        return Err("STEPS must be greater than zero".into());
    }
    let prompt_csv = cli_args::value(args, "prompt")?
        .or_else(|| positionals.get(3).cloned());
    let prompt: Vec<u32> = match prompt_csv {
        Some(csv) => parse_prompt_tokens(&csv)?,
        None => vec![2005],
    };
    if prompt.is_empty() {
        return Err("PROMPT_CSV must be non-empty".into());
    }
    Ok(RunConfig {
        target,
        mtp,
        steps,
        prompt,
    })
}

fn parse_prompt_tokens(csv: &str) -> Result<Vec<u32>, String> {
    csv.split(',')
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.trim()
                .parse::<u32>()
                .map_err(|error| format!("invalid prompt token '{part}': {error}"))
        })
        .collect()
}