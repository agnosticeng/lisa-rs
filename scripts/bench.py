#!/usr/bin/env python3
"""MLX reference + lisa-rs native benchmark on a prompt.

Runs the Qwen MTP reference (mlx_vlm.generate) and optionally the Rust native
engine (`cli run`) on the SAME prompt, and prints both side by side with a delta.

Prompt selection:
  --prompt <text>          a full prompt string (default: a short prompt)
  --prompt-file <path>     read the prompt text from a file
  --prompt-len N           generate a deterministic pseudo-random prompt of ~N
                           tokens on the fly (native runs get the exact token
                           CSV, mlx re-tokenizes the decoded text). Default 0
                           (use --prompt / --prompt-file / short default).

By default both engines run on a SHORT prompt so a quick mlx-vs-lisa comparison
finishes in under a minute. Pass `--prompt-len 4096` for a heavy prefill run.

Run modes:
  (default)                MLX reference + native engine (--runs runs)
  --mlx-only / --native-only

Usage examples:
  python3 scripts/bench.py --runs 1              # MLX + native, short prompt
  python3 scripts/bench.py --prompt-len 4096 --runs 3
  python3 scripts/bench.py --prompt "hello world" --native-only
"""
import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Optional

# Bootstrap the repo MLX venv (installs mlx/mlx-lm/mlx-vlm on first use) and
# re-exec under it, so a bare `python3 scripts/bench.py` has no dependency on
# any pre-installed Python environment.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx_env  # noqa: E402

mlx_env.reexec()

HOME = os.path.expanduser("~")
TARGET_CACHE = f"{HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots"
DRAFT_CACHE = f"{HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-MTP-4bit/snapshots"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROMPT = "What is the capital of France? Answer in one sentence."


def snapshot(cache):
    matches = sorted(glob.glob(f"{cache}/*/config.json"))
    if not matches:
        raise SystemExit(f"no cached model under {cache}")
    return os.path.dirname(matches[-1])


def target_snapshot():
    return snapshot(TARGET_CACHE)


# ---------------------------------------------------------------------------
# Synthetic prompt (tokens generated on the fly, deterministic)
# ---------------------------------------------------------------------------

# Deterministic LCG so every run/engine sees the same token stream.
def _gen_tokens(n: int, seed: int = 0xC0FFEE) -> list[int]:
    tokens = []
    state = seed
    for _ in range(n):
        # Modulus must stay < VOCAB (248320) and > 4 so a decode always
        # embeds a real row; keep values in the mid-range of common ids.
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        tokens.append(1000 + (state % 100_000))
    return tokens


def resolve_prompt(args):
    """Return `(text, tokens)` where `tokens` is a list when the native side
    must run on a fixed synthetic prompt (--prompt-len), else None."""
    if args.prompt_len:
        tokens = _gen_tokens(args.prompt_len)
        # Best-effort decode for mlx: replace unknown ids keeps a token stream.
        try:
            from mlx_lm.utils import load_tokenizer
            tokenizer = load_tokenizer(target_snapshot())
            text = tokenizer.decode(tokens)
        except Exception:
            text = " ".join(str(t) for t in tokens[:200])
        return text, tokens
    if args.prompt_file:
        with open(args.prompt_file) as f:
            return f.read(), None
    return args.prompt, None


# ---------------------------------------------------------------------------
# Native engine
# ---------------------------------------------------------------------------

def run_native(target, draft, tokens_to_gen, prompt_csv):
    """Run the Rust native benchmark once and parse its single-line JSON.
    `prompt_csv` is a comma-joined token-id string (empty -> CLI default)."""
    if prompt_csv:
        command = [
            "cargo", "run", "--release", "--bin", "cli", "--",
            "run", target, draft, str(tokens_to_gen), prompt_csv,
        ]
    else:
        command = [
            "cargo", "run", "--release", "--bin", "cli", "--",
            "run", target, draft, str(tokens_to_gen),
        ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, cwd=ROOT
    )
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"no JSON in native benchmark output:\n{result.stdout}{result.stderr}")


# ---------------------------------------------------------------------------
# MLX reference (mlx_vlm.generate)
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    tokens_per_second: float
    peak_memory_gb: float
    prompt_tokens_per_second: float
    accepted_tokens_per_round: Optional[float] = None
    accepted_drafts_per_round: Optional[float] = None
    acceptance_percent: Optional[float] = None
    rounds: Optional[int] = None


def parse_metrics(output):
    generation = re.search(r"Generation: \d+ tokens, ([0-9.]+) tokens-per-sec", output)
    prompt = re.search(r"Prompt: \d+ tokens, ([0-9.]+) tokens-per-sec", output)
    memory = re.search(r"Peak memory: ([0-9.]+) GB", output)
    if not generation or not prompt or not memory:
        raise RuntimeError(f"could not parse mlx_vlm metrics:\n{output}")
    speculative = re.search(
        r"Speculative decoding: ([0-9.]+) accepted tokens/round "
        r"\(([0-9.]+) accepted drafts/round, ([0-9.]+)% of drafted, "
        r"avg draft [0-9.]+\) over (\d+) rounds",
        output,
    )
    metrics = Metrics(
        tokens_per_second=float(generation.group(1)),
        peak_memory_gb=float(memory.group(1)),
        prompt_tokens_per_second=float(prompt.group(1)),
    )
    if speculative:
        metrics.accepted_tokens_per_round = float(speculative.group(1))
        metrics.accepted_drafts_per_round = float(speculative.group(2))
        metrics.acceptance_percent = float(speculative.group(3))
        metrics.rounds = int(speculative.group(4))
    return metrics


def run_mlx(target, draft, prompt, tokens, show_output):
    command = [
        sys.executable, "-m", "mlx_vlm.generate",
        "--model", target,
        "--prompt", prompt,
        "--max-tokens", str(tokens),
        "--temperature", "0",
        "--verbose",
    ]
    if draft:
        command += [
            "--draft-model", draft,
            "--draft-kind", "mtp",
            "--draft-block-size", "3",
        ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if show_output:
        print(output)
    return parse_metrics(output)


def prompt_token_csv(model_path, prompt):
    """Tokenize the prompt with the model's own tokenizer and return a CSV of ids."""
    from mlx_lm.utils import load_tokenizer
    tokenizer = load_tokenizer(model_path)
    return ",".join(str(token_id) for token_id in tokenizer.encode(prompt))


def median(values, field):
    return statistics.median(getattr(value, field) for value in values)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=64, help="generated tokens")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt text")
    ap.add_argument("--prompt-file", help="read prompt text from a file")
    ap.add_argument("--prompt-len", type=int, default=0,
                    help="generate a synthetic ~N-token prompt on the fly")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--show-output", action="store_true")
    ap.add_argument("--json", action="store_true", dest="json_output")
    ap.add_argument("--output", help="write the JSON summary to this path")
    ap.add_argument("--native", action="store_true", help="also run the Rust native benchmark")
    ap.add_argument("--native-only", action="store_true", help="run only native")
    ap.add_argument("--mlx-only", action="store_true", help="run only MLX")
    args = ap.parse_args()
    if args.runs < 1:
        ap.error("--runs must be positive")

    if args.native_only and args.mlx_only:
        ap.error("--native-only and --mlx-only are mutually exclusive")
    do_native = args.native or args.native_only or (not args.mlx_only)
    do_mlx = args.mlx_only or (not args.native_only)
    if not do_mlx:
        do_native = True

    target = snapshot(TARGET_CACHE)
    draft = snapshot(DRAFT_CACHE)

    target = snapshot(TARGET_CACHE)
    draft = snapshot(DRAFT_CACHE)

    prompt_text, synth_tokens = resolve_prompt(args)
    ntoks = len(synth_tokens) if synth_tokens else len(prompt_text.split())
    label = (
        f"{os.path.basename(args.prompt_file)}"
        if args.prompt_file
        else f"len={len(synth_tokens)} tokens"
        if synth_tokens
        else args.prompt[:48]
    )
    print(f"prompt: ~{ntoks} tokens  ({label}…)", flush=True)
    native_prompt_csv = ""
    if do_native:
        native_prompt_csv = (
            ",".join(str(t) for t in synth_tokens)
            if synth_tokens
            else prompt_token_csv(target, prompt_text)
        )

    # ---- collect MLX runs ----
    target_runs, mtp_runs = [], []
    if do_mlx and not args.native_only:
        for index in range(args.runs):
            print(f"run {index + 1}/{args.runs}: target", flush=True)
            t = run_mlx(target, None, prompt_text, args.tokens, args.show_output)
            target_runs.append(t)
            print(f"  {t.tokens_per_second:.3f} tok/s", flush=True)
            print(f"run {index + 1}/{args.runs}: MTP block 3", flush=True)
            m = run_mlx(target, draft, prompt_text, args.tokens, args.show_output)
            mtp_runs.append(m)
            print(f"  {m.tokens_per_second:.3f} tok/s, acceptance {m.acceptance_percent:.1f}%", flush=True)

    target_tps = median(target_runs, "tokens_per_second") if target_runs else None
    mtp_tps = median(mtp_runs, "tokens_per_second") if mtp_runs else None

    summary = {
        "tokens": args.tokens,
        "runs": args.runs,
        "prompt_tokens": ntoks,
        "target_snapshot": target,
        "draft_snapshot": draft,
        "target_median_tokens_per_second": target_tps,
        "mtp_median_tokens_per_second": mtp_tps,
        "speedup": (mtp_tps / target_tps) if target_tps and mtp_tps else None,
        "target_median_peak_memory_gb": median(target_runs, "peak_memory_gb") if target_runs else None,
        "mtp_median_peak_memory_gb": median(mtp_runs, "peak_memory_gb") if mtp_runs else None,
        "mtp_median_acceptance_percent": median(mtp_runs, "acceptance_percent") if mtp_runs else None,
        "target_runs": [asdict(v) for v in target_runs],
        "mtp_runs": [asdict(v) for v in mtp_runs],
    }

    # ---- native runs ----
    native = None
    if do_native:
        native_runs = []
        for index in range(args.runs):
            print(f"native {index + 1}/{args.runs}", flush=True)
            nr = run_native(target, draft, args.tokens, native_prompt_csv)
            native_runs.append(nr)
            print(
                f"  target {nr['target_tok_s']:.3f} tok/s, "
                f"MTP {nr['speculative_tok_s']:.3f} tok/s, "
                f"prefill target {nr['target_prefill_tok_s']:.1f}/spec {nr['speculative_prefill_tok_s']:.1f} tok/s, "
                f"acceptance {nr['acceptance'] * 100:.1f}%, "
                f"exact_match={nr['exact_match']}",
                flush=True,
            )
        native = {
            "prompt_tokens": ntoks,
            "target_tokens_per_second": [r["target_tok_s"] for r in native_runs],
            "mtp_tokens_per_second": [r["speculative_tok_s"] for r in native_runs],
            "target_median_tokens_per_second": statistics.median(r["target_tok_s"] for r in native_runs),
            "mtp_median_tokens_per_second": statistics.median(r["speculative_tok_s"] for r in native_runs),
            "target_median_prefill_tokens_per_second": statistics.median(r["target_prefill_tok_s"] for r in native_runs),
            "speculative_median_prefill_tokens_per_second": statistics.median(r["speculative_prefill_tok_s"] for r in native_runs),
            "acceptance": native_runs[0]["acceptance"],
            "exact_match": all(r["exact_match"] for r in native_runs),
            "runs": native_runs,
        }
        summary["native"] = native

    # ---- comparison ----
    print("\n=== comparison ===")
    if target_tps and mtp_tps:
        print(f"target median : {target_tps:.3f} tok/s")
        print(f"MTP median    : {mtp_tps:.3f} tok/s")
        print(f"speedup       : {summary['speedup']:.3f}x")
        print(f"acceptance    : {summary['mtp_median_acceptance_percent']:.1f}%")
    if native and target_tps and mtp_tps:
        print(
            f"prefill        : MLX {median(mtp_runs, 'prompt_tokens_per_second'):.1f} tok/s "
            f"<< native target {native['target_median_prefill_tokens_per_second']:.1f} tok/s, "
            f"native spec {native['speculative_median_prefill_tokens_per_second']:.1f} tok/s"
        )
        print(
            f"decode         : MLX target {target_tps:.1f} / spec {mtp_tps:.1f} tok/s "
            f"<< native target {native['target_median_tokens_per_second']:.1f} / spec {native['mtp_median_tokens_per_second']:.1f} tok/s"
        )
        print("\n== native vs mlx delta ==")
        print(f"  target decode : {(native['target_median_tokens_per_second']/target_tps-1)*100:+.1f}%")
        print(f"  mtp  decode   : {(native['mtp_median_tokens_per_second']/mtp_tps-1)*100:+.1f}%")
    if target_runs:
        print(
            "peak memory  : "
            f"{summary['target_median_peak_memory_gb']:.3f} GB target, "
            f"{summary['mtp_median_peak_memory_gb']:.3f} GB MTP"
        )

    if args.json_output:
        print(json.dumps(summary, indent=2))
    if args.output:
        with open(args.output, "w") as output_file:
            json.dump(summary, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()