#!/usr/bin/env python3
"""Run the reproducible MLX target-only and Qwen MTP reference benchmark."""
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

HOME = os.path.expanduser("~")
TARGET_CACHE = f"{HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots"
DRAFT_CACHE = f"{HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-MTP-4bit/snapshots"
DEFAULT_PROMPT = "Implement an in-place stable merge sort in Rust and explain its complexity."


def snapshot(cache):
    matches = sorted(glob.glob(f"{cache}/*/config.json"))
    if not matches:
        raise SystemExit(f"no cached model under {cache}")
    return os.path.dirname(matches[-1])


def prompt_token_csv(model_path, prompt):
    """Tokenize the prompt with the model's own tokenizer and return a CSV of ids."""
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(model_path)
    return ",".join(str(token_id) for token_id in tokenizer.encode(prompt))


def run_native(target, draft, tokens, prompt_csv):
    """Run the Rust native benchmark once and parse its single-line JSON."""
    command = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "cli",
        "--",
        "run",
        target,
        draft,
        str(tokens),
        prompt_csv,
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, cwd=os.path.dirname(__file__) + "/.."
    )
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"no JSON in native benchmark output:\n{result.stdout}{result.stderr}")


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


def run(target, draft, prompt, tokens, show_output):
    command = [
        sys.executable,
        "-m",
        "mlx_vlm.generate",
        "--model",
        target,
        "--prompt",
        prompt,
        "--max-tokens",
        str(tokens),
        "--temperature",
        "0",
        "--verbose",
    ]
    if draft:
        command += [
            "--draft-model",
            draft,
            "--draft-kind",
            "mtp",
            "--draft-block-size",
            "3",
        ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if show_output:
        print(output)
    return parse_metrics(output)


def median(values, field):
    return statistics.median(getattr(value, field) for value in values)


parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, default=64)
parser.add_argument("--prompt", default=DEFAULT_PROMPT)
parser.add_argument("--runs", type=int, default=3)
parser.add_argument("--show-output", action="store_true")
parser.add_argument("--json", action="store_true", dest="json_output")
parser.add_argument("--output", help="write the JSON summary to this path")
parser.add_argument(
    "--native", action="store_true", help="also run the Rust native benchmark (same prompt)"
)
args = parser.parse_args()
if args.runs < 1:
    parser.error("--runs must be positive")
target = snapshot(TARGET_CACHE)
draft = snapshot(DRAFT_CACHE)

target_runs = []
mtp_runs = []
for index in range(args.runs):
    print(f"run {index + 1}/{args.runs}: target", flush=True)
    target_metric = run(target, None, args.prompt, args.tokens, args.show_output)
    target_runs.append(target_metric)
    print(f"  {target_metric.tokens_per_second:.3f} tok/s", flush=True)
    print(f"run {index + 1}/{args.runs}: MTP block 3", flush=True)
    mtp_metric = run(target, draft, args.prompt, args.tokens, args.show_output)
    mtp_runs.append(mtp_metric)
    print(
        f"  {mtp_metric.tokens_per_second:.3f} tok/s, "
        f"acceptance {mtp_metric.acceptance_percent:.1f}%",
        flush=True,
    )

target_tps = median(target_runs, "tokens_per_second")
mtp_tps = median(mtp_runs, "tokens_per_second")
summary = {
    "tokens": args.tokens,
    "runs": args.runs,
    "target_snapshot": target,
    "draft_snapshot": draft,
    "target_median_tokens_per_second": target_tps,
    "mtp_median_tokens_per_second": mtp_tps,
    "speedup": mtp_tps / target_tps,
    "target_median_peak_memory_gb": median(target_runs, "peak_memory_gb"),
    "mtp_median_peak_memory_gb": median(mtp_runs, "peak_memory_gb"),
    "mtp_median_acceptance_percent": median(mtp_runs, "acceptance_percent"),
    "target_runs": [asdict(value) for value in target_runs],
    "mtp_runs": [asdict(value) for value in mtp_runs],
}

if args.native:
    prompt_csv = prompt_token_csv(target, args.prompt)
    native_runs = []
    for index in range(args.runs):
        print(f"native {index + 1}/{args.runs}", flush=True)
        native = run_native(target, draft, args.tokens, prompt_csv)
        native_runs.append(native)
        print(
            f"  target {native['target_tok_s']:.3f} tok/s, "
            f"MTP {native['speculative_tok_s']:.3f} tok/s, "
            f"prefill target {native['target_prefill_tok_s']:.1f}/spec {native['speculative_prefill_tok_s']:.1f} tok/s, "
            f"acceptance {native['acceptance'] * 100:.1f}%, "
            f"exact_match={native['exact_match']}",
            flush=True,
        )
    summary["native"] = {
        "prompt_tokens": prompt_csv.count(",") + 1,
        "target_tokens_per_second": [r["target_tok_s"] for r in native_runs],
        "mtp_tokens_per_second": [r["speculative_tok_s"] for r in native_runs],
        "target_median_tokens_per_second": statistics.median(
            r["target_tok_s"] for r in native_runs
        ),
        "mtp_median_tokens_per_second": statistics.median(
            r["speculative_tok_s"] for r in native_runs
        ),
        "target_prefill_tokens_per_second": [r["target_prefill_tok_s"] for r in native_runs],
        "speculative_prefill_tokens_per_second": [
            r["speculative_prefill_tok_s"] for r in native_runs
        ],
        "target_median_prefill_tokens_per_second": statistics.median(
            r["target_prefill_tok_s"] for r in native_runs
        ),
        "speculative_median_prefill_tokens_per_second": statistics.median(
            r["speculative_prefill_tok_s"] for r in native_runs
        ),
        "acceptance": native_runs[0]["acceptance"],
        "exact_match": all(r["exact_match"] for r in native_runs),
        "runs": native_runs,
    }

print("\n=== comparison ===")
print(f"target median : {target_tps:.3f} tok/s")
print(f"MTP median    : {mtp_tps:.3f} tok/s")
print(f"speedup       : {summary['speedup']:.3f}x")
print(f"acceptance    : {summary['mtp_median_acceptance_percent']:.1f}%")
if args.native:
    nat = summary["native"]
    print(
        f"prefill        : MLX {median(mtp_runs, 'prompt_tokens_per_second'):.1f} tok/s "
        f"<< native target {nat['target_median_prefill_tokens_per_second']:.1f} tok/s, "
        f"native spec {nat['speculative_median_prefill_tokens_per_second']:.1f} tok/s"
    )
    print(
        f"decode         : MLX target {target_tps:.1f} / spec {mtp_tps:.1f} tok/s "
        f"<< native target {nat['target_median_tokens_per_second']:.1f} / spec {nat['mtp_median_tokens_per_second']:.1f} tok/s"
    )
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
