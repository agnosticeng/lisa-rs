# Benchmarks

How lisa-rs compares against the MLX reference (`mlx_vlm.generate`) on the same
machine and prompt.

## Running

```bash
python3 scripts/bench.py --native        # short quick comparison
python3 scripts/bench.py --prompt-len 512 --runs 2   # 512-token prompt
lisa run mlx-community/Qwen3.8-27B-4bit mlx-community/Qwen3.8-27B-MTP-4bit 64
```

- `python3 scripts/bench.py` needs no pre-installed env: it bootstraps a private
  MLX venv (`~/.cache/lisa-rs/mlxenv`, pinned `scripts/requirements-mlx.txt`)
  on first use and re-executes itself under it.
- `lisa run` reports a single-line JSON splitting target vs speculative, prefill
  vs decode.
- `bench.py` runs the MLX reference and the native benchmark on the same prompt
  end to end and prints a delta (`--native` adds native; `--prompt-len N`
  generates a deterministic synthetic prompt of ~N tokens on the fly; by default
  both engines run on a short prompt for a quick comparison).

## Greedy decode (64 tokens, 14-token prompt, Apple M5 Max)

Medians over several runs:

| Implementation | target | MTP block-3 | Speedup | Acceptance |
|----------------|--------|-------------|---------|-----------|
| **MLX reference** | 29.35 tok/s | 51.77 tok/s | 1.764× | 61.4% |
| **lisa-rs (native)** | **~31.3 tok/s** | **~51.0 tok/s** | ~1.63× | 63.0% |

The target pass is at/above the MLX reference; MTP within ~1.5%.

## Recent session (512-token prompt, `--runs 2`)

`python3 scripts/bench.py --prompt-len 512 --runs 2`:

| Metric | MLX | Native | Delta |
|--------|-----|--------|-------|
| target decode | 32.1 tok/s | 30.9 tok/s | **-3.7%** |
| MTP decode | 52.0 tok/s | 41.2 tok/s | **-20.8%** |
| prefill (target) | 836.9 tok/s | 696.7 tok/s | **-16.8%** |
| prefill (spec) | n/a | 518.6 tok/s | |
| acceptance | 56.7% | 72.0% | +15pp |

Notes:

- **Target decode near-parity** (-3.7%), in the expected range.
- **MTP decode is the weak spot** (-20.8%): native spec is 41 vs MLX 52.
  Native acceptance is higher (72% vs 57%) but the draft forward cost
  dominates, draft-side work is where the next win lives.
- **Prefill still behind** (-16.8%): GEMM-bound, chunked at `PREFILL_BATCH=512`
  vs MLX's whole-prompt GEMM. See AGENTS.md's prefill roadmap.

## How it stays fast

- **q4_affine_u4**, 16 values/thread byte-mask nibble decode,
  `scale·sum(q·x) + bias·sum(x)`, MLX's own layout.
- **Fused dispatch families**, gate/up (MLP), Q/K/V (attention),
  qkv+Δz+a+b (GDN) in a single compute pass.
- **GPU argmax**, the 248k-logit head and the MTP head reduce on-device; only
  the index crosses the bus.
- **Native glue**, RMSNorm 256-thread multigroup; attention decode q/k-norm,
  KV store and streaming SDPA lifted to 32/256-lane kernels.