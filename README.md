<p align="left">
  <img src="https://img.shields.io/github/license/agnosticeng/lisa-rs?color=brightgreen" alt="License: MIT">
  <img src="https://img.shields.io/badge/language-Rust-orange" alt="Rust">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/version-0.1.0-blue" alt="version">
</p>

# lisa-rs

**LLM Inference for Silicon Architecture**

A clean-room **Rust + Metal** inference engine for `Qwen3.8-27B-4bit` on Apple
Silicon. NAX tensor-core GEMM, a hybrid GDN + full-attention forward pass, and
MTP speculative decoding with a live terminal dashboard.

Built from the silicon up, no bindings to MLX, no frameworks. Rust talking
directly to the Metal C API, and 14 hand-written `.metal` shaders.

## What it does

| Feature | Details |
|---------|---------|
| **Maximal throughput** | ~51 tok/s speculative, ~31 tok/s target on M5 Max, at or above the MLX reference |
| **MTP speculative decode** | block-3 batch verification with GPU state rollback, ~1.63x over greedy target-only |
| **NAX tensor cores** | direct access to the M5's tensor units via Metal compute |
| **OpenAI-compatible API** | drop-in for opencode, Cursor, `@ai-sdk/openai-compatible`, curl |
| **Clean-room, low-dependency** | core inference deps on `serde_json` + `memmap2`; everything else hand-rolled |

## Install

Two installs on any Apple Silicon Mac:

```bash
brew install hf                                                        # Hugging Face CLI (model weights)
brew install agnosticeng/lisa-rs/lisa                                  # this project
```

`brew` puts `lisa` on your PATH automatically.

## Run

Grab the weights, then start a server:

```bash
hf download mlx-community/Qwen3.8-27B-4bit
hf download mlx-community/Qwen3.8-27B-MTP-4bit    # optional MTP drafter

lisa serve mlx-community/Qwen3.8-27B-4bit mlx-community/Qwen3.8-27B-MTP-4bit --port 8000
```

The weights land in `~/.cache/huggingface/hub`, auto-resolved by ID
([Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)
· [Qwen3.8-27B-MTP-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-4bit)).
The drafter is optional, skip it to run target-only.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Explain MTP speculation in one sentence."}],
    "max_tokens": 64
  }'
```

Point any OpenAI-compatible client at `http://localhost:8000/v1` with model
`qwen3.8-27b`.

For benchmark commands and performance numbers, see [BENCHMARK.md](BENCHMARK.md).

## Architecture

```
src/
  cache.rs      HF cache snapshot resolution (org/name → local dir)
  device/       Minimal Metal host bridge (MTLCreateSystemDefaultDevice,
                objc_msgSend, hazard-tagged buffers), P0
  format/       safetensors container + dtype (bf16/u4) mmap parser, P0
  kernels/      .metal shaders + host dispatch
    shaders/    per-family Metal kernels: gemm (prefill GEMM) · qmv (decode
                QMV) · mlp · attention · gdn · norm · embed
  model/        64-layer hybrid runner: 48 GDN + 16 full attention, P3
  speculative/  MTP drafter (accept_greedy, block-3 verify, reconcile), P4
  tokenizer.rs  Qwen3 chat tokenizer + template (bit-exact vs HF)
  serve/       OpenAI-compatible HTTP layer (tokio + axum)
    metrics.rs Telemetry store
  cli/          serve + run subcommands (src/cli/main.rs, the only binary)
```

### The hybrid decoder

`is_linear = (layer_idx + 1) % 4 != 0` → **full causal attention** at layers
3, 7, …, 63, **Gated Delta Net** elsewhere.

- **GDN**, Causal K4 conv1d + SiLU, q/k RMSNorm + grouped scales, gated-delta
  recurrence in **f32** state `[48,128,128]`, gated output `RMSNorm(y) · silu(z)`.
- **Attention**, 24 Q-heads × 512, gated (`attn_output_gate`), 4 KV heads ×
  256, **partial rotary** (64 dims, theta 1e7).
- **Speculative**, the MTP drafter (31 tensors) proposes 3 tokens; the target
  verifies them in one batched **M=3 pass** (weights shared across the 3 rows),
  accepts greedily, and replays/trims as needed.

### Zero-framework Metal

The bridge binds `MTLCreateSystemDefaultDevice` and the Objective-C runtime
directly, no `metal-rs`. Weight shards are mmap'd and copied into
**shared + hazard-untracked** GPU buffers loaded once at startup. Compile
options enable fast-math on the NAX/tensor-unit shaders.

## Tests

```bash
cargo test                       # fast units + GPGPU tests
cargo test -- --ignored          # real-weight correctness (needs the cache)
```

| Suite | What it proves | Anchor |
|-------|----------------|--------|
| `metal_*` | host bridge, queues, NAX compile/run | unit |
| `mlp_decode` / `gdn_decode` / `attn_decode` | layer math on real weights | `/tmp/mlx_*.raw`, `2e-4 » 8e-3` |
| `fwd_decode` / `io_decode` | block + final norm/head | argmax exact |
| `full_runner` | 64-layer target M=1 token-2005 | argmax exact |
| `mtp_runner` | MTP drafter vs ML | argmax exact |
| `runner_checkpoint` | GDN state rollback + `commit_verified_prefix` (1/2/3 rows) | argmax-exact |
| `speculative_loop` | prefill + block-3 loop == target-only output | `exact_match` |
| `qmv` | q4 QMV parity; fast==wide3 bit-exact | |

Integration tests must run **one at a time** (resident 16 GB shards + shared
heap): `cargo test --test <name> -- --ignored`.

## References

| Project | Role |
|---------|------|
| [mlx](https://github.com/ml-explore/mlx) | reference engine, `v` arithmetic |
| [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) | Qwen3.5 attention / GDN / MTP modules |
| [saragossa](https://github.com/azerozero/saragossa) | NAX tensor-core + GDN port |
| [lattice](https://github.com/ohdearquant/lattice) | GEMM / attention / fusion kernels |
| [mlx-native](https://github.com/robertelee78/mlx-native) | Rust/MLX hybrid port |
| [candle](https://github.com/huggingface/candle) | concise Metal kernel cases |

## License

MIT, see [LICENSE](LICENSE). A reference implementation of Apple's Metal
tensor units and Meta's Qwen3.5 building blocks; the model weights and any
upstream shaders are subject to their own terms.