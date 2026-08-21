<div align="center">

# **lisa-rs**

### *L*LM *I*nference for *S*ilicon *A*rchitecture

[![Rust](https://img.shields.io/badge/Rust-1.97+-orange.svg)](https://www.rust-lang.org) [![Apple Silicon](https://img.shields.io/badge/Apple-M5%20Max-000000?style=flat&logo=apple&logoColor=white)]() [![Metal](https://img.shields.io/badge/Metal-NAX%20tensor--core-5856D6)]()

A clean-room **Rust + Metal** inference engine for `Qwen3.8-27B-4bit` on Apple Silicon — NAX tensor-core GEMM, a hybrid GDN + full-attention forward pass, and **MTP speculative decoding** with a live terminal dashboard.

Built from the silicon up: no bindings to MLX, no frameworks — just Rust talking to the Metal C API, and 14 hand-written `.metal` shaders.

</div>

---

## ✨ Highlights

| Feature | Details |
|---------|---------|
| ⚡ **Maximal throughput** | ~51 tok/s speculative, ~31 tok/s target on M5 Max — on par with the MLX reference (actually *faster* on the target pass) |
| 🧠 **MTP speculative decode** | block-3 batch verification with GPU state rollback, ~1.63× speedup over greedy target-only |
| 🔬 **NAX tensor cores** | direct access to the M5's tensor units via Metal compute |
| 🌐 **OpenAI-compatible API** | drop-in for opencode, Cursor, `@ai-sdk/openai-compatible`, curl |
| 🏗️ **Clean-room, low-dependency** | core inference deps on `serde_json` + `memmap2` — everything else hand-rolled |

---

## 🚀 Quickstart

### Install

Two installs on any Apple Silicon Mac, then you're done:

```bash
# the Hugging Face CLI (for the model weights)
brew install hf

# this project — the prebuilt binary, via the Homebrew tap
brew install agnosticeng/lisa-rs/lisa
```

`brew` puts `lisa` on your PATH automatically — no shell config needed.

### Run

Grab the weights, then start an OpenAI-compatible server:

```bash
# download the models from Hugging Face
hf download mlx-community/Qwen3.8-27B-4bit
hf download mlx-community/Qwen3.8-27B-MTP-4bit    # optional MTP drafter

# start the server
lisa serve mlx-community/Qwen3.8-27B-4bit mlx-community/Qwen3.8-27B-MTP-4bit --port 8000
```

The weights land in `~/.cache/huggingface/hub`, which the engine auto-resolves
by ID (see the model pages for details: [Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)
· [Qwen3.8-27B-MTP-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-4bit)).
The drafter is optional (skip it to run target-only).

### Try it

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Explain MTP speculation in one sentence."}],
    "max_tokens": 64
  }'
```

Or point any OpenAI-compatible client at it (opencode, Cursor,
`@ai-sdk/openai-compatible`, ...): base URL `http://localhost:8000/v1`, model
`qwen3.8-27b`.

### Benchmark

```bash
# Native target-only + MTP block-3, 64 tokens
lisa run mlx-community/Qwen3.8-27B-4bit mlx-community/Qwen3.8-27B-MTP-4bit 64

# Head-to-head vs the MLX reference on the same prompt
/tmp/mlxenv/bin/python scripts/bench_mtp.py --native
```

---

## 🏗️ Architecture

```
src/
  cache.rs     HF cache snapshot resolution (org/name → local dir)
  device/      Minimal Metal host bridge (MTLCreateSystemDefaultDevice,
               objc_msgSend, hazard-tagged buffers) — P0
  format/      safetensors container + dtype (bf16/u4) mmap parser — P0
  kernels/     .metal shaders + host dispatch
    shaders/   nax_affine_u4 (NAX tensor-core q4 GEMM + QMV decode families)
               mlp · attn.one · gdn · norm · embed
  model/       64-layer hybrid runner: 48 GDN + 16 full attention — P3
  speculative/ MTP drafter — accept_greedy, block-3 verify, reconcile — P4
  tokenizer.rs Qwen3 chat tokenizer + template (bit-exact vs HF)
  serve/       OpenAI-compatible HTTP layer (tokio + axum)
    metrics.rs Telemetry store
  cli/         `serve` + `run` subcommands (the only binary: src/bin/cli.rs)
```

### The hybrid decoder

`is_linear = (layer_idx + 1) % 4 != 0` → **full causal attention** at layers
3, 7, …, 63 and **Gated Delta Net** elsewhere:

- **GDN** — Causal K4 conv1d + SiLU, q/k RMSNorm + grouped scales, gated-delta
  recurrence in **f32** state `[48,128,128]`, gated output `RMSNorm(y) · silu(z)`.
- **Attention** — 24 Q-heads × 512, gated (`attn_output_gate`), 4 KV heads ×
  256, **partial rotary** (64 dims, theta 1e7).
- **Speculative** — the MTP drafter (31 tensors) proposes 3 tokens; the target
  verifies them in one batched **M=3 pass** (weights shared across the 3 rows),
  accepts greedily, and replays/trims as needed.

### Zero-framework Metal

The bridge binds `MTLCreateSystemDefaultDevice` and the Objective-C runtime
directly — no `metal-rs`. Weight shards are mmap'd and copied into
**shared + hazard-untracked** GPU buffers loaded once at startup. Compile
options enable fast-math for the NAX/tensor-unit shaders.

---

## ✅ Tests

```bash
cargo test                       # fast unit + GPGPU tests
cargo test -- --ignored          # real-weight correctness (needs the cache)
```

| Suite | What it proves | Anchor |
|------------|--------------------|----------|
| `metal_*` | host bridge, queues, NAX compile/run | unit |
| `mlp_decode` / `gdn_decode` / `attn_decode` | layer math on real weights | `/tmp/mlx_*.raw`, `2e-4 » 8e-3` |
| `fwd_decode` / `io_decode` | block + final norm/head | argmax exact |
| `full_runner` | 64-layer target M=1 token-2005 | argmax exact |
| `mtp_runner` | MTP drafter vs ML | argmax exact |
| `runner_checkpoint` | GDN state rollback + `commit_verified_prefix` (1/2/3 rows) | argmax-exact |
| `speculative_loop` | prefill + block-3 loop == target-only output | `exact_match` |
| `qmv` | q4 QMV parity; fast==wide3 bit-exact | — |

Integration tests must run **one at a time** (resident 16 GB shards + shared
heap): `cargo test --test <name> -- --ignored`.

---

## 📈 Performance

Greedy 64-token decode on Apple M5 Max, same 14-token prompt, medians:

| Implementation | target | MTP block-3 | Speedup | Acceptance | Verify |
|----------------|--------:|-------------|:---:|---:|:---|
| **MLX reference** (Meta) | 29.35 tok/s | 51.77 tok/s | 1.764× | 61.4% | batched block 3 |
| **lisa-rs (native)** | **~31.3 tok/s** | **~51.0 tok/s** | ~1.63× | 63.0% | batched M=3 + fused QMV |

- Target pass is *at/above* the MLX reference; MTP within ~1.5%.
- Raw results: `benchmarks/mtp_native_m5_max.json`, `mtp_mlx_m5_max.json`.

### How it stays fast

- **q4_qvm_fast** — 16 values/thread byte-mask nibble decode, `scale·sum(q·x)+bias·sum(x)` arithmetic, MLX's own layout.
- **Fused dispatch families** — gate/up (MLP), Q/K/V (attention), qkv-Δz-a-b (GDN) in one compute pass.
- **GPU argmax** — the 248k-logit head and the MTP head reduce on-device; only the index crosses the bus.
- **Parallelized glue** — RMSNorm 256-thread multigroup; attention decode q/k-norm, KV store and streaming SDPA lifted from scalar 1-thread/head to 32/256-lane kernels.

---

## 🔗 References & sources

| Project | Role |
|---------|------|
| [mlx](https://github.com/ml-explore/mlx) | reference engine, `qvm_*` arithmetic |
| [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) | Qwen3.5 attention / GDN / MTP modules (source of truth) |
| [saragossa](https://github.com/azerozero/saragossa) | NAX tensor-core + GDN shader port |
| [lattice](https://github.com/ohdearquant/lattice) | GEMM / attention / fusion kernels |
| [mlx-native](https://github.com/robertelee78/mlx-native) | Rust/MLX hybrid port |
| [candle](https://github.com/huggingface/candle) | concise Metal kernel cases |

---

## License

MIT — see [LICENSE](LICENSE). It is a reference implementation of Apple's Metal
tensor units and Meta's Qwen3.5 building blocks; the model weights and any
upstream shaders are subject to their own terms.
