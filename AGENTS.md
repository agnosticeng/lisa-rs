# AGENTS.md

Working guide for the coding agent on this repo. Read before any modification.
This file is the single source of truth: goal, architecture (current design
decisions), the blockers/correctness gotchas we have hit, model geometry, and
the exact commands. Keep it updated as steps turn green.

Sections: Goal · Repository map · Model geometry/constants · Data flow (sliced
decode, near-tie drift) · MLX specifics · Architecture choices (forward, Metal
discipline, MTP device-resident, batched prefill, tiled GEMM, session cache,
benchmarks) · Prefill performance state · Blockers/gotchas ·
Environment/commands · Reference implementation repos (GitHub).

## Goal

**lisa-rs (LLM Inference for Silicon Architecture)**: clean-room Rust + Metal
inference engine for
`mlx-community/Qwen3.8-27B-4bit` (+ `-MTP-4bit` drafter) on Apple M5 Max:
NAX tensor-core, hybrid forward, MTP speculative decode. Priorities: fastest
prefill/decode.

No external Metal/GPU crate: `src/device/metal.rs` binds the Metal C API +
Objective-C runtime directly. The inference core depends only on `serde_json` +
`memmap2` (host-side parsing); the serving layer (`src/serve`, `src/tokenizer`)
adds `tokio`/`axum` (HTTP) and `tokenizers` (the HF tokenizer, loaded from
`tokenizer.json`) — those are optional and do not touch the Metal path.

## Repository map (file-by-file)

```
src/
  lib.rs                 crate root: module decls.
  cache.rs               resolve_snapshot(model): turn a HF id (`org/name`) or an
                         explicit directory into a local snapshot dir via the HF
                         cache (`~/.cache/huggingface/hub`, or `$HF_HOME/hub`);
                         prefers `refs/main`, falls back to any dir with .safetensors.
  device/                Host Metal bridge (P0)
    metal.rs             Minimal Metal host bridge. Binds MTLCreateSystemDefaultDevice,
                         objc_msgSend/objc_getClass/sel_registerName via #[link]. Types:
                         MetalDevice, MetalBuffer, CommandQueue, CommandBuffer, Library,
                         Function, ComputePipeline, ComputeEncoder, Heap. Buffers are
                         MTLResourceStorageModeShared; new_untracked_buffer adds
                         HAZARD_TRACKING_UNTRACKED (used for immutable weight shards).
                         Compile options enable fast-math. Test prelude in-file. Owns
                         with_autorelease_pool, ns_string, sel, ns_to_string helpers.
    mod.rs               pub mod metal;
  format/                safetensors container + dtype (P0)
    dtype.rs             Dtype enum, dtype_from_name, bf16_to_f32, f32_to_bf16
                         (truncation, no rounding).
    safetensor.rs        safetensors parser: [u64 LE header_len][header json][data bytes].
                         open() -> Shard { mmap, data_start, data_len, tensors }.
                         parse_header skips __metadata__/__header__.
    mod.rs               pub mod dtype; pub mod safetensor;
  kernels/               .metal shaders + host dispatch (P1/P2)
    linear.rs            include_str! of the seven shader sources, exported as const &
                         (NAX_GEMM_SHADER, NAX_QMV_SHADER, plus NAX_AFFINE_U4_SHADER =
                         concat of the two for the smoke tests, MLP_SHADER,
                         ATTENTION_SHADER, GDN_SHADER, NORM_SHADER, EMBED_SHADER).
                         Rebuild required for shader changes (include_str! caching).
    shaders/             one file per compute family, split by execution phase:
      gemm.metal         Batched prefill NAX tensor-core affine-U4 GEMM kernels.
                         q4_gemm_nax_tiled_align: the prefill GEMM (DEFAULT) —
                         branchless 64×64 cooperative tile (128 thr, 4 SIMD),
                         MLX QuantizedBlockLoader-style coalesced dequant into a
                         padded threadgroup tile, per-SIMD 32x32 C sub-tiles,
                         NO per-lane predication. Requires M,N % 64 == 0 (all
                         projections except GDN a/b gates). ~55 TFLOPS vs ~33
                         for the masked variant. q4_gemm_nax_tiled: same, with
                         masked M/N tails, used ONLY for unaligned gates.
                         q4_gemm_nax_tiled_align_fused2: gate+up (or any two
                         same-geometry projections) in one dispatch via grid.z.
                         q4_gemm_nax_coop (16x32, 1 SIMD, dequant-in-register)
                         kept as reference (bit-exactness oracle).
      qmv.metal           Decode M=1/M=3 affine-U4 matrix-vector kernels.
                         q4_qmv_fast/_impl: M=1 matrix-vector (4 simdgroups,
                         16 rows/TG, packed-word 8 nibbles/load). q4_qmv_wide3/_impl:
                         M=3, one weight pass reused by 3 rows, x streamed per row.
                         Fused dispatch families (grid.y selects independent weights):
                         fused2/3/4 (M=1) and fused2/3/4_wide3 (M=3). Fused2=MLP
                         gate/up, fused3=attention Q/K/V, fused4=GDN qkv/z/a/b.
      mlp.metal           silu_mul_up: SiLU(gate)*up -> bf16 (MLP SwiGLU intermediate).
      attention.metal     Qwen3.5 full attention (24x512 q, 4x256 kv, dim 256, partial
                         rotary 0.25 -> 64 dims, theta 1e7, NeoX). rms_norm_rope helper,
                         qk_norm_gate_rope, k_norm_rope (legacy batch), decode block:
                         q_norm_gate_rope_decode/_block3, kv_cache_store_decode/_block3,
                         sdpa_decode_streaming/_block3 (legacy serial online softmax),
                         sdpa_decode_partial/_final (FLASH-STYLE SLICED decode, the runner
                         uses these, see below), gate_out_bf16 (out*sigmoid(gate)),
sdpa_scalar.
      gdn.metal           Gated Delta Net (saragossa port). gdn_conv_norm_gates (legacy,
                          stride-64 pads) + gdn_conv_norm_gates_bf16_weights (runner,
                          stride-48, in-buffer BF16 aux weights). gdn_update_conv_state
                          (K=4 history, state-slot capture guarded by slot_stride>0).
                          gdn_recurrence (f32 state [48,128,128], GQA vh->kh, simd_sum
                          reduction, state-slot capture). gdn_rms_gate +
                          gdn_rms_gate_bf16_weight (y rms-norm * silu(z)).
      norm.metal          rms_norm_rows (256-thread multi-simdgroup, 2-level simd_sum +
                          shared reduction), residual_add, residual_add_rms_norm_rows (fused
                          residual+norm, DISABLED in runner, see gotchas), clear_bytes,
                          copy_bytes (16B/thread device copy), f32_to_bf16,
                          rms_norm_rows_stride (strided variant for the MTP fc input).
      embed.metal         affine_u4_lookup: indexed affine-U4 embedding row lookup.
                          argmax_f32_partial: hierarchical GPU argmax over f32 logits
                          (256-thread threadgroups -> per-TG partials).
    mod.rs               pub mod linear;
  model/                 model geometry + runners (P3/P4)
    mod.rs               LAYERS=64, HIDDEN=5120, VOCAB=248320. LayerKind{Gdn,Attention},
                         layer_kind(i)=Attention iff (i+1)%4==0 (layers 3..63). Unit test
                         asserts 16 attention / 48 GDN.
    weights.rs           WeightIndex: opens all shards, name -> WeightSlot{shard, offset,
                         len, dtype, shape}. bytes(), rows(), shard_count(), shard_bytes().
                         The runner mmaps the shards, validates every tensor
                         (start<=end<=data_len, numel*bits == end-start), then copies each
                         shard's live byte range into ONE StorageModeShared Metal buffer —
                         header-validated, lazy-mmap, eager-copy weight arena (weights stay
                         resident; kernels address by (shard_idx, byte_offset)).
    runner.rs            THE core file (~2600 lines). QwenRunner (64-layer M=1/M=3
                         target), MtpRunner (31-tensor MTP head), Pipelines (all compiled
                         compute pipelines), ScratchLayout/MtpScratchLayout (offsets into
                         one reusable scratch buffer), encode_* kernels for every stage,
                         load/build_layers/build_mtp_weights, read_f32/read_bf16/argmax
                         helpers, forward_token_decode + verify_block3_decode (GPU argmax,
                         no logits readback), forward_prefill (batched tiled-NAX-GEMM prefill in
                         PREFILL_BATCH=512 chunks), MtpRunner::forward_position_batch (batched
                         MTP-head prefill)/forward_position_decode (GPU
                         MTP head argmax), profile_token_m1_diagnostic (per-family timing).
  speculative/           MTP logic (P4)
    mod.rs               accept_greedy, MtpReconcilePlan/mtp_reconcile_plan (mirrors mlx
                         accept_verified_tokens), Gemini-like run_block3_loop (target M=3
                         verify_block3 + mtp trim/replay/replacement), plus prefill_prompt
                         (batched target+MTD prefill via forward_prefill +
                         forward_position_batch when the prompt is >=4 tokens, else M=1;
                         mirrors mlx prefill_from_target_hidden), generate_greedy_block3_prefilled,
                         and generate_greedy_block3_prefilled_streaming (per-token callback
                         + optional EOS early stop, used by the serving layer).
  tokenizer.rs           Qwen3 chat tokenizer: loads `tokenizer.json` via the HF
                         `tokenizers` crate; encode/decode; ported Qwen3 chat template
                         (text path, enable_thinking + reasoning_effort) and `split_thinking`.
                         Bit-exactness vs transformers guarded by
                         `tokenizer::tests::chat_template_matches_transformers_reference`.
serve/                 OpenAI-compatible HTTP layer (tokio + axum)
    mod.rs               /v1/models, /v1/chat/completions (SSE + non-streaming),
                         /v1/completions. `Model` owns QwenRunner + MtpRunner + ChatTokenizer
                         behind a mutex (one generation at a time, on the blocking pool);
                         returns `reasoning_content` (split from ` thinking… response`) and
                         `content`. `complete` times prefill/decode separately and records
                         into shared metrics. Session prefix cache: holds last msg-only ids
                         + a (target, mtp) messages-end checkpoint; on an exact prefix
                         extension it restores them and prefills only the delta instead of
                         re-prefilling the whole conversation every turn (see below).
    metrics.rs           Telemetry `Metrics` store (SharedMetrics = Arc<Mutex<Metrics>>):
                         begin/prefill_done/tick/finish/fail record per-query prefill tok/s,
                         decode tok/s, acceptance, draft ratio, target forwards + aggregates.
    ui.rs                mactop-style ratatui dashboard: renders 4 gauges (Prefill Speed,
                         Token Speed, Speculative Acceptance, Draft Ratio) + a query table.
                         `preview_draw` prints a frame to stdout headless (TestBackend).

src/
  cli/                   library module for the single CLI binary (subcommands)
    mod.rs               pub mod args; pub mod run; pub mod serve;
    args.rs              shared named/positional arg parser (`--name value`, `--name=value`).
    serve.rs             `serve` subcommand body: `` serve <TARGET> <MTP> [options]
                         (named `--target`/`--mtp` also accepted). Resolves snapshots,
                         builds Model, binds axum on 0.0.0.0:PORT.
    run.rs               `run` subcommand body: native target-only vs MTP block-3
                         benchmark. Positional `run <T> <M> [STEPS] [PROMPT_CSV]` or
                         named `--target/--mtp/--steps/--prompt`. Prints a single-line
                         JSON (the fair-comparison report consumed by `bench.py`).
                         Warms up the weight shards before timing (cold first pass is
                         ~2.5x slower and would skew the numbers).

src/cli/main.rs
              single binary (bin target name `cli`, path declared in Cargo.toml).
              `cli serve <T> <M> [..]` runs the OpenAI-compatible HTTP server;
              `cli run <T> <M> [..]` runs the native benchmark. The only binary.

tests/ (integration, each = own process; symbolic names)
  metal_library/queue/vadd/nax_compile/nax_run   P0/P1 metal host + NAX smoke.
  qmv.rs                q4_qmv_fast parity vs CPU reference + fast==wide3 row-0
                       bit-exactness (incl. fused + large-N + K=5120).
  argmax.rs             argmax_f32_partial vs CPU argmax (full vocab + partial tail).
  real_nax.rs           NAX GEMM on real down_proj (skips without cache).
  attn_compile.rs       Compiles the full-attention shader set.
  attn_cache.rs         Attention KV cache kernels (+ compiles sdpa partial/final/block3).
  attn_sliced.rs        [ignored] sliced decode attention parity vs CPU + original
                       kernel, and the M=1-vs-M=3 per-row BIT-IDENTICAL invariant.
  attn_decode.rs        Layer-3 attention vs /tmp/mlx_attn_truth.raw (+gated).
  gdn_state.rs, gdn_recurrence_state.rs   GDN conv state + recurrence state kernels.
  gdn_decode.rs         Layer-17 GDN vs /tmp/mlx_gdn_out.raw.
  mlp_decode.rs         Layer-17 MLP vs /tmp/mlx_out.raw.
  fwd_decode.rs         Full layer-17 block vs /tmp/mlx_fwd_truth.raw.
  io_decode.rs          Embed + final norm + lm_head slice vs /tmp/mlx_io_*.raw.
  residual_norm.rs      residual_add_rms_norm_rows fusion correctness.
  batched_prefill.rs    [ignored] batched target prefill argmax == sequential decode
                       argmax AND batched MTP prefill argmax == sequential MTP argmax.
  full_runner.rs        [ignored] 64-layer M=1 token 2005 vs /tmp/mlx_full_token_2005.raw.
  mtp_runner.rs         [ignored] MTP M=1 fixture vs /tmp/mlx_mtp_{hidden,logits}.raw, plus
                       forward_position_decode (GPU argmax == full-logits CPU argmax).
  runner_checkpoint.rs  [ignored] GDN/conv checkpoint restore + KV logical rewind. The
                       block-3 test asserts argmax equality + hidden/logits within a
                       small tolerance (M=1/M=3 are argmax-consistent, not bit-exact)
                       and exercises commit_verified_prefix for rows=1/2/3.
speculative_loop.rs   [ignored] prefill_prompt + generate_greedy_block3_prefilled must
                        reproduce the sequential target-only token stream (exact_match).
  session_cache.rs      [ignored] cached two-turn completion reproduces the exact token
                        stream of a fresh full re-prefill (and is ~5x faster on prefill).
  spec_diag.rs          [ignored] reproduces `cli run`'s exact_match check on the long
                        1641-token prompt; on divergence prints the position + M=1 top-2
                        margin to tell a near-tie artifact from a real bug.
  tui_render_preview.rs [ignored] renders the serve dashboard into a TestBackend buffer
                        and prints it as text, for eyeballing the mactop-style layout.
  probe.rs              [ignored] mmap-enumerates real HF shards; asserts the affine-q4
                        layout for layer-3 projections. Diagnostic only.
  profile_qwen_m1.rs    [ignored] renders QwenRunner::profile_token_m1_diagnostic to JSON
                        (per-family wall times) and checks argmax-vs-normal consistency.
  profile_ablate.rs     [ignored] real fused-path ablation: times forward_token_ablate with
                        each stage disabled and prints per-stage ms/token deltas.
  prefill_fast.rs       [ignored] fast prefill-only harness: loads the runner once, times
                        full `forward_prefill` passes on a fixed prompt (env PREFILL_N
                        /PREFILL_PASSES). The right tool for PREFILL_BATCH sweeps and
                        prefill GEMM A/B (~45 s per 3-pass run, vs ~4 min full bench).
  gemm_solo.rs          [ignored] isolated tiled/align/coop GEMM micro-benchmark on
                        real gate/down weights (M sweep, prints TFLOPS) + the
                        align-vs-coop correctness check. The right tool for GEMM
                        kernel A/B work.
  weight_index.rs       WeightIndex slot resolution/shards.

scripts/ (invoke with plain `python3` — the MLX venv is bootstrapped on demand)
  mlx_env.py            Self-bootstraps the MLX reference venv: creates
                        ~/.cache/lisa-rs/mlxenv (override: LISA_MLX_VENV) with a
                        modern python (3.12+, prefers 3.14) and pip-installs the
                        pinned requirements-mlx.txt (mlx 0.32.1, mlx-lm 0.31.3,
                        mlx-vlm 0.6.15) on first use. `mlx_env.reexec()` in
                        every mlx-requiring script re-runs itself under it, so
                        nothing depends on a pre-installed environment.
  mlx_gdn_ref.py, mlx_mlp_ref.py, mlx_attn_truth.py, mlx_fwd_truth.py, mlx_io_truth.py,
  mlx_full_token_truth.py, mlx_mtp_truth.py  -> write the /tmp/mlx_*_*.raw references
  (each test compares only when its matching .raw exists, skips otherwise).
  bench.py              Unified MLX-reference + native benchmark (self-bootstraps
                        via mlx_env). By default both
                        engines run on a short prompt; `--prompt-len N` generates
                        a deterministic synthetic ~N-token prompt on the fly
                        (native gets the exact token CSV, mlx decodes the text).
                        `--native`/`--native-only`/`--mlx-only` restrict a side;
                        `--json`/`--output` write the summary.

README.md               Public-facing overview + test table + performance.
```

## Model geometry / constants (defined in `runner.rs` / `model/mod.rs`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `LAYERS` / `HIDDEN` / `VOCAB` | 64 / 5120 / 248320 | decoder depth / hidden / vocab |
| `INTERMEDIATE` | 17408 | MLP gate/up width |
| `GDN_LAYERS` / `ATTN_LAYERS` | 48 / 16 | hybrid schedule |
| GDN key/value heads, head dim | 16 / 48 / 128 | `GDN_KEY_DIM`=2048, `GDN_VALUE_DIM`=6144 |
| `GDN_CONV_DIM` | 10240 | 2*key_dim + value_dim |
| `GDN_STATE_ELEMENTS` | 48*128*128 | f32 SSM state per GDN layer |
| `ATTN_HEADS` / `ATTN_KV_HEADS` / `ATTN_HEAD_DIM` | 24 / 4 / 256 | `ATTN_Q_OUT`=12288 (24*512), `ATTN_KV_OUT`=1024 (4*256), `ATTN_OUT`=6144 (24*256) |
| `EPS` | 1e-6 | RMSNorm epsilon |

Weight layout (affine-U4, gs64, MLX `affine`): `weight U32[out, in/8]` (8
nibbles/u32, low index first), `scales BF16[out, in/64]`, `biases BF16[out,
in/64]`. `deq = q*scale + bias`. Embeds/head: `U32[248320, 640]` + BF16
scales/biases, untied, no embed_scale.

Layer 17 tensor locality: GDN/norms are in shard 1, MLP in shard 2 (3 shards total).

## Data flow

Target forward (`QwenRunner::forward_token_with_hidden`, one fused command buffer):
`embed -> per-layer [input_norm -> branch -> residual -> post_norm -> mlp -> residual]`
`-> final_norm -> lm_head qmv -> (CPU) read logits/hidden/argmax`. Branch is GDN
or attention per `layer_kind`. `verify_block3` runs the same with batch=3 and
`capture=true` (GDN/conv intermediate slots snapshotted for rollback).

GDN branch: fused4 qmv(qkv,z,a,b) -> `gdn_conv_norm_gates_bf16_weights` (conv+SiLU,
q/k rms-norm+scale, beta/decay) -> `gdn_update_conv_state` -> `gdn_recurrence`
-> `gdn_rms_gate_bf16_weight` -> out_proj qmv.

Attention branch: fused3 qmv(q,k,v) -> `q_norm_gate_rope_decode` +
`kv_cache_store_decode` + **sliced decode attention** -> `gate_out_bf16` -> o_proj qmv.
Block-3 uses the `_block3` variants with absolute positions and causal block attention.

### Sliced decode attention (sdpa_decode_partial/_final)

The hot decode-path bottleneck at long context was the serial online-softmax
scan (`sdpa_decode_streaming*`), which ran the WHOLE KV prefix in one 32-lane
threadgroup per (head,row) — cost grew linearly with context, dragging decode
from ~51 tok/s (14 ctx) down to ~13 tok/s at 2.2k ctx. The runner now uses a
flash-style sliced pair:

- `sdpa_decode_partial`: dispatch grid `(24 heads, batch rows, grid_slices)`; each
  threadgroup scans a fixed `TOKENS_PER_SLICE`-token chunk with the exact same
  per-token online-softmax numerics as the serial kernel, writing a per-slice
  accumulator + `(denom, max)`.
- `sdpa_decode_final`: merges `used = ceil(context/TOKENS_PER_SLICE)` slices
  flash-style (`w_s = exp2(m_s - gmax)`, `Σ w*O / Σ w*den`).

**Invariant (do not break):** the slice partition is a PURE FUNCTION of each row's
own context (`t0 = slice*TOKENS_PER_SLICE`, fixed chunks) — never of the dispatch
batch. So row 0 of an M=3 `verify_block3` batch and an M=1 `forward_token` produce
**bit-identical** attention for identical inputs (`tests/attn_sliced.rs` asserts
this). That is what lets the M=1 greedy decode and M=3 speculative verify stay
argmax-consistent at ANY context length. The old code dispatched `slices`
dependent on `position+batch`, which split the SAME row differently between M=1
and M=3 and flipped argmaxes at long context.

Knobs (runner.rs): `ATTN_SLICES=256` = max buffered slices (decode scratch sized
for it; prefill layouts pass 0 to skip the buffers), `TOKENS_PER_SLICE=64` =
tokens per slice, `SLICED_ATTN=true` A/B switch back to the serial kernels.

### Near-tie exact_match drift (long context)

M=1 target decode and M=3 verify use different QMV reductions (qmv_fast vs
qmv_wide3) and GDN batching, so hidden/logits are only argmax-consistent, never
bit-exact (gotcha 4). At long context, a logit sometimes lands within ~0.2 of a
rival and the wider M=3 reduction flips it — `speculative.tokens` then differs
from the greedy `target_tokens` (observed at token 3, margin 0.21; the serial
baseline likewise drifts, at token 16, margin 1.01). This mirrors MLX ("argmax-
consistent, not bit-exact", per its own notes). `cli run` therefore REPORTS
`exact_match` (true/false) instead of
hard-failing, so long-prompt benchmarks still produce numbers; short prompts
(where margins are clear) still report `exact_match=true`.

MTP (`MtpRunner::forward_position`): reuses target embed/lm_head shard buffers
(shared `Rc`), 31 own tensors, pre-fc fuses token-embed + target-hidden, then a
single attention+MLP block and full-vocab head. `trim_state` rewind is logical.

Speculative loop (`speculative::generate_greedy_block3_prefilled`): target `verify_block3`
of [bonus, draft0, draft1] -> greedy acceptance -> `commit_verified_prefix` ->
`mtp_reconcile_plan` + `trim_state` + replay -> next seed.

## MLX/Qwen3.5 specifics (diverges from a typical backbone)

- `text_config.rope_parameters = {"mrope_interleaved": True, "mrope_section": [11,11,10], "partial_rotary_factor": 0.25, "rope_theta": 1e7, "rope_type": "default"}`.
- Attention rotary **dim = round(head_dim * 0.25) = 64** (NOT 256), **base = 1e7** (NOT 1e6), NeoX half-split pairing.
- 64 layers, `is_linear = (layer_idx+1)%4 != 0` -> full attention only at 3,7,…,63; GDN elsewhere.
- GDN recurrence: decay = `exp(-exp(A_log) * softplus(a+dt_bias))`, f32 state [48,128,128], GQA vh->kh (repeat=3); q/k scales `[inv², inv]`, `inv=1/sqrt(128)`.

## Architecture choices

### Forward pipeline
- One **fused command buffer per forward** (embed + all 64 layers + final norm +
  lm_head). Every kernel reads the output of the previous via the same layout
  offsets; only the final `(argmax, hidden)` is read back.
- **GPU argmax** for the decode path: `argmax_f32_partial` reduces the 248320-wide
  lm_head logits (target: `forward_token_decode`/`verify_block3_decode`) and the MTP
  head (MTP: `forward_position_decode`) on-device, so the full-vocab F32 logits
  (1 MB) + a 248320-element CPU argmax are never needed on the decode/speculative
  paths (acceptance is argmax-only). `forward_token_with_hidden`/`forward_position`
  still read full logits for the truth tests.
- **Decode QMV** (`q4_qmv_fast`, M=1; `q4_qmv_wide3`, M=3): 16 values/thread
  contiguous loads, byte-mask nibble decode with pre-scaled input (`x/16` for odd
  nibbles), MLX `scale*sum(q*x)+bias*sum(x)` arithmetic. `q4_qmv_wide3` re-reads x
  per input row from L2 (2 weight words live per row, ~20 regs) instead of holding
  `xs[3][16]` (48 floats) — this keeps M=3 fast and the two kernels bit-identical
  per row. Earlier regressions (ushort masks, 2 simdgroups/8 outputs, scale
  `simd_broadcast_first`) are reverted — see blockers.
- **Fused dispatch families** (grid.y selects independent weights): fused2 = MLP
  gate/up, fused3 = attention q/k/v, fused4 = GDN qkv/z/a/b; M=3 wide-3 variants.
- **rms_norm_rows** uses 256 threads (multi-simdgroup, 2-level simd_sum + shared
  reduction, MLX `rms_looped`); attention decode/block3 q/k-norm + KV store use
  256 threads, streaming SDPA 32 lanes/head (8 dims each, `simd_sum` score).
- Immutable weight shard buffers are **shared + hazard-untracked**; scratch is a
  single hazard-tracked shared buffer.

### Metal engineering discipline (Apple M5 Max, g17)

- **Dispatch overlap amortizes ramp/drain.** Back-to-back independent dispatches
  in one encoder cost ~0.33 ms each vs ~1.13 ms isolated (~3.4x). A prefill is a
  serial RAW chain, so small kernels pay near-single-dispatch latency — batch
  per-token kernels (RMSNorm, rotary, conv, gate/up) into one dispatch.
- **`dispatch_threads` counts TOTAL threads, not threadgroups.** Passing
  threadgroup counts under-dispatches massively (a `(32,48,1)` grid with tg
  `(32,4,1)` = 12 groups; a GDN norm covers 1% of output). Use
  `dispatch_thread_groups` for grid-in-groups kernels, `dispatch_threads` only
  for flat 1-D blasts.
- **Metal buffer offsets are BYTES** — every element offset is `index * width`
  before `set_buffer`.
- **`#pragma unroll` is unreliable on mma loop nests** — fragment arrays on the
  stack trail ≈4x slow. Manually unroll the mma calls with literal indices.

### Device-resident speculative decode (MTP, `run_block3_loop`)

The MTP speculative loop carries the target and MTP **hidden states on-device**
instead of round-tripping them through the host. Each MTP verify/draft forward
leaves its per-row hidden (bf16) in GPU scratch; the next draft conditions on it
directly:

- `QwenRunner::verify_block3_decode_device` — identical batch-3 fused forward to
  `verify_block3_decode`, but reads back ONLY the three GPU argmaxes and leaves
  the rows' hidden (bf16) at `hidden_offset(row)`.
- `MtpRunner::forward_position_decode_device(token, cond, cond_off)` — the draft
  forward reads its conditioning hidden from an explicit device buffer
  (`cond=None` = own `normalized_offset` for draft chaining; `Some(target.scratch())`
  = the target's verified hidden for the replacement), leaves its own hidden on
  device, and reads back only the argmax.

The old host loop did `bf16→f32` readback + `f32→bf16` reconvert + host write +
a full GPU sync between every verify and every MTP forward — measurable
per-round latency (the MTP path was ~13% behind MLX at long context). Carrying
hidden on device removes those round-trips; together with skipping the wasted
replay rematerializations (only the final `plan.replay` entry seeds the next
round — earlier `Draft(index)` entries were computed then discarded), long-context
MTP decode went from ~52 to ~56-57 tok/s against MLX ~59.7.

### Batched prefill (forward_prefill)

The target prefill is a batched NAX-GEMM pass instead of N sequential decodes
(`QwenRunner::forward_prefill`, in `runner.rs`). It mirrors MLX's batched
affine-QMM prefill the way omlx does: GEMM every projection over the prompt,
batched causal attention, and the GDN recurrence looped over `steps` in one
dispatch (the GDN conv/update/recurrence/gate kernels were already
`steps`/`batch`-parameterized).

- `PREFILL_BATCH = 512` (see Prefill performance below for the sweep and the
  sweet spot). `prefill_scratch`/`prefill_layout` are a second `ScratchLayout`
  sized for `PREFILL_BATCH` rows; the real chunk `m` (<= batch) is passed to the
  GEMM and masked (rows >= m never computed/stored, N-tail masked in-store).
- `encode_gemm` dispatches the NAX **tiled** `q4_gemm_nax_tiled` or the
  branchless **`q4_gemm_nax_tiled_align`** (see below; 128-thread threadgroup,
  grid `(M/64, N/64)`, selected by `output % 64 == 0`) for every linear
  projection (MLP gate/up/down, GDN qkv/z/a/b/out, attention q/k/v/o, lm_head).
  **gate+up are fused** into one dispatch (`q4_gemm_nax_tiled_align_fused2`,
  grid.z picks the weight/output pair). The lm_head argmax is **batched**:
  `argmax_f32_rows` reduces all `m` rows in ONE dispatch (grid `(n/256, m)`)
  instead of `m` per-row dispatches (target + MTP prefill paths).
- Batched attention uses `q_norm_gate_rope_prefill` / `kv_cache_store_prefill` /
  `sdpa_decode_prefill` (generalized `_block3` kernels, arbitrary batch, causal
  over the full prefix `base_position + row + 1`, absolute positions).
- GDN prefill reuses `gdn_conv_norm_gates_bf16_weights` (batch) ->
  `gdn_update_conv_state` (steps) -> `gdn_recurrence` (steps) ->
  `gdn_rms_gate_bf16_weight` (batch) in one fused buffer. Recurrent GDN/conv/KV
  state carries across 64-token chunks.
- `forward_prefill` returns `(argmax, hidden)` per token (hidden is what the MTP
  drafter conditions on); `speculative::prefill_prompt` calls it when
  `prompt.len() >= 4`.

### Tiled cooperative prefill GEMM (`q4_gemm_nax_tiled` / `_align`)

The prefill GEMM is a 64×64 cooperative tile (128 threads, 4 SIMD groups,
`BM=BN=BK=64, WM=WN=2`), a faithful port of MLX's `qmm_t_nax_tgp` structure.
It replaced the old 16×32 `q4_gemm_nax_coop` (1 SIMD, dequant-in-register) as
the GEMM pipelines used by both the target and MTP prefill dispatchers.
**`q4_gemm_nax_tiled_align` is the default** for every projection whose output
is a multiple of 64 (all MLP/GDN/attention/lm_head) — it drops all per-lane
bounds checking and is **~55 TFLOPS isolated vs ~33 for the masked kernel**;
`q4_gemm_nax_tiled` (masked M/N tails) is used only for the 48-wide GDN a/b
gates. This branchless path closed most of the per-GEMM gap to MLX's
`qmm_t_nax` (~52-56 TFLOPS).

- **Coalesced cooperative dequant** (MLX `QuantizedBlockLoader` recipe): each
  thread dequants 16 contiguous bytes (32 q4 values from 2 u32) of one weight
  row into a **padded threadgroup tile** `Ws[64 × 72]` (BK_PAD=72 avoids bank
  conflicts), with one scale/bias load per output row per 64-col k-block. The
  4 SIMD groups then all read the shared padded tile → dequant work is done
  once per tile, not redundantly per SIMD.
- Each SIMD group computes a 32×32 C sub-tile via two 16×16 `NaxFrag` MMA
  fragments (2 m-rows × 2 n-cols, double-B `tile_matmad`).
- **grid** `(M/64, N/64)`, 128 threads/group. `MNK = (m64, N, K)` where `m64`
  is M rounded up to 64. The masked keeper (`q4_gemm_nax_tiled`) guards rows
  >= M (`load_a_masked`) and clamps store columns to `N` (the 48-wide GDN
  gates never write OOB).
- **Bit-exact** vs the old coop kernel (verified: `tiled-vs-coop max_abs_diff =
  0.00000` on real gate/down projections, all M; both prefill argmax tests
  report 0 mismatches). The branchless align kernel is argmax-consistent on
  those too (the remaining max_abs_diff ~3e0 comes from the padded tail rows,
  which are never consumed).
- Measured kernel ceilings (isolated, real gate/down weights,
  `tests/gemm_solo.rs`): masked tiled ~33, align branchless ~50-56; the
  dropped predication roughly doubled effective TFLOPS at M>=512. The prefill
  effective rate went from ~18-20 to ~28-31 TFLOPS (444 → ~600 tok/s on the
  `prefill_fast` 3713-token harness).

The correctness gotcha that cost the most debugging: the dequant loop must index
threads with `thread_index_in_threadgroup` **directly** (`lid`, already 0..127
for a 128-thread group). Computing the index as `sgid*32 + lid` and using the
result for the W-row / byte-offset mapping silently **left half the tile rows
unwritten** (odd rows zero), producing a kernel that was fast but wrong. The MMA
cooperative tensor uses `simdgroup_index_in_threadgroup` per-SIMD for its own
lane math, but the cooperative dequant is flat over all 128 threads.

Fragment ground truth: lane→coord is `qid=lane>>2; fm=(qid&4)|((lane>>1)&3);
fn=((qid&2)|(lane&1))*4`; the 16×16 A fragment is one `metal::vec<bfloat,8>`,
B is a 16×16 pair fed as two vec halves (double-B `mma`, 16×32×16 via
`mpp::tensor_ops::matmul2d`). Q4 affine dequant (gs64):
`q=(packed[nn*K8+(kk>>3)]>>((kk&7)*4))&0xF`, `g=kk/64`,
`dst=bf16(q*scale[nn*groups+g]+bias[nn*groups+g])`. Prefer one concrete kernel
per config over runtime switches — explicit template args (`instantiate_kernel`
/ `host_name`) let the compiler bake constants and never spill fragments.

### MTP drafter prefill is batched too

`MtpRunner::forward_position_batch` processes the MTP positions (shifted
`prompt[1..]` + the target's per-position hidden) in the same fused chunked
style (GEMM projections + batched causal attention + per-row GPU argmax) over
the MTP head's single attention+MLP block. The pre-fc input is built in place
by `rms_norm_rows_stride`, writing the two normed halves (embedding and target
hidden) directly into the interleaved `[m, 2*HIDDEN]` fc layout — mirrors the
M=1 contiguous `pre_fc`+`pre_fc_hidden` trick without a separate concat pass
(see blockers for why a `concat_rows` kernel was rejected).

Future MTP work (not yet implemented, after the prefill roadmap): **windowed
draft KV** for 20k-100k contexts — let the drafter read a sliding KV window
while the target verifies on the full context, cutting draft cost to
O(window) — and mirror MLX's acceptance-rate-aware block size (draft depth
shrinks when the prefix-hit rate < 0.65).

### Session prefix cache (Model::complete)

The serving layer no longer re-prefills the whole conversation every turn.
`Model::complete` renders the messages twice — once with and once without the
generation-prompt suffix. Because `apply_chat_template(..., add_generation_prompt)`
only appends the suffix at the very end (new `add_generation_prompt` param in
`tokenizer.rs`), the messages-only render is an exact token-prefix of the full
prompt, so `msg_len` is the reusable boundary.

- Request defaults & debug: `enable_thinking` defaults to `true`, but to
  **false when tools are present** (agentic clients like pi declare
  `reasoning:false` and burn the token budget on free-thinking turns in tool
  loops); the server default `max_tokens` is 4096 (not 512, which truncated
  reasoning-heavy answers mid-thought). `LISA_LOG_REQUESTS=1` prints a
  per-request stderr line (message roles/tool-call counts, full/messages token
  counts, whether the prefix cache was reused) — run the server with it and it
  shows exactly what any client sends and whether the whole history was
  prefilled.

- `Model` keeps `last_msg_ids` (messages-only ids) + `last_msgs_checkpoint =`
  `(QwenStateCheckpoint, MtpCheckpoint)` captured at the messages-end boundary.
- On the next request, if `msg_ids` is an **exact prefix extension** of
  `last_msg_ids`, it restores both checkpoints (position-relative
  `forward_prefill`/`forward_position_batch` continue from the restored
  position), prefills only the delta (`prefill_prefix_until`) + the generation
  segment (`prefill_prompt_from`), and re-checkpoints the new messages boundary.
  Any restore/prefill error falls back to a full `reset_state` + re-prefill.
- `speculative::prefill_prompt` is now a thin wrapper over
  `prefill_prompt_from(target, mtp, prompt, 0)`; `prefill_prefix_until` advances
  to a strict prefix boundary without producing a generation seed.
- `tests/session_cache.rs` asserts that a cached two-turn completion reproduces
  the exact token stream of a fresh full re-prefill (and is ~5x faster on the
  prefill portion).

### Benchmarks / comparisons
- `cli run` reports a single-line JSON splitting **prefill / decoding /
  speculative**: `target_prefill_tok_s`, `target_decode_tok_s` (=`target_tok_s`),
  `speculative_prefill_tok_s`, `speculative_decode_tok_s` (=`speculative_tok_s`),
  `*_e2e_tok_s` (including prefill), plus ms fields. Target and MTP prefill are
  both batched GEMM; the decode loop emits `steps` tokens. A warmup pass touches
  the GBs of weight shards first (cold first pass is ~2.5x slower).
- `scripts/bench.py --native` runs the Rust native benchmark and the MLX
  reference (`mlx_vlm.generate`) side by side. Both run on a short default
  prompt; `--prompt-len N` synthesizes an ~N-token prompt on the fly (native
  gets the exact token CSV, mlx decodes the text).

### Prefill performance state (vs mlx-vlm) — ~mid-2026

The 4k-prompt numbers (`--prompt-len 4096`, ~3713 tokens) native vs
`mlx_vlm.generate` (fresh run, mains-powered M5 Max):

| metric | native | mlx-vlm | note |
|---|---|---|---|
| target_prefill | ~450-620 tok/s | ~590-880 | mixed GEMM (batched), env/thermal noise ±10-15% |
| spec prefill | ~360-489 | ~740 | target + MTP warmup |
| target decode | ~28-31 | ~32 | near parity (-10%) |
| spec decode | ~42-52 | ~55 | ~-10-25% |

Prefill is the last big gap: **~1.2-1.7x behind** (was 3.3x before
`q4_gemm_nax_tiled`, 2x before `q4_gemm_nax_tiled_align`). It is GEMM-bound —
188 TFLOP per 4k prefill, so target ≈ 45+ TFLOPS sustained required for
parity; the branchless prefill GEMM now measures ~30-34 effective
(`QwenRunner`), up from ~18-20. The remaining gap over the isolated kernel
(~50-56) is the non-GEMM tail (norms/GDN/attention ~15-20% of wall time) plus
inter-chunk GPU ramp.

Measured kernel ceiling (isolated, real weights; `tests/gemm_solo.rs`, plus
the `tests/prefill_fast.rs` harness):
- old `q4_gemm_nax_tiled` masked (64×64 tail-guarded): **~33 TFLOPS**
  at M≥512 on gate (N=17408) / down (K=17408).
- new `q4_gemm_nax_tiled_align` branchless (default): **~50-56 TFLOPS**
  isolated, bit-exactness checked vs coop; `gemm_solo.rs` prints it.
- `mlx.quantized_matmul` reference on the same shapes: **~52-56 TFLOPS** at
  M≥512 (measured with `mlx_lm`'s affine-q4 gemm on gate_proj).

So the native tiled kernel is ~65% of MLX's per-GEMM efficiency. The prefill gap
(=2x) is larger than the per-GEMM gap because: (1) our GEMMs run at chunked
M <= 512 while MLX processes the whole prompt as one GEMM (M=3712),
(2) normalizations/attention/GDN also account for wall-time.

Next levers to reach / beat MLX prefill (in effort order):
1. **`PREFILL_BATCH` sweep is DONE** — 512 confirmed best on the real 4k
   (`tests/prefill_fast.rs`, env `PREFILL_N`/`PREFILL_PASSES`): 128-1024 all
   ~392-415 tok/s on the short-prompt harness, 2048 regresses (366). The real
   4k bench still favors 512 over 1024.
2. **Feed the tiled GEMM more data per dispatch** — MLX does one GEMM per
   projection over the whole prompt; we chunk at PREFILL_BATCH rows. Avoid the
   per-chunk surge (each chunk issues ~60 dispatches + barrier). The lm_head
   argmax is DONE (single batched `argmax_f32_rows` dispatch per chunk); the
   lm_head GEMM itself (N=248320) matters disproportionately next.
3. **Split-K cooperation** (MLX `affine_qmm_*_splitK`): for small-M large-K
   GEMMs, split K across threadgroups and reduce — MLX uses it when
   `n_tiles*m_tiles < ~512`. Unlocks the 5120-K projections of the GDN/MLP
   branch on large prompts.
4. **Grid orientation** A/B (M-major vs N-major dispatch) and **A staged in
   threadgroup** (MLX stages both Xs and Ws), plus `BK_PAD` tuning.
5. Non-GEMM: batch the attention prefill along the chunk (already `_prefill`
   kernels) and shave the per-chunk tail (norms/residuals are bandwidth-bound;
   the fused `residual_add_rms_norm_rows` kernel exists but is DISABLED, see
   gotcha 10).
6. **ANE/GPU hybrid prefill** (oMLX experiment — stretch goal, NOT planned now):
   fixed-shape INT8 `conv` MIL programs on the Neural Engine run output-channel
   prefixes of MLP/GDN projections while the GPU NAX QMM runs the suffix
   (IOSurface zero-copy bridge + event sync). ~+18.9% prefill @32k on M3 Ultra
   (split ~53% MLP / 50% GDN); on M5/NAX the ANE win sits **below** the classic
   50% optimum, so it trails items 1-5 above.

Keep an eye on: power/thermal — the M5 Max drops ~10% prefill sustained after
repeated runs; always compare medians of >=3 alternated runs.

## Blockers / correctness gotchas (do not regress)

1. **NAX GEMM N-tail**: `q4_gemm_nax_tiled` (64×64 tile, grid `(M/64, N/64)`)
   clamps every store column to `N` (projections whose `N` is not a multiple of
   64, e.g. the 48-wide GDN `a`/`b` gates, never write out of bounds). The old
   16×32 `q4_gemm_nax_coop` guarded its two 16-column halves with
   `n0+16 <= N` / `n0+32 <= N` (N always a multiple of 16) — keep those guards
   if the coop kernel is ever re-enabled. The tiled kernel also zero-fills the
   W-tile rows for `n >= N` so masked columns contribute 0.
2. **NAX GEMM M-tail**: a tail chunk (`m % 64 != 0`) must still produce the last
   partial tile — the runner rounds the GEMM's M up to 64 (`m64`) and both
   `load_a_masked` (zero out rows >= real M) and `store_c_masked` (skip rows >=
   real M) guard it. Buffers are sized for `PREFILL_BATCH` rows; downstream
   kernels only consume the real `m` rows. Never dispatch a GEMM with the raw
   `m` in `MNK.x` — the padded `m64` is required or the tail rows stay zero.
3. **GDN dispatch width**: `gdn_conv_norm_gates_bf16_weights` and
   `gdn_recurrence` must be dispatched with **32 threads/threadgroup** (their
   `simd_sum` reduction is written for 32 lanes). Dispatching 1024 threads wrote
   out of bounds / did partial reductions → NaN logits (argmax 248319) and subtly
   wrong M=1 GDN.
4. **M=1 vs M=3 is argmax-consistent, not bit-exact** (mirrors MLX). The QMV
   kernels are bit-exact in isolation (`qmv_fast == wide3` row 0), but the full
   64-layer M=3 verify drifts ~1 bf16 ulp in some hidden elements (observed max
   ~2.0 hidden / ~0.2 logits; argmax always exact). `runner_checkpoint` asserts
   argmax equality + tolerance (4.0 / 2.0); `cli run` still reports
   `exact_match=true` (speculative tokens == greedy tokens).
5. **GDN recurrence grouping**: keep the strided 4-cell grouping. A contiguous
   `float4` rewrite changed the 32-lane reduction grouping, crossing a bf16
   quantization threshold differently for M=1 vs M=3 and flipping an argmax
   (`exact_match=false`). The `(32,4,1)` threadgroup pack and "skip last state
   snapshot" variants were perf-neutral and reverted.
6. **`rms_norm_rows_stride`, not `concat_rows`**: a `concat_rows` kernel for the
   MTP fc input was correct in isolation but reliably mis-read GPU-written
   sources when dispatched from the runner. The strided-RMSNorm approach writes
   into the interleaved 2H layout using the same norm-write -> GEMM-read pattern
   the rest of the engine relies on and is argmax-exact.
7. **GDN stride contracts**: the legacy M=16 tests pad A/B projections to stride
   64, the native runner uses compact stride 48. `gdn_conv_norm_gates` must keep
   stride 64; `gdn_conv_norm_gates_bf16_weights` must keep stride 48.
8. **GDN state-slot capture**: optional GDN/conv state-slot writes are guarded by
   `slot_stride > 0`; legacy tests bind a valid dummy buffer and zero stride.
9. **Metal threadgroup attribute types**: `threadgroup_position_in_grid` /
   `thread_position_in_threadgroup` / `threads_per_threadgroup` must all be
   scalar or all `uint3` in the same kernel — the block3 kernels use `uint3`.
10. **`residual_add_rms_norm_rows` fusion is DISABLED** in the runner. It is
    bit-exact in isolation but perf-neutral in the fused forward (saves one
    dispatch, loses residual-add parallelism); the standalone kernel + test remain.
11. **Shaders are `include_str!`-compiled**: any `.metal` change requires a
    rebuild (the source is baked into the binary).
12. **Real-weight integration tests MUST run one at a time** (shared heap storage
    segfaults in one process): `for t in attn_decode gdn_decode fwd_decode mlp_decode; do cargo test --test "$t"; done`.

## Environment / commands

- **Model (cache)**: `~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots` (3 shards; layer-17 GDN/norms in shard 1, MLP in shard 2). MTP: `...--Qwen3.8-27B-MTP-4bit/snapshots`.
- **MLX reference env**: self-bootstrapped — `scripts/mlx_env.py` creates
  `~/.cache/lisa-rs/mlxenv` on first use and installs the pinned
  `requirements-mlx.txt` (mlx 0.32.1 + mlx-lm + mlx-vlm). Every mlx-requiring
  script (bench.py, mlx_*.py) re-runs itself under it, so a bare
  `python3 scripts/bench.py ...` / `python3 scripts/mlx_*.py` works with no
  pre-installed environment. Regenerate `.raw` truths with
  `python3 scripts/mlx_*.py` (each writes the `/tmp/mlx_*_*.raw` its test
  reads; tests skip if the file is absent).
- **Tests / build**: `cargo test` and `cargo build`.
- **OpenAI-compatible server** (loads tokenizer + 16 GB target + MTP, then serves on port 8000).
  On a real terminal it opens a mactop-style live dashboard (gauges for prefill
  speed / token speed / speculative acceptance / draft ratio + a query list);
  press `q` to quit, `r` to reset, `↑/↓` to scroll. `--no-ui` runs headless:
  ```
  cargo run --release --bin cli -- serve mlx-community/Qwen3.8-27B-4bit mlx-community/Qwen3.8-27B-MTP-4bit --port 8000
  ```
  Named args also accepted: `cli serve --target <T> --mtp <M> [--port 8000] [--capacity 32768] [--model-id qwen3.8-27b] [--no-ui]`.
  `TARGET`/`MTP` are snapshot directories or HF ids resolved from the local cache.
  Endpoints: `GET /v1/models`, `POST /v1/chat/completions` (stream + non-stream),
  `POST /v1/completions`. The opencode provider is registered in
  `~/.config/opencode/opencode.json` as `engine/qwen3.8-27b`.
- **Dashboard preview** (headless): `cargo test --test tui_render_preview --release -- --ignored --nocapture`.
- **Native target + MTP benchmark**:
  ```
  cargo run --release --bin cli -- run mlx-community/Qwen3.8-27B-4bit mlx-community/Qwen3.8-27B-MTP-4bit 64
  ```
  Positional: `run <T> <M> [STEPS] [PROMPT_CSV]`; named: `--target/--mtp/--steps/--prompt`.
- **Fair MLX-vs-native comparison (4k context)**: `scripts/bench.py --native --prompt-len 4096 --runs 3`
  (see Architecture choices). `--prompt-len N` synthesizes a deterministic
  ~N-token prompt on the fly (no corpus files to keep in sync).
- **Diagnostic profile**: `cargo test --test profile_qwen_m1 --release -- --ignored --nocapture`.
- **Real-fused ablation**: `cargo test --test profile_ablate --release -- --ignored --nocapture`.
- **Heavy correctness tests** (run separately):
  ```
  cargo test --test full_runner -- --ignored --nocapture
  cargo test --test mtp_runner -- --ignored --nocapture
  cargo test --test runner_checkpoint -- --ignored --nocapture
  ```

## Reference implementation repos (GitHub)

Optional, read-only dev references for shader/geometry work (NOT required for
`cargo build`, `cargo test`, or any script).

- **mlx** (Apple) — steel NAX GEMM / steel_attention:
  https://github.com/ml-explore/mlx
- **mlx-vlm** (canonical MLX Qwen3.5) — `mlx_vlm/models/qwen3_5/language.py`
  (Qwen3_5Attention, Qwen3_5GatedDeltaNet, Qwen3_5DecoderLayer):
  https://github.com/Blaizzy/mlx-vlm
- **mlx-native** (Rust MLX hybrid port):
  https://github.com/robertelee78/mlx-native
- **saragossa** (NAX + GDN shader port): https://github.com/azerozero/saragossa
- **lattice** (GEMM / attention / fusion kernels): https://github.com/ohdearquant/lattice
- **candle** (small Metal kernels, `candle-metal-kernels/src/metal_src/`):
  https://github.com/huggingface/candle
- **mistral.rs** (Candle-based Rust LLM engine, Metal kernels;
  `mistralrs-quant/src/metal_kernels/` and `mistralrs-paged-attn/src/metal/kernels/`):
  https://github.com/EricLBuehler/mistral.rs
- **omlx** (ANE/GPU hybrid prefill, experimental): https://github.com/jundot/omlx