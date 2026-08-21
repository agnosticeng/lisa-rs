# AGENTS.md

Working guide for the coding agent on this repo. Read before any modification.
This file is the single source of truth: goal, architecture (current design
decisions), the blockers/correctness gotchas we have hit, model geometry, and
the exact commands. Keep it updated as steps turn green.

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
    linear.rs            include_str! of the six shader sources, exported as const &
                         (NAX_AFFINE_U4_SHADER, MLP_SHADER, ATTENTION_SHADER,
                         GDN_SHADER, NORM_SHADER, EMBED_SHADER). Rebuild required for
                         shader changes (include_str! caching).
    shaders/
      nax_affine_u4.metal  NAX tensor-core affine-U4 GEMM + decode QMV families.
                           q4_gemm_nax_coop (16x32 tile, mpp cooperative tensors,
                           col/M tail guards).
                           q4_qmv_fast/_impl: M=1 matrix-vector (4 simdgroups,
                           16 rows/TG, packed-word 8 nibbles/load). q4_qmv_wide3/_impl:
                           M=3, one weight pass reused by 3 rows, x streamed per row.
                           Fused dispatch families (grid.y selects independent weights):
                           fused2/3/4 (M=1) and fused2/3/4_wide3 (M=3). Fused2=MLP
                           gate/up, fused3=attention Q/K/V, fused4=GDN qkv/z/a/b.
      mlp.metal           silu_mul_up: SiLU(gate)*up -> bf16 (MLP SwiGLU intermediate).
      attn.one.metal      Qwen3.5 full attention (24x512 q, 4x256 kv, dim 256, partial
                          rotary 0.25 -> 64 dims, theta 1e7, NeoX). rms_norm_rope helper,
                          qk_norm_gate_rope, k_norm_rope (legacy batch), decode block:
                          q_norm_gate_rope_decode/_block3, kv_cache_store_decode/_block3,
                          sdpa_decode_streaming/_block3 (online softmax, 32 lanes/head),
                          gate_out_bf16 (out*sigmoid(gate)), sdpa_scalar.
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
                         The runner copies each shard into one GPU buffer and addresses
                         weights by (shard, byte offset).
    runner.rs            THE core file (~2600 lines). QwenRunner (64-layer M=1/M=3
                         target), MtpRunner (31-tensor MTP head), Pipelines (all compiled
                         compute pipelines), ScratchLayout/MtpScratchLayout (offsets into
                         one reusable scratch buffer), encode_* kernels for every stage,
                         load/build_layers/build_mtp_weights, read_f32/read_bf16/argmax
                         helpers, forward_token_decode + verify_block3_decode (GPU argmax,
                         no logits readback), forward_prefill (batched NAX-GEMM prefill in
                         PREFILL_BATCH=64 chunks), MtpRunner::forward_position_batch (batched
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
                         into shared metrics.
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
                         JSON (the fair-comparison report consumed by `bench_mtp.py`).
                         Warms up the weight shards before timing (cold first pass is
                         ~2.5x slower and would skew the numbers).

src/bin/
  cli.rs                 single binary. `cli serve <T> <M> [..]` runs the OpenAI-
                         compatible HTTP server; `cli run <T> <M> [..]` runs the
                         native benchmark. The only binary in the crate.

tests/ (integration, each = own process; symbolic names)
  metal_library/queue/vadd/nax_compile/nax_run   P0/P1 metal host + NAX smoke.
  qmv.rs                q4_qmv_fast parity vs CPU reference + fast==wide3 row-0
                       bit-exactness (incl. fused + large-N + K=5120).
  argmax.rs             argmax_f32_partial vs CPU argmax (full vocab + partial tail).
  real_nax.rs           NAX GEMM on real down_proj (skips without cache).
  attn_compile.rs       Compiles the full-attention shader set.
  attn_cache.rs         Attention KV cache kernels.
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
  tui_render_preview.rs  [ignored] renders the serve dashboard into a TestBackend buffer
                       and prints it as text, for eyeballing the mactop-style layout.
  probe.rs              [ignored] mmap-enumerates real HF shards; asserts the affine-q4
                       layout for layer-3 projections. Diagnostic only.
  profile_qwen_m1.rs    [ignored] renders QwenRunner::profile_token_m1_diagnostic to JSON
                       (per-family wall times) and checks argmax-vs-normal consistency.
  profile_ablate.rs     [ignored] real fused-path ablation: times forward_token_ablate with
                       each stage disabled and prints per-stage ms/token deltas.
  weight_index.rs       WeightIndex slot resolution/shards.

scripts/ (run with /tmp/mlxenv/bin/python)
  mlx_gdn_ref.py, mlx_mlp_ref.py, mlx_attn_truth.py, mlx_fwd_truth.py, mlx_io_truth.py,
  mlx_full_token_truth.py, mlx_mtp_truth.py  -> write the /tmp/mlx_*_*.raw references
  (each test compares only when its matching .raw exists, skips otherwise).
  bench_mtp.py          MLX reference benchmark -> benchmarks/mtp_mlx_m5_max.json. With
                        `--native` it also runs the Rust native benchmark on the same prompt.

benchmarks/             mtp_native_m5_max.json, mtp_mlx_m5_max.json (see Performance below).
research.md             Design notes + empirical Metal/NAX/GDN/MTP findings + section index
                        at the end (500 lines). Read before kernel work.
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
`kv_cache_store_decode` + `sdpa_decode_streaming` -> `gate_out_bf16` -> o_proj qmv.
Block-3 uses the `_block3` variants with absolute positions and causal block attention.

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

### Batched prefill (forward_prefill)

The target prefill is a batched NAX-GEMM pass instead of N sequential decodes
(`QwenRunner::forward_prefill`, in `runner.rs`). It mirrors MLX's batched
affine-QMM prefill the way omlx does: GEMM every projection over the prompt,
batched causal attention, and the GDN recurrence looped over `steps` in one
dispatch (the GDN conv/update/recurrence/gate kernels were already
`steps`/`batch`-parameterized).

- `PREFILL_BATCH = 64`; `prefill_scratch`/`prefill_layout` are a second
  `ScratchLayout` sized for 64 rows (decode keeps its own batch=3 scratch).
- `encode_gemm` dispatches the NAX `q4_gemm_nax_coop` (32-thread threadgroup,
  grid `(M/16, N/32)`) for every linear projection (MLP gate/up/down, GDN
  qkv/z/a/b/out, attention q/k/v/o, lm_head).
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

### MTP drafter prefill is batched too

`MtpRunner::forward_position_batch` processes the MTP positions (shifted
`prompt[1..]` + the target's per-position hidden) in the same fused chunked
style (GEMM projections + batched causal attention + per-row GPU argmax) over
the MTP head's single attention+MLP block. The pre-fc input is built in place
by `rms_norm_rows_stride`, writing the two normed halves (embedding and target
hidden) directly into the interleaved `[m, 2*HIDDEN]` fc layout — mirrors the
M=1 contiguous `pre_fc`+`pre_fc_hidden` trick without a separate concat pass
(see blockers for why a `concat_rows` kernel was rejected).

### Benchmarks / comparisons
- `cli run` reports a single-line JSON splitting **prefill / decoding /
  speculative**: `target_prefill_tok_s`, `target_decode_tok_s` (=`target_tok_s`),
  `speculative_prefill_tok_s`, `speculative_decode_tok_s` (=`speculative_tok_s`),
  `*_e2e_tok_s` (including prefill), plus ms fields. Target and MTP prefill are
  both batched GEMM; the decode loop emits `steps` tokens. A warmup pass touches
  the GBs of weight shards first (cold first pass is ~2.5x slower).
- `scripts/bench_mtp.py --native` runs the MLX reference (`mlx_vlm.generate`) AND
  the Rust native benchmark on the SAME tokenized prompt and prints prefill and
  decode side by side.

## Blockers / correctness gotchas (do not regress)

1. **NAX GEMM N-tail**: `q4_gemm_nax_coop` computes a 16x32 tile; a projection
   whose `N` is not a multiple of 32 (e.g. the 48-wide GDN `a`/`b` gates) would
   have its second n-tile read out-of-range weights (cols >= N) and scatter
   NaN/garbage into the next row. The two 16-column halves store only when
   `n0+16 <= N` / `n0+32 <= N` (N is always a multiple of 16).
2. **NAX GEMM M-tail**: a tail chunk (`m % 16 != 0`) must still fill the last
   16-row tile — the runner rounds the GEMM's M up to the next multiple of 16
   (buffers are sized for `PREFILL_BATCH`; downstream kernels use only the real
   `m` rows). Otherwise the `m0+16 > M` guard skips the tile → zero output.
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
- **Reference venv**: `/tmp/mlxenv` (py3.14, mlx 0.32, mlx_vlm, mlx_lm). Regenerate `.raw` truths with `/tmp/mlxenv/bin/python scripts/mlx_*.py` (each truth script writes the `/tmp/mlx_*_*.raw` its test reads; tests skip if the file is absent). Do not break the venv.
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
- **Fair MLX-vs-native comparison (same prompt, end to end)**: `scripts/bench_mtp.py --native [--runs 3]`
  (see Architecture choices).
- **Diagnostic profile**: `cargo test --test profile_qwen_m1 --release -- --ignored --nocapture`.
- **Real-fused ablation**: `cargo test --test profile_ablate --release -- --ignored --nocapture`.
- **Heavy correctness tests** (run separately):
  ```
  cargo test --test full_runner -- --ignored --nocapture
  cargo test --test mtp_runner -- --ignored --nocapture
  cargo test --test runner_checkpoint -- --ignored --nocapture
  ```

## Local reference sources (repos cloned locally under /tmp)

Use as source of truth for shaders/structures. Prefer absolute paths.

- **mlx-vlm** (canonical MLX Qwen3.5): `/tmp/mlx-vlm` — `mlx_vlm/models/qwen3_5/language.py` (Qwen3_5Attention, Qwen3_5GatedDeltaNet, Qwen3_5DecoderLayer).
- **mlx** (Apple): `/tmp/mlx` — steel NAX GEMM / steel_attention.
- **mlx-native** (Rust MLX hybrid port): `/tmp/mlx-native`.
- **saragossa** (NAX + GDN shader port): `/tmp/saragossa`.
- **lattice** (GEMM/attention/fusion kernels): `/tmp/lattice` (flash attention, fused qk-norm+RoPE, gemv decode core).
- **candle** (small Metal kernels): `/tmp/candle` (`candle-metal-kernels/src/metal_src/`).
- **mistral.rs** (Candle-based Rust LLM engine, Metal kernels): `/tmp/mistral.rs`
  (`mistralrs-quant/src/metal_kernels/` — quantized GEMM/gemv, rotary, flash_attn,
  scan/moe/bitwise; `mistralrs-paged-attn/src/metal/kernels/` — paged attention).
- **omlx** (ANE/GPU hybrid prefill, experimental): `/tmp/omlx-latest` (legacy snapshot `/tmp/omlx`).
- **mlxenv**: `/tmp/mlxenv` (py3.14 venv, mlx 0.32).