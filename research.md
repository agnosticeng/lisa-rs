# Research: Rust + Metal high-performance inference design notes

Design notes for building a high-performance Rust + Metal inference runtime for
the Qwen3.8-27B hybrid (GDN + full-attention) model, compiled from studying
mature native implementations: MLX itself plus `mlx-native`, `saragossa`,
`lattice`, `candle`, and `omlx`. Everything below is forward-looking design
guidance and correctness traps to avoid, derived from those codebases.

Reference repos (cloned to `/tmp`):

- `saragossa` — `na_gemm.metal` is a faithful port of MLX's steel NAX
  quantized tensor-unit GEMM; `steel_attention.metal` is a verbatim MLX
  steel_attention shader file. Best source for tensor-core fragment mechanics
  and MLX shader idioms.
- `lattice` — clean M1-style simdgroup GEMM/GEMV suite, fused flash attention
  with online softmax, fused qk-norm+RoPE kernel; tiled q4 simdgroup GEMM.
  Best source for compact idiomatic kernels and a `qwen35.metal` engine.
- `mlx-native` — full Rust re-implementation of MLX with the same hybrid model
  family (gated-delta-net / qwen3.5). Rich test/ADR suite for fused epilogues
  and parity discipline.
- `candle` (candle-metal-kernels) — clean small kernels for gemv / gemm /
  quantized / affine / sdpa; compile-many-small-libraries instead of one
  monolithic shader.
- `omlx` — MLX inference server. Its experimental **ANE/GPU hybrid prefill**
  uses private AppleNeuralEngine + MIL `conv` programs + IOSurface to run
  output-channel slices of MLP/GDN projections on the ANE in parallel with the
  GPU NAX QMM suffix. Source: `omlx/custom_kernels/qwen35_prefill/`.
  MTP/doc source: `mlx_lm` (`speculative/mtp.py`) + `mlx-vlm`
  (`speculative/drafters/qwen3_5_mtp`, `speculative/mtp.py`).

---

## 1. Metal compute model (empirical ground truths)

Verified on Apple M5 Max / macOS 26.5 (arch `applegpu_g17s`):

1. **Consecutive dispatches in ONE encoder serialize on data but overlap
   ramp/drain.** Back-to-back independent dispatches overlap: 50 identical QMMs
   in one encoder = 0.33 ms each vs 1.13 ms isolated = ~3.4x. A prefill is a
   serial RAW chain of hundreds of kernels, so every kernel pays a
   near-single-dispatch cost. Compute-heavy kernels hide this; tiny kernels
   don't.
2. **`dispatch_threads` counts *total threads*, not threadgroups.** Passing
   threadgroup counts yields massive under-dispatch (e.g. a `(32,48,1)` grid
   with threadgroup `(32,4,1)` = only 12 threadgroups; a GDN norm then covers
   1% of its output). Maintain two strict dispatch helpers:
   `run1d`/`run3d` = `dispatch_threads` (grid in threads) and
   `run1d_tg`/`run3d_groups` = `dispatch_thread_groups` (grid in threadgroups).
   Most kernels want the threadgroup form. This bug silently produces
   approximately-right-looking but quantitatively wrong output.
3. **Metal buffer offsets are BYTES.** Every element offset must be multiplied
   by the element width before `set_buffer`/`set_bytes`.
4. **`#pragma unroll` is NOT reliable on mma loop nests.** Loop nests calling a
   big `mma` function body leave fragment arrays (`D[4]`, `A[4]`, `Bt[4]`) in
   stack memory → ~4x slowdown. **Fix: manually unroll the mma calls with
   literal indices.** In MLX/saragossa this is what closes most of the gap
   against a naive tensor-unit port.
5. **Shaders compiled via `include_str!` + `new_library_with_source`** — source
   changes require a rebuild (cargo caches the dependency). Touch the shader
   file when a change "does not take effect".
6. **Keeping two full model engines alive at once causes memory thrash.** For
   A/B validation, run both paths in ONE engine (run one, then reset + run the
   other), and only load the model once.
7. **Precompiled shader libraries build faster than one monolithic source.**
   A single ~2000-line `shaders.metal` recompiles wholesale on any edit; split
   logical op families into separate libraries (candle's `KernelManager` caches
   per-`(source, name, constants)` pipelines with double-checked locking).

---

## 1b. Model format, loader, and weight layout

This section pins down the on-disk format and the load/manipulation strategy so
an implementation can be written without inspecting the weights manually.

### Model identity

- **Target checkpoint**: `mlx-community/Qwen3.8-27B-4bit` (MLX-native affine q4,
  group size 64). Files: `config.json`, `tokenizer.json`, `model.safetensors`
  (`+ model.safetensors.index.json` for the sharded case).
- **Drafter checkpoint**: `mlx-community/Qwen3.8-27B-MTP-4bit`
  (`model_type=qwen3_5_mtp`, `block_size=3`) — see section 8. It borrows the
  target's embed and lm_head, so it is loaded next to the target, not alone.

### safetensors container format

The file is:

```
[u64 LE header_byte_len][header JSON][tensor data blob...]
```

- `header_byte_len` is a little-endian `u64`.
- Header JSON maps each tensor name to
  `{"dtype": "...", "shape": [...], "data_offsets": [start, end]}`.
  `data_offsets` are byte offsets **relative to the end of the JSON header**
  (not relative to file start). Modern files use `data_offsets: [start,end]`;
  very old files use a single `data_offset`.
- Dtype byte widths (bcp table): `BOOL/U8/I8/F8_E4M3/F8_E5M2` = 1,
  `F16/BF16/U16/I16` = 2, `F32/I32/U32` = 4, `F64/I64/U64` = 8.

Header must be validated before any data is copied:
- `start <= end`, `end <= data_len`.
- `numel * bits`, must be byte-aligned and equal `end - start`.
- For a single shard, tensor byte ranges must tile the data region exactly
  (contiguous, no overlap, no trailing gap). This rejects corrupt/truncated
  checkpoints before a large allocation.

### Affine q4 weight layout (MLX)

For a quantized linear, three tensors per layer projection:

- `weight`: **U32 `[out, in/8]`** — each u32 packs 8 nibbles. Stores the
  low indices first.
- `scales`: **BF16 `[out, in/64]`**
- `biases`: **BF16 `[out, in/64]`**

Per output row `o`, per group `g` over input columns `[g*64, (g+1)*64)`:
`scale = scales[o, g]`, `bias = biases[o, g]`, and element `j` in the group is
`nibble * scale + bias` where `nibble ∈ {0..15}` is read from the packed weight
as `(weight[o, in/8] >> (4 * (col % 8))) & 0xF`. (See the dequant in section 2.)

Verification from the safetensors metadata: `in = shape[weight][1] * 8`,
`out = shape[weight][0]`, `groups = in / 64`.

### BF16 <-> F32

The model ships BF16. Conversion is a bit-shift, no rounding:
- `f32 -> bf16 bits`: `(f32.to_bits() >> 16) as u16` (truncate).
- `bf16 bits -> f32`: `f32::from_bits((u16 as u32) << 16)`.

### Load/manipulation strategy (decided)

Adopt the **lazy mmap + one page-aligned Metal arena** pattern (what lattice and
MLX both converge on, tuning for correctness-first):

1. `memmap2::Mmap` the safetensors files (never `read` into RAM up front).
2. Parse the header JSON; build `name -> (shard, byte_offset, shape, dtype)`
   and validate every tensor's layout per the rules above, but **do not copy
   anything yet**.
3. Verify required tensor names exist with the expected shapes before
   allocating (lattice's `load_owned_tensor_checked` calls `has_tensor`/shape
   first, so a bad model is rejected before a ~GB copy).
4. Compute per-shard byte offsets, 64-KiB-align each, and copy the live byte
   range of each shard into a single `StorageModeShared` Metal buffer per
   shard. Keep the mmap alive (lazily) and do not hand mmap pages directly to
   Metal as storage — copy them into page-aligned Metal buffers so weights are
   always resident and safe for `set_buffer` with arbitrary tensor offsets.
5. GPU kernels address weights through `(shard_idx, byte_offset)`; the layer
   table is built once at load and never reparsed.

For maximum determinism this is a **one-time eager** load of the ~13 GB into
Metal (unified memory; acceptable on 128 GB). Do not preload the drafter eagerly;
wire its weights to the target's embed/lm_head buffers at runtime (section 8).

---

## 2. Tensor-unit (NAX) GEMM — the foundation for quantized matmul

On g17+ (tensor-unit available, arch `applegpu_g17s`),
`mpp::tensor_ops::matmul2d` is usable at runtime compile:

```metal
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_tensor>
using namespace mpp;
constexpr auto desc = tensor_ops::matmul2d_descriptor(
    64, 32, static_cast<int>(dynamic_extent), false, false, false);
tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
```

MLX's affine suite (steel `quantized_nax.h`) achieves ~45-48 TFLOPS at M=128.
Key mechanics (from `saragossa/src/metal_backend/na_gemm.metal`, a faithful
`BaseNAXFrag` transcription):

- **Fragment 16x16, 8 bfloat16 elems/thread, lane→coord bit-twiddle:**
  `qid = lane >> 2; fm = (qid & 4) | ((lane >> 1) & 3); fn = ((qid & 2) | (lane & 1)) * 4`.
- **MMA 16x32x16** through `NaxFrag::mma(c0a,c0b, af, b0, b1)` where the
  fragment is split across two `metal::vec<bfloat,8>` halves fed as a 16x16 pair.
- **Q4 affine dequant** (`load_b_quant_u4`, group size 64):
  `packed_cols = K/8`, `word = packed[nn*packed_cols + (kk>>3)]`,
  `q = (word >> ((kk & 7) * 4)) & 0xF`, `g = kk/64`,
  `dst = bfloat(q*scale[nn*groups+g] + bias[nn*groups+g])`.
  MLX affine layout: `U32[out, in/8]` weight + BF16 `scales`, `biases`
  `[out, in/64]`.
- **Tiled kernel shape** (`gemm_nax_coop_qb_tiled_u4`): BM=BN=BK=64, WM=WN=2,
  one threadgroup per 64x64 tile, dequant tile `Ws[BN*BK]` into threadgroup
  memory, 4 simdgroups compute (sg_m,sg_n), K loop of 16-step mma. This is
  MLX's steel `QuantizedBlockLoader`/`steel_gemm` recipe.
- **Explicit template instantiation** (`instantiate_kernel` / `host_name` with
  concrete template args) lets the compiler bake all constants and never spill
  fragments. Prefer emitting one concrete kernel name per config over runtime
  switches.
- Useful kernel variants: plain bf16 out, fused residual add, f32 logits out,
  and a paired form that runs two projections in one dispatch (e.g. gate+up,
  `tid.z` selects the weight) to halve the MLP dispatch count. Fuse SiLU(g)*u
  as a separate cheap kernel over the paired result.

---

## 3. Attention recipes

### Online-softmax flash attention (lattice `flash_attention.metal`)

- One threadgroup per (kv_head × q-tile of FA_TILE_Q), `FA_THREADS = Q*GQA*32`.
- Q in registers; K/V tiles (float4) double-buffered in threadgroup mem.
- Causal masking done **per row** by clamping tile length:
  `last_k_exclusive = qi+1; row_tile_len = min(TILE_K, last_k_exclusive-k_start)`.
- Running max/sum rescale: `alpha=exp(m_i-m_new)`, `o = o*alpha + Σ p_ij V`,
  `l = l*alpha + Σ p_ij`.
- **Fail-closed finalize**: if the running sum `l_i` is not finite or `<= 0`,
  write literal `0.0f` (never `o * (1/l)`), because `NaN * o` stays NaN.

### Fused QK-norm + RoPE (lattice `fused_qk_norm_rope`)

- One threadgroup per (pos, head, family==Q|K); threadgroup reduction over the
  head dim; `inv_rms = rsqrt(sumsq/D + eps)` then rotate with precomputed
  `rope_cos/rope_sin[pos*HALF + tid]`. This turns 4 dispatches (q-norm, k-norm,
  q-rope, k-rope) into 1.

### Attention layout and dispatch correctness traps

The full-attention core must be validated bit-identically against a scalar
reference. The mechanical traps:

- Raw Q layout is **24 heads × 512 = 12288/token**, where each head's first 256
  values are the query and the next 256 are the gate
  (`attn_output_gate=true`). Raw K is **4 kv heads × 256 = 1024/token but
  stored with a 2048 row stride**. The rotary-norm kernels must index
  `row*12288 + head*512` (q) and `row*2048 + head*256` (k). Any off-layout
  stride silently corrupts every head from token 1 onward.
- Rotation covers `partial_rotary_factor * head_dim` dims (0.25 * 256 = 64):
  split the head into two 32-element halves and apply cos/sin in-place.
- The norm/rot prefills must run on real threadgroups
  (`dispatch_thread_groups` with e.g. 256 threads), not `dispatch_threads` of
  size 1. With a 1-thread "group", the reduction `tgs.x/32==0` gives
  `inv = 1/sqrt(eps) = 1000` → every post-attention value is ~1000x too large
  while argmax still "matches".
- KV cache is `[kv_head, pos, dim]` so a positional scan is contiguous;
  dispatch with actual threadgroup grids.

---

## 4. GEMM/GEMV shapes

### GEMV M=1 decode path (lattice `gemv_decode_core`)

- One threadgroup per output element; `float4`/`half4` vector loads when K and
  the leading dimension are 4-aligned; `simdgroup_reduce_add_f32` tree
  reduction over lanes. `simd_shuffle_down` is the portable 32-lane idiom.
- Fast paths ordered: `(K&3)==0 && (ld&3)==0` → float4; `(K&1)` → float2; else
  scalar.

### Quantized q4 tiled GEMM (lattice `gemm_q4_tiled.metal`)

- Q4 stored as u8 rows, dequantized on load into a padded threadgroup tile
  `half Wtg[32][40]` (BN_PAD=40 pads the 32-lane loads), then
  `simdgroup_multiply_accumulate` with `half8x8` tiles. 8 accumulate tiles per
  threadgroup. Shows the padding trick for unaligned tile loads.

---

## 5. GDN (gated delta net) layer geometry

For Qwen3.8-27B, from the safetensors metadata (see section 1b for how these
are read):

- GDN projection `in_proj_qkv` is `[10240, 5120]` (q=2048 + k=2048 + v=6144),
  `in_proj_z`=6144, `in_proj_a/b`=48, `out_proj`=`[5120, 6144]`; 16 key heads,
  48 value heads, key/value head dim 128, conv kernel dim 4, group size 64.
- Storage math: q scale `1/128`, k scale `1/sqrt(128)`, decay
  `exp(-exp(A_log) * softplus(a + dt_bias))`.
- Precision boundaries: activations, conv state, KV, and scratch stay BF16;
  the persistent GDN recurrent state is F32; final logits are F32.
- `mlx-native` provides a full Rust port of the same model family, including a
  fused gate+up+SiLU (`ops/fused_gate_up_silu_iq4_nl.rs`) and a causal conv
  (`ssm_conv.rs`) — good cross-check sources.

---

## 6. Performance engineering checklist

- One encoder per chunk, one sync per chunk; add instrumentation for per-section
  GPU timing so the GPU-bound vs launch-bound regions are visible.
- Batch per-token kernels (RMSNorm, rotary, conv, gate/up) into one dispatch.
- Fuse the residual into the main q4 projection (removes a separate add
  dispatch and an intermediate write).
- For the lm_head at M=1, compute only the last position's logits row (saves a
  large write + GFLOPs).
- Batched kernels must still preserve the scalar row's arithmetic order. e.g. a
  128-thread RMSNorm that reduces via 4 simd-group partials is bit-identical to
  the scalar path; a 256-thread reduction is not.
- Reduce the dependent-chain ramp: the heavy NAX QMMs dominate, but a serial
  prefill chain of ~640 dispatches makes each pay a near-single-dispatch cost.
  Cross-device overlap (ANE/GPU) and decode-time speculative batching (MTP
  verify of multiple rows in one forward) are the levers that break the M=1 /
  serial-chain ceiling — intra-encoder reordering does not help because the
  layers are RAW-dependent.
- Argmax: a multiple-thread reduction over the logits instead of a single-thread
  scan.
- Audit every dispatch helper call site; reductions over a tile need real
  threadgroups.

---

## 6b. ANE/GPU hybrid prefill (oMLX, experimental)

oMLX 0.6.3 adds hybrid prefill: fixed-shape INT8 programs on the **Neural
Engine** compute output-channel slices of the MLP gate/up and GDN projections
while the **GPU runs the remaining channels on NAX qmm kernels**. Validated on
M3 Ultra (Qwen3.8-27B oQ4e): prefill +18.9% @32K, +17.8% @16K, +3.4% @4K, and a
`61.12ms → 48.45ms` layer-0 MLP (1.26x). Mechanics:

- **MIL program = a 1x1 `conv`** (`int8_linear_mil`): weights are
  per-*output-row* INT8 (symmetric scale = max|w|/127),
  `constexpr_blockwise_shift_scale` dequantizes to fp16, fp16 in/out tensors
  `[1, seq, 1, hidden]`. Weights are requantized to INT8, so the path is
  **approximate** (top-1 stable, cosine 0.999). A plain `fp16_linear_mil`
  exists for exact mode, and `fp16_swiglu_down_mil` fuses gate/up/down into one
  conv chain.
- **Private runtime**: `dlopen` AppleNeuralEngine.framework;
  `_ANEInMemoryModelDescriptor` → `modelWithMILText:weights:optionsPlist:`;
  `_ANEInMemoryModel` → `inMemoryModelWithDescriptor:`,
  `compileWithQoS:options:error:`, `loadWithQoS:...`; `_ANERequest` +
  `_ANEIOSurfaceObject`. Verify availability with
  `device respondsToSelector:newBufferWithIOSurface:`.
- **Zero-copy bridge**: IOSurface-backed Metal buffers (`newBufferWithIOSurface:`)
  are the model's input/output buffers; a Metal `pack_input` kernel writes x,
  the ANE `eval` runs, and Metal reads the output surfaces back.
- **Overlap/synchronization**: producer command buffer `encodeSignalEvent(ready)`;
  a CPU thread blocks on the ANE then signals `done`; consumer
  `encodeWaitForEvent(done)`. On M3 Ultra the blocking form beat the
  completion-callback variant (the callback delayed ANE until after the GPU
  suffix, killing overlap).
- **Split model**: ANE handles the output-channel prefix `[0, ane_n)`, GPU the
  suffix `[ane_n, total)`. A merge kernel interleaves the ANE planar output with
  the GPU rows; a SwiGLU variant applies `gate*up/(1+exp(-gate))` inline during
  the merge so no full gate/up is materialized. Tuned split is **53% for MLP,
  50% for GDN** on M3 Ultra; on M5/NAX it sits well below 50% — tune per machine.
- **Dual ANE**: M3 Ultra exposes ANE instances 1,2; the same op evaluates twice
  (once per ANE, pinned) in parallel, merging a prefix and a middle slice.
- **Banking**: 112 fixed-shape procedures (64 MLP + 48 GDN) pack into **two
  resident programs** (one per ANE) sharing one weight blob; ~4 GiB address
  window per ANE. On single-die chips (M3 Max) two banks don't fit → retry with
  split banks. Eager compile adds ~40s startup.
- **Tuner**: measure a few representative layers, predict the split, then verify
  once (≤3 full runs instead of 11).
- Weight blob format: 64-byte header + `0xefbeadde`-chunked sections holding the
  int8 data and fp16 scales, referenced by `BLOBFILE(...offset=uint64(N))`.

For a Rust runtime: the GPU side uses NAX QMM for the suffix; the ANE side
requires an MPS/MIL `conv` program plus the IOSurface bridge and private ObjC
(via `ffi`/`objc` crates). On M5-family the ANE win is smaller than the classic
~50% optimum, so this is a stretch goal behind MTP batching.

---

## 7. Correctness tooling for kernel changes

- Keep a scalar/CPU reference path and compare per-layer hidden states (and,
  for attention layers, per-head q/k/v) against the accelerated path on a
  single model load. A layer-0 bit-exact result means the kernels match; any
  silent per-token divergence shows up immediately.
- Bisect by running progressively more layers. In this model attention layers
  appear at indices `3, 7, ..., 63`; a divergence introduced by a GDN layer and
  one introduced by an attention layer are different bug classes.
- Use a small synthetic random-weight model with the same GDN/attention geometry
  to iterate in seconds instead of waiting on the full 13 GB download. Its q4
  layout must match the affine group-64 layout the kernels consume.
- `mlx-native/tests/test_gated_delta_net*` parity suites are the template for
  verifying a GDN recurrence against a CPU reference (compare before/after
  per-token state, not just the final logits).

---

## 8. MTP (Multi-Token Prediction)

The main 27B model has **no embedded MTP head**. The drafter is a separate
module `mlx-community/Qwen3.8-27B-MTP-4bit` (`model_type=qwen3_5_mtp`,
`block_size=3`). It is tiny (~1 transformer layer) and **borrows the target's
embeddings and lm_head at runtime** (`mtp_use_dedicated_embeddings=false`), so
the target produces the logits per draft position.

### MTP head architecture (from `mlx_lm` `qwen3_5_mtp.py`)

Tensors (`fc` is an affine-q4 `Linear(2*H, H)`, group 64):
`fc.{weight,scales,biases}` shape `[5120, 1280] U32` + `[5120,160] BF16`
(= K=10240 → **`fc` concatenates two H=5120 vectors**), `pre_fc_norm_embedding`,
`pre_fc_norm_hidden` (H=5120 RMSNorm gains), `norm.weight` (final RMSNorm),
and one full decoder layer `layers.0` =
`input_layernorm`,
`self_attn.{q_proj[12288,640], k_proj[1024,640], v_proj[1024,640], o_proj[5120,768], q_norm[256], k_norm[256]}`,
`post_attention_layernorm`, `mlp.{gate,up,down}` (17408 intermediate). With
`full_attention_interval=1` the single layer is **full attention** (not GDN),
the same Qwen3.5Attention as the target layers (24 heads, 4 kv heads,
head_dim 256, partial rotary 0.25).

Forward (per draft position):
```
token_embed = target.embed_tokens(tok) * target_embed_scale   # H
h = fc( concat( pre_fc_norm_embedding(token_embed),
                pre_fc_norm_hidden(target_last_hidden) ))      # H
h = layers.0(h)          # full-attention decoder layer (attn + mlp)
h = norm(h)
logits = target.lm_head(h)   # borrow target head
```

Because the MTP layer is one ordinary full-attention decoder layer reusing the
target's projections, vocab head, and attention geometry, it shares every
kernel the target uses. What must be added: the two `pre_fc_norm*` RMSNorm
dispatches, a `fc` q4 GEMM over the concatenated K=10240 input, one full-attn
layer dispatch, and the final `norm` + lm_head.

### Native MTP speculative loop (from `mlx_lm` `mtp.py`)

`block_size=3` → each round drafts `block_size-1 = 2` tokens autoregressively
through the single MTP layer, then the **target verifies one batch of
`[bonus, draft_0, draft_1]` in a single forward**. Walk draft vs target logits
(from each row of the verify hidden), accept the matching prefix, and **rewind
the target cache by `bs - (accepted+1)`** positions. Key mechanics:

1. **Separate drafter KV**: Qwen MTP ignores the generic `shared_kv_states`
   argument and owns a KV cache for its learned q/k/v projections. Prompt
   priming fills that cache from shifted target tokens and target hidden states.
   The target and drafter share embeddings, lm_head, and target hidden states,
   not projected K/V tensors.
2. **Position math**: `_mtp_draft_position(kv_valid_len) = kv_valid_len - 1`;
   the verify input is `[b, draft_tokens]` (length `bs`), and the hidden slices
   for the next draft seed are taken from the **accepted slot** of the target
   verify hidden (`hidden[:, accepted:accepted+1, :]`).
3. **Rewind after rejection**: `cache.trim(num_draft - num_accept)` trims the
   target cache to the accepted prefix only (a trimmable KVCache, no KV copies).
   The generic loop rewinds the model cache by `num_draft - n` and the draft
   cache by `max(num_draft - n - 1, 0)` every round.
4. **Emit**: accepted drafts, then the first non-accepted target token; the next
   round's seed is `[last accepted draft token] + first rejected target token`
   only when all drafts were accepted.
5. **Acceptance-rate-aware block size**: shrink the draft depth when the recent
   prefix-hit rate < 0.65, capping at the configured `block_size`.
6. **Parallel streams**: draft rounds run on a generation stream; the target
   verify is a single batched forward so GPU utilization stays high.

### MTP + windowed KV (future work)

At long contexts the autoregressive drafter rereads the full target KV each
position. A windowed drafter KV (the draft reads only a sliding window of the
shared KV; the target still validates on the full context) cuts draft cost to
`O(window)` while keeping exact target verification. This matters for 20k-100k
token contexts on M5 Max.

### Integration design

- Load the drafter weights alongside the target and reuse the target's
  embed/lm_head buffers.
- The single MTP layer reuses the target's attention-decode kernels (full-attn,
  GQA6, partial rotary); the `fc` GEMM is the same affine-q4 matmul with
  `K=10240`, fed by a small kernel that concatenates the two normed halves.
- Batch the target "verify" of `block_size` positions as one decode with
  `rows=block_size` (q4 matmul for projections + KV store + GQA6 decode), the
  same shape class as a chunked prefill but length `bs`, ending with a cache
  trim instead of rollback copies.
- Use a rolling/trimmable KV cache so rejection never copies KV (trim advances
  a per-head start offset).

---

## 9. Architecture decisions and priorities

Ordered by expected tok/s impact on an M5 Max target for the 27B hybrid
(prerequisites first):

0. **Loader + weight arena** (section 1b): mmap + validate + one page-aligned
   Metal buffer. Everything depends on correctly reading the affine q4 layout.
1. **Native MTP speculative decode** (highest decode lever). Draft 2 tokens per
   round through the single MTP layer, verify `[bonus, d0, d1]` at once, trim
   KV. MLX/DFlash measured decode speedups ~1.4x at 60-70% acceptance. See
   section 8.
2. **Batched target verify** breaks M=1 serial decode: one forward over
   `rows=block_size` (q4 matmul projections + KV store + GQA6 decode).
3. **Steel_attention prefill** port (flash attention over the batch) caps
   attention-prefill cost; a prerequisite for holding prefill tok/s high as
   context grows.
4. **Compile shaders per logical library** (candle style) to cut monolithic
   recompile time and enable per-kernel `host_name` template instantiation for
   the NAX fragments.
5. **ANE/GPU hybrid prefill** (section 6b) — the prefill lever beyond the GPU
   NAX QMM chain; on M5-family the win sits below the classic ~50% optimum, so
   pursue after 1-4. Requires the private ANE runtime + IOSurface bridge.
6. **Windowed draft KV** for MTP at 20k-100k contexts: the drafter reads a
   sliding KV window while the target verifies on the full context.

Explicitly NOT planned: a separate draft-model speculative decoder (the target's
own MTP head is cheaper and higher-acceptance), speculative chains through the
whole 64-layer model per draft step, and CPU-side rollback copies for the target
cache (use `trim`).

### Reference benchmark (M5 Max, MLX 0.32, greedy, 64 generated tokens)

Prompt: `Implement an in-place stable merge sort in Rust and explain its complexity.`

- Median of three runs with subprocess output captured symmetrically.
- Target only: 29.352 tok/s, 18.574 GB peak.
- Qwen3.8-27B-MTP-4bit, block size 3: 51.766 tok/s, 17.403 GB peak.
- Speedup: 1.764x.
- Acceptance: 1.21 accepted drafts/round, 61.4% of drafted tokens,
  2.21 emitted tokens/round over 29 rounds.
- Raw result: `benchmarks/mtp_mlx_m5_max.json`.

### Native baseline before batched verification

The exact native block-3 loop currently verifies `[bonus,d0,d1]` with three
sequential target M=1 forwards and GPU-resident GDN checkpoints. On M5 Max,
64 greedy tokens produce exactly the target-only sequence:

- Target only: 10.531 tok/s.
- Sequential-M=1 speculative: 6.977 tok/s.
- Acceptance: 55.17% (32/58 drafts), 29 rounds.
- Speedup: 0.663x; 90 target forwards versus 64 in the baseline.
- Raw result: `benchmarks/mtp_native_m5_max.json`.

This is the correctness baseline for the pending target M=3 batched verify; it
must not be presented as speculative acceleration.
