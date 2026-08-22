#include <metal_stdlib>
using namespace metal;

// Full-attention layer (Qwen3.5 hybrid): 24 Q-heads × 4 K/V-heads, head_dim 256,
// partial rotary 0.25 (proportional/NeoX pairing across the two 128 halves),
// and an output gate (attn_output_gate) that is NOT normed.
//
// Raw q from q_proj is [L,12288] = 24 heads × 512 (256 query + 256 gate per
// head). Raw k is [L,1024] = 4 heads × 256. All kernels are scalar (one thread
// per (row,head)) — correctness-first; exactness is validated later vs a fused
// thread-group variant.

#define HEAD_DIM 256
#define HALF_DIM 128
#define ROT_PAIRS 32 // partial_rotary 0.25 * head_dim = 64 dims -> 32 pairs
#define ROT_DIM 64   // rotary dimension (dims [0,64) rotated)
#define THETA 10000000.0f // rope_parameters.rope_theta = 1e7
#define QHEADS 24
#define KVHEADS 4
#define QSTRIDE (12288)    // q row byte stride in elements (24*512)
#define KSTRIDE (1024)     // k raw row stride (4*256)
#define OSTRIDE (6144)     // query / gate output row stride (24*256)
#define KOUTSTRIDE (2048)  // k output row stride (4*256)
// Flash-style sliced decode: fixed-size context chunks + the max number of
// slices the runner's buffers are sized for (must match runner ATTN_SLICES).
#define TOKENS_PER_SLICE 64
#define ATTN_SLICES 256

// RMS-normalize one head of 256 dims against the given gain, then apply the
// partial rotary: rotate pairs (i, i+ROT_DIM/2) for i in 0..ROT_PAIRS (covering
// dims [0,64)), with freq = pos / theta^(2i/ROT_DIM). Matches mlx-vlm
// Qwen3_5RotaryEmbedding (dim=64, base=1e7, NeoX/half-split pairing).
static void rms_norm_rope(device const float* src,    // per-head base pointer
                          device const bfloat* gain,  // [256]
                          device bfloat*       dst,   // per-head output base
                          float rms_eps,
                          uint row) {
  float s = 0.0f;
  for (uint j = 0; j < HEAD_DIM; j++) {
    // QuantizedLinear returns model-dtype BF16 in MLX. q4_qmv_fast retains F32
    // for accumulation, so restore the model boundary before normalization.
    float v = float(bfloat(src[j]));
    s += v * v;
  }
  float rms = rsqrt(s / 256.0f + rms_eps);
  thread float n[HEAD_DIM];
  for (uint j = 0; j < HEAD_DIM; j++)
    n[j] = float(bfloat(src[j])) * rms * float(gain[j]);
  for (uint j = 0; j < ROT_PAIRS; j++) {
    float x0 = n[j];
    float x1 = n[j + ROT_DIM / 2];
    float freq = float(row) / pow(THETA, float(2u * j) / float(ROT_DIM));
    float c = cos(freq);
    float s_ = sin(freq);
    dst[j]            = bfloat(x0 * c - x1 * s_);
    dst[j + ROT_DIM / 2] = bfloat(x1 * c + x0 * s_);
  }
  for (uint j = ROT_DIM; j < HEAD_DIM; j++)
    dst[j] = bfloat(n[j]);
}

// Split raw q [L,12288] into normed/roped query [L,6144] (bf16) and raw gate
// [L,6144] (f32). Handles the gate WITHOUT norming it.
kernel void qk_norm_gate_rope(device const float* q    [[buffer(0)]], // [L,12288]
                              device const bfloat* qn   [[buffer(1)]], // [256]
                              device bfloat*       oq    [[buffer(2)]], // [L,6144]
                              device float*        og    [[buffer(3)]], // [L,6144]
                              constant uint&       L     [[buffer(4)]],
                              constant float&      eps   [[buffer(5)]],
                              uint2 gid [[threadgroup_position_in_grid]]) {
  uint head = uint(gid.y);
  uint row  = uint(gid.x);
  if (row >= L || head >= QHEADS) return;
  uint qb = row * QSTRIDE + head * 512u;
  uint ob = row * OSTRIDE + head * HEAD_DIM;
  rms_norm_rope(q + qb, qn, oq + ob, eps, row);
  for (uint j = 0; j < HEAD_DIM; j++)
    og[ob + j] = float(bfloat(q[qb + HEAD_DIM + j]));
}

// Normalize + rope the 4 K-heads. Raw k f32 [L,1024], output bf16 [L,2048].
kernel void k_norm_rope(device const float* k     [[buffer(0)]], // [L,1024]
                        device const bfloat* kng   [[buffer(1)]], // [256]
                        device bfloat*       ok    [[buffer(2)]], // [L,2048]
                        constant uint&       L     [[buffer(3)]],
                        constant float&      eps   [[buffer(4)]],
                        uint2 gid [[threadgroup_position_in_grid]]) {
  uint head = uint(gid.y);
  uint row  = uint(gid.x);
  if (row >= L || head >= KVHEADS) return;
  uint kb = row * KSTRIDE + head * HEAD_DIM;
  uint ob = row * KOUTSTRIDE + head * HEAD_DIM;
  rms_norm_rope(k + kb, kng, ok + ob, eps, row);
}

// Two-level reduction of per-lane sum-of-squares across all `lsize` threads of
// a threadgroup, returning rsqrt(mean + eps). Mirrors rms_norm_rows in
// norm.metal; used by the parallel (multi-simdgroup) decode heads.
static inline float reduce_rms_inv(
    float s, uint simd_lane, uint simd_group,
    threadgroup float* local_sums, threadgroup float* local_inv,
    uint cols, float eps) {
  s = simd_sum(s);
  if (simd_group == 0u) local_sums[simd_lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0u) local_sums[simd_group] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0u) {
    s = simd_sum(local_sums[simd_lane]);
    if (simd_lane == 0u) local_inv[0] = rsqrt(s / float(cols) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return local_inv[0];
}

// Decode one query row at an absolute sequence position. 256 threads cooperate
// on the 256-dim norm (one dim each); lanes 0..31 also apply the 32 rotary pairs.
kernel void q_norm_gate_rope_decode(
    device const float* q [[buffer(0)]],       // [12288]
    device const bfloat* qn [[buffer(1)]],     // [256]
    device bfloat* query [[buffer(2)]],        // [24,256]
    device float* gate [[buffer(3)]],          // [24,256]
    constant uint& position [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (head >= QHEADS) return;
  const uint qb = head * 512u;
  const uint ob = head * HEAD_DIM;
  threadgroup float local_sums[32];
  threadgroup float local_inv[1];

  float s = 0.0f;
  for (uint j = lid; j < HEAD_DIM; j += lsize) {
    const float v = float(bfloat(q[qb + j]));
    s += v * v;
    gate[ob + j] = float(bfloat(q[qb + HEAD_DIM + j]));
  }
  const float inv = reduce_rms_inv(s, simd_lane, simd_group, local_sums, local_inv, HEAD_DIM, eps);
  for (uint j = lid; j < HEAD_DIM; j += lsize) {
    const float n = float(bfloat(q[qb + j])) * inv * float(qn[j]);
    if (j < ROT_PAIRS) {
      const float n1 = float(bfloat(q[qb + j + ROT_PAIRS])) * inv * float(qn[j + ROT_PAIRS]);
      const float angle = float(position) * (1.0f / pow(THETA, float(2u * j) / float(ROT_DIM)));
      const float c = cos(angle), sn = sin(angle);
      query[ob + j] = bfloat(n * c - n1 * sn);
      query[ob + j + ROT_PAIRS] = bfloat(n1 * c + n * sn);
    } else if (j >= ROT_DIM) {
      query[ob + j] = bfloat(n);
    }
  }
}

// Normalize/rotate one K row and append K/V to compact persistent caches.
kernel void kv_cache_store_decode(
    device const float* k [[buffer(0)]],        // [4,256]
    device const float* v [[buffer(1)]],        // [4,256]
    device const bfloat* kng [[buffer(2)]],     // [256]
    device bfloat* key_cache [[buffer(3)]],     // [4,capacity,256]
    device bfloat* value_cache [[buffer(4)]],   // [4,capacity,256]
    constant uint& position [[buffer(5)]],
    constant uint& capacity [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (head >= KVHEADS || position >= capacity) return;
  const uint input = head * HEAD_DIM;
  const uint output = (head * capacity + position) * HEAD_DIM;
  threadgroup float local_sums[32];
  threadgroup float local_inv[1];

  float s = 0.0f;
  for (uint j = lid; j < HEAD_DIM; j += lsize) {
    const float kv = float(bfloat(k[input + j]));
    s += kv * kv;
    value_cache[output + j] = bfloat(v[input + j]);
  }
  const float inv = reduce_rms_inv(s, simd_lane, simd_group, local_sums, local_inv, HEAD_DIM, eps);
  for (uint j = lid; j < HEAD_DIM; j += lsize) {
    const float n = float(bfloat(k[input + j])) * inv * float(kng[j]);
    if (j < ROT_PAIRS) {
      const float n1 = float(bfloat(k[input + j + ROT_PAIRS])) * inv * float(kng[j + ROT_PAIRS]);
      const float angle = float(position) * (1.0f / pow(THETA, float(2u * j) / float(ROT_DIM)));
      const float c = cos(angle), sn = sin(angle);
      key_cache[output + j] = bfloat(n * c - n1 * sn);
      key_cache[output + j + ROT_PAIRS] = bfloat(n1 * c + n * sn);
    } else if (j >= ROT_DIM) {
      key_cache[output + j] = bfloat(n);
    }
  }
}

// Streaming decode attention. 32 lanes per head (8 dims each); the per-token
// score is a simd_sum over the 8-dim partials, online softmax stays scalar.
kernel void sdpa_decode_streaming(
    device const bfloat* query [[buffer(0)]],      // [24,256]
    device const bfloat* key_cache [[buffer(1)]],  // [4,capacity,256]
    device const bfloat* value_cache [[buffer(2)]],// [4,capacity,256]
    device float* out [[buffer(3)]],               // [24,256]
    constant uint& context [[buffer(4)]],
    constant uint& capacity [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (head >= QHEADS || context == 0u || context > capacity) return;
  const uint kvh = head / (QHEADS / KVHEADS);
  const uint qb = head * HEAD_DIM;
  float accum[8];
  for (uint d = 0; d < 8u; ++d) accum[d] = 0.0f;
  float max_score = -INFINITY;
  float denominator = 0.0f;

  for (uint token = 0; token < context; ++token) {
    const uint cache_base = (kvh * capacity + token) * HEAD_DIM;
    float partial = 0.0f;
    for (uint d = 0; d < 8u; ++d) {
      const uint dim = lane * 8u + d;
      partial += float(query[qb + dim]) * float(key_cache[cache_base + dim]);
    }
    const float score = simd_sum(partial) * 0.0625f;
    const float next_max = max(max_score, score);
    const float old_scale = exp2(1.4426950408889634f * (max_score - next_max));
    const float new_scale = exp2(1.4426950408889634f * (score - next_max));
    denominator = denominator * old_scale + new_scale;
    for (uint d = 0; d < 8u; ++d) {
      const uint dim = lane * 8u + d;
      accum[d] = accum[d] * old_scale + new_scale * float(value_cache[cache_base + dim]);
    }
    max_score = next_max;
  }
  const float inv = 1.0f / denominator;
  for (uint d = 0; d < 8u; ++d) {
    const uint dim = lane * 8u + d;
    out[qb + dim] = accum[d] * inv;
  }
}

// Three-row cached decode. Positions are absolute, and row r attends through
// base_position+r, so the newly appended block is strictly causal.
kernel void q_norm_gate_rope_block3(
    device const float* q [[buffer(0)]], device const bfloat* qn [[buffer(1)]],
    device bfloat* query [[buffer(2)]], device float* gate [[buffer(3)]],
    constant uint& base_position [[buffer(4)]], constant float& eps [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint head = tg.x, row = tg.y;
  if (head >= QHEADS || row >= 3u) return;
  const uint qb = row * QSTRIDE + head * 512u;
  const uint ob = row * OSTRIDE + head * HEAD_DIM;
  threadgroup float local_sums[32];
  threadgroup float local_inv[1];

  float s = 0.0f;
  for (uint j = lid.x; j < HEAD_DIM; j += lsize.x) {
    const float v = float(bfloat(q[qb + j]));
    s += v * v;
    gate[ob + j] = float(bfloat(q[qb + HEAD_DIM + j]));
  }
  const float inv = reduce_rms_inv(s, simd_lane, simd_group, local_sums, local_inv, HEAD_DIM, eps);
  const uint position = base_position + row;
  for (uint j = lid.x; j < HEAD_DIM; j += lsize.x) {
    const float n = float(bfloat(q[qb + j])) * inv * float(qn[j]);
    if (j < ROT_PAIRS) {
      const float n1 = float(bfloat(q[qb + j + ROT_PAIRS])) * inv * float(qn[j + ROT_PAIRS]);
      const float angle = float(position) * (1.0f / pow(THETA, float(2u * j) / float(ROT_DIM)));
      const float c = cos(angle), sn = sin(angle);
      query[ob + j] = bfloat(n * c - n1 * sn);
      query[ob + j + ROT_PAIRS] = bfloat(n1 * c + n * sn);
    } else if (j >= ROT_DIM) {
      query[ob + j] = bfloat(n);
    }
  }
}

kernel void kv_cache_store_block3(
    device const float* k [[buffer(0)]], device const float* v [[buffer(1)]],
    device const bfloat* kng [[buffer(2)]], device bfloat* key_cache [[buffer(3)]],
    device bfloat* value_cache [[buffer(4)]], constant uint& base_position [[buffer(5)]],
    constant uint& capacity [[buffer(6)]], constant float& eps [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint head = tg.x, row = tg.y, position = base_position + row;
  if (head >= KVHEADS || row >= 3u || position >= capacity) return;
  const uint input = row * KSTRIDE + head * HEAD_DIM;
  const uint output = (head * capacity + position) * HEAD_DIM;
  threadgroup float local_sums[32];
  threadgroup float local_inv[1];

  float s = 0.0f;
  for (uint j = lid.x; j < HEAD_DIM; j += lsize.x) {
    const float kv = float(bfloat(k[input + j]));
    s += kv * kv;
    value_cache[output + j] = bfloat(v[input + j]);
  }
  const float inv = reduce_rms_inv(s, simd_lane, simd_group, local_sums, local_inv, HEAD_DIM, eps);
  for (uint j = lid.x; j < HEAD_DIM; j += lsize.x) {
    const float n = float(bfloat(k[input + j])) * inv * float(kng[j]);
    if (j < ROT_PAIRS) {
      const float n1 = float(bfloat(k[input + j + ROT_PAIRS])) * inv * float(kng[j + ROT_PAIRS]);
      const float angle = float(position) * (1.0f / pow(THETA, float(2u * j) / float(ROT_DIM)));
      const float c = cos(angle), sn = sin(angle);
      key_cache[output + j] = bfloat(n * c - n1 * sn);
      key_cache[output + j + ROT_PAIRS] = bfloat(n1 * c + n * sn);
    } else if (j >= ROT_DIM) {
      key_cache[output + j] = bfloat(n);
    }
  }
}

kernel void sdpa_decode_streaming_block3(
    device const bfloat* query [[buffer(0)]], device const bfloat* key_cache [[buffer(1)]],
    device const bfloat* value_cache [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& base_position [[buffer(4)]], constant uint& capacity [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y;
  if (head >= QHEADS || row >= 3u || base_position + row >= capacity) return;
  const uint context = base_position + row + 1u;
  const uint kvh = head / (QHEADS / KVHEADS);
  const uint qb = row * OSTRIDE + head * HEAD_DIM;
  float accum[8];
  for (uint d = 0; d < 8u; ++d) accum[d] = 0.0f;
  float max_score = -INFINITY, denominator = 0.0f;
  for (uint token = 0; token < context; ++token) {
    const uint cache_base = (kvh * capacity + token) * HEAD_DIM;
    float partial = 0.0f;
    for (uint d = 0; d < 8u; ++d) {
      const uint dim = lane * 8u + d;
      partial += float(query[qb + dim]) * float(key_cache[cache_base + dim]);
    }
    const float score = simd_sum(partial) * 0.0625f;
    const float next_max = max(max_score, score);
    const float old_scale = exp2(1.4426950408889634f * (max_score - next_max));
    const float new_scale = exp2(1.4426950408889634f * (score - next_max));
    denominator = denominator * old_scale + new_scale;
    for (uint d = 0; d < 8u; ++d) {
      const uint dim = lane * 8u + d;
      accum[d] = accum[d] * old_scale + new_scale * float(value_cache[cache_base + dim]);
    }
    max_score = next_max;
  }
  const float inv = 1.0f / denominator;
  for (uint d = 0; d < 8u; ++d) {
    const uint dim = lane * 8u + d;
    out[qb + dim] = accum[d] * inv;
  }
}

// ---- Batched prefill variants: process `batch` consecutive rows in one
// dispatch, causal over the full prefix. `base_position` is the absolute
// sequence position of row 0; row r attends through base_position + r + 1.

kernel void q_norm_gate_rope_prefill(
    device const float* q [[buffer(0)]], device const bfloat* qn [[buffer(1)]],
    device bfloat* query [[buffer(2)]], device float* gate [[buffer(3)]],
    constant uint& base_position [[buffer(4)]], constant float& eps [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint head = tg.x, row = tg.y;
  if (head >= QHEADS || row >= batch) return;
  const uint qb = row * QSTRIDE + head * 512u;
  const uint ob = row * OSTRIDE + head * HEAD_DIM;
  threadgroup float local_sums[32];
  threadgroup float local_inv[1];

  float s = 0.0f;
  for (uint j = lid.x; j < HEAD_DIM; j += lsize.x) {
    const float v = float(bfloat(q[qb + j]));
    s += v * v;
    gate[ob + j] = float(bfloat(q[qb + HEAD_DIM + j]));
  }
  const float inv = reduce_rms_inv(s, simd_lane, simd_group, local_sums, local_inv, HEAD_DIM, eps);
  const uint position = base_position + row;
  for (uint j = lid.x; j < HEAD_DIM; j += lsize.x) {
    const float n = float(bfloat(q[qb + j])) * inv * float(qn[j]);
    if (j < ROT_PAIRS) {
      const float n1 = float(bfloat(q[qb + j + ROT_PAIRS])) * inv * float(qn[j + ROT_PAIRS]);
      const float angle = float(position) * (1.0f / pow(THETA, float(2u * j) / float(ROT_DIM)));
      const float c = cos(angle), sn = sin(angle);
      query[ob + j] = bfloat(n * c - n1 * sn);
      query[ob + j + ROT_PAIRS] = bfloat(n1 * c + n * sn);
    } else if (j >= ROT_DIM) {
      query[ob + j] = bfloat(n);
    }
  }
}

kernel void kv_cache_store_prefill(
    device const float* k [[buffer(0)]], device const float* v [[buffer(1)]],
    device const bfloat* kng [[buffer(2)]], device bfloat* key_cache [[buffer(3)]],
    device bfloat* value_cache [[buffer(4)]], constant uint& base_position [[buffer(5)]],
    constant uint& capacity [[buffer(6)]], constant float& eps [[buffer(7)]],
    constant uint& batch [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint head = tg.x, row = tg.y, position = base_position + row;
  if (head >= KVHEADS || row >= batch || position >= capacity) return;
  const uint input = row * KSTRIDE + head * HEAD_DIM;
  const uint output = (head * capacity + position) * HEAD_DIM;
  threadgroup float local_sums[32];
  threadgroup float local_inv[1];

  float s = 0.0f;
  for (uint j = lid.x; j < HEAD_DIM; j += lsize.x) {
    const float kv = float(bfloat(k[input + j]));
    s += kv * kv;
    value_cache[output + j] = bfloat(v[input + j]);
  }
  const float inv = reduce_rms_inv(s, simd_lane, simd_group, local_sums, local_inv, HEAD_DIM, eps);
  for (uint j = lid.x; j < HEAD_DIM; j += lsize.x) {
    const float n = float(bfloat(k[input + j])) * inv * float(kng[j]);
    if (j < ROT_PAIRS) {
      const float n1 = float(bfloat(k[input + j + ROT_PAIRS])) * inv * float(kng[j + ROT_PAIRS]);
      const float angle = float(position) * (1.0f / pow(THETA, float(2u * j) / float(ROT_DIM)));
      const float c = cos(angle), sn = sin(angle);
      key_cache[output + j] = bfloat(n * c - n1 * sn);
      key_cache[output + j + ROT_PAIRS] = bfloat(n1 * c + n * sn);
    } else if (j >= ROT_DIM) {
      key_cache[output + j] = bfloat(n);
    }
  }
}

kernel void sdpa_decode_prefill(
    device const bfloat* query [[buffer(0)]], device const bfloat* key_cache [[buffer(1)]],
    device const bfloat* value_cache [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& base_position [[buffer(4)]], constant uint& capacity [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y;
  if (head >= QHEADS || row >= batch || base_position + row >= capacity) return;
  const uint context = base_position + row + 1u;
  const uint kvh = head / (QHEADS / KVHEADS);
  const uint qb = row * OSTRIDE + head * HEAD_DIM;
  float accum[8];
  for (uint d = 0; d < 8u; ++d) accum[d] = 0.0f;
  float max_score = -INFINITY, denominator = 0.0f;
  for (uint token = 0; token < context; ++token) {
    const uint cache_base = (kvh * capacity + token) * HEAD_DIM;
    float partial = 0.0f;
    for (uint d = 0; d < 8u; ++d) {
      const uint dim = lane * 8u + d;
      partial += float(query[qb + dim]) * float(key_cache[cache_base + dim]);
    }
    const float score = simd_sum(partial) * 0.0625f;
    const float next_max = max(max_score, score);
    const float old_scale = exp2(1.4426950408889634f * (max_score - next_max));
    const float new_scale = exp2(1.4426950408889634f * (score - next_max));
    denominator = denominator * old_scale + new_scale;
    for (uint d = 0; d < 8u; ++d) {
      const uint dim = lane * 8u + d;
      accum[d] = accum[d] * old_scale + new_scale * float(value_cache[cache_base + dim]);
    }
    max_score = next_max;
  }
  const float inv = 1.0f / denominator;
  for (uint d = 0; d < 8u; ++d) {
    const uint dim = lane * 8u + d;
    out[qb + dim] = accum[d] * inv;
  }
}

// Scalar SDPA across one (row, head): causal scores against the 4 KV heads,
// softmax, weighted sum of v. qn bf16 [L,6144], kn bf16 [L,2048], v f32
// [L,1024], out f32 [L,6144].
kernel void sdpa_scalar(device const bfloat* qn  [[buffer(0)]], // [L,6144]
                        device const bfloat* kn  [[buffer(1)]], // [L,2048]
                        device const float*  v   [[buffer(2)]], // [L,1024]
                        device float*        out [[buffer(3)]], // [L,6144]
                        constant uint&       L   [[buffer(4)]],
                        uint2 gid [[threadgroup_position_in_grid]]) {
  uint head = uint(gid.y);
  uint row  = uint(gid.x);
  if (row >= L || head >= QHEADS) return;
  uint kvh = head / (QHEADS / KVHEADS); // 24 Q-heads share 4 KV-heads (gqa=6)
  const uint qb = row * OSTRIDE + head * HEAD_DIM;
  float sc[16];
  float mx = -1e30f;
  for (uint i = 0; i <= row; i++) {
    const uint kb = i * KOUTSTRIDE + kvh * HEAD_DIM;
    float acc = 0.0f;
    for (uint d = 0; d < HEAD_DIM; d++)
      acc += float(qn[qb + d]) * float(kn[kb + d]);
    sc[i] = acc * 0.0625f; // 1/sqrt(256)
    if (sc[i] > mx) mx = sc[i];
  }
  float w[16];
  float sum = 0.0f;
  for (uint i = 0; i <= row; i++) { w[i] = exp2(1.4426950408889634f * (sc[i] - mx)); sum += w[i]; }
  float iz = 1.0f / sum;
  uint ob = row * OSTRIDE + head * HEAD_DIM;
  for (uint d = 0; d < HEAD_DIM; d++) {
    float acc = 0.0f;
    for (uint i = 0; i <= row; i++)
      acc += w[i] * float(bfloat(v[i * KSTRIDE + kvh * HEAD_DIM + d]));
    out[ob + d] = acc * iz;
  }
}

// out * sigmoid(gate) -> bf16, the pre-o_proj handoff.
kernel void gate_out_bf16(device const float* out  [[buffer(0)]], // f32
                            device const float* gate [[buffer(1)]], // f32
                            device bfloat*      gated [[buffer(2)]], // bf16
                            constant uint&      n     [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {
  if (tid < n) {
    float o = out[tid];
    float g = gate[tid];
    float sg = 1.0f / (1.0f + exp2(-1.4426950408889634f * g));
    gated[tid] = bfloat(o * sg);
  }
}

// ---------------------------------------------------------------------------
// Flash-style sliced decode attention. The KV context is partitioned into
// fixed-size token slices (each slice covers TOKENS_PER_SLICE contiguous keys);
// each threadgroup (head,row,slice) runs the exact same per-token online-softmax
// loop over its slice using 32 lanes over the 256 dims (identical numerics to
// sdpa_decode_streaming within a slice), and writes its partial per-dim
// accumulator + running (max, denom) stats.
// A tiny finalize kernel then merges the slice stats flash-style:
//   gmax = max_s (max of slice);  tot_den = Σ w_s * den_s;  out = Σ w_s * O_s / den_s
//
// CRITICAL: the slice partition is a PURE FUNCTION of the row's own context
// (context = base_position + row + 1, fixed 64-token chunks), never of the
// dispatch batch. The partial for (head,row,slice) therefore depends only on
// that row, so the same row produces bit-identical attention whether it is
// decoded in isolation (M=1) or as part of a batch-3 verify — the speculative
// acceptance invariant (M=1 == M=3 argmax) is preserved. The finalize kernel
// merges only `used = ceil(context / TPS)` slices.
kernel void sdpa_decode_partial(
    device const bfloat* query [[buffer(0)]], // [B,24,256]
    device const bfloat* key_cache [[buffer(1)]],  // [4,capacity,256]
    device const bfloat* value_cache [[buffer(2)]],// [4,capacity,256]
    device float* partial_out [[buffer(3)]],  // [B,24,ATTN_SLICES*256] per-slice acc
    device float* stat_denmax [[buffer(4)]],  // [B,24,ATTN_SLICES,2] = {denom, max}
    constant uint& base_position [[buffer(5)]],
    constant uint& capacity [[buffer(6)]],
    constant uint& batch [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],       // head,row,slice
    uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y, slice = tg.z;
  if (head >= QHEADS || row >= batch) return;
  const uint position = base_position + row;
  if (position >= capacity) return;
  const uint context = position + 1u;
  const uint t0 = slice * TOKENS_PER_SLICE;
  if (t0 >= context) return;   // inactive slice: never written, never merged
  const uint t1 = min(t0 + TOKENS_PER_SLICE, context);
  const uint kvh = head / (QHEADS / KVHEADS);
  const uint qb = row * OSTRIDE + head * HEAD_DIM;

  float accum[8];
  for (uint d = 0; d < 8u; ++d) accum[d] = 0.0f;
  float max_score = -INFINITY;
  float denom = 0.0f;

  for (uint token = t0; token < t1; ++token) {
    const uint cache_base = (kvh * capacity + token) * HEAD_DIM;
    float partial = 0.0f;
    for (uint d = 0; d < 8u; ++d) {
      const uint dim = lane * 8u + d;
      partial += float(query[qb + dim]) * float(key_cache[cache_base + dim]);
    }
    const float score = simd_sum(partial) * 0.0625f;
    const float next_max = max(max_score, score);
    const float old_scale = exp2(1.4426950408889634f * (max_score - next_max));
    const float new_scale = exp2(1.4426950408889634f * (score - next_max));
    denom = denom * old_scale + new_scale;
    for (uint d = 0; d < 8u; ++d) {
      const uint dim = lane * 8u + d;
      accum[d] = accum[d] * old_scale + new_scale * float(value_cache[cache_base + dim]);
    }
    max_score = next_max;
  }

  const uint sbase = ((row * QHEADS + head) * ATTN_SLICES + slice) * 2u;
  if (lane == 0u) {
    stat_denmax[sbase] = denom;
    stat_denmax[sbase + 1u] = max_score;
  }
  const uint pbase = (row * QHEADS + head) * ATTN_SLICES + slice;
  for (uint d = 0; d < 8u; ++d)
    partial_out[pbase * HEAD_DIM + lane * 8u + d] = accum[d];
}

// Merge the per-slice partial accumulators + stats for each head (flash style):
// find the global max across this row's USED slices, then rescale+sum each
// slice's O and denominator by exp2(slice_max - global_max), and normalize.
// Only `used = ceil(context/TPS)` slices are merged (in ascending order), so
// the merge is a pure function of the row's context and M=1 == M=3 per row.
kernel void sdpa_decode_final(
    device const float* partial_out [[buffer(0)]], // [B,24,ATTN_SLICES,256]
    device const float* stat_denmax [[buffer(1)]], // [B,24,ATTN_SLICES,2]
    device float* out [[buffer(2)]],               // [B,24,256]
    constant uint& base_position [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],       // head,row,unused
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]]) {
  const uint head = tg.x, row = tg.y;
  if (head >= QHEADS) return;
  const uint context = base_position + row + 1u;
  const uint used = (context + TOKENS_PER_SLICE - 1u) / TOKENS_PER_SLICE;
  const uint ob = row * OSTRIDE + head * HEAD_DIM;
  const uint stat = (row * QHEADS + head) * ATTN_SLICES;
  float gmax = -INFINITY;
  for (uint s = 0; s < used; ++s) gmax = max(gmax, stat_denmax[(stat + s) * 2u + 1u]);
  float total_den = 0.0f;
  float res = 0.0f;
  for (uint d = lid.x; d < HEAD_DIM; d += lsize.x) {
    total_den = 0.0f;
    res = 0.0f;
    for (uint s = 0; s < used; ++s) {
      const float m = stat_denmax[(stat + s) * 2u + 1u];
      const float dn = stat_denmax[(stat + s) * 2u];
      const float w = exp2(1.4426950408889634f * (m - gmax));
      total_den += w * dn;
      res += w * partial_out[(stat + s) * 256u + d];
    }
    out[ob + d] = res / total_den;
  }
}
