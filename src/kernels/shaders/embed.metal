#include <metal_stdlib>
using namespace metal;

// Indexed affine-U4 embedding lookup. The buffers may contain the full table
// or a row slice as long as token_ids index the bound rows.
kernel void affine_u4_lookup(
    device const uint* token_ids [[buffer(0)]],
    device const uint* weight    [[buffer(1)]], // [rows, hidden/8]
    device const bfloat* scales  [[buffer(2)]], // [rows, hidden/64]
    device const bfloat* biases  [[buffer(3)]], // [rows, hidden/64]
    device bfloat* output        [[buffer(4)]], // [tokens, hidden]
    constant uint& hidden        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint col = gid.x;
  const uint token_pos = gid.y;
  if (col >= hidden) return;

  const uint row = token_ids[token_pos];
  const uint packed_cols = hidden / 8u;
  const uint groups = hidden / 64u;
  const uint word = weight[row * packed_cols + (col >> 3u)];
  const uint q = (word >> ((col & 7u) * 4u)) & 0x0fu;
  const uint group = col >> 6u;
  const float value = float(q) * float(scales[row * groups + group])
                    + float(biases[row * groups + group]);
  output[token_pos * hidden + col] = bfloat(value);
}

// Hierarchical global argmax over n f32 values. One 256-thread threadgroup
// (8 simdgroups) reduces 256 values to a (max, index) pair, then writes it to
// per-threadgroup partial buffers; the host reduces the ~n/256 partials. The
// simd shuffle ladder keeps the index synced with the running max.
kernel void argmax_f32_partial(
    device const float* x [[buffer(0)]],
    device float* partial_max [[buffer(1)]],
    device uint* partial_idx [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]]) {
  threadgroup float smv[8];
  threadgroup uint smi[8];
  const uint i = tg * 256u + sg * 32u + lane;
  float v = (i < n) ? x[i] : -FLT_MAX;
  uint idx = i;
  for (uint s = 16u; s > 0u; s >>= 1u) {
    const float other = simd_shuffle_down(v, s);
    const uint oi = simd_shuffle_down(idx, s);
    if (other > v) {
      v = other;
      idx = oi;
    }
  }
  if (lane == 0u) {
    smv[sg] = v;
    smi[sg] = idx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && lane == 0u) {
    float best = smv[0];
    uint bi = smi[0];
    for (uint j = 1u; j < 8u; ++j) {
      if (smv[j] > best) {
        best = smv[j];
        bi = smi[j];
      }
    }
    partial_max[tg] = best;
    partial_idx[tg] = bi;
  }
}
