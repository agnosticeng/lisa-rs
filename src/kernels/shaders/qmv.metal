#include <metal_stdlib>
using namespace metal;

// Decode M=1 / M=3 affine-U4 matrix-vector kernels (QMV):
//   output[N] = x[K]^bf16 · deq(W[N,K/8]^u4, gs64)^T[:, :]
// Shared arithmetic for the single and multi-projection M=1 entry points.
// Keep this as the sole implementation so fused dispatches remain bit-identical
// to q4_qmv_fast.
static inline void q4_qmv_fast_impl(
    device const bfloat* x,
    device const uint* packed,
    device const bfloat* scales,
    device const bfloat* biases,
    device float* output,
    uint N,
    uint K,
    uint lane,
    uint simdgroup,
    uint group) {
  const uint packed_cols = K / 8u;
  const uint groups = K / 64u;
  const uint row_base = group * 16u + simdgroup * 4u;
  float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  const uint aligned_end = (K / 512u) * 512u;
  for (uint block = 0u; block < aligned_end; block += 512u) {
    const uint lane_off = block + lane * 16u;
    float xt[16];
    float sum16 = 0.0f;
    for (uint i = 0; i < 16u; i += 2u) {
      const float xe = float(x[lane_off + i]);
      const float xo = float(x[lane_off + i + 1u]);
      xt[i] = xe;
      xt[i + 1u] = xo / 16.0f;
      sum16 += xe + xo;
    }
    const uint g = block / 64u + lane / 4u;
    for (uint result = 0; result < 4u; ++result) {
      const uint row = row_base + result;
      if (row < N) {
        const uint w0 = packed[row * packed_cols + block / 8u + lane * 2u];
        const uint w1 = packed[row * packed_cols + block / 8u + lane * 2u + 1u];
        const float scale = float(scales[row * groups + g]);
        const float bias = float(biases[row * groups + g]);
        float acc16 = 0.0f;
        for (uint b = 0; b < 4u; ++b) {
          const uint byte0 = (w0 >> (8u * b)) & 0xffu;
          const uint byte1 = (w1 >> (8u * b)) & 0xffu;
          acc16 += xt[2u * b] * float(byte0 & 0x0fu) + xt[2u * b + 1u] * float(byte0 & 0xf0u);
          acc16 += xt[8u + 2u * b] * float(byte1 & 0x0fu) + xt[8u + 2u * b + 1u] * float(byte1 & 0xf0u);
        }
        accum[result] += scale * acc16 + bias * sum16;
      }
    }
  }
  for (uint k = aligned_end + lane; k < K; k += 32u) {
    const float xv = float(x[k]);
    const uint word_col = k >> 3u;
    const uint shift = (k & 7u) * 4u;
    const uint scale_col = k >> 6u;
    for (uint result = 0; result < 4u; ++result) {
      const uint row = row_base + result;
      if (row < N) {
        const uint q = (packed[row * packed_cols + word_col] >> shift) & 0x0fu;
        const float scale = float(scales[row * groups + scale_col]);
        const float bias = float(biases[row * groups + scale_col]);
        accum[result] += xv * (float(q) * scale + bias);
      }
    }
  }
  for (uint result = 0; result < 4u; ++result) {
    const float value = simd_sum(accum[result]);
    const uint row = row_base + result;
    if (lane == 0u && row < N) output[row] = value;
  }
}

// Bandwidth-oriented affine-U4 matrix-vector product for decode M=1.
// Four SIMD groups compute 16 output rows per threadgroup with explicit N
// bounds. Inputs and dequantized weights are BF16; accumulation/output are F32.
kernel void q4_qmv_fast(
    device const bfloat* x [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint2& nk [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  q4_qmv_fast_impl(x, packed, scales, biases, output, nk.x, nk.y,
                   lane, simdgroup, group);
}

// Multiple independent projections of the same M=1 input. Grid y selects a
// weight triple and output; grid x spans the widest projection.
kernel void q4_qmv_fused2(
    device const bfloat* x [[buffer(0)]],
    device const uint* p0 [[buffer(1)]], device const bfloat* s0 [[buffer(2)]],
    device const bfloat* b0 [[buffer(3)]], device float* o0 [[buffer(4)]],
    device const uint* p1 [[buffer(5)]], device const bfloat* s1 [[buffer(6)]],
    device const bfloat* b1 [[buffer(7)]], device float* o1 [[buffer(8)]],
    constant uint* dims [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  if (group.y == 0u)
    q4_qmv_fast_impl(x, p0, s0, b0, o0, dims[0], dims[2], lane, simdgroup, group.x);
  else
    q4_qmv_fast_impl(x, p1, s1, b1, o1, dims[1], dims[2], lane, simdgroup, group.x);
}

kernel void q4_qmv_fused3(
    device const bfloat* x [[buffer(0)]],
    device const uint* p0 [[buffer(1)]], device const bfloat* s0 [[buffer(2)]],
    device const bfloat* b0 [[buffer(3)]], device float* o0 [[buffer(4)]],
    device const uint* p1 [[buffer(5)]], device const bfloat* s1 [[buffer(6)]],
    device const bfloat* b1 [[buffer(7)]], device float* o1 [[buffer(8)]],
    device const uint* p2 [[buffer(9)]], device const bfloat* s2 [[buffer(10)]],
    device const bfloat* b2 [[buffer(11)]], device float* o2 [[buffer(12)]],
    constant uint* dims [[buffer(13)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  if (group.y == 0u)
    q4_qmv_fast_impl(x, p0, s0, b0, o0, dims[0], dims[3], lane, simdgroup, group.x);
  else if (group.y == 1u)
    q4_qmv_fast_impl(x, p1, s1, b1, o1, dims[1], dims[3], lane, simdgroup, group.x);
  else
    q4_qmv_fast_impl(x, p2, s2, b2, o2, dims[2], dims[3], lane, simdgroup, group.x);
}

kernel void q4_qmv_fused4(
    device const bfloat* x [[buffer(0)]],
    device const uint* p0 [[buffer(1)]], device const bfloat* s0 [[buffer(2)]],
    device const bfloat* b0 [[buffer(3)]], device float* o0 [[buffer(4)]],
    device const uint* p1 [[buffer(5)]], device const bfloat* s1 [[buffer(6)]],
    device const bfloat* b1 [[buffer(7)]], device float* o1 [[buffer(8)]],
    device const uint* p2 [[buffer(9)]], device const bfloat* s2 [[buffer(10)]],
    device const bfloat* b2 [[buffer(11)]], device float* o2 [[buffer(12)]],
    device const uint* p3 [[buffer(13)]], device const bfloat* s3 [[buffer(14)]],
    device const bfloat* b3 [[buffer(15)]], device float* o3 [[buffer(16)]],
    constant uint* dims [[buffer(17)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  if (group.y == 0u)
    q4_qmv_fast_impl(x, p0, s0, b0, o0, dims[0], dims[4], lane, simdgroup, group.x);
  else if (group.y == 1u)
    q4_qmv_fast_impl(x, p1, s1, b1, o1, dims[1], dims[4], lane, simdgroup, group.x);
  else if (group.y == 2u)
    q4_qmv_fast_impl(x, p2, s2, b2, o2, dims[2], dims[4], lane, simdgroup, group.x);
  else
    q4_qmv_fast_impl(x, p3, s3, b3, o3, dims[3], dims[4], lane, simdgroup, group.x);
}

// Shared arithmetic for M=3. Fused entry points only select independent
// weight/output buffers with grid y; weights are never concatenated.
static inline void q4_qmv_wide3_impl(
    device const bfloat* x,
    device const uint* packed,
    device const bfloat* scales,
    device const bfloat* biases,
    device float* output,
    uint N,
    uint K,
    uint lane,
    uint simdgroup,
    uint group) {
  const uint packed_cols = K / 8u;
  const uint groups = K / 64u;
  const uint row_base = group * 16u + simdgroup * 4u;
  float accum[3][4] = {{0.0f, 0.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 0.0f, 0.0f}};

  const uint aligned_end = (K / 512u) * 512u;
  for (uint block = 0u; block < aligned_end; block += 512u) {
    const uint lane_off = block + lane * 16u;
    const uint g = block / 64u + lane / 4u;
    for (uint result = 0; result < 4u; ++result) {
      const uint row = row_base + result;
      if (row < N) {
        const uint w0 = packed[row * packed_cols + block / 8u + lane * 2u];
        const uint w1 = packed[row * packed_cols + block / 8u + lane * 2u + 1u];
        const float scale = float(scales[row * groups + g]);
        const float bias = float(biases[row * groups + g]);
        for (uint ir = 0; ir < 3u; ++ir) {
          const device bfloat* xr = x + ir * K;
          float xt[16];
          float sum16 = 0.0f;
          for (uint i = 0; i < 16u; i += 2u) {
            const float xe = float(xr[lane_off + i]);
            const float xo = float(xr[lane_off + i + 1u]);
            xt[i] = xe;
            xt[i + 1u] = xo / 16.0f;
            sum16 += xe + xo;
          }
          float acc16 = 0.0f;
          for (uint b = 0; b < 4u; ++b) {
            const uint byte0 = (w0 >> (8u * b)) & 0xffu;
            const uint byte1 = (w1 >> (8u * b)) & 0xffu;
            acc16 += xt[2u * b] * float(byte0 & 0x0fu) + xt[2u * b + 1u] * float(byte0 & 0xf0u);
            acc16 += xt[8u + 2u * b] * float(byte1 & 0x0fu) + xt[8u + 2u * b + 1u] * float(byte1 & 0xf0u);
          }
          accum[ir][result] += scale * acc16 + bias * sum16;
        }
      }
    }
  }
  for (uint k = aligned_end + lane; k < K; k += 32u) {
    const float x0 = float(x[k]), x1 = float(x[K + k]), x2 = float(x[2u * K + k]);
    const uint word_col = k >> 3u, shift = (k & 7u) * 4u, scale_col = k >> 6u;
    for (uint result = 0; result < 4u; ++result) {
      const uint row = row_base + result;
      if (row < N) {
        const uint q = (packed[row * packed_cols + word_col] >> shift) & 0x0fu;
        const float scale = float(scales[row * groups + scale_col]);
        const float bias = float(biases[row * groups + scale_col]);
        const float weight = float(q) * scale + bias;
        accum[0][result] += x0 * weight;
        accum[1][result] += x1 * weight;
        accum[2][result] += x2 * weight;
      }
    }
  }
  for (uint result = 0; result < 4u; ++result) {
    const uint row = row_base + result;
    for (uint input_row = 0; input_row < 3u; ++input_row) {
      const float value = simd_sum(accum[input_row][result]);
      if (lane == 0u && row < N) output[input_row * N + row] = value;
    }
  }
}

// Decode-wide affine-U4 product specialized for exactly three input rows.
// Each weight is unpacked/dequantized once and reused across all three rows.
// Four SIMD groups compute 16 output columns; all accesses are N-tail safe.
kernel void q4_qmv_wide3(
    device const bfloat* x [[buffer(0)]],       // [3,K]
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device float* output [[buffer(4)]],         // [3,N]
    constant uint2& nk [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  q4_qmv_wide3_impl(x, packed, scales, biases, output, nk.x, nk.y,
                    lane, simdgroup, group);
}

kernel void q4_qmv_fused2_wide3(
    device const bfloat* x [[buffer(0)]],
    device const uint* p0 [[buffer(1)]], device const bfloat* s0 [[buffer(2)]],
    device const bfloat* b0 [[buffer(3)]], device float* o0 [[buffer(4)]],
    device const uint* p1 [[buffer(5)]], device const bfloat* s1 [[buffer(6)]],
    device const bfloat* b1 [[buffer(7)]], device float* o1 [[buffer(8)]],
    constant uint* dims [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  if (group.y == 0u)
    q4_qmv_wide3_impl(x, p0, s0, b0, o0, dims[0], dims[2], lane, simdgroup, group.x);
  else
    q4_qmv_wide3_impl(x, p1, s1, b1, o1, dims[1], dims[2], lane, simdgroup, group.x);
}

kernel void q4_qmv_fused3_wide3(
    device const bfloat* x [[buffer(0)]],
    device const uint* p0 [[buffer(1)]], device const bfloat* s0 [[buffer(2)]],
    device const bfloat* b0 [[buffer(3)]], device float* o0 [[buffer(4)]],
    device const uint* p1 [[buffer(5)]], device const bfloat* s1 [[buffer(6)]],
    device const bfloat* b1 [[buffer(7)]], device float* o1 [[buffer(8)]],
    device const uint* p2 [[buffer(9)]], device const bfloat* s2 [[buffer(10)]],
    device const bfloat* b2 [[buffer(11)]], device float* o2 [[buffer(12)]],
    constant uint* dims [[buffer(13)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  if (group.y == 0u)
    q4_qmv_wide3_impl(x, p0, s0, b0, o0, dims[0], dims[3], lane, simdgroup, group.x);
  else if (group.y == 1u)
    q4_qmv_wide3_impl(x, p1, s1, b1, o1, dims[1], dims[3], lane, simdgroup, group.x);
  else
    q4_qmv_wide3_impl(x, p2, s2, b2, o2, dims[2], dims[3], lane, simdgroup, group.x);
}

kernel void q4_qmv_fused4_wide3(
    device const bfloat* x [[buffer(0)]],
    device const uint* p0 [[buffer(1)]], device const bfloat* s0 [[buffer(2)]],
    device const bfloat* b0 [[buffer(3)]], device float* o0 [[buffer(4)]],
    device const uint* p1 [[buffer(5)]], device const bfloat* s1 [[buffer(6)]],
    device const bfloat* b1 [[buffer(7)]], device float* o1 [[buffer(8)]],
    device const uint* p2 [[buffer(9)]], device const bfloat* s2 [[buffer(10)]],
    device const bfloat* b2 [[buffer(11)]], device float* o2 [[buffer(12)]],
    device const uint* p3 [[buffer(13)]], device const bfloat* s3 [[buffer(14)]],
    device const bfloat* b3 [[buffer(15)]], device float* o3 [[buffer(16)]],
    constant uint* dims [[buffer(17)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  if (group.y == 0u)
    q4_qmv_wide3_impl(x, p0, s0, b0, o0, dims[0], dims[4], lane, simdgroup, group.x);
  else if (group.y == 1u)
    q4_qmv_wide3_impl(x, p1, s1, b1, o1, dims[1], dims[4], lane, simdgroup, group.x);
  else if (group.y == 2u)
    q4_qmv_wide3_impl(x, p2, s2, b2, o2, dims[2], dims[4], lane, simdgroup, group.x);
  else
    q4_qmv_wide3_impl(x, p3, s3, b3, o3, dims[3], dims[4], lane, simdgroup, group.x);
}
