#include <metal_stdlib>
using namespace metal;

// Model-level RMSNorm over [M, H] with a per-element gain, and residual add
// f32-onto-bf16. All values stored bf16 (model dtype); arithmetic in f32.

// y[row] = RMSNorm(x[row], w, eps)  (bf16 in/out)
// Multi-simdgroup: `lsize` threads (256 in the runner) cooperate on one row,
// with a two-level simd_sum + shared-memory reduction (MLX `rms_looped`).
kernel void rms_norm_rows(device const bfloat* x [[buffer(0)]], // [M,H] bf16
                          device const bfloat* w [[buffer(1)]], // [H] bf16
                          device bfloat* y       [[buffer(2)]], // [M,H] bf16
                          constant uint& cols    [[buffer(3)]],
                          constant float& eps    [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint lid [[thread_position_in_threadgroup]],
                          uint lsize [[threads_per_threadgroup]],
                          uint simd_lane [[thread_index_in_simdgroup]],
                          uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float local_sums[32];
  threadgroup float local_inv[1];

  float s = 0.0f;
  for (uint j = lid; j < cols; j += lsize) {
    float v = float(x[row * cols + j]);
    s += v * v;
  }
  s = simd_sum(s);
  if (simd_group == 0u) {
    local_sums[simd_lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0u) {
    local_sums[simd_group] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0u) {
    s = simd_sum(local_sums[simd_lane]);
    if (simd_lane == 0u) {
      local_inv[0] = rsqrt(s / float(cols) + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = local_inv[0];
  for (uint j = lid; j < cols; j += lsize)
    y[row * cols + j] = bfloat(float(x[row * cols + j]) * inv * float(w[j]));
}

// y = x_bf16 + o_f32  (residual add), stored bf16. Flat n = M*H.
kernel void residual_add(device const bfloat* x [[buffer(0)]],
                         device const float* o  [[buffer(1)]],
                         device bfloat* y       [[buffer(2)]],
                         constant uint& n       [[buffer(3)]],
                         uint tid [[thread_position_in_grid]]) {
  if (tid < n)
    y[tid] = bfloat(float(x[tid]) + float(bfloat(o[tid])));
}

// Fuses the model's exact BF16 residual boundary with the following RMSNorm.
// The normalized output is computed from the stored BF16 residual, matching
// residual_add followed by rms_norm_rows bit for bit. Multi-simdgroup (256
// threads) so the residual add is as parallel as the standalone elementwise
// kernel; `lsize` threads cooperate per row via a two-level reduction.
kernel void residual_add_rms_norm_rows(
    device const bfloat* x [[buffer(0)]],       // [M,H] bf16
    device const float* o [[buffer(1)]],        // [M,H] f32
    device const bfloat* w [[buffer(2)]],       // [H] bf16
    device bfloat* residual [[buffer(3)]],      // [M,H] bf16
    device bfloat* normalized [[buffer(4)]],    // [M,H] bf16
    constant uint& cols [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint base = row * cols;
  threadgroup float local_sums[32];
  threadgroup float local_inv[1];

  float s = 0.0f;
  for (uint j = lid; j < cols; j += lsize) {
    const bfloat r = bfloat(float(x[base + j]) + float(bfloat(o[base + j])));
    residual[base + j] = r;
    const float v = float(r);
    s += v * v;
  }
  s = simd_sum(s);
  if (simd_group == 0u) {
    local_sums[simd_lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0u) {
    local_sums[simd_group] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0u) {
    s = simd_sum(local_sums[simd_lane]);
    if (simd_lane == 0u) {
      local_inv[0] = rsqrt(s / float(cols) + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = local_inv[0];
  for (uint j = lid; j < cols; j += lsize)
    normalized[base + j] = bfloat(float(residual[base + j]) * inv * float(w[j]));
}

kernel void clear_bytes(device uchar* output [[buffer(0)]],
                        constant ulong& n [[buffer(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
  const ulong index = ulong(gid.x) + ulong(gid.y) * (1ul << 30);
  if (index < n) output[index] = 0;
}

// Device-to-device state snapshots without staging the recurrent state on CPU.
// Each thread copies 16 bytes; the scalar tail supports arbitrary byte counts.
kernel void copy_bytes(device const uchar* input [[buffer(0)]],
                       device uchar* output [[buffer(1)]],
                       constant ulong& n [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
  const ulong chunk = ulong(gid.x) + ulong(gid.y) * (1ul << 30);
  const ulong offset = chunk * 16ul;
  if (offset + 16ul <= n) {
    *reinterpret_cast<device uint4*>(output + offset) =
        *reinterpret_cast<device const uint4*>(input + offset);
  } else {
    for (ulong i = offset; i < n; ++i) output[i] = input[i];
  }
}

kernel void f32_to_bf16(device const float* input [[buffer(0)]],
                        device bfloat* output [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {
  if (tid < n) output[tid] = bfloat(input[tid]);
}

// RMSNorm variant that writes into a strided 2H row (the MTP pre-fc input),
// column offset `col_base` is the element offset within the 2H row (0 or H).
// Keeps the exact rms_norm_rows numerics so the MTP prefill matches M=1.
kernel void rms_norm_rows_stride(device const bfloat* x [[buffer(0)]],
                                 device const bfloat* w [[buffer(1)]],
                                 device bfloat* y       [[buffer(2)]],
                                 constant uint& cols    [[buffer(3)]],
                                 constant float& eps    [[buffer(4)]],
                                 constant uint2& layout [[buffer(5)]], // stride, col_base
                                 uint row [[threadgroup_position_in_grid]],
                                 uint lid [[thread_position_in_threadgroup]],
                                 uint lsize [[threads_per_threadgroup]],
                                 uint simd_lane [[thread_index_in_simdgroup]],
                                 uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float local_sums[32];
  threadgroup float local_inv[1];

  float s = 0.0f;
  for (uint j = lid; j < cols; j += lsize) {
    float v = float(x[row * cols + j]);
    s += v * v;
  }
  s = simd_sum(s);
  if (simd_group == 0u) {
    local_sums[simd_lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0u) {
    local_sums[simd_group] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0u) {
    s = simd_sum(local_sums[simd_lane]);
    if (simd_lane == 0u) {
      local_inv[0] = rsqrt(s / float(cols) + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = local_inv[0];
  const uint out_base = row * layout.x + layout.y;
  for (uint j = lid; j < cols; j += lsize)
    y[out_base + j] = bfloat(float(x[row * cols + j]) * inv * float(w[j]));
}
