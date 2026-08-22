#include <metal_stdlib>
using namespace metal;

// Gated Delta Net (linear_attn) layer — exact port of saragossa's fused
// Qwen3.5 kernels (linear-attn conv+norm+gates / gated_delta / rms_gate).
// Geometry: 16 key heads x 128, 48 value heads x 128, conv kernel 4, gs64.
// All activations/state f32; only the leaf projections (NAX) are bf16-fed.
//
// conv over the fused qkv (2*key_dim + value_dim = 10240 channels), SiLU;
// q/k then RMS-normed and scaled: q *= 1/128, k *= 1/sqrt(128).
// Sizes (from real tensors layer 17): key_dim=2048, value_dim=6144,
// key_heads=16, value_heads=48, head dim 128 both.

#define KHD 128

static inline float gdn_silu(float x) { return x / (1.0f + exp(-x)); }
static inline float gdn_softplus(float x) {
  return (x > 20.0f) ? x : log(1.0f + exp(x));
}

// Causal K=4 conv over a single channel of the fused qkv, with SiLU. Fresh
// state (leading pads) read from conv_state0[3, conv_dim].
static inline float gdn_conv_batch_channel(
    device const float* qkv, device const float* conv_weight,
    device const bfloat* conv_state0, uint conv_dim, uint channel,
    uint token) {
  const uint w = channel * 4u;
  float acc = 0.0f;
  for (int k = 0; k < 4; k++) {
    int p = int(token) - 3 + k;
    float x;
    if (p >= 0) {
      x = float(bfloat(qkv[uint(p) * conv_dim + channel]));
    } else {
      x = conv_state0[uint(p + 3) * conv_dim + channel];
    }
    acc += x * conv_weight[w + uint(k)];
  }
  float conv = float(bfloat(acc));
  return float(bfloat(gdn_silu(conv)));
}

// Fused: conv(qkv) + SiLU, RMS-norm+scale q/k, beta=sigmoid(b), decay.
// Grid: (groups = max(key,value) heads, tokens); one 32-lane threadgroup
// computes a full 128-dim head.
kernel void gdn_conv_norm_gates(
    device const float* qkv         [[buffer(0)]], // [M, conv_dim]
    device const float* beta_input  [[buffer(1)]], // [M, value_heads]
    device const float* gate_input  [[buffer(2)]], // [M, value_heads]
    device const float* conv_weight [[buffer(3)]], // [conv_dim, 4]
    device const bfloat* conv_state0 [[buffer(4)]], // [3, conv_dim]
    device const float* a_log       [[buffer(5)]], // [value_heads]  A_log
    device const float* dt_bias     [[buffer(6)]], // [value_heads]
    device float* conv_out          [[buffer(7)]], // [M, conv_dim] (v region used)
    device float* q_norm            [[buffer(8)]], // [M, key_dim]
    device float* k_norm            [[buffer(9)]], // [M, key_dim]
    device float* beta              [[buffer(10)]],// [M, value_heads]
    device float* decay             [[buffer(11)]],// [M, value_heads]
    constant uint4& dims            [[buffer(12)]],
    constant float2& scales         [[buffer(13)]],
    constant uint& batch            [[buffer(14)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg  [[threadgroup_position_in_grid]]) {
  const uint key_heads   = dims.x;
  const uint value_heads = dims.y;
  const uint key_head_dim = dims.z;
  const uint value_head_dim = dims.w;
  if (key_head_dim != 128u || value_head_dim != 128u) return;
  const uint item = tg.x;
  const uint token = tg.y;
  if (token >= batch) return;
  const uint key_dim = key_heads * KHD;
  const uint conv_dim = (2u * key_dim) + (value_heads * KHD);
  const uint qrow = token * key_dim;
  const uint crow = token * conv_dim;
  const uint grow = token * value_heads;
  const uint proj_grow = token * 64u; // legacy NAX test pads 48 outputs to 64

  if (item < key_heads) {
    const uint base = item * KHD;
    const uint l0 = base + lane, l1 = l0 + 32u, l2 = l0 + 64u, l3 = l0 + 96u;
    const uint kc0 = key_dim + l0, kc1 = key_dim + l1, kc2 = key_dim + l2, kc3 = key_dim + l3;
    float q0 = gdn_conv_batch_channel(qkv, conv_weight, conv_state0, conv_dim, l0, token);
    float q1 = gdn_conv_batch_channel(qkv, conv_weight, conv_state0, conv_dim, l1, token);
    float q2 = gdn_conv_batch_channel(qkv, conv_weight, conv_state0, conv_dim, l2, token);
    float q3 = gdn_conv_batch_channel(qkv, conv_weight, conv_state0, conv_dim, l3, token);
    float k0 = gdn_conv_batch_channel(qkv, conv_weight, conv_state0, conv_dim, kc0, token);
    float k1 = gdn_conv_batch_channel(qkv, conv_weight, conv_state0, conv_dim, kc1, token);
    float k2 = gdn_conv_batch_channel(qkv, conv_weight, conv_state0, conv_dim, kc2, token);
    float k3 = gdn_conv_batch_channel(qkv, conv_weight, conv_state0, conv_dim, kc3, token);
    float q_ss = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
    float k_ss = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
    float q_inv = rsqrt(simd_sum(q_ss) / float(key_head_dim) + 1.0e-6f) * scales.x;
    float k_inv = rsqrt(simd_sum(k_ss) / float(key_head_dim) + 1.0e-6f) * scales.y;
    q_norm[qrow + l0] = float(bfloat(q0 * q_inv));
    q_norm[qrow + l1] = float(bfloat(q1 * q_inv));
    q_norm[qrow + l2] = float(bfloat(q2 * q_inv));
    q_norm[qrow + l3] = float(bfloat(q3 * q_inv));
    k_norm[qrow + l0] = float(bfloat(k0 * k_inv));
    k_norm[qrow + l1] = float(bfloat(k1 * k_inv));
    k_norm[qrow + l2] = float(bfloat(k2 * k_inv));
    k_norm[qrow + l3] = float(bfloat(k3 * k_inv));
  }

  if (item < value_heads) {
    const uint vbase = (2u * key_dim) + (item * KHD);
    const uint l0 = vbase + lane, l1 = l0 + 32u, l2 = l0 + 64u, l3 = l0 + 96u;
    conv_out[crow + l0] = gdn_conv_batch_channel(qkv, conv_weight, conv_state0, conv_dim, l0, token);
    conv_out[crow + l1] = gdn_conv_batch_channel(qkv, conv_weight, conv_state0, conv_dim, l1, token);
    conv_out[crow + l2] = gdn_conv_batch_channel(qkv, conv_weight, conv_state0, conv_dim, l2, token);
    conv_out[crow + l3] = gdn_conv_batch_channel(qkv, conv_weight, conv_state0, conv_dim, l3, token);
    if (lane == 0u) {
      float beta_in = float(bfloat(beta_input[proj_grow + item]));
      beta[grow + item] = float(bfloat(1.0f / (1.0f + exp(-beta_in))));
      float gate_in = float(bfloat(gate_input[proj_grow + item]));
      float dt_arg = float(bfloat(gate_in + dt_bias[item]));
      float dt = float(bfloat(gdn_softplus(dt_arg)));
      decay[grow + item] = exp(-exp(a_log[item]) * dt);
    }
  }
}

// Native runner variant: immutable checkpoint auxiliaries stay BF16 in their
// resident shard buffer rather than being expanded into per-layer F32 copies.
kernel void gdn_conv_norm_gates_bf16_weights(
    device const float* qkv         [[buffer(0)]],
    device const float* beta_input  [[buffer(1)]],
    device const float* gate_input  [[buffer(2)]],
    device const bfloat* conv_weight [[buffer(3)]],
    device const bfloat* conv_state0 [[buffer(4)]],
    device const bfloat* a_log       [[buffer(5)]],
    device const bfloat* dt_bias     [[buffer(6)]],
    device float* conv_out          [[buffer(7)]],
    device float* q_norm            [[buffer(8)]],
    device float* k_norm            [[buffer(9)]],
    device float* beta              [[buffer(10)]],
    device float* decay             [[buffer(11)]],
    constant uint4& dims            [[buffer(12)]],
    constant float2& scales         [[buffer(13)]],
    constant uint& batch            [[buffer(14)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg  [[threadgroup_position_in_grid]]) {
  const uint key_heads = dims.x, value_heads = dims.y;
  const uint key_head_dim = dims.z, value_head_dim = dims.w;
  if (key_head_dim != 128u || value_head_dim != 128u) return;
  const uint item = tg.x, token = tg.y;
  if (token >= batch) return;
  const uint key_dim = key_heads * KHD;
  const uint conv_dim = 2u * key_dim + value_heads * KHD;
  const uint qrow = token * key_dim, crow = token * conv_dim;
  const uint grow = token * value_heads, proj_grow = token * value_heads;

  if (item < key_heads) {
    const uint base = item * KHD;
    const uint l0 = base + lane, l1 = l0 + 32u, l2 = l0 + 64u, l3 = l0 + 96u;
    const uint kc0 = key_dim + l0, kc1 = key_dim + l1, kc2 = key_dim + l2, kc3 = key_dim + l3;
    float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
    float k0 = 0.0f, k1 = 0.0f, k2 = 0.0f, k3 = 0.0f;
    // Inline the four-tap convolution because the legacy helper accepts F32 weights.
    const uint channels[8] = {l0, l1, l2, l3, kc0, kc1, kc2, kc3};
    thread float values[8];
    for (uint c = 0; c < 8u; ++c) {
      float acc = 0.0f;
      for (int tap = 0; tap < 4; ++tap) {
        int p = int(token) - 3 + tap;
        float x = p >= 0 ? float(bfloat(qkv[uint(p) * conv_dim + channels[c]]))
                         : float(conv_state0[uint(p + 3) * conv_dim + channels[c]]);
        acc += x * float(conv_weight[channels[c] * 4u + uint(tap)]);
      }
      values[c] = float(bfloat(gdn_silu(float(bfloat(acc)))));
    }
    q0 = values[0]; q1 = values[1]; q2 = values[2]; q3 = values[3];
    k0 = values[4]; k1 = values[5]; k2 = values[6]; k3 = values[7];
    float q_ss = q0*q0 + q1*q1 + q2*q2 + q3*q3;
    float k_ss = k0*k0 + k1*k1 + k2*k2 + k3*k3;
    float q_inv = rsqrt(simd_sum(q_ss) / float(key_head_dim) + 1.0e-6f) * scales.x;
    float k_inv = rsqrt(simd_sum(k_ss) / float(key_head_dim) + 1.0e-6f) * scales.y;
    q_norm[qrow+l0]=float(bfloat(q0*q_inv)); q_norm[qrow+l1]=float(bfloat(q1*q_inv));
    q_norm[qrow+l2]=float(bfloat(q2*q_inv)); q_norm[qrow+l3]=float(bfloat(q3*q_inv));
    k_norm[qrow+l0]=float(bfloat(k0*k_inv)); k_norm[qrow+l1]=float(bfloat(k1*k_inv));
    k_norm[qrow+l2]=float(bfloat(k2*k_inv)); k_norm[qrow+l3]=float(bfloat(k3*k_inv));
  }

  if (item < value_heads) {
    const uint vbase = 2u * key_dim + item * KHD;
    const uint channels[4] = {vbase + lane, vbase + lane + 32u,
                              vbase + lane + 64u, vbase + lane + 96u};
    for (uint c = 0; c < 4u; ++c) {
      float acc = 0.0f;
      for (int tap = 0; tap < 4; ++tap) {
        int p = int(token) - 3 + tap;
        float x = p >= 0 ? float(bfloat(qkv[uint(p) * conv_dim + channels[c]]))
                         : float(conv_state0[uint(p + 3) * conv_dim + channels[c]]);
        acc += x * float(conv_weight[channels[c] * 4u + uint(tap)]);
      }
      conv_out[crow + channels[c]] = float(bfloat(gdn_silu(float(bfloat(acc)))));
    }
    if (lane == 0u) {
      float beta_in = float(bfloat(beta_input[proj_grow + item]));
      beta[grow + item] = float(bfloat(1.0f / (1.0f + exp(-beta_in))));
      float gate_in = float(bfloat(gate_input[proj_grow + item]));
      float dt_arg = float(bfloat(gate_in + float(dt_bias[item])));
      float dt = float(bfloat(gdn_softplus(dt_arg)));
      decay[grow + item] = exp(-exp(float(a_log[item])) * dt);
    }
  }
}

// Advance the K=4 convolution history after processing `steps` QKV rows.
// One thread owns all three state slots for a channel, avoiding in-place races
// for the decode cases steps=1 and steps=2.
kernel void gdn_update_conv_state(
    device const float* qkv [[buffer(0)]],       // [steps, conv_dim]
    device bfloat* conv_state [[buffer(1)]],     // [3, conv_dim]
    constant uint& conv_dim [[buffer(2)]],
    constant uint& steps [[buffer(3)]],
    device bfloat* state_slots [[buffer(4)]],
    constant uint& slot_stride [[buffer(5)]],
    uint channel [[thread_position_in_grid]]) {
  if (channel >= conv_dim || steps == 0u) return;

  const bfloat old0 = conv_state[channel];
  const bfloat old1 = conv_state[conv_dim + channel];
  const bfloat old2 = conv_state[2u * conv_dim + channel];
  bfloat next0 = old0, next1 = old1, next2 = old2;
  for (uint token = 0; token < steps; ++token) {
    next0 = next1;
    next1 = next2;
    next2 = bfloat(qkv[token * conv_dim + channel]);
    if (slot_stride > 0u) {
      const uint slot = token * slot_stride;
      state_slots[slot + channel] = next0;
      state_slots[slot + conv_dim + channel] = next1;
      state_slots[slot + 2u * conv_dim + channel] = next2;
    }
  }
  conv_state[channel] = next0;
  conv_state[conv_dim + channel] = next1;
  conv_state[2u * conv_dim + channel] = next2;
}

// Gated-delta recurrence over `steps` positional tokens (one dispatch).
// state layout [value_heads, 128, 128] indexed value_index*128+d shared as
// [value_index][d_k]; GQA k_head = value_head / repeat.
kernel void gdn_recurrence(
    device const float* conv_out [[buffer(0)]], // [M, conv_dim]
    device const float* q_norm   [[buffer(1)]], // [M, key_dim]
    device const float* k_norm   [[buffer(2)]], // [M, key_dim]
    device const float* beta     [[buffer(3)]], // [M, value_heads]
    device const float* decay    [[buffer(4)]], // [M, value_heads]
    device float* ssm_state      [[buffer(5)]], // [value_heads][128][128]
    device float* y              [[buffer(6)]], // [M, value_dim]
    constant uint4& dims         [[buffer(7)]],
    constant uint& steps         [[buffer(8)]],
    device float* state_slots    [[buffer(9)]],
    constant uint& slot_stride   [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
  const uint value_heads   = dims.x;
  const uint value_head_dim = dims.y;
  const uint key_head_dim  = dims.z;
  const uint repeat        = dims.w;
  const uint value_col = gid.y;   // 0..value_head_dim-1
  const uint value_head = gid.z;  // 0..value_heads-1
  const uint lane = tid.x;        // 0..31 (dk lanes)
  if (value_head >= value_heads || value_col >= value_head_dim || repeat == 0u || lane >= 32u || key_head_dim != 128u) return;
  const uint key_heads = value_heads / repeat;
  const uint key_dim = key_heads * 128u;
  const uint value_dim = value_heads * value_head_dim;
  const uint conv_dim = (2u * key_dim) + value_dim;
  const uint key_head = value_head / repeat;
  const uint key_base = key_head * 128u;
  const uint value_index = value_head * value_head_dim + value_col;
  const uint state_base = value_index * 128u;
  const uint idx0 = state_base + lane;
  const uint idx1 = idx0 + 32u;
  const uint idx2 = idx0 + 64u;
  const uint idx3 = idx0 + 96u;
  const uint key0 = key_base + lane;
  const uint key1 = key0 + 32u;
  const uint key2 = key0 + 64u;
  const uint key3 = key0 + 96u;

  float s0 = ssm_state[idx0];
  float s1 = ssm_state[idx1];
  float s2 = ssm_state[idx2];
  float s3 = ssm_state[idx3];

  for (uint t = 0u; t < steps; ++t) {
    const uint key_offset = t * key_dim;
    const uint value_offset = t * value_dim;
    const uint conv_offset = t * conv_dim;
    const uint gate_offset = t * value_heads;
    float d = decay[gate_offset + value_head];
    float bv = beta[gate_offset + value_head];
    s0 *= d; s1 *= d; s2 *= d; s3 *= d;
    const float kv0 = k_norm[key_offset + key0];
    const float kv1 = k_norm[key_offset + key1];
    const float kv2 = k_norm[key_offset + key2];
    const float kv3 = k_norm[key_offset + key3];
    float kv_part = s0*kv0 + s1*kv1 + s2*kv2 + s3*kv3;
    const float kv_mem = simd_sum(kv_part);
    const float v = conv_out[conv_offset + (2u * key_dim) + value_index];
    const float delta = (v - kv_mem) * bv;
    s0 += delta * kv0; s1 += delta * kv1; s2 += delta * kv2; s3 += delta * kv3;
    const float q0 = q_norm[key_offset + key0];
    const float q1 = q_norm[key_offset + key1];
    const float q2 = q_norm[key_offset + key2];
    const float q3 = q_norm[key_offset + key3];
    float y_part = s0*q0 + s1*q1 + s2*q2 + s3*q3;
    const float outv = simd_sum(y_part);
    if (lane == 0u) {
      y[value_offset + value_index] = float(bfloat(outv));
    }
    if (slot_stride > 0u) {
      const uint slot_base = t * slot_stride + state_base;
      state_slots[slot_base + lane] = s0;
      state_slots[slot_base + lane + 32u] = s1;
      state_slots[slot_base + lane + 64u] = s2;
      state_slots[slot_base + lane + 96u] = s3;
    }
  }

  ssm_state[idx0] = s0;
  ssm_state[idx1] = s1;
  ssm_state[idx2] = s2;
  ssm_state[idx3] = s3;
}

// out = RMSNorm(norm_weight)(y) * silu(z), per value head of 128 dims.
kernel void gdn_rms_gate(
    device const float* y           [[buffer(0)]], // [M, value_dim]
    device const float* z           [[buffer(1)]], // [M, value_dim]
    device const float* norm_weight [[buffer(2)]], // [128]
    device bfloat* gated            [[buffer(3)]], // [M, value_dim]
    constant uint3& dims            [[buffer(4)]],
    constant float& eps             [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg  [[threadgroup_position_in_grid]]) {
  const uint value_heads = dims.x;
  const uint value_head_dim = dims.y;
  const uint batch = dims.z;
  const uint value_head = tg.x;
  const uint token = tg.y;
  if (value_head >= value_heads || token >= batch || value_head_dim != 128u) return;
  const uint base = token * value_heads * 128u + value_head * 128u;
  const uint idx0 = base + lane;
  const uint idx1 = idx0 + 32u;
  const uint idx2 = idx0 + 64u;
  const uint idx3 = idx0 + 96u;
  float y0 = y[idx0], y1 = y[idx1], y2 = y[idx2], y3 = y[idx3];
  float ss = y0*y0 + y1*y1 + y2*y2 + y3*y3;
  float mean = simd_sum(ss) / float(value_head_dim);
  float inv = rsqrt(mean + eps);
  float z0 = float(bfloat(z[idx0])), z1 = float(bfloat(z[idx1]));
  float z2 = float(bfloat(z[idx2])), z3 = float(bfloat(z[idx3]));
  float n0 = float(bfloat(y0 * inv * norm_weight[lane]));
  float n1 = float(bfloat(y1 * inv * norm_weight[lane + 32u]));
  float n2 = float(bfloat(y2 * inv * norm_weight[lane + 64u]));
  float n3 = float(bfloat(y3 * inv * norm_weight[lane + 96u]));
  gated[idx0] = bfloat(n0 * gdn_silu(z0));
  gated[idx1] = bfloat(n1 * gdn_silu(z1));
  gated[idx2] = bfloat(n2 * gdn_silu(z2));
  gated[idx3] = bfloat(n3 * gdn_silu(z3));
}

kernel void gdn_rms_gate_bf16_weight(
    device const float* y [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const bfloat* norm_weight [[buffer(2)]],
    device bfloat* gated [[buffer(3)]],
    constant uint3& dims [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  const uint value_heads = dims.x, value_head_dim = dims.y, batch = dims.z;
  const uint value_head = tg.x, token = tg.y;
  if (value_head >= value_heads || token >= batch || value_head_dim != 128u) return;
  const uint base = token * value_heads * 128u + value_head * 128u;
  const uint i0=base+lane, i1=i0+32u, i2=i0+64u, i3=i0+96u;
  float y0=y[i0], y1=y[i1], y2=y[i2], y3=y[i3];
  float inv=rsqrt(simd_sum(y0*y0+y1*y1+y2*y2+y3*y3)/128.0f+eps);
  float z0=float(bfloat(z[i0])), z1=float(bfloat(z[i1]));
  float z2=float(bfloat(z[i2])), z3=float(bfloat(z[i3]));
  float n0=float(bfloat(y0*inv*float(norm_weight[lane])));
  float n1=float(bfloat(y1*inv*float(norm_weight[lane+32u])));
  float n2=float(bfloat(y2*inv*float(norm_weight[lane+64u])));
  float n3=float(bfloat(y3*inv*float(norm_weight[lane+96u])));
  gated[i0]=bfloat(n0*gdn_silu(z0)); gated[i1]=bfloat(n1*gdn_silu(z1));
  gated[i2]=bfloat(n2*gdn_silu(z2)); gated[i3]=bfloat(n3*gdn_silu(z3));
}
