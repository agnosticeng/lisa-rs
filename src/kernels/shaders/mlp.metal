#include <metal_stdlib>
#include <metal_tensor>
using namespace metal;

// SiLU(g) * u  ->  store bf16 (MLP SwiGLU intermediate, group-64 active path).
kernel void silu_mul_up(device const float* gate [[buffer(0)]],
                        device const float* up   [[buffer(1)]],
                        device bfloat*      out  [[buffer(2)]],
                        constant uint&      n    [[buffer(3)]],
                        uint tid [[thread_position_in_grid]]) {
  if (tid < n) {
    float g = float(bfloat(gate[tid]));
    float u = float(bfloat(up[tid]));
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float si = g / (1.0f + exp2(-1.4426950408889634f * g));
    out[tid] = bfloat(si * u);
  }
}
