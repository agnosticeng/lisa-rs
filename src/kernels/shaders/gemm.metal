#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

// NAX tensor-core affine int4 GEMM, group size 64 (MLX layout).
//   A bf16 [M,K], Bp uint [N,K/8] (8 nibbles/u32, low first),
//   Bs/Bb bf16 [N,K/64], C f32 [M,N].  C = A · deq(B)^T.
// One threadgroup (a single SIMD group) computes a 16×32 tile of C, k in
// chunks of 16, weights dequantized in-register (no full unquantized material).
// Fragment: BaseNAXFrag transcription — 16×16, 8 bf16/thread, lane→coord
// bit-twiddle, MMA 16x32x16 via mpp cooperative tensors.

struct NaxFrag {
  static short2 get_coord() {
    ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    short qid = lane >> 2;
    short fm = (qid & 4) | ((lane >> 1) & 3);
    short fn = ((qid & 2) | (lane & 1)) * 4;
    return short2(fn, fm);
  }

  static void load_a_masked(thread metal::vec<bfloat,8>& dst, const device bfloat* src,
                            uint m_base, uint k_base, uint K, uint M) {
    short2 sc = get_coord();
    for (short i = 0; i < 2; i++) {
      uint row = m_base + uint(sc.y) + uint(i) * 8u;
      for (short j = 0; j < 4; j++) {
        uint col = k_base + uint(sc.x) + uint(j);
        dst[i * 4 + j] = (row < M) ? src[row * K + col] : bfloat(0.0f);
      }
    }
  }

  static void load_b_quant_u4(thread metal::vec<bfloat,8>& dst, const device uint* packed,
                              const device bfloat* scales, const device bfloat* biases,
                              uint n_base, uint k_base, uint K) {
    short2 sc = get_coord();
    uint groups = K / 64u;
    uint packed_cols = K / 8u;
    for (short i = 0; i < 2; i++) {
      uint nn = n_base + uint(sc.y) + uint(i) * 8u;
      for (short j = 0; j < 4; j++) {
        uint kk = k_base + uint(sc.x) + uint(j);
        uint word = packed[nn * packed_cols + (kk >> 3)];
        uint q = (word >> ((kk & 7u) * 4u)) & 0x0fu;
        uint g = kk / 64u;
        float s = float(scales[nn * groups + g]);
        float b = float(biases[nn * groups + g]);
        dst[i * 4 + j] = bfloat(float(q) * s + b);
      }
    }
  }

  static void store_c(const thread metal::vec<float,8>& src, device float* dst, int str_x) {
    short2 sc = get_coord();
    dst += sc.y * str_x + sc.x;
    for (short i = 0; i < 2; i++)
      for (short j = 0; j < 4; j++)
        dst[(i * 8) * str_x + j] = src[i * 4 + j];
  }

  // Aligned fast load of A: M and N both multiples of 64, so every tile is full
// (no row/col masking). Mirrors MLX's kAlignedM/kAlignedN constant paths which
// drop all per-lane bounds checks/predication.
  static void load_a_fast(thread metal::vec<bfloat,8>& dst, const device bfloat* src,
                          uint m_base, uint k_base, uint K) {
    short2 sc = get_coord();
    for (short i = 0; i < 2; i++) {
      const device bfloat* r = src + (m_base + uint(sc.y) + uint(i) * 8u) * K + k_base + sc.x;
      for (short j = 0; j < 4; j++) dst[i * 4 + j] = r[j];
    }
  }

  static void load_b_tg(thread metal::vec<bfloat,8>& dst, const threadgroup bfloat* src,
                        int str_x, uint off_x, uint off_y) {
    short2 sc = get_coord();
    const threadgroup bfloat* b = src + (off_x + sc.y) * str_x + off_y + sc.x;
    for (short i = 0; i < 2; i++) {
      const threadgroup bfloat* r = b + i * 8 * str_x;
      for (short j = 0; j < 4; j++) dst[i * 4 + j] = r[j];
    }
  }

  static void store_c_masked(thread metal::vec<float,8>& src, device float* dst,
                             uint str_x, uint m_base, uint n_base, uint M, uint N) {
    short2 sc = get_coord();
    for (short i = 0; i < 2; i++) {
      uint row = m_base + uint(sc.y) + uint(i) * 8u;
      if (row < M) {
        for (short j = 0; j < 4; j++) {
          uint col = n_base + uint(sc.x) + uint(j);
          if (col < N) dst[row * str_x + col] = src[i * 4 + j];
        }
      }
    }
  }

  static void mma(thread metal::vec<float,8>& Cn0, thread metal::vec<float,8>& Cn1,
                  const thread metal::vec<bfloat,8>& A,
                  const thread metal::vec<bfloat,8>& Bn0, const thread metal::vec<bfloat,8>& Bn1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
    for (short i = 0; i < 8; i++) ct_a[i] = A[i];
    for (short i = 0; i < 8; i++) { ct_b[i] = Bn0[i]; ct_b[8 + i] = Bn1[i]; }
    for (short i = 0; i < 8; i++) { ct_c[i] = Cn0[i]; ct_c[8 + i] = Cn1[i]; }
    op.run(ct_a, ct_b, ct_c);
    for (short i = 0; i < 8; i++) { Cn0[i] = ct_c[i]; Cn1[i] = ct_c[8 + i]; }
  }
};

// C[16x32 tile] = A^bf16 · deq(B^u4,gs64)^T. Grid: (M/16, N/32), 1 SIMD group.
// The whole 16-row tile stores only when it is fully within M (tail tiles with
// m0+15 >= M are skipped so store_c never writes past row M-1).
kernel void q4_gemm_nax_coop(device const bfloat* A   [[buffer(0)]],
                             device const uint*   Bp  [[buffer(1)]],
                             device const bfloat* Bs  [[buffer(2)]],
                             device const bfloat* Bb  [[buffer(3)]],
                             device float*        C   [[buffer(4)]],
                             constant uint3& nk       [[buffer(5)]],
                             uint2 tgid [[threadgroup_position_in_grid]]) {
  uint N = nk.x, K = nk.y, M = nk.z;
  uint m0 = tgid.x * 16u, n0 = tgid.y * 32u;
  if (n0 >= N || m0 + 16u > M) return;
  metal::vec<float,8> Cn0 = float(0), Cn1 = float(0);
  for (uint k = 0; k < K; k += 16u) {
    metal::vec<bfloat,8> Af, Bn0, Bn1;
    NaxFrag::load_a_masked(Af, A, m0, k, K, m0 + 16u);
    NaxFrag::load_b_quant_u4(Bn0, Bp, Bs, Bb, n0, k, K);
    NaxFrag::load_b_quant_u4(Bn1, Bp, Bs, Bb, n0 + 16u, k, K);
    NaxFrag::mma(Cn0, Cn1, Af, Bn0, Bn1);
  }
  // N is always a multiple of 16, so store the two 16-column halves only when
  // they are within bounds; a half-tile at the tail (N % 32 != 0) would
  // otherwise read out-of-range weights and scatter garbage into the next row.
  if (n0 + 16u <= N) NaxFrag::store_c(Cn0, C + m0 * N + n0, int(N));
  if (n0 + 32u <= N) NaxFrag::store_c(Cn1, C + m0 * N + n0 + 16u, int(N));
}

// Coalesced cooperative 64x64 tile for batched prefill GEMMs. Each thread
// dequants 16 contiguous bytes (32 q4 values) of one weight row into a padded
// threadgroup tile; the 4 SIMD groups then each compute a 32x32 C sub-tile from
// the shared tile (reuse the dequant, no duplicate work). Bit-exact vs
// q4_gemm_nax_coop; ~15-35% faster at M>=128 on MLP shapes. Grid: (M/64, N/64),
// 128 threads. M,N padded up by the host (masked rows stored nowhere).
kernel void q4_gemm_nax_tiled(device const bfloat* A  [[buffer(0)]],
                              device const uint*   W  [[buffer(1)]],
                              device const bfloat* Sc [[buffer(2)]],
                              device const bfloat* Bs [[buffer(3)]],
                              device float*        C  [[buffer(4)]],
                              constant uint3&      MNK [[buffer(5)]],
                              uint2 tgid [[threadgroup_position_in_grid]],
                              uint sgid [[simdgroup_index_in_threadgroup]],
                              uint lid [[thread_index_in_threadgroup]]) {
  const uint M = MNK.x, N = MNK.y, K = MNK.z;
  constexpr uint BK = 64, BK_PAD = 72;
  threadgroup bfloat Ws[64 * BK_PAD];

  const uint m_base = tgid.x * 64u;
  const uint n_base = tgid.y * 64u;
  const uint num_outs = min(64u, N - n_base);
  const uint m0 = m_base + (sgid / 2u) * 32u;
  const uint n0 = n_base + (sgid % 2u) * 32u;

  // MLX-style coalesced loader: 16 bytes/thread (32 q4) per k-group.
  const uint rbi = (16u * lid) / 32u;
  const uint rbj = (16u * lid) % 32u;
  const uint packed_cols = K / 8u;
  const uint gpw = K / 64u;

  metal::vec<float,8> c00=float(0), c01=float(0), c10=float(0), c11=float(0);

  for (uint kb = 0; kb < K; kb += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nn = n_base + rbi;
    float ss = 0.0f, bb = 0.0f;
    if (rbi < num_outs) { ss = float(Sc[nn*gpw + kb/64u]); bb = float(Bs[nn*gpw + kb/64u]); }
threadgroup bfloat* dst = Ws + rbi*BK_PAD + rbj*2u;
    const device uint8_t* wp = (const device uint8_t*)(W + nn*packed_cols) + (kb>>1u) + rbj;
    for (uint i = 0; i < 16u; i++) {
      const uint8_t byte = (rbi < num_outs) ? wp[i] : 0u;
      dst[2u*i]   = bfloat(ss * float(byte & 0x0fu) + bb);
      dst[2u*i+1] = bfloat(ss/16.0f * float(byte & 0xf0u) + bb);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0; kk < BK; kk += 16u) {
      metal::vec<bfloat,8> a0, a1, b0, b1;
      NaxFrag::load_a_masked(a0, A, m0,      kb+kk, K, M);
      NaxFrag::load_a_masked(a1, A, m0 + 16u, kb+kk, K, M);
      NaxFrag::load_b_tg(b0, Ws, (int)BK_PAD, n0 - n_base, kk);
      NaxFrag::load_b_tg(b1, Ws, (int)BK_PAD, n0 - n_base + 16u, kk);
      NaxFrag::mma(c00, c01, a0, b0, b1);
      NaxFrag::mma(c10, c11, a1, b0, b1);
    }
  }
  NaxFrag::store_c_masked(c00, C, (uint)N, m0,      n0,      M, N);
  NaxFrag::store_c_masked(c01, C, (uint)N, m0,      n0 + 16u, M, N);
  NaxFrag::store_c_masked(c10, C, (uint)N, m0 + 16u, n0,      M, N);
  NaxFrag::store_c_masked(c11, C, (uint)N, m0 + 16u, n0 + 16u, M, N);
}

// Branchless variant of q4_gemm_nax_tiled for when M and N are both multiples
// of 64 (all model projections: MLP gate/up/down, GDN, attn, lm_head). Zero
// explicit per-lane predication on every load/store — mirrors MLX's
// kAlignedM/kAlignedN compile-time paths.
kernel void q4_gemm_nax_tiled_align(device const bfloat* A  [[buffer(0)]],
                                    device const uint*   W  [[buffer(1)]],
                                    device const bfloat* Sc [[buffer(2)]],
                                    device const bfloat* Bs [[buffer(3)]],
                                    device float*        C  [[buffer(4)]],
                                    constant uint3&      MNK [[buffer(5)]],
                                    uint2 tgid [[threadgroup_position_in_grid]],
                                    uint sgid [[simdgroup_index_in_threadgroup]],
                                    uint lid [[thread_index_in_threadgroup]]) {
  const uint M = MNK.x, N = MNK.y, K = MNK.z;
  constexpr uint BK = 64, BK_PAD = 72;
  threadgroup bfloat Ws[64 * BK_PAD];

  const uint m_base = tgid.x * 64u;
  const uint n_base = tgid.y * 64u;
  const uint m0 = m_base + (sgid / 2u) * 32u;
  const uint n0 = n_base + (sgid % 2u) * 32u;

  const uint rbi = (16u * lid) / 32u;
  const uint rbj = (16u * lid) % 32u;
  const uint packed_cols = K / 8u;
  const uint gpw = K / 64u;

  metal::vec<float,8> c00=float(0), c01=float(0), c10=float(0), c11=float(0);

  for (uint kb = 0; kb < K; kb += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nn = n_base + rbi;
    const uint g = kb / 64u;
    const float ss = float(Sc[nn*gpw + g]);
    const float bb = float(Bs[nn*gpw + g]);
    threadgroup bfloat* dst = Ws + rbi*BK_PAD + rbj*2u;
    const device uint8_t* wp = (const device uint8_t*)(W + nn*packed_cols) + (kb>>1u) + rbj;
    for (uint i = 0; i < 16u; i++) {
      const uint8_t byte = wp[i];
      dst[2u*i]   = bfloat(ss * float(byte & 0x0fu) + bb);
      dst[2u*i+1] = bfloat(ss/16.0f * float(byte & 0xf0u) + bb);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0; kk < BK; kk += 16u) {
      metal::vec<bfloat,8> a0, a1, b0, b1;
      NaxFrag::load_a_fast(a0, A, m0,      kb+kk, K);
      NaxFrag::load_a_fast(a1, A, m0 + 16u, kb+kk, K);
      NaxFrag::load_b_tg(b0, Ws, (int)BK_PAD, n0 - n_base, kk);
      NaxFrag::load_b_tg(b1, Ws, (int)BK_PAD, n0 - n_base + 16u, kk);
      NaxFrag::mma(c00, c01, a0, b0, b1);
      NaxFrag::mma(c10, c11, a1, b0, b1);
    }
  }
  NaxFrag::store_c(c00, C + m0 * N + n0, (int)N);
  NaxFrag::store_c(c01, C + m0 * N + n0 + 16u, (int)N);
  NaxFrag::store_c(c10, C + (m0 + 16u) * N + n0, (int)N);
  NaxFrag::store_c(c11, C + (m0 + 16u) * N + n0 + 16u, (int)N);
}

// Fused two-projection variant of q4_gemm_nax_tiled_align: gate/up (or any
// two same-geometry bilinear projections) in ONE dispatch. grid.z selects the
// weight/output pair; both tiles are otherwise identical so the whole 64x64
// cooperative tile is shared. Same branchless numerics as the single kernel.
kernel void q4_gemm_nax_tiled_align_fused2(
    device const bfloat* A  [[buffer(0)]],
    device const uint*   W0 [[buffer(1)]],
    device const bfloat* Sc0 [[buffer(2)]],
    device const bfloat* Bs0 [[buffer(3)]],
    device float*        C0  [[buffer(4)]],
    device const uint*   W1 [[buffer(5)]],
    device const bfloat* Sc1 [[buffer(6)]],
    device const bfloat* Bs1 [[buffer(7)]],
    device float*        C1  [[buffer(8)]],
    constant uint3&      MNK [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_threadgroup]]) {
  const uint M = MNK.x, N = MNK.y, K = MNK.z;
  constexpr uint BK = 64, BK_PAD = 72;

  const bool use_w1 = tgid.z == 1u;
  const device uint*   W  = use_w1 ? W1 : W0;
  const device bfloat* Sc = use_w1 ? Sc1 : Sc0;
  const device bfloat* Bs = use_w1 ? Bs1 : Bs0;
  device float*        C  = use_w1 ? C1 : C0;

  threadgroup bfloat Ws_[64 * BK_PAD];

  const uint m_base = tgid.x * 64u;
  const uint n_base = tgid.y * 64u;
  const uint m0 = m_base + (sgid / 2u) * 32u;
  const uint n0 = n_base + (sgid % 2u) * 32u;

  const uint rbi = (16u * lid) / 32u;
  const uint rbj = (16u * lid) % 32u;
  const uint packed_cols = K / 8u;
  const uint gpw = K / 64u;

  threadgroup bfloat* Ws = Ws_;

  metal::vec<float,8> c00=float(0), c01=float(0), c10=float(0), c11=float(0);

  for (uint kb = 0; kb < K; kb += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nn = n_base + rbi;
    const uint g = kb / 64u;
    const float ss = float(Sc[nn*gpw + g]);
    const float bb = float(Bs[nn*gpw + g]);
    threadgroup bfloat* dst = Ws + rbi*BK_PAD + rbj*2u;
    const device uint8_t* wp = (const device uint8_t*)(W + nn*packed_cols) + (kb>>1u) + rbj;
    for (uint i = 0; i < 16u; i++) {
      const uint8_t byte = wp[i];
      dst[2u*i]   = bfloat(ss * float(byte & 0x0fu) + bb);
      dst[2u*i+1] = bfloat(ss/16.0f * float(byte & 0xf0u) + bb);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0; kk < BK; kk += 16u) {
      metal::vec<bfloat,8> a0, a1, b0, b1;
      NaxFrag::load_a_fast(a0, A, m0,      kb+kk, K);
      NaxFrag::load_a_fast(a1, A, m0 + 16u, kb+kk, K);
      NaxFrag::load_b_tg(b0, Ws, (int)BK_PAD, n0 - n_base, kk);
      NaxFrag::load_b_tg(b1, Ws, (int)BK_PAD, n0 - n_base + 16u, kk);
      NaxFrag::mma(c00, c01, a0, b0, b1);
      NaxFrag::mma(c10, c11, a1, b0, b1);
    }
  }
  NaxFrag::store_c(c00, C + m0 * N + n0, (int)N);
  NaxFrag::store_c(c01, C + m0 * N + n0 + 16u, (int)N);
  NaxFrag::store_c(c10, C + (m0 + 16u) * N + n0, (int)N);
  NaxFrag::store_c(c11, C + (m0 + 16u) * N + n0 + 16u, (int)N);
}

