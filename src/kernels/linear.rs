// Host side of the linear kernels: shader sources live here so the integration
// tests can compile them through the Metal bridge.
//
// The quantized-affine path is split by execution phase:
//   - `gemm.metal`  : batched prefill GEMM kernels (`q4_gemm_*`, NAX tensor cores)
//   - `qmv.metal`   : decode M=1 / M=3 matrix-vector kernels (`q4_qmv_*`)
pub const NAX_GEMM_SHADER: &str = include_str!("shaders/gemm.metal");
pub const NAX_QMV_SHADER: &str = include_str!("shaders/qmv.metal");
// Concatenated source, kept for the low-level smoke tests and any consumer
// that wants a single library with both families.
pub const NAX_AFFINE_U4_SHADER: &str =
    concat!(include_str!("shaders/gemm.metal"), include_str!("shaders/qmv.metal"));
pub const MLP_SHADER: &str = include_str!("shaders/mlp.metal");
pub const ATTENTION_SHADER: &str = include_str!("shaders/attention.metal");
pub const GDN_SHADER: &str = include_str!("shaders/gdn.metal");
pub const NORM_SHADER: &str = include_str!("shaders/norm.metal");
pub const EMBED_SHADER: &str = include_str!("shaders/embed.metal");
