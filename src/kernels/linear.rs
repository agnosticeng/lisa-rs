// Host side of the linear kernels: shader sources live here so the integration
// tests can compile them through the Metal bridge.
pub const NAX_AFFINE_U4_SHADER: &str = include_str!("shaders/nax_affine_u4.metal");
pub const MLP_SHADER: &str = include_str!("shaders/mlp.metal");
pub const ATTENTION_SHADER: &str = include_str!("shaders/attn.one.metal");
pub const GDN_SHADER: &str = include_str!("shaders/gdn.metal");
pub const NORM_SHADER: &str = include_str!("shaders/norm.metal");
pub const EMBED_SHADER: &str = include_str!("shaders/embed.metal");
