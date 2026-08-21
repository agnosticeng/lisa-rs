// Kernel shaders and their host-side wrappers.
// One shader file per logical op family (candle style), recompiled
// independently; host code lives here next to each family.
pub mod linear;
