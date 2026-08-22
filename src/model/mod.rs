// Model: config table, hybrid GDN + full-attention layers, KV cache.
// P1 (quantized GEMM / quantization), P2 (attention), P3 (forward/decode).
pub mod runner;
pub mod weights;

pub const LAYERS: usize = 64;
pub const HIDDEN: usize = 5120;
pub const VOCAB: usize = 248320;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerKind {
    Gdn,
    Attention,
}

pub const fn layer_kind(index: usize) -> LayerKind {
    if (index + 1).is_multiple_of(4) {
        LayerKind::Attention
    } else {
        LayerKind::Gdn
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen35_layer_schedule() {
        assert_eq!(layer_kind(0), LayerKind::Gdn);
        assert_eq!(layer_kind(3), LayerKind::Attention);
        assert_eq!(layer_kind(63), LayerKind::Attention);
        assert_eq!(
            (0..LAYERS)
                .filter(|index| layer_kind(*index) == LayerKind::Attention)
                .count(),
            16
        );
    }
}
