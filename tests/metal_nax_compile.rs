// Compile-validates the NAX tensor-core affine u4 GEMM shader through the real
// Metal compiler (catches any MSL/mpp syntax errors). Full dispatch + parity is
// the next step once the fragment pipeline state builds.
use lisa_rs::device::metal::MetalDevice;

#[test]
fn nax_affine_u4_compiles() {
    let dev = MetalDevice::default();
    let lib = dev.new_library(lisa_rs::kernels::linear::NAX_AFFINE_U4_SHADER);
    let f = lib.function_named("q4_gemm_nax_coop");
    let pipe = dev.new_compute_pipeline(&f);
    assert!(!pipe.id.is_null());
}
