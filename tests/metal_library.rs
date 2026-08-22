// Compiles a trivial shader, looks up a function, builds a compute pipeline.
// Own process (integration) to keep the device sub-allocator heap isolated.
use lisa_rs::device::metal::MetalDevice;

#[test]
fn compiles_shader_and_pipeline() {
    let dev = MetalDevice::default();
    let src = r#"
        kernel void vadd(device float* a, device float* b, device float* c,
                         constant uint& n, uint gid [[thread_position_in_grid]]) {
            if (gid < n) { c[gid] = a[gid] + b[gid]; }
        }
    "#;
    let lib = dev.new_library(src);
    assert!(!lib.id.is_null());
    let f = lib.function_named("vadd");
    assert!(!f.id.is_null());
    let pipe = dev.new_compute_pipeline(&f);
    assert!(!pipe.id.is_null());
}
