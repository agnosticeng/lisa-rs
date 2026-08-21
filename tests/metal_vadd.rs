// Full compute: write A/B into shared buffers, run vadd on the GPU, read C back.
use lisa_rs::device::metal::{MTLSize, MetalDevice};

const N: usize = 16;

#[test]
fn gpu_vector_add() {
    let dev = MetalDevice::default();
    let src = r#"
        kernel void vadd(device float* a, device float* b, device float* c,
                         constant uint& n, uint gid [[thread_position_in_grid]]) {
            if (gid < n) { c[gid] = a[gid] + b[gid]; }
        }
    "#;
    let lib = dev.new_library(src);
    let pipe = dev.new_compute_pipeline(&lib.function_named("vadd"));
    let q = dev.new_command_queue();

    // Carve A/B/C from one shared MTLHeap to avoid the device sub-allocator.
    let heap = dev.new_heap(N * 4 * 3 + 4096);
    let a = heap.new_buffer(N * 4);
    let b = heap.new_buffer(N * 4);
    let c = heap.new_buffer(N * 4);

    let mut av = Vec::<u8>::new();
    let mut bv = Vec::<u8>::new();
    for i in 0..N {
        av.extend_from_slice(&(i as f32).to_le_bytes());
        bv.extend_from_slice(&(i as f32 * 100.0).to_le_bytes());
    }
    a.write_bytes(0, &av);
    b.write_bytes(0, &bv);

    let cb = q.new_command_buffer();
    let enc = cb.compute_compute_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(&a, 0, 0);
    enc.set_buffer(&b, 0, 1);
    enc.set_buffer(&c, 0, 2);
    enc.set_bytes(&(N as u64).to_le_bytes(), 3);
    enc.dispatch_threads(
        MTLSize {
            width: N as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let out = c.read_bytes(0, N * 4);
    for i in 0..N {
        let b4: [u8; 4] = [out[i * 4], out[i * 4 + 1], out[i * 4 + 2], out[i * 4 + 3]];
        let v = f32::from_le_bytes(b4);
        let expected = 101.0 * i as f32;
        assert!(
            (v - expected).abs() < 1e-3,
            "c[{i}] = {v}, expected {expected}"
        );
    }
}
