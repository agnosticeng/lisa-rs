// Executes the NAX tensor-core affine-u4 GEMM on the GPU. With A all-zero the
// output must be exactly zero → proves dequant+mma+store actually run. Full
// numerical parity vs MLX/saragossa is the next validation step.
use lisa_rs::device::metal::{MTLSize, MetalDevice};
use lisa_rs::kernels::linear::NAX_AFFINE_U4_SHADER;

#[test]
fn nax_gemm_executes() {
    let m: usize = 16;
    let n: usize = 32;
    let k: usize = 64;
    let packed_cols = k / 8;
    let groups = k / 64;

    let dev = MetalDevice::default();
    let lib = dev.new_library(NAX_AFFINE_U4_SHADER);
    let pipe = dev.new_compute_pipeline(&lib.function_named("q4_gemm_nax_coop"));
    let q = dev.new_command_queue();

    // One heap buffer, sub-regions per argument (avoids the multi-subresource
    // cache-mode path that aborts).
    let a_len = m * k * 2;
    let wp_len = n * packed_cols * 4;
    let ws_len = n * groups * 2;
    let wb_len = n * groups * 2;
    let c_len = m * n * 4;
    let a0 = 0usize;
    let wp0 = a0 + a_len;
    let ws0 = wp0 + wp_len;
    let wb0 = ws0 + ws_len;
    let c0 = wb0 + wb_len;

    let heap = dev.new_heap(c0 + c_len + 4096);
    let buf = heap.new_buffer(c0 + c_len); // single allocation
    buf.write_bytes(0, &vec![0u8; c0 + c_len]); // zero A + weights

    let cb = q.new_command_buffer();
    let enc = cb.compute_compute_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(&buf, a0 as u64, 0); // A bf16 [M,K] = 0
    enc.set_buffer(&buf, wp0 as u64, 1); // weight u4
    enc.set_buffer(&buf, ws0 as u64, 2); // scales bf16
    enc.set_buffer(&buf, wb0 as u64, 3); // biases bf16
    enc.set_buffer(&buf, c0 as u64, 4); // C f32 out
    let mut nk = Vec::<u8>::new();
    nk.extend_from_slice(&(n as u32).to_le_bytes());
    nk.extend_from_slice(&(k as u32).to_le_bytes());
    nk.extend_from_slice(&16u32.to_le_bytes());
    enc.set_bytes(&nk, 5);
    enc.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let out = buf.read_bytes(c0 as u64, c_len);
    for (i, chunk) in out.chunks_exact(4).enumerate() {
        let b: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let v = f32::from_le_bytes(b);
        assert!(v == 0.0, "C[{i}] nonzero ({v}) with zero A");
    }
}
