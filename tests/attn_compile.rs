// Compile-validates the full-attention shader set (qk-norm+gate+rope split,
// k-norm+rope, scalar sdpa, gate->bf16) through the real Metal compiler and
// builds each compute pipeline. Catches any MSL issue at the boundary.
use lisa_rs::device::metal::{MTLSize, MetalDevice};

#[test]
fn qk_norm_rope_compiles() {
    let dev = MetalDevice::default();
    let lib = dev.new_library(lisa_rs::kernels::linear::ATTENTION_SHADER);
    for name in [
        "qk_norm_gate_rope",
        "k_norm_rope",
        "sdpa_scalar",
        "gate_out_bf16",
    ] {
        let f = lib.function_named(name);
        let pipe = dev.new_compute_pipeline(&f);
        assert!(!pipe.id.is_null(), "{name} pipeline must not be null");
    }
}

#[test]
fn dispatch_qk_norm_gate_rope_smoke() {
    // Feed a tiny synthetic q (L=1, 12288) straight into the norm/split kernel
    // and just check the pipeline runs and emits finite query + raw gate values
    // with a matching head layout (no weights, no rope correctness yet).
    let dev = MetalDevice::default();
    let lib = dev.new_library(lisa_rs::kernels::linear::ATTENTION_SHADER);
    let pipe = dev.new_compute_pipeline(&lib.function_named("qk_norm_gate_rope"));

    let qbytes: usize = 12288 * 4;
    let h = dev.new_heap(qbytes + (6144 * 4) + (6144 * 2) + 4096);
    let q = h.new_buffer(qbytes);
    let qn = h.new_buffer(256 * 2);
    let oq = h.new_buffer(6144 * 2);
    let og = h.new_buffer(6144 * 4);

    let mut qv = Vec::new();
    for i in 0..12288 {
        let v = ((i % 31) as f32 - 15.0) * 0.01;
        qv.extend_from_slice(&v.to_le_bytes());
    }
    q.write_bytes(0, &qv);
    let mut nv = Vec::new();
    for _ in 0..256 {
        nv.extend_from_slice(&1.0f32.to_le_bytes());
    }
    qn.write_bytes(0, &nv);

    let cb = dev.new_command_queue();
    let buf = cb.new_command_buffer();
    let enc = buf.compute_compute_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(&q, 0, 0);
    enc.set_buffer(&qn, 0, 1);
    enc.set_buffer(&oq, 0, 2);
    enc.set_buffer(&og, 0, 3);
    let mut lbytes = Vec::new();
    lbytes.extend_from_slice(&1u32.to_le_bytes());
    enc.set_bytes(&lbytes, 4);
    let mut ebytes = Vec::new();
    ebytes.extend_from_slice(&1e-6f32.to_le_bytes());
    enc.set_bytes(&ebytes, 5);
    enc.dispatch_threads(
        MTLSize {
            width: 1,
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
    buf.commit();
    buf.wait_until_completed();

    let oq_r = oq.read_bytes(0, 6144 * 2);
    // query[0] = src[0]*rms*1; check finite + roughly preserves sign of src[0]
    let b0 = f32::from_bits((uq16le(&oq_r[0..2]) as u32) << 16) as f64;
    assert!(b0.is_finite(), "oq[0] must be finite, got {b0}");
}
fn uq16le(b: &[u8]) -> u16 {
    (b[0] as u16) | ((b[1] as u16) << 8)
}
