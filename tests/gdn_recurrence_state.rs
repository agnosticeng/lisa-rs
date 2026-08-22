use lisa_rs::device::metal::{MTLSize, MetalBuffer, MetalDevice};
use lisa_rs::kernels::linear::GDN_SHADER;

const STEPS: usize = 16;
const KH: usize = 16;
const VH: usize = 48;
const HD: usize = 128;
const KEY_DIM: usize = KH * HD;
const VALUE_DIM: usize = VH * HD;
const CONV_DIM: usize = 2 * KEY_DIM + VALUE_DIM;
const STATE_LEN: usize = VH * HD * HD;

fn write_f32(buffer: &MetalBuffer, values: &[f32]) {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    buffer.write_bytes(0, &bytes);
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    enc: &lisa_rs::device::metal::ComputeEncoder,
    pipe: &lisa_rs::device::metal::ComputePipeline,
    conv: &MetalBuffer,
    conv_offset: usize,
    q: &MetalBuffer,
    q_offset: usize,
    k: &MetalBuffer,
    k_offset: usize,
    beta: &MetalBuffer,
    beta_offset: usize,
    decay: &MetalBuffer,
    decay_offset: usize,
    state: &MetalBuffer,
    output: &MetalBuffer,
    output_offset: usize,
    steps: u32,
) {
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(conv, conv_offset as u64, 0);
    enc.set_buffer(q, q_offset as u64, 1);
    enc.set_buffer(k, k_offset as u64, 2);
    enc.set_buffer(beta, beta_offset as u64, 3);
    enc.set_buffer(decay, decay_offset as u64, 4);
    enc.set_buffer(state, 0, 5);
    enc.set_buffer(output, output_offset as u64, 6);
    enc.set_buffer(state, 0, 9);
    let dims = [VH as u32, HD as u32, HD as u32, (VH / KH) as u32];
    let dims: Vec<u8> = dims.iter().flat_map(|value| value.to_le_bytes()).collect();
    enc.set_bytes(&dims, 7);
    enc.set_bytes(&steps.to_le_bytes(), 8);
    enc.set_bytes(&0u32.to_le_bytes(), 10);
    enc.set_buffer(state, 0, 9);
    enc.set_bytes(&0u32.to_le_bytes(), 10);
    enc.dispatch_threads(
        MTLSize {
            width: 32,
            height: HD as u64,
            depth: VH as u64,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
}

#[test]
fn recurrence_batch_matches_persistent_decode() {
    let dev = MetalDevice::default();
    let pipe =
        dev.new_compute_pipeline(&dev.new_library(GDN_SHADER).function_named("gdn_recurrence"));

    let conv_values: Vec<f32> = (0..STEPS * CONV_DIM)
        .map(|index| {
            let col = index % CONV_DIM;
            if col < 2 * KEY_DIM {
                0.0
            } else {
                ((index % 127) as f32 - 63.0) * 0.002
            }
        })
        .collect();
    let q_values: Vec<f32> = (0..STEPS * KEY_DIM)
        .map(|index| ((index % 61) as f32 - 30.0) * 0.00002)
        .collect();
    let k_values: Vec<f32> = (0..STEPS * KEY_DIM)
        .map(|index| ((index % 67) as f32 - 33.0) * 0.0002)
        .collect();
    let beta_values: Vec<f32> = (0..STEPS * VH)
        .map(|index| 0.35 + (index % 11) as f32 * 0.01)
        .collect();
    let decay_values: Vec<f32> = (0..STEPS * VH)
        .map(|index| 0.91 + (index % 7) as f32 * 0.005)
        .collect();

    let conv = dev.new_buffer(conv_values.len() * 4);
    let q = dev.new_buffer(q_values.len() * 4);
    let k = dev.new_buffer(k_values.len() * 4);
    let beta = dev.new_buffer(beta_values.len() * 4);
    let decay = dev.new_buffer(decay_values.len() * 4);
    write_f32(&conv, &conv_values);
    write_f32(&q, &q_values);
    write_f32(&k, &k_values);
    write_f32(&beta, &beta_values);
    write_f32(&decay, &decay_values);

    let state_batch = dev.new_buffer(STATE_LEN * 4);
    let state_decode = dev.new_buffer(STATE_LEN * 4);
    let output_batch = dev.new_buffer(STEPS * VALUE_DIM * 4);
    let output_decode = dev.new_buffer(STEPS * VALUE_DIM * 4);
    let zeros = vec![0; STATE_LEN * 4];
    state_batch.write_bytes(0, &zeros);
    state_decode.write_bytes(0, &zeros);

    let queue = dev.new_command_queue();
    let command = queue.new_command_buffer();
    let enc = command.compute_compute_encoder();
    dispatch(
        &enc,
        &pipe,
        &conv,
        0,
        &q,
        0,
        &k,
        0,
        &beta,
        0,
        &decay,
        0,
        &state_batch,
        &output_batch,
        0,
        STEPS as u32,
    );
    for token in 0..STEPS {
        dispatch(
            &enc,
            &pipe,
            &conv,
            token * CONV_DIM * 4,
            &q,
            token * KEY_DIM * 4,
            &k,
            token * KEY_DIM * 4,
            &beta,
            token * VH * 4,
            &decay,
            token * VH * 4,
            &state_decode,
            &output_decode,
            token * VALUE_DIM * 4,
            1,
        );
    }
    enc.end_encoding();
    command.commit();
    command.wait_until_completed();

    assert_eq!(
        output_batch.read_bytes(0, STEPS * VALUE_DIM * 4),
        output_decode.read_bytes(0, STEPS * VALUE_DIM * 4),
        "batch and persistent decode outputs differ"
    );
    assert_eq!(
        state_batch.read_bytes(0, STATE_LEN * 4),
        state_decode.read_bytes(0, STATE_LEN * 4),
        "batch and persistent decode states differ"
    );
}
