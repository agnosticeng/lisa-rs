use lisa_rs::device::metal::{MTLSize, MetalDevice};
use lisa_rs::format::dtype::{bf16_to_f32, f32_to_bf16};
use lisa_rs::kernels::linear::GDN_SHADER;

const CONV_DIM: usize = 10240;

fn round_to_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    bf16_to_f32((rounded >> 16) as u16)
}

fn run_case(steps: usize) {
    let dev = MetalDevice::default();
    let pipe = dev.new_compute_pipeline(
        &dev.new_library(GDN_SHADER)
            .function_named("gdn_update_conv_state"),
    );
    let qkv = dev.new_buffer(steps * CONV_DIM * 4);
    let state = dev.new_buffer(3 * CONV_DIM * 2);

    let qkv_values: Vec<f32> = (0..steps * CONV_DIM)
        .map(|index| 10.0 + index as f32 * 0.0001)
        .collect();
    let qkv_bytes: Vec<u8> = qkv_values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    qkv.write_bytes(0, &qkv_bytes);

    let old_values: Vec<f32> = (0..3 * CONV_DIM)
        .map(|index| -3.0 + index as f32 * 0.0001)
        .collect();
    let old_bytes: Vec<u8> = old_values
        .iter()
        .flat_map(|value| f32_to_bf16(*value).to_le_bytes())
        .collect();
    state.write_bytes(0, &old_bytes);

    let queue = dev.new_command_queue();
    let command = queue.new_command_buffer();
    let enc = command.compute_compute_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(&qkv, 0, 0);
    enc.set_buffer(&state, 0, 1);
    enc.set_buffer(&state, 0, 4);
    enc.set_bytes(&(CONV_DIM as u32).to_le_bytes(), 2);
    enc.set_bytes(&(steps as u32).to_le_bytes(), 3);
    enc.set_bytes(&0u32.to_le_bytes(), 5);
    enc.set_buffer(&state, 0, 4);
    enc.set_bytes(&0u32.to_le_bytes(), 5);
    enc.dispatch_threads(
        MTLSize {
            width: CONV_DIM as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    command.commit();
    command.wait_until_completed();

    let actual = state.read_bytes(0, 3 * CONV_DIM * 2);
    for slot in 0..3 {
        for channel in [0, 17, CONV_DIM - 1] {
            let source = steps + slot;
            let expected = if source < 3 {
                bf16_to_f32(f32_to_bf16(old_values[source * CONV_DIM + channel]))
            } else {
                round_to_bf16(qkv_values[(source - 3) * CONV_DIM + channel])
            };
            let index = (slot * CONV_DIM + channel) * 2;
            let value = bf16_to_f32(u16::from_le_bytes([actual[index], actual[index + 1]]));
            assert_eq!(
                value, expected,
                "steps={steps} slot={slot} channel={channel}"
            );
        }
    }
}

#[test]
fn conv_state_advances_for_decode_and_prefill() {
    for steps in [1, 2, 3, 16] {
        run_case(steps);
    }
}
