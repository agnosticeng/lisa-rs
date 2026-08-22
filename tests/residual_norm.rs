use lisa_rs::device::metal::{MTLSize, MetalDevice};
use lisa_rs::format::dtype::f32_to_bf16;
use lisa_rs::kernels::linear::NORM_SHADER;

const COLS: usize = 6_144;
const EPS: f32 = 1e-6;

fn bf16_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values
        .flat_map(|value| f32_to_bf16(value).to_le_bytes())
        .collect()
}

#[test]
fn fused_residual_norm_is_bit_exact_for_m1_and_m3() {
    let device = MetalDevice::default();
    let library = device.new_library(NORM_SHADER);
    let residual = device.new_compute_pipeline(&library.function_named("residual_add"));
    let norm = device.new_compute_pipeline(&library.function_named("rms_norm_rows"));
    let fused = device.new_compute_pipeline(&library.function_named("residual_add_rms_norm_rows"));
    let queue = device.new_command_queue();

    for rows in [1usize, 3] {
        let elements = rows * COLS;
        let x = device.new_buffer(elements * 2);
        let branch = device.new_buffer(elements * 4);
        let weight = device.new_buffer(COLS * 2);
        let reference_residual = device.new_buffer(elements * 2);
        let reference_norm = device.new_buffer(elements * 2);
        let fused_residual = device.new_buffer(elements * 2);
        let fused_norm = device.new_buffer(elements * 2);

        x.write_bytes(
            0,
            &bf16_bytes((0..elements).map(|i| ((i % 509) as f32 - 254.0) / 97.0)),
        );
        branch.write_bytes(
            0,
            &(0..elements)
                .flat_map(|i| {
                    let value = ((i * 31 % 1021) as f32 - 510.0) / 131.0;
                    value.to_le_bytes()
                })
                .collect::<Vec<_>>(),
        );
        weight.write_bytes(
            0,
            &bf16_bytes((0..COLS).map(|i| 0.5 + (i % 127) as f32 / 193.0)),
        );

        let command = queue.new_command_buffer();
        let encoder = command.compute_compute_encoder();
        encoder.set_compute_pipeline_state(&residual);
        encoder.set_buffer(&x, 0, 0);
        encoder.set_buffer(&branch, 0, 1);
        encoder.set_buffer(&reference_residual, 0, 2);
        encoder.set_bytes(&(elements as u32).to_le_bytes(), 3);
        encoder.dispatch_threads(
            MTLSize {
                width: elements as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
        encoder.set_compute_pipeline_state(&norm);
        encoder.set_buffer(&reference_residual, 0, 0);
        encoder.set_buffer(&weight, 0, 1);
        encoder.set_buffer(&reference_norm, 0, 2);
        encoder.set_bytes(&(COLS as u32).to_le_bytes(), 3);
        encoder.set_bytes(&EPS.to_le_bytes(), 4);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: rows as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.set_compute_pipeline_state(&fused);
        encoder.set_buffer(&x, 0, 0);
        encoder.set_buffer(&branch, 0, 1);
        encoder.set_buffer(&weight, 0, 2);
        encoder.set_buffer(&fused_residual, 0, 3);
        encoder.set_buffer(&fused_norm, 0, 4);
        encoder.set_bytes(&(COLS as u32).to_le_bytes(), 5);
        encoder.set_bytes(&EPS.to_le_bytes(), 6);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: rows as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();

        assert_eq!(
            fused_residual.read_bytes(0, elements * 2),
            reference_residual.read_bytes(0, elements * 2),
            "M={rows} residual differs"
        );
        assert_eq!(
            fused_norm.read_bytes(0, elements * 2),
            reference_norm.read_bytes(0, elements * 2),
            "M={rows} normalized output differs"
        );
    }
}
