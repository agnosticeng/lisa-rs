use lisa_rs::device::metal::{MTLSize, MetalDevice};
use lisa_rs::kernels::linear::EMBED_SHADER;

fn gpu_argmax(n: usize, values: &[f32]) -> u32 {
    const TG: usize = 256;
    let dev = MetalDevice::default();
    let pipe = dev.new_compute_pipeline(
        &dev.new_library(EMBED_SHADER)
            .function_named("argmax_f32_partial"),
    );
    let partials = n.div_ceil(TG);
    let x = dev.new_buffer(n * 4);
    let pmax = dev.new_buffer(partials * 4);
    let pidx = dev.new_buffer(partials * 4);
    x.write_bytes(
        0,
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let queue = dev.new_command_queue();
    let command = queue.new_command_buffer();
    let enc = command.compute_compute_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(&x, 0, 0);
    enc.set_buffer(&pmax, 0, 1);
    enc.set_buffer(&pidx, 0, 2);
    enc.set_bytes(&(n as u32).to_le_bytes(), 3);
    enc.dispatch_thread_groups(
        MTLSize {
            width: partials as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: TG as u64,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    command.commit();
    command.wait_until_completed();

    let maxes = pmax.read_bytes(0, partials * 4);
    let idxes = pidx.read_bytes(0, partials * 4);
    let mut best = f32::NEG_INFINITY;
    let mut best_index = 0u32;
    for i in 0..partials {
        let v = f32::from_le_bytes(maxes[i * 4..i * 4 + 4].try_into().unwrap());
        if v > best {
            best = v;
            best_index = u32::from_le_bytes(idxes[i * 4..i * 4 + 4].try_into().unwrap());
        }
    }
    best_index
}

#[test]
fn argmax_matches_cpu_over_full_vocab() {
    const N: usize = 248320;
    let mut values: Vec<f32> = (0..N)
        .map(|i| ((i as f32) * 0.00137).sin() * 0.9)
        .collect();
    values[123456] = 42.0;
    let cpu: u32 = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap();
    assert_eq!(gpu_argmax(N, &values), cpu);
    assert_eq!(cpu, 123456);
}

#[test]
fn argmax_matches_cpu_on_partial_tail() {
    const N: usize = 1000;
    let mut values: Vec<f32> = (0..N)
        .map(|i| ((i as f32) * 0.0137).cos() * 0.5)
        .collect();
    values[999] = 3.5;
    let cpu: u32 = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap();
    assert_eq!(gpu_argmax(N, &values), cpu);
    assert_eq!(cpu, 999);
}
