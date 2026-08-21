use lisa_rs::device::metal::{MTLSize, MetalDevice};
use lisa_rs::format::dtype::{bf16_to_f32, f32_to_bf16};
use lisa_rs::kernels::linear::NAX_AFFINE_U4_SHADER;

#[test]
fn qmv_matches_affine_reference_with_tail_rows() {
    const N: usize = 19;
    const K: usize = 128;
    let x_values: Vec<f32> = (0..K).map(|k| (k as f32 - 63.0) * 0.003).collect();
    let x_bf16: Vec<u16> = x_values.iter().map(|value| f32_to_bf16(*value)).collect();
    let packed: Vec<u32> = (0..N * K / 8)
        .map(|index| {
            let mut word = 0u32;
            for nibble in 0..8 {
                word |= (((index * 3 + nibble * 5) % 16) as u32) << (nibble * 4);
            }
            word
        })
        .collect();
    let scales = [f32_to_bf16(0.0125); N * K / 64];
    let biases = [f32_to_bf16(-0.08); N * K / 64];

    let dev = MetalDevice::default();
    let pipe = dev.new_compute_pipeline(
        &dev.new_library(NAX_AFFINE_U4_SHADER)
            .function_named("q4_qmv_fast"),
    );
    let x = dev.new_buffer(K * 2);
    let w = dev.new_buffer(packed.len() * 4);
    let s = dev.new_buffer(scales.len() * 2);
    let b = dev.new_buffer(biases.len() * 2);
    let y = dev.new_buffer(N * 4);
    x.write_bytes(
        0,
        &x_bf16
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    w.write_bytes(
        0,
        &packed
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    s.write_bytes(
        0,
        &scales
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    b.write_bytes(
        0,
        &biases
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );

    let queue = dev.new_command_queue();
    let command = queue.new_command_buffer();
    let enc = command.compute_compute_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(&x, 0, 0);
    enc.set_buffer(&w, 0, 1);
    enc.set_buffer(&s, 0, 2);
    enc.set_buffer(&b, 0, 3);
    enc.set_buffer(&y, 0, 4);
    let mut nk = Vec::with_capacity(8);
    nk.extend_from_slice(&(N as u32).to_le_bytes());
    nk.extend_from_slice(&(K as u32).to_le_bytes());
    enc.set_bytes(&nk, 5);
    enc.dispatch_thread_groups(
        MTLSize {
            width: N.div_ceil(16) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    command.commit();
    command.wait_until_completed();

    let output = y.read_bytes(0, N * 4);
    for row in 0..N {
        let mut expected = 0.0f32;
        for col in 0..K {
            let word = packed[row * (K / 8) + col / 8];
            let q = ((word >> ((col % 8) * 4)) & 0xf) as f32;
            let group = row * (K / 64) + col / 64;
            let weight = q * bf16_to_f32(scales[group]) + bf16_to_f32(biases[group]);
            expected += bf16_to_f32(x_bf16[col]) * weight;
        }
        let index = row * 4;
        let actual = f32::from_le_bytes([
            output[index],
            output[index + 1],
            output[index + 2],
            output[index + 3],
        ]);
        assert!(
            (actual - expected).abs() < 2e-4,
            "row {row}: {actual} != {expected}"
        );
    }
}

#[test]
fn qmv_wide3_matches_three_affine_references_with_tail_rows() {
    const M: usize = 3;
    const N: usize = 19;
    const K: usize = 128;
    let x_bf16: Vec<u16> = (0..M * K)
        .map(|index| {
            let row = index / K;
            let col = index % K;
            f32_to_bf16((col as f32 - 61.0 + row as f32 * 7.0) * 0.0025)
        })
        .collect();
    let packed: Vec<u32> = (0..N * K / 8)
        .map(|index| {
            (0..8).fold(0u32, |word, nibble| {
                word | (((index * 7 + nibble * 3) % 16) as u32) << (nibble * 4)
            })
        })
        .collect();
    let scales = [f32_to_bf16(0.011); N * K / 64];
    let biases = [f32_to_bf16(-0.075); N * K / 64];
    let dev = MetalDevice::default();
    let pipe = dev.new_compute_pipeline(
        &dev.new_library(NAX_AFFINE_U4_SHADER)
            .function_named("q4_qmv_wide3"),
    );
    let x = dev.new_buffer(M * K * 2);
    let w = dev.new_buffer(packed.len() * 4);
    let s = dev.new_buffer(scales.len() * 2);
    let b = dev.new_buffer(biases.len() * 2);
    let y = dev.new_buffer(M * N * 4);
    x.write_bytes(
        0,
        &x_bf16
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    w.write_bytes(
        0,
        &packed
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    s.write_bytes(
        0,
        &scales
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    b.write_bytes(
        0,
        &biases
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let queue = dev.new_command_queue();
    let command = queue.new_command_buffer();
    let enc = command.compute_compute_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(&x, 0, 0);
    enc.set_buffer(&w, 0, 1);
    enc.set_buffer(&s, 0, 2);
    enc.set_buffer(&b, 0, 3);
    enc.set_buffer(&y, 0, 4);
    let mut nk = Vec::with_capacity(8);
    nk.extend_from_slice(&(N as u32).to_le_bytes());
    nk.extend_from_slice(&(K as u32).to_le_bytes());
    enc.set_bytes(&nk, 5);
    enc.dispatch_thread_groups(
        MTLSize {
            width: N.div_ceil(16) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    command.commit();
    command.wait_until_completed();
    let output = y.read_bytes(0, M * N * 4);
    for input_row in 0..M {
        for row in 0..N {
            let expected = (0..K).fold(0.0f32, |sum, col| {
                let word = packed[row * (K / 8) + col / 8];
                let q = ((word >> ((col % 8) * 4)) & 0xf) as f32;
                let group = row * (K / 64) + col / 64;
                let weight = q * bf16_to_f32(scales[group]) + bf16_to_f32(biases[group]);
                sum + bf16_to_f32(x_bf16[input_row * K + col]) * weight
            });
            let offset = (input_row * N + row) * 4;
            let actual = f32::from_le_bytes(output[offset..offset + 4].try_into().unwrap());
            assert!(
                (actual - expected).abs() < 2e-4,
                "row {input_row},{row}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn qmv_fast_matches_wide3_row0_bit_exact() {
    const N: usize = 19;
    const K: usize = 512;
    let x_bf16: Vec<u16> = (0..K)
        .map(|index| {
            let col = index % K;
            f32_to_bf16((col as f32 - 255.0) * 0.007)
        })
        .collect();
    let x3_bf16: Vec<u16> = x_bf16.repeat(3);
    let packed: Vec<u32> = (0..N * K / 8)
        .map(|index| {
            (0..8).fold(0u32, |word, nibble| {
                word | (((index * 3 + nibble * 5) % 16) as u32) << (nibble * 4)
            })
        })
        .collect();
    let scales = [f32_to_bf16(0.0125); N * K / 64];
    let biases = [f32_to_bf16(-0.08); N * K / 64];

    let dev = MetalDevice::default();
    let library = dev.new_library(NAX_AFFINE_U4_SHADER);
    let fast_pipe = dev.new_compute_pipeline(&library.function_named("q4_qmv_fast"));
    let wide_pipe = dev.new_compute_pipeline(&library.function_named("q4_qmv_wide3"));

    let x = dev.new_buffer(K * 2);
    let x3 = dev.new_buffer(3 * K * 2);
    let w = dev.new_buffer(packed.len() * 4);
    let s = dev.new_buffer(scales.len() * 2);
    let b = dev.new_buffer(biases.len() * 2);
    let y_fast = dev.new_buffer(N * 4);
    let y_wide = dev.new_buffer(3 * N * 4);
    x.write_bytes(
        0,
        &x_bf16
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    x3.write_bytes(
        0,
        &x3_bf16
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    w.write_bytes(
        0,
        &packed
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    s.write_bytes(
        0,
        &scales
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    b.write_bytes(
        0,
        &biases
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );

    let queue = dev.new_command_queue();
    let command = queue.new_command_buffer();
    let enc = command.compute_compute_encoder();
    enc.set_compute_pipeline_state(&fast_pipe);
    enc.set_buffer(&x, 0, 0);
    enc.set_buffer(&w, 0, 1);
    enc.set_buffer(&s, 0, 2);
    enc.set_buffer(&b, 0, 3);
    enc.set_buffer(&y_fast, 0, 4);
    let mut nk = Vec::with_capacity(8);
    nk.extend_from_slice(&(N as u32).to_le_bytes());
    nk.extend_from_slice(&(K as u32).to_le_bytes());
    enc.set_bytes(&nk, 5);
    enc.dispatch_thread_groups(
        MTLSize {
            width: N.div_ceil(16) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    enc.set_compute_pipeline_state(&wide_pipe);
    enc.set_buffer(&x3, 0, 0);
    enc.set_buffer(&w, 0, 1);
    enc.set_buffer(&s, 0, 2);
    enc.set_buffer(&b, 0, 3);
    enc.set_buffer(&y_wide, 0, 4);
    enc.set_bytes(&nk, 5);
    enc.dispatch_thread_groups(
        MTLSize {
            width: N.div_ceil(16) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    command.commit();
    command.wait_until_completed();

    let fast_out = y_fast.read_bytes(0, N * 4);
    let wide_out = y_wide.read_bytes(0, 3 * N * 4);
    for row in 0..N {
        assert_eq!(
            fast_out[row * 4..row * 4 + 4],
            wide_out[row * 4..row * 4 + 4],
            "fast row {row} != wide3 row 0"
        );
    }
}

#[test]
fn qmv_fused_wide3_dispatches_are_bit_exact_to_separate_calls() {
    const M: usize = 3;
    const K: usize = 128;
    const NS: [usize; 4] = [19, 33, 7, 24];
    let dev = MetalDevice::default();
    let library = dev.new_library(NAX_AFFINE_U4_SHADER);
    let separate = dev.new_compute_pipeline(&library.function_named("q4_qmv_wide3"));
    let fused = [
        dev.new_compute_pipeline(&library.function_named("q4_qmv_fused2_wide3")),
        dev.new_compute_pipeline(&library.function_named("q4_qmv_fused3_wide3")),
        dev.new_compute_pipeline(&library.function_named("q4_qmv_fused4_wide3")),
    ];
    let x_values: Vec<u16> = (0..M * K)
        .map(|index| {
            let row = index / K;
            let col = index % K;
            f32_to_bf16((col as f32 - 59.0 + row as f32 * 11.0) * 0.0031)
        })
        .collect();
    let x = dev.new_buffer(M * K * 2);
    x.write_bytes(
        0,
        &x_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    );

    let packed: Vec<_> = NS
        .iter()
        .enumerate()
        .map(|(projection, n)| {
            let values: Vec<u32> = (0..n * K / 8)
                .map(|index| {
                    (0..8).fold(0u32, |word, nibble| {
                        word | (((index * (projection + 3) + nibble * 5 + projection) % 16) as u32)
                            << (nibble * 4)
                    })
                })
                .collect();
            let buffer = dev.new_buffer(values.len() * 4);
            buffer.write_bytes(
                0,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            buffer
        })
        .collect();
    let scales: Vec<_> = NS
        .iter()
        .enumerate()
        .map(|(projection, n)| {
            let values = vec![f32_to_bf16(0.009 + projection as f32 * 0.001); n * K / 64];
            let buffer = dev.new_buffer(values.len() * 2);
            buffer.write_bytes(
                0,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            buffer
        })
        .collect();
    let biases: Vec<_> = NS
        .iter()
        .enumerate()
        .map(|(projection, n)| {
            let values = vec![f32_to_bf16(-0.07 + projection as f32 * 0.004); n * K / 64];
            let buffer = dev.new_buffer(values.len() * 2);
            buffer.write_bytes(
                0,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            buffer
        })
        .collect();
    let expected: Vec<_> = NS.iter().map(|n| dev.new_buffer(M * n * 4)).collect();
    let queue = dev.new_command_queue();
    let command = queue.new_command_buffer();
    let enc = command.compute_compute_encoder();
    for projection in 0..4 {
        enc.set_compute_pipeline_state(&separate);
        enc.set_buffer(&x, 0, 0);
        enc.set_buffer(&packed[projection], 0, 1);
        enc.set_buffer(&scales[projection], 0, 2);
        enc.set_buffer(&biases[projection], 0, 3);
        enc.set_buffer(&expected[projection], 0, 4);
        let mut dims = Vec::with_capacity(8);
        dims.extend_from_slice(&(NS[projection] as u32).to_le_bytes());
        dims.extend_from_slice(&(K as u32).to_le_bytes());
        enc.set_bytes(&dims, 5);
        enc.dispatch_thread_groups(
            MTLSize {
                width: NS[projection].div_ceil(16) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }
    enc.end_encoding();
    command.commit();
    command.wait_until_completed();

    for count in 2..=4 {
        let actual: Vec<_> = NS[..count]
            .iter()
            .map(|n| dev.new_buffer(M * n * 4))
            .collect();
        let command = queue.new_command_buffer();
        let enc = command.compute_compute_encoder();
        enc.set_compute_pipeline_state(&fused[count - 2]);
        enc.set_buffer(&x, 0, 0);
        let mut dims = Vec::with_capacity((count + 1) * 4);
        for projection in 0..count {
            let base = 1 + projection as u64 * 4;
            enc.set_buffer(&packed[projection], 0, base);
            enc.set_buffer(&scales[projection], 0, base + 1);
            enc.set_buffer(&biases[projection], 0, base + 2);
            enc.set_buffer(&actual[projection], 0, base + 3);
            dims.extend_from_slice(&(NS[projection] as u32).to_le_bytes());
        }
        dims.extend_from_slice(&(K as u32).to_le_bytes());
        enc.set_bytes(&dims, 1 + count as u64 * 4);
        enc.dispatch_thread_groups(
            MTLSize {
                width: NS[..count].iter().copied().max().unwrap().div_ceil(16) as u64,
                height: count as u64,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        command.commit();
        command.wait_until_completed();
        for projection in 0..count {
            assert_eq!(
                actual[projection].read_bytes(0, M * NS[projection] * 4),
                expected[projection].read_bytes(0, M * NS[projection] * 4),
                "fused{count}_wide3 projection {projection} differs"
            );
        }
    }
}

#[test]
fn qmv_fused_dispatches_match_separate_qmv_calls() {
    const K: usize = 128;
    const NS: [usize; 4] = [19, 33, 7, 24];
    let x_bf16: Vec<u16> = (0..K)
        .map(|col| f32_to_bf16((col as f32 - 59.0) * 0.0031))
        .collect();
    let dev = MetalDevice::default();
    let library = dev.new_library(NAX_AFFINE_U4_SHADER);
    let separate = dev.new_compute_pipeline(&library.function_named("q4_qmv_fast"));
    let fused = [
        dev.new_compute_pipeline(&library.function_named("q4_qmv_fused2")),
        dev.new_compute_pipeline(&library.function_named("q4_qmv_fused3")),
        dev.new_compute_pipeline(&library.function_named("q4_qmv_fused4")),
    ];
    let x = dev.new_buffer(K * 2);
    x.write_bytes(
        0,
        &x_bf16
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    );

    let packed_values: Vec<Vec<u32>> = NS
        .iter()
        .enumerate()
        .map(|(projection, n)| {
            (0..n * K / 8)
                .map(|index| {
                    (0..8).fold(0u32, |word, nibble| {
                        word | (((index * (projection + 3) + nibble * 5 + projection) % 16) as u32)
                            << (nibble * 4)
                    })
                })
                .collect()
        })
        .collect();
    let scale_values: Vec<Vec<u16>> = NS
        .iter()
        .enumerate()
        .map(|(projection, n)| vec![f32_to_bf16(0.009 + projection as f32 * 0.001); n * K / 64])
        .collect();
    let bias_values: Vec<Vec<u16>> = NS
        .iter()
        .enumerate()
        .map(|(projection, n)| vec![f32_to_bf16(-0.07 + projection as f32 * 0.004); n * K / 64])
        .collect();
    let packed: Vec<_> = packed_values
        .iter()
        .map(|values| {
            let buffer = dev.new_buffer(values.len() * 4);
            buffer.write_bytes(
                0,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            buffer
        })
        .collect();
    let scales: Vec<_> = scale_values
        .iter()
        .map(|values| {
            let buffer = dev.new_buffer(values.len() * 2);
            buffer.write_bytes(
                0,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            buffer
        })
        .collect();
    let biases: Vec<_> = bias_values
        .iter()
        .map(|values| {
            let buffer = dev.new_buffer(values.len() * 2);
            buffer.write_bytes(
                0,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            buffer
        })
        .collect();
    let expected: Vec<_> = NS.iter().map(|n| dev.new_buffer(n * 4)).collect();
    let queue = dev.new_command_queue();
    let command = queue.new_command_buffer();
    let enc = command.compute_compute_encoder();
    for projection in 0..4 {
        enc.set_compute_pipeline_state(&separate);
        enc.set_buffer(&x, 0, 0);
        enc.set_buffer(&packed[projection], 0, 1);
        enc.set_buffer(&scales[projection], 0, 2);
        enc.set_buffer(&biases[projection], 0, 3);
        enc.set_buffer(&expected[projection], 0, 4);
        let mut dims = Vec::with_capacity(8);
        dims.extend_from_slice(&(NS[projection] as u32).to_le_bytes());
        dims.extend_from_slice(&(K as u32).to_le_bytes());
        enc.set_bytes(&dims, 5);
        enc.dispatch_thread_groups(
            MTLSize {
                width: NS[projection].div_ceil(16) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }
    enc.end_encoding();
    command.commit();
    command.wait_until_completed();

    for count in 2..=4 {
        let actual: Vec<_> = NS[..count].iter().map(|n| dev.new_buffer(n * 4)).collect();
        let command = queue.new_command_buffer();
        let enc = command.compute_compute_encoder();
        enc.set_compute_pipeline_state(&fused[count - 2]);
        enc.set_buffer(&x, 0, 0);
        let mut dims = Vec::with_capacity((count + 1) * 4);
        for projection in 0..count {
            let base = 1 + projection as u64 * 4;
            enc.set_buffer(&packed[projection], 0, base);
            enc.set_buffer(&scales[projection], 0, base + 1);
            enc.set_buffer(&biases[projection], 0, base + 2);
            enc.set_buffer(&actual[projection], 0, base + 3);
            dims.extend_from_slice(&(NS[projection] as u32).to_le_bytes());
        }
        dims.extend_from_slice(&(K as u32).to_le_bytes());
        enc.set_bytes(&dims, 1 + count as u64 * 4);
        enc.dispatch_thread_groups(
            MTLSize {
                width: NS[..count].iter().copied().max().unwrap().div_ceil(16) as u64,
                height: count as u64,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        command.commit();
        command.wait_until_completed();
        for projection in 0..count {
            assert_eq!(
                actual[projection].read_bytes(0, NS[projection] * 4),
                expected[projection].read_bytes(0, NS[projection] * 4),
                "fused{count} projection {projection} differs"
            );
        }
    }
}
