use lisa_rs::device::metal::{MTLSize, MetalDevice};
use lisa_rs::format::dtype::{bf16_to_f32, f32_to_bf16};
use lisa_rs::kernels::linear::ATTENTION_SHADER;

const L: usize = 16;
const QHEADS: usize = 24;
const KVHEADS: usize = 4;
const HD: usize = 256;
const QSTRIDE: usize = QHEADS * HD * 2;
const KSTRIDE: usize = KVHEADS * HD;
const OSTRIDE: usize = QHEADS * HD;
const KOUTSTRIDE: usize = 2048;

#[test]
fn persistent_cache_matches_prefill_last_token() {
    let dev = MetalDevice::default();
    let library = dev.new_library(ATTENTION_SHADER);
    let q_prefill = dev.new_compute_pipeline(&library.function_named("qk_norm_gate_rope"));
    let k_prefill = dev.new_compute_pipeline(&library.function_named("k_norm_rope"));
    let sdpa_prefill = dev.new_compute_pipeline(&library.function_named("sdpa_scalar"));
    let q_decode = dev.new_compute_pipeline(&library.function_named("q_norm_gate_rope_decode"));
    let kv_store = dev.new_compute_pipeline(&library.function_named("kv_cache_store_decode"));
    let sdpa_decode = dev.new_compute_pipeline(&library.function_named("sdpa_decode_streaming"));

    let q = dev.new_buffer(L * QSTRIDE * 4);
    let k = dev.new_buffer(L * KSTRIDE * 4);
    let v = dev.new_buffer(L * KSTRIDE * 4);
    let gain = dev.new_buffer(HD * 2);
    let query_prefill = dev.new_buffer(L * OSTRIDE * 2);
    let gate_prefill = dev.new_buffer(L * OSTRIDE * 4);
    let key_prefill = dev.new_buffer(L * KOUTSTRIDE * 2);
    let out_prefill = dev.new_buffer(L * OSTRIDE * 4);
    let query_decode = dev.new_buffer(OSTRIDE * 2);
    let gate_decode = dev.new_buffer(OSTRIDE * 4);
    let key_cache = dev.new_buffer(KVHEADS * L * HD * 2);
    let value_cache = dev.new_buffer(KVHEADS * L * HD * 2);
    let out_decode = dev.new_buffer(OSTRIDE * 4);

    let q_values: Vec<u8> = (0..L * QSTRIDE)
        .flat_map(|index| (((index % 113) as f32 - 56.0) * 0.01).to_le_bytes())
        .collect();
    let k_values: Vec<u8> = (0..L * KSTRIDE)
        .flat_map(|index| (((index % 97) as f32 - 48.0) * 0.012).to_le_bytes())
        .collect();
    let v_values: Vec<u8> = (0..L * KSTRIDE)
        .flat_map(|index| (((index % 89) as f32 - 44.0) * 0.008).to_le_bytes())
        .collect();
    let gains: Vec<u8> = (0..HD)
        .flat_map(|_| f32_to_bf16(1.0).to_le_bytes())
        .collect();
    q.write_bytes(0, &q_values);
    k.write_bytes(0, &k_values);
    v.write_bytes(0, &v_values);
    gain.write_bytes(0, &gains);

    let queue = dev.new_command_queue();
    let command = queue.new_command_buffer();
    let enc = command.compute_compute_encoder();
    enc.set_compute_pipeline_state(&q_prefill);
    enc.set_buffer(&q, 0, 0);
    enc.set_buffer(&gain, 0, 1);
    enc.set_buffer(&query_prefill, 0, 2);
    enc.set_buffer(&gate_prefill, 0, 3);
    enc.set_bytes(&(L as u32).to_le_bytes(), 4);
    enc.set_bytes(&1e-6f32.to_le_bytes(), 5);
    enc.dispatch_thread_groups(
        MTLSize {
            width: L as u64,
            height: QHEADS as u64,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    enc.set_compute_pipeline_state(&k_prefill);
    enc.set_buffer(&k, 0, 0);
    enc.set_buffer(&gain, 0, 1);
    enc.set_buffer(&key_prefill, 0, 2);
    enc.set_bytes(&(L as u32).to_le_bytes(), 3);
    enc.set_bytes(&1e-6f32.to_le_bytes(), 4);
    enc.dispatch_thread_groups(
        MTLSize {
            width: L as u64,
            height: KVHEADS as u64,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    enc.set_compute_pipeline_state(&sdpa_prefill);
    enc.set_buffer(&query_prefill, 0, 0);
    enc.set_buffer(&key_prefill, 0, 1);
    enc.set_buffer(&v, 0, 2);
    enc.set_buffer(&out_prefill, 0, 3);
    enc.set_bytes(&(L as u32).to_le_bytes(), 4);
    enc.dispatch_thread_groups(
        MTLSize {
            width: L as u64,
            height: QHEADS as u64,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );

    for position in 0..L {
        enc.set_compute_pipeline_state(&kv_store);
        enc.set_buffer(&k, (position * KSTRIDE * 4) as u64, 0);
        enc.set_buffer(&v, (position * KSTRIDE * 4) as u64, 1);
        enc.set_buffer(&gain, 0, 2);
        enc.set_buffer(&key_cache, 0, 3);
        enc.set_buffer(&value_cache, 0, 4);
        enc.set_bytes(&(position as u32).to_le_bytes(), 5);
        enc.set_bytes(&(L as u32).to_le_bytes(), 6);
        enc.set_bytes(&1e-6f32.to_le_bytes(), 7);
        enc.dispatch_thread_groups(
            MTLSize {
                width: KVHEADS as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }
    enc.set_compute_pipeline_state(&q_decode);
    enc.set_buffer(&q, ((L - 1) * QSTRIDE * 4) as u64, 0);
    enc.set_buffer(&gain, 0, 1);
    enc.set_buffer(&query_decode, 0, 2);
    enc.set_buffer(&gate_decode, 0, 3);
    enc.set_bytes(&((L - 1) as u32).to_le_bytes(), 4);
    enc.set_bytes(&1e-6f32.to_le_bytes(), 5);
    enc.dispatch_thread_groups(
        MTLSize {
            width: QHEADS as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    enc.set_compute_pipeline_state(&sdpa_decode);
    enc.set_buffer(&query_decode, 0, 0);
    enc.set_buffer(&key_cache, 0, 1);
    enc.set_buffer(&value_cache, 0, 2);
    enc.set_buffer(&out_decode, 0, 3);
    enc.set_bytes(&(L as u32).to_le_bytes(), 4);
    enc.set_bytes(&(L as u32).to_le_bytes(), 5);
    enc.dispatch_thread_groups(
        MTLSize {
            width: QHEADS as u64,
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
    command.commit();
    command.wait_until_completed();

    let expected = out_prefill.read_bytes(((L - 1) * OSTRIDE * 4) as u64, OSTRIDE * 4);
    let actual = out_decode.read_bytes(0, OSTRIDE * 4);
    let max_diff = expected
        .chunks_exact(4)
        .zip(actual.chunks_exact(4))
        .map(|(a, b)| {
            (f32::from_le_bytes([a[0], a[1], a[2], a[3]])
                - f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .abs()
        })
        .fold(0.0, f32::max);
    assert!(max_diff < 2e-5, "cached decode drifted by {max_diff}");
}

#[test]
fn block3_sdpa_is_causal_against_old_and_new_cache_rows() {
    const CAPACITY: usize = 6;
    const BASE: usize = 3;
    let dev = MetalDevice::default();
    let pipe = dev.new_compute_pipeline(
        &dev.new_library(ATTENTION_SHADER)
            .function_named("sdpa_decode_streaming_block3"),
    );
    let query_values: Vec<u16> = (0..3 * OSTRIDE)
        .map(|index| f32_to_bf16(((index % 37) as f32 - 18.0) * 0.01))
        .collect();
    let cache_values: Vec<u16> = (0..KVHEADS * CAPACITY * HD)
        .map(|index| f32_to_bf16(((index % 41) as f32 - 20.0) * 0.008))
        .collect();
    let value_values: Vec<u16> = (0..KVHEADS * CAPACITY * HD)
        .map(|index| f32_to_bf16(((index % 43) as f32 - 21.0) * 0.006))
        .collect();
    let query = dev.new_buffer(query_values.len() * 2);
    let keys = dev.new_buffer(cache_values.len() * 2);
    let values = dev.new_buffer(value_values.len() * 2);
    let output = dev.new_buffer(3 * OSTRIDE * 4);
    query.write_bytes(
        0,
        &query_values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    keys.write_bytes(
        0,
        &cache_values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    values.write_bytes(
        0,
        &value_values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let queue = dev.new_command_queue();
    let command = queue.new_command_buffer();
    let enc = command.compute_compute_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(&query, 0, 0);
    enc.set_buffer(&keys, 0, 1);
    enc.set_buffer(&values, 0, 2);
    enc.set_buffer(&output, 0, 3);
    enc.set_bytes(&(BASE as u32).to_le_bytes(), 4);
    enc.set_bytes(&(CAPACITY as u32).to_le_bytes(), 5);
    enc.dispatch_thread_groups(
        MTLSize {
            width: QHEADS as u64,
            height: 3,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    command.commit();
    command.wait_until_completed();
    let actual = output.read_bytes(0, 3 * OSTRIDE * 4);
    for row in 0..3 {
        for head in 0..QHEADS {
            let kv_head = head / (QHEADS / KVHEADS);
            let context = BASE + row + 1;
            let scores: Vec<f32> = (0..context)
                .map(|token| {
                    (0..HD).fold(0.0, |sum, dim| {
                        sum + bf16_to_f32(query_values[row * OSTRIDE + head * HD + dim])
                            * bf16_to_f32(cache_values[(kv_head * CAPACITY + token) * HD + dim])
                    }) * 0.0625
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|score| (*score - max).exp()).collect();
            let denominator: f32 = weights.iter().sum();
            for dim in [0, 73, HD - 1] {
                let expected = (0..context).fold(0.0, |sum, token| {
                    sum + weights[token]
                        * bf16_to_f32(value_values[(kv_head * CAPACITY + token) * HD + dim])
                }) / denominator;
                let offset = (row * OSTRIDE + head * HD + dim) * 4;
                let got = f32::from_le_bytes(actual[offset..offset + 4].try_into().unwrap());
                assert!(
                    (got - expected).abs() < 2e-5,
                    "row={row} head={head} dim={dim}: {got} != {expected}"
                );
            }
        }
    }
}
