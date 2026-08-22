// Parity tests for the flash-style sliced decode attention (sdpa_decode_partial +
// sdpa_decode_final) against BOTH the original scalar online-softmax kernel
// (sdpa_decode_streaming) and a CPU softmax reference, across several contexts.
//
// Key invariant: the slice partition is a pure function of each row's own
// context (fixed TOKENS_PER_SLICE chunks), so a row produced in isolation
// (batch=1) must be BIT-IDENTICAL to the same row produced inside a batch-3
// dispatch. That is what keeps M=1 target decode and M=3 speculative verify
// argmax-consistent at any context length.
use lisa_rs::device::metal::{ComputePipeline, MetalBuffer, MTLSize, MetalDevice};
use lisa_rs::format::dtype::{bf16_to_f32, f32_to_bf16};
use lisa_rs::kernels::linear::ATTENTION_SHADER;

const QHEADS: usize = 24;
const KVHEADS: usize = 4;
const HD: usize = 256;
const OSTRIDE: usize = QHEADS * HD;
const TOKENS_PER_SLICE: usize = 64;
const LAYOUT_SLICES: usize = 256;

fn gen_f16(seed: usize, items: usize) -> Vec<u16> {
    (0..items)
        .map(|index| f32_to_bf16(((index * seed * 13 % 131) as f32 - 65.0) * 0.01 + (seed % 7) as f32))
        .collect()
}

fn pack(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>()
}

fn run_sliced(
    dev: &MetalDevice,
    partial_pipe: &ComputePipeline,
    final_pipe: &ComputePipeline,
    query: &MetalBuffer,
    keys: &MetalBuffer,
    values: &MetalBuffer,
    partial_out: &MetalBuffer,
    stat: &MetalBuffer,
    sliced_out: &MetalBuffer,
    base: usize,
    batch: usize,
) {
    let grid_slices = ((base + batch) / TOKENS_PER_SLICE
        + if (base + batch) % TOKENS_PER_SLICE == 0 { 0 } else { 1 })
        .min(LAYOUT_SLICES);
    let capacity = base + batch;

    let queue = dev.new_command_queue();
    let command = queue.new_command_buffer();
    let enc = command.compute_compute_encoder();

    enc.set_compute_pipeline_state(partial_pipe);
    enc.set_buffer(query, 0, 0);
    enc.set_buffer(keys, 0, 1);
    enc.set_buffer(values, 0, 2);
    enc.set_buffer(partial_out, 0, 3);
    enc.set_buffer(stat, 0, 4);
    enc.set_bytes(&(base as u32).to_le_bytes(), 5);
    enc.set_bytes(&(capacity as u32).to_le_bytes(), 6);
    enc.set_bytes(&(batch as u32).to_le_bytes(), 7);
    enc.dispatch_thread_groups(
        MTLSize { width: QHEADS as u64, height: batch as u64, depth: grid_slices as u64 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );

    enc.set_compute_pipeline_state(final_pipe);
    enc.set_buffer(partial_out, 0, 0);
    enc.set_buffer(stat, 0, 1);
    enc.set_buffer(sliced_out, 0, 2);
    enc.set_bytes(&(base as u32).to_le_bytes(), 3);
    enc.dispatch_thread_groups(
        MTLSize { width: QHEADS as u64, height: batch as u64, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );

    enc.end_encoding();
    command.commit();
    command.wait_until_completed();
}

fn run_orig(
    dev: &MetalDevice,
    pipe: &ComputePipeline,
    query: &MetalBuffer,
    keys: &MetalBuffer,
    values: &MetalBuffer,
    out: &MetalBuffer,
    ctx: usize,
) -> Vec<u8> {
    let queue = dev.new_command_queue();
    let command = queue.new_command_buffer();
    let enc = command.compute_compute_encoder();
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(query, 0, 0);
    enc.set_buffer(keys, 0, 1);
    enc.set_buffer(values, 0, 2);
    enc.set_buffer(out, 0, 3);
    enc.set_bytes(&(ctx as u32).to_le_bytes(), 4);
    enc.set_bytes(&(ctx as u32).to_le_bytes(), 5);
    enc.dispatch_thread_groups(
        MTLSize { width: QHEADS as u64, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
    enc.end_encoding();
    command.commit();
    command.wait_until_completed();
    out.read_bytes(0, OSTRIDE * 4)
}

fn cpu_expected(query_values: &[u16], keys_values: &[u16], values_values: &[u16], ctx: usize, head: usize, dim: usize) -> f32 {
    let kv_head = head / (QHEADS / KVHEADS);
    let scores: Vec<f32> = (0..ctx)
        .map(|token| {
            (0..HD).fold(0.0, |sum, c| {
                sum
                    + bf16_to_f32(query_values[head * HD + c])
                        * bf16_to_f32(keys_values[(kv_head * ctx + token) * HD + c])
            }) * 0.0625
        })
        .collect();
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = scores.iter().map(|s| (*s - max).exp()).collect();
    let denom: f32 = weights.iter().sum();
    (0..ctx).fold(0.0, |sum, token| sum + weights[token]
        * bf16_to_f32(values_values[(kv_head * ctx + token) * HD + dim])) / denom
}

#[test]
fn sliced_matches_cpu_reference_and_original() {
    let dev = MetalDevice::default();
    let library = dev.new_library(ATTENTION_SHADER);
    let partial_pipe = dev.new_compute_pipeline(&library.function_named("sdpa_decode_partial"));
    let final_pipe = dev.new_compute_pipeline(&library.function_named("sdpa_decode_final"));
    let orig_pipe = dev.new_compute_pipeline(&library.function_named("sdpa_decode_streaming"));

    for ctx in [1, 16, 63, 64, 129, 1000] {
        let capacity = ctx;
        let query_values = gen_f16(1, OSTRIDE);
        let keys_values = gen_f16(2, KVHEADS * capacity * HD);
        let values_values = gen_f16(3, KVHEADS * capacity * HD);
        let query = dev.new_buffer(3 * OSTRIDE * 2);
        let keys = dev.new_buffer(keys_values.len() * 2);
        let values = dev.new_buffer(values_values.len() * 2);
        let partial_out = dev.new_buffer(QHEADS * LAYOUT_SLICES * HD * 4);
        let stat = dev.new_buffer(QHEADS * LAYOUT_SLICES * 2 * 4);
        let sliced_out = dev.new_buffer(3 * OSTRIDE * 4);
        let orig_out = dev.new_buffer(OSTRIDE * 4);
        query.write_bytes(0, &pack(&query_values.repeat(3)));
        keys.write_bytes(0, &pack(&keys_values));
        values.write_bytes(0, &pack(&values_values));

        // standalone row (base=ctx-1, batch=1) attending over ctx keys
        run_sliced(&dev, &partial_pipe, &final_pipe,
            &query, &keys, &values, &partial_out, &stat, &sliced_out,
            ctx - 1, 1);
        let standalone = sliced_out.read_bytes(0, OSTRIDE * 4);

        // batch-3 dispatch whose row 2 attends over exactly ctx keys (base=ctx-3)
        let (batch3_row2, batch3_ok) = if ctx >= 3 {
            run_sliced(&dev, &partial_pipe, &final_pipe,
                &query, &keys, &values, &partial_out, &stat, &sliced_out,
                ctx - 3, 3);
            (Some(sliced_out.read_bytes((2 * OSTRIDE * 4).try_into().unwrap(), OSTRIDE * 4)), true)
        } else {
            (None, false)
        };

        // original scalar kernel for the same ctx row
        let orig = run_orig(&dev, &orig_pipe, &query, &keys, &values, &orig_out, ctx);

        for head in 0..QHEADS {
            for dim in (0..HD).step_by(37) {
                let off = (head * HD + dim) * 4;
                let e = cpu_expected(&query_values, &keys_values, &values_values, ctx, head, dim);
                let gs = f32::from_le_bytes(standalone[off..off + 4].try_into().unwrap());
                let go = f32::from_le_bytes(orig[off..off + 4].try_into().unwrap());
                assert!((gs - e).abs() < 1e-4, "ctx={ctx} head={head} dim={dim}: sliced {gs} vs cpu {e}");
                assert!((gs - go).abs() < 1e-5,
                    "ctx={ctx} head={head} dim={dim}: sliced {gs} vs orig {go}");
                if let Some(b3) = &batch3_row2 {
                    let g3 = f32::from_le_bytes(b3[off..off + 4].try_into().unwrap());
                    assert_eq!(gs.to_bits(), g3.to_bits(),
                        "ctx={ctx} head={head} dim={dim}: batch-3 row2 must be bit-identical to standalone ({gs} vs {g3})");
                }
            }
        }
    }
}