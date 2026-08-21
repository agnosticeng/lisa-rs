use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::device::metal::{
    CommandQueue, ComputeEncoder, ComputePipeline, MTLSize, MetalBuffer, MetalDevice,
};
use crate::format::dtype::{Dtype, bf16_to_f32, f32_to_bf16};
use crate::kernels::linear::{
    ATTENTION_SHADER, EMBED_SHADER, GDN_SHADER, MLP_SHADER, NAX_AFFINE_U4_SHADER, NORM_SHADER,
};

use super::weights::{WeightIndex, WeightSlot};
use super::{HIDDEN, LAYERS, LayerKind, VOCAB, layer_kind};

const INTERMEDIATE: usize = 17_408;
const GDN_LAYERS: usize = 48;
const ATTN_LAYERS: usize = 16;
const GDN_KEY_HEADS: usize = 16;
const GDN_VALUE_HEADS: usize = 48;
const GDN_HEAD_DIM: usize = 128;
const GDN_KEY_DIM: usize = GDN_KEY_HEADS * GDN_HEAD_DIM;
const GDN_VALUE_DIM: usize = GDN_VALUE_HEADS * GDN_HEAD_DIM;
const GDN_CONV_DIM: usize = 2 * GDN_KEY_DIM + GDN_VALUE_DIM;
const GDN_STATE_ELEMENTS: usize = GDN_VALUE_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM;
const ATTN_Q_OUT: usize = 12_288;
const ATTN_KV_OUT: usize = 1_024;
const ATTN_HEADS: usize = 24;
const ATTN_KV_HEADS: usize = 4;
const ATTN_HEAD_DIM: usize = 256;
const ATTN_OUT: usize = ATTN_HEADS * ATTN_HEAD_DIM;
const EPS: f32 = 1e-6;

/// Maximum number of prompt tokens processed in one fused prefill command
/// buffer. All prefill scratch tensors are sized for this batch and actual
/// chunk sends use the same layout offsets (zeros beyond the real rows).
const PREFILL_BATCH: usize = 64;

#[derive(Clone, Copy)]
struct TensorRef {
    shard: usize,
    offset: u64,
}

#[derive(Clone, Copy)]
struct Q4Linear {
    weight: TensorRef,
    scales: TensorRef,
    biases: TensorRef,
    input: usize,
    output: usize,
}

struct MlpWeights {
    gate: Q4Linear,
    up: Q4Linear,
    down: Q4Linear,
}

struct CommonLayer {
    input_norm: TensorRef,
    post_norm: TensorRef,
    mlp: MlpWeights,
}

struct GdnWeights {
    qkv: Q4Linear,
    z: Q4Linear,
    a: Q4Linear,
    b: Q4Linear,
    output: Q4Linear,
    conv_weight: TensorRef,
    a_log: TensorRef,
    dt_bias: TensorRef,
    norm: TensorRef,
    state_index: usize,
}

struct AttentionWeights {
    q: Q4Linear,
    k: Q4Linear,
    v: Q4Linear,
    output: Q4Linear,
    q_norm: TensorRef,
    k_norm: TensorRef,
    cache_index: usize,
}

enum BranchWeights {
    Gdn(GdnWeights),
    Attention(AttentionWeights),
}

struct LayerWeights {
    common: CommonLayer,
    branch: BranchWeights,
}

struct Pipelines {
    qmv: ComputePipeline,
    qmv_fused2: ComputePipeline,
    qmv_fused3: ComputePipeline,
    qmv_fused4: ComputePipeline,
    qmv_fused2_wide3: ComputePipeline,
    qmv_fused3_wide3: ComputePipeline,
    qmv_fused4_wide3: ComputePipeline,
    qmv_wide3: ComputePipeline,
    q4_gemm: ComputePipeline,
    embed: ComputePipeline,
    rms: ComputePipeline,
    residual: ComputePipeline,
    clear: ComputePipeline,
    copy_bytes: ComputePipeline,
    cast_bf16: ComputePipeline,
    rms_stride: ComputePipeline,
    silu: ComputePipeline,
    gdn_conv: ComputePipeline,
    gdn_update_conv: ComputePipeline,
    gdn_recurrence: ComputePipeline,
    gdn_gate: ComputePipeline,
    attn_q: ComputePipeline,
    attn_store: ComputePipeline,
    attn_sdpa: ComputePipeline,
    attn_gate: ComputePipeline,
    attn_q_block3: ComputePipeline,
    attn_store_block3: ComputePipeline,
    attn_sdpa_block3: ComputePipeline,
    attn_q_prefill: ComputePipeline,
    attn_store_prefill: ComputePipeline,
    attn_sdpa_prefill: ComputePipeline,
    argmax: ComputePipeline,
}

#[derive(Clone, Copy)]
struct ScratchLayout {
    token: usize,
    hidden: usize,
    normalized: usize,
    branch_output: usize,
    residual: usize,
    post_norm: usize,
    mlp_gate: usize,
    mlp_up: usize,
    mlp_active: usize,
    projection: usize,
    aux0: usize,
    aux1: usize,
    small0: usize,
    small1: usize,
    conv_output: usize,
    q_norm: usize,
    k_norm: usize,
    beta: usize,
    decay: usize,
    state_output: usize,
    query: usize,
    attention_gate: usize,
    attention_output: usize,
    gated: usize,
    logits: usize,
    argmax_partial: usize,
    len: usize,
}

impl ScratchLayout {
    fn new(batch: usize) -> Self {
        let mut next = 0usize;
        let mut take = |bytes: usize| {
            next = next.div_ceil(16) * 16;
            let offset = next;
            next += bytes;
            offset
        };
        let token = take(batch * 4);
        let hidden = take(batch * HIDDEN * 2);
        let normalized = take(batch * HIDDEN * 2);
        let branch_output = take(batch * HIDDEN * 4);
        let residual = take(batch * HIDDEN * 2);
        let post_norm = take(batch * HIDDEN * 2);
        let mlp_gate = take(batch * INTERMEDIATE * 4);
        let mlp_up = take(batch * INTERMEDIATE * 4);
        let mlp_active = take(batch * INTERMEDIATE * 2);
        let projection = take(batch * ATTN_Q_OUT * 4);
        let aux0 = take(batch * GDN_VALUE_DIM * 4);
        let aux1 = take(batch * GDN_VALUE_DIM * 4);
        let small0 = take(batch * 64 * 4);
        let small1 = take(batch * 64 * 4);
        let conv_output = take(batch * GDN_CONV_DIM * 4);
        let q_norm = take(batch * GDN_KEY_DIM * 4);
        let k_norm = take(batch * GDN_KEY_DIM * 4);
        let beta = take(batch * GDN_VALUE_HEADS * 4);
        let decay = take(batch * GDN_VALUE_HEADS * 4);
        let state_output = take(batch * GDN_VALUE_DIM * 4);
        let query = take(batch * ATTN_OUT * 2);
        let attention_gate = take(batch * ATTN_OUT * 4);
        let attention_output = take(batch * ATTN_OUT * 4);
        let gated = take(batch * ATTN_OUT * 2);
        let logits = take(batch * VOCAB * 4);
        let argmax_partial = take(batch * VOCAB.div_ceil(256) * 2 * 4);
        Self {
            token,
            hidden,
            normalized,
            branch_output,
            residual,
            post_norm,
            mlp_gate,
            mlp_up,
            mlp_active,
            projection,
            aux0,
            aux1,
            small0,
            small1,
            conv_output,
            q_norm,
            k_norm,
            beta,
            decay,
            state_output,
            query,
            attention_gate,
            attention_output,
            gated,
            logits,
            argmax_partial,
            len: next,
        }
    }
}

/// Persistent M=1 native runner for the text-only Qwen3.8-27B target model.
pub struct QwenRunner {
    _device: MetalDevice,
    _index: WeightIndex,
    shards: Arc<Vec<MetalBuffer>>,
    queue: CommandQueue,
    pipelines: Pipelines,
    layers: Vec<LayerWeights>,
    embed: Q4Linear,
    final_norm: TensorRef,
    head: Q4Linear,
    scratch: MetalBuffer,
    scratch_layout: ScratchLayout,
    prefill_scratch: MetalBuffer,
    prefill_layout: ScratchLayout,
    gdn_state: MetalBuffer,
    conv_state: MetalBuffer,
    verify_gdn_slots: MetalBuffer,
    verify_conv_slots: MetalBuffer,
    key_cache: MetalBuffer,
    value_cache: MetalBuffer,
    capacity: usize,
    position: usize,
    verify_base_position: Option<usize>,
}

/// Ablation bitmask flags for `forward_token_ablate`.
pub const ABLATE_GDN: u32 = 1 << 0;
pub const ABLATE_ATTENTION: u32 = 1 << 1;
pub const ABLATE_MLP: u32 = 1 << 2;
pub const ABLATE_LM_HEAD: u32 = 1 << 3;
pub const ABLATE_READBACK: u32 = 1 << 4;

/// One target decode result. `hidden` is the last decoder state before the
/// target final norm, which is the conditioning input expected by Qwen MTP.
pub struct TargetOutput {
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub argmax: u32,
}

/// Wall-clock timing for one complete decoder layer in the synchronized M=1
/// diagnostic path.
#[derive(Debug)]
pub struct DiagnosticLayerTiming {
    pub layer: usize,
    pub kind: LayerKind,
    pub wall: Duration,
}

/// Synchronized wall-clock timings for one M=1 token. Every GPU section uses
/// its own command buffer and wait, so these numbers diagnose cost attribution
/// and intentionally do not represent normal decode throughput.
pub struct DiagnosticTokenProfile {
    pub output: TargetOutput,
    pub embedding_wall: Duration,
    pub layers: Vec<DiagnosticLayerTiming>,
    pub gdn_wall: Duration,
    pub attention_wall: Duration,
    pub final_norm_lm_head_wall: Duration,
    pub cpu_readback_argmax_wall: Duration,
    pub total_wall: Duration,
}

/// GPU-resident snapshot of the target's recurrent state. Attention cache rows
/// are not copied: restoring `position` makes speculative rows logically stale
/// and the next forwards overwrite them in place.
pub struct QwenStateCheckpoint {
    gdn_state: MetalBuffer,
    conv_state: MetalBuffer,
    position: usize,
}

impl QwenStateCheckpoint {
    pub fn position(&self) -> usize {
        self.position
    }
}

impl QwenRunner {
    pub fn load(snapshot: &Path, capacity: usize) -> Result<Self, String> {
        if capacity == 0 {
            return Err("KV capacity must be greater than zero".into());
        }
        if capacity > u32::MAX as usize {
            return Err("KV capacity exceeds the kernel index range".into());
        }
        let mut paths: Vec<PathBuf> = std::fs::read_dir(snapshot)
            .map_err(|error| format!("read snapshot {}: {error}", snapshot.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "safetensors")
            })
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(format!("no safetensors shards in {}", snapshot.display()));
        }

        let index = WeightIndex::open(&paths);
        let layers = build_layers(&index)?;
        let embed = q4(&index, "language_model.model.embed_tokens", HIDDEN, VOCAB)?;
        let final_norm = bf16(&index, "language_model.model.norm.weight", &[HIDDEN])?;
        let head = q4(&index, "language_model.lm_head", HIDDEN, VOCAB)?;

        let device = MetalDevice::default();
        let mut shards = Vec::with_capacity(index.shard_count());
        for shard in 0..index.shard_count() {
            let bytes = index
                .shard_bytes(shard)
                .ok_or_else(|| format!("missing shard {shard}"))?;
            let buffer = device.new_untracked_buffer(bytes.len());
            buffer.write_bytes(0, bytes);
            shards.push(buffer);
        }

        let pipelines = Pipelines::new(&device);
        let queue = device.new_command_queue();
        let scratch_layout = ScratchLayout::new(3);
        let scratch = device.new_buffer(scratch_layout.len);
        let prefill_layout = ScratchLayout::new(PREFILL_BATCH);
        let prefill_scratch = device.new_buffer(prefill_layout.len);
        let gdn_state = device.new_buffer(GDN_LAYERS * GDN_STATE_ELEMENTS * 4);
        let conv_state = device.new_buffer(GDN_LAYERS * 3 * GDN_CONV_DIM * 2);
        let verify_gdn_slots = device.new_buffer(3 * GDN_LAYERS * GDN_STATE_ELEMENTS * 4);
        let verify_conv_slots = device.new_buffer(3 * GDN_LAYERS * 3 * GDN_CONV_DIM * 2);
        let cache_bytes = ATTN_LAYERS
            .checked_mul(ATTN_KV_HEADS)
            .and_then(|value| value.checked_mul(capacity))
            .and_then(|value| value.checked_mul(ATTN_HEAD_DIM * 2))
            .ok_or_else(|| "KV capacity overflow".to_owned())?;
        let key_cache = device.new_buffer(cache_bytes);
        let value_cache = device.new_buffer(cache_bytes);
        let mut runner = Self {
            _device: device,
            _index: index,
            shards: Arc::new(shards),
            queue,
            pipelines,
            layers,
            embed,
            final_norm,
            head,
            scratch,
            scratch_layout,
            prefill_scratch,
            prefill_layout,
            gdn_state,
            conv_state,
            verify_gdn_slots,
            verify_conv_slots,
            key_cache,
            value_cache,
            capacity,
            position: 0,
            verify_base_position: None,
        };
        runner.reset_state();
        Ok(runner)
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn reset_state(&mut self) {
        let command = self.queue.new_command_buffer();
        let encoder = command.compute_compute_encoder();
        self.clear(
            &encoder,
            &self.gdn_state,
            GDN_LAYERS * GDN_STATE_ELEMENTS * 4,
        );
        self.clear(
            &encoder,
            &self.conv_state,
            GDN_LAYERS * 3 * GDN_CONV_DIM * 2,
        );
        let cache_len = ATTN_LAYERS * ATTN_KV_HEADS * self.capacity * ATTN_HEAD_DIM * 2;
        self.clear(&encoder, &self.key_cache, cache_len);
        self.clear(&encoder, &self.value_cache, cache_len);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        self.position = 0;
        self.verify_base_position = None;
    }

    /// Capture all mutable recurrent state with device-to-device copies.
    pub fn checkpoint_state(&self) -> QwenStateCheckpoint {
        let gdn_state = self._device.new_buffer(GDN_LAYERS * GDN_STATE_ELEMENTS * 4);
        let conv_state = self._device.new_buffer(GDN_LAYERS * 3 * GDN_CONV_DIM * 2);
        let command = self.queue.new_command_buffer();
        let encoder = command.compute_compute_encoder();
        self.copy_bytes(
            &encoder,
            &self.gdn_state,
            &gdn_state,
            GDN_LAYERS * GDN_STATE_ELEMENTS * 4,
        );
        self.copy_bytes(
            &encoder,
            &self.conv_state,
            &conv_state,
            GDN_LAYERS * 3 * GDN_CONV_DIM * 2,
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        QwenStateCheckpoint {
            gdn_state,
            conv_state,
            position: self.position,
        }
    }

    /// Restore a recurrent checkpoint. Future attention slots remain in the KV
    /// buffers but are excluded by the restored logical position.
    pub fn restore_state(&mut self, checkpoint: &QwenStateCheckpoint) -> Result<(), String> {
        if checkpoint.position > self.capacity {
            return Err(format!(
                "checkpoint position {} exceeds KV capacity {}",
                checkpoint.position, self.capacity
            ));
        }
        let command = self.queue.new_command_buffer();
        let encoder = command.compute_compute_encoder();
        self.copy_bytes(
            &encoder,
            &checkpoint.gdn_state,
            &self.gdn_state,
            GDN_LAYERS * GDN_STATE_ELEMENTS * 4,
        );
        self.copy_bytes(
            &encoder,
            &checkpoint.conv_state,
            &self.conv_state,
            GDN_LAYERS * 3 * GDN_CONV_DIM * 2,
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        self.position = checkpoint.position;
        self.verify_base_position = None;
        Ok(())
    }

    pub fn forward_token(&mut self, token: u32) -> Result<Vec<f32>, String> {
        self.forward_token_with_hidden(token)
            .map(|output| output.logits)
    }

    pub fn forward_token_with_hidden(&mut self, token: u32) -> Result<TargetOutput, String> {
        if usize::try_from(token).map_err(|_| "token does not fit usize")? >= VOCAB {
            return Err(format!("token {token} is outside vocabulary"));
        }
        if self.position >= self.capacity {
            return Err(format!("KV capacity {} exhausted", self.capacity));
        }
        let s = self.scratch_layout;
        self.scratch
            .write_bytes(s.token as u64, &token.to_le_bytes());
        let command = self.queue.new_command_buffer();
        let encoder = command.compute_compute_encoder();

        self.encode_embed(&encoder, 1);
        for layer in &self.layers {
            self.encode_norm(&encoder, s.hidden, layer.common.input_norm, s.normalized, 1);
            match &layer.branch {
                BranchWeights::Gdn(weights) => self.encode_gdn(&encoder, weights, 1, false),
                BranchWeights::Attention(weights) => self.encode_attention(&encoder, weights, 1),
            }
            self.encode_residual(&encoder, s.hidden, s.branch_output, s.residual, 1);
            self.encode_norm(&encoder, s.residual, layer.common.post_norm, s.post_norm, 1);
            self.encode_mlp(&encoder, &layer.common.mlp, 1);
            self.encode_residual(&encoder, s.residual, s.branch_output, s.hidden, 1);
        }
        self.encode_norm(&encoder, s.hidden, self.final_norm, s.normalized, 1);
        self.encode_qmv(&encoder, s.normalized, &self.head, s.logits, 1);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();

        let logits = read_f32(&self.scratch, s.logits, VOCAB);
        let hidden = read_bf16(&self.scratch, s.hidden, HIDDEN);
        let argmax = argmax(&logits);
        self.position += 1;
        Ok(TargetOutput {
            hidden,
            logits,
            argmax,
        })
    }

    /// Decode-only forward: same fused M=1 command buffer as
    /// `forward_token_with_hidden`, but the lm_head argmax is reduced on the GPU
    /// so the full-vocab logits are never read back. Returns `(argmax, hidden)`.
    pub fn forward_token_decode(&mut self, token: u32) -> Result<(u32, Vec<f32>), String> {
        if usize::try_from(token).map_err(|_| "token does not fit usize")? >= VOCAB {
            return Err(format!("token {token} is outside vocabulary"));
        }
        if self.position >= self.capacity {
            return Err(format!("KV capacity {} exhausted", self.capacity));
        }
        let s = self.scratch_layout;
        self.scratch
            .write_bytes(s.token as u64, &token.to_le_bytes());
        let command = self.queue.new_command_buffer();
        let encoder = command.compute_compute_encoder();

        self.encode_embed(&encoder, 1);
        for layer in &self.layers {
            self.encode_norm(&encoder, s.hidden, layer.common.input_norm, s.normalized, 1);
            match &layer.branch {
                BranchWeights::Gdn(weights) => self.encode_gdn(&encoder, weights, 1, false),
                BranchWeights::Attention(weights) => self.encode_attention(&encoder, weights, 1),
            }
            self.encode_residual(&encoder, s.hidden, s.branch_output, s.residual, 1);
            self.encode_norm(&encoder, s.residual, layer.common.post_norm, s.post_norm, 1);
            self.encode_mlp(&encoder, &layer.common.mlp, 1);
            self.encode_residual(&encoder, s.residual, s.branch_output, s.hidden, 1);
        }
        self.encode_norm(&encoder, s.hidden, self.final_norm, s.normalized, 1);
        self.encode_qmv(&encoder, s.normalized, &self.head, s.logits, 1);
        self.encode_argmax(&encoder, s.logits, 0);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();

        let argmax = self.read_argmax(0);
        let hidden = read_bf16(&self.scratch, s.hidden, HIDDEN);
        self.position += 1;
        Ok((argmax, hidden))
    }

    /// Ablation forward: identical fused M=1 command buffer to
    /// `forward_token_with_hidden`, but `mask` disables whole stages so their
    /// real (non-synchronized) GPU cost can be measured by difference. Output
    /// values are garbage when a stage is disabled; only wall-clock matters.
    pub fn forward_token_ablate(&mut self, token: u32, mask: u32) -> Result<TargetOutput, String> {
        if usize::try_from(token).map_err(|_| "token does not fit usize")? >= VOCAB {
            return Err(format!("token {token} is outside vocabulary"));
        }
        if self.position >= self.capacity {
            return Err(format!("KV capacity {} exhausted", self.capacity));
        }
        let s = self.scratch_layout;
        self.scratch
            .write_bytes(s.token as u64, &token.to_le_bytes());
        let command = self.queue.new_command_buffer();
        let encoder = command.compute_compute_encoder();

        self.encode_embed(&encoder, 1);
        for layer in &self.layers {
            self.encode_norm(&encoder, s.hidden, layer.common.input_norm, s.normalized, 1);
            match &layer.branch {
                BranchWeights::Gdn(weights) => {
                    if mask & ABLATE_GDN == 0 {
                        self.encode_gdn(&encoder, weights, 1, false)
                    }
                }
                BranchWeights::Attention(weights) => {
                    if mask & ABLATE_ATTENTION == 0 {
                        self.encode_attention(&encoder, weights, 1)
                    }
                }
            }
            self.encode_residual(&encoder, s.hidden, s.branch_output, s.residual, 1);
            self.encode_norm(&encoder, s.residual, layer.common.post_norm, s.post_norm, 1);
            if mask & ABLATE_MLP == 0 {
                self.encode_mlp(&encoder, &layer.common.mlp, 1);
            }
            self.encode_residual(&encoder, s.residual, s.branch_output, s.hidden, 1);
        }
        self.encode_norm(&encoder, s.hidden, self.final_norm, s.normalized, 1);
        if mask & ABLATE_LM_HEAD == 0 {
            self.encode_qmv(&encoder, s.normalized, &self.head, s.logits, 1);
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();

        let (logits, hidden, argmax) = if mask & ABLATE_READBACK != 0 {
            (Vec::new(), Vec::new(), 0)
        } else {
            let logits = read_f32(&self.scratch, s.logits, VOCAB);
            let hidden = read_bf16(&self.scratch, s.hidden, HIDDEN);
            let argmax = argmax(&logits);
            (logits, hidden, argmax)
        };
        self.position += 1;
        Ok(TargetOutput {
            hidden,
            logits,
            argmax,
        })
    }

    /// Profile one M=1 token with a command-buffer boundary and blocking wait
    /// around every reported GPU section. This is diagnostic instrumentation,
    /// not a throughput benchmark; the normal forward path remains fully fused
    /// into one command buffer.
    pub fn profile_token_m1_diagnostic(
        &mut self,
        token: u32,
    ) -> Result<DiagnosticTokenProfile, String> {
        if usize::try_from(token).map_err(|_| "token does not fit usize")? >= VOCAB {
            return Err(format!("token {token} is outside vocabulary"));
        }
        if self.position >= self.capacity {
            return Err(format!("KV capacity {} exhausted", self.capacity));
        }

        let s = self.scratch_layout;
        self.scratch
            .write_bytes(s.token as u64, &token.to_le_bytes());
        let total_start = Instant::now();
        let embedding_wall = self.timed_command(|encoder| self.encode_embed(encoder, 1));

        let mut layers = Vec::with_capacity(LAYERS);
        let mut gdn_wall = Duration::ZERO;
        let mut attention_wall = Duration::ZERO;
        for (index, layer) in self.layers.iter().enumerate() {
            let kind = layer_kind(index);
            let wall = self.timed_command(|encoder| {
                self.encode_norm(encoder, s.hidden, layer.common.input_norm, s.normalized, 1);
                match &layer.branch {
                    BranchWeights::Gdn(weights) => self.encode_gdn(encoder, weights, 1, false),
                    BranchWeights::Attention(weights) => self.encode_attention(encoder, weights, 1),
                }
                self.encode_residual(encoder, s.hidden, s.branch_output, s.residual, 1);
                self.encode_norm(encoder, s.residual, layer.common.post_norm, s.post_norm, 1);
                self.encode_mlp(encoder, &layer.common.mlp, 1);
                self.encode_residual(encoder, s.residual, s.branch_output, s.hidden, 1);
            });
            match kind {
                LayerKind::Gdn => gdn_wall += wall,
                LayerKind::Attention => attention_wall += wall,
            }
            layers.push(DiagnosticLayerTiming {
                layer: index,
                kind,
                wall,
            });
        }

        let final_norm_lm_head_wall = self.timed_command(|encoder| {
            self.encode_norm(encoder, s.hidden, self.final_norm, s.normalized, 1);
            self.encode_qmv(encoder, s.normalized, &self.head, s.logits, 1);
        });
        let readback_start = Instant::now();
        let logits = read_f32(&self.scratch, s.logits, VOCAB);
        let hidden = read_bf16(&self.scratch, s.hidden, HIDDEN);
        let argmax = argmax(&logits);
        let cpu_readback_argmax_wall = readback_start.elapsed();
        let total_wall = total_start.elapsed();
        self.position += 1;

        Ok(DiagnosticTokenProfile {
            output: TargetOutput {
                hidden,
                logits,
                argmax,
            },
            embedding_wall,
            layers,
            gdn_wall,
            attention_wall,
            final_norm_lm_head_wall,
            cpu_readback_argmax_wall,
            total_wall,
        })
    }

    /// Verify three consecutive tokens in one 64-layer batched pass. Call
    /// `commit_verified_prefix` with the accepted prefix plus replacement row.
    pub fn verify_block3(&mut self, tokens: [u32; 3]) -> Result<[TargetOutput; 3], String> {
        if tokens.iter().any(|token| *token as usize >= VOCAB) {
            return Err("verify token is outside vocabulary".into());
        }
        if self.position + 3 > self.capacity {
            return Err(format!("KV capacity {} exhausted", self.capacity));
        }
        let base = self.position;
        let s = self.scratch_layout;
        let token_bytes: Vec<u8> = tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect();
        self.scratch.write_bytes(s.token as u64, &token_bytes);
        let command = self.queue.new_command_buffer();
        let encoder = command.compute_compute_encoder();
        self.encode_embed(&encoder, 3);
        for layer in &self.layers {
            self.encode_norm(&encoder, s.hidden, layer.common.input_norm, s.normalized, 3);
            match &layer.branch {
                BranchWeights::Gdn(weights) => self.encode_gdn(&encoder, weights, 3, true),
                BranchWeights::Attention(weights) => self.encode_attention(&encoder, weights, 3),
            }
            self.encode_residual(&encoder, s.hidden, s.branch_output, s.residual, 3);
            self.encode_norm(&encoder, s.residual, layer.common.post_norm, s.post_norm, 3);
            self.encode_mlp(&encoder, &layer.common.mlp, 3);
            self.encode_residual(&encoder, s.residual, s.branch_output, s.hidden, 3);
        }
        self.encode_norm(&encoder, s.hidden, self.final_norm, s.normalized, 3);
        self.encode_qmv(&encoder, s.normalized, &self.head, s.logits, 3);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();

        let logits = read_f32(&self.scratch, s.logits, 3 * VOCAB);
        let hidden = read_bf16(&self.scratch, s.hidden, 3 * HIDDEN);
        self.position += 3;
        self.verify_base_position = Some(base);
        Ok(std::array::from_fn(|row| {
            let row_logits = logits[row * VOCAB..(row + 1) * VOCAB].to_vec();
            TargetOutput {
                argmax: argmax(&row_logits),
                logits: row_logits,
                hidden: hidden[row * HIDDEN..(row + 1) * HIDDEN].to_vec(),
            }
        }))
    }

    /// Verify-only variant of `verify_block3` for the speculative decode loop:
    /// reduces each of the three lm_head argmaxes on the GPU and reads back only
    /// `(argmax, hidden)` per row, never the full-vocab logits. Acceptance is
    /// argmax-only, so the logits are never needed on this path.
    pub fn verify_block3_decode(
        &mut self,
        tokens: [u32; 3],
    ) -> Result<[(u32, Vec<f32>); 3], String> {
        if tokens.iter().any(|token| *token as usize >= VOCAB) {
            return Err("verify token is outside vocabulary".into());
        }
        if self.position + 3 > self.capacity {
            return Err(format!("KV capacity {} exhausted", self.capacity));
        }
        let base = self.position;
        let s = self.scratch_layout;
        let token_bytes: Vec<u8> = tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect();
        self.scratch.write_bytes(s.token as u64, &token_bytes);
        let command = self.queue.new_command_buffer();
        let encoder = command.compute_compute_encoder();
        self.encode_embed(&encoder, 3);
        for layer in &self.layers {
            self.encode_norm(&encoder, s.hidden, layer.common.input_norm, s.normalized, 3);
            match &layer.branch {
                BranchWeights::Gdn(weights) => self.encode_gdn(&encoder, weights, 3, true),
                BranchWeights::Attention(weights) => self.encode_attention(&encoder, weights, 3),
            }
            self.encode_residual(&encoder, s.hidden, s.branch_output, s.residual, 3);
            self.encode_norm(&encoder, s.residual, layer.common.post_norm, s.post_norm, 3);
            self.encode_mlp(&encoder, &layer.common.mlp, 3);
            self.encode_residual(&encoder, s.residual, s.branch_output, s.hidden, 3);
        }
        self.encode_norm(&encoder, s.hidden, self.final_norm, s.normalized, 3);
        self.encode_qmv(&encoder, s.normalized, &self.head, s.logits, 3);
        for row in 0..3 {
            self.encode_argmax(&encoder, s.logits + row * VOCAB * 4, row);
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();

        let argmax0 = self.read_argmax(0);
        let argmax1 = self.read_argmax(1);
        let argmax2 = self.read_argmax(2);
        let hidden = read_bf16(&self.scratch, s.hidden, 3 * HIDDEN);
        self.position += 3;
        self.verify_base_position = Some(base);
        Ok([
            (argmax0, hidden[0..HIDDEN].to_vec()),
            (argmax1, hidden[HIDDEN..2 * HIDDEN].to_vec()),
            (argmax2, hidden[2 * HIDDEN..3 * HIDDEN].to_vec()),
        ])
    }

    pub fn commit_verified_prefix(&mut self, rows: usize) -> Result<(), String> {
        let base = self
            .verify_base_position
            .take()
            .ok_or_else(|| "no pending block-3 verification".to_owned())?;
        if !(1..=3).contains(&rows) {
            return Err("verified prefix must contain 1 to 3 rows".into());
        }
        if rows < 3 {
            let gdn_bytes = GDN_LAYERS * GDN_STATE_ELEMENTS * 4;
            let conv_bytes = GDN_LAYERS * 3 * GDN_CONV_DIM * 2;
            let command = self.queue.new_command_buffer();
            let encoder = command.compute_compute_encoder();
            self.copy_bytes_from_offset(
                &encoder,
                &self.verify_gdn_slots,
                (rows - 1) * gdn_bytes,
                &self.gdn_state,
                gdn_bytes,
            );
            self.copy_bytes_from_offset(
                &encoder,
                &self.verify_conv_slots,
                (rows - 1) * conv_bytes,
                &self.conv_state,
                conv_bytes,
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
        }
        self.position = base + rows;
        Ok(())
    }

    fn weight_buffer(&self, tensor: TensorRef) -> &MetalBuffer {
        &self.shards[tensor.shard]
    }

    fn timed_command(&self, encode: impl FnOnce(&ComputeEncoder)) -> Duration {
        let start = Instant::now();
        let command = self.queue.new_command_buffer();
        let encoder = command.compute_compute_encoder();
        encode(&encoder);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        start.elapsed()
    }

    fn clear(&self, encoder: &ComputeEncoder, buffer: &MetalBuffer, len: usize) {
        const ROW: u64 = 1 << 30;
        encoder.set_compute_pipeline_state(&self.pipelines.clear);
        encoder.set_buffer(buffer, 0, 0);
        encoder.set_bytes(&(len as u64).to_le_bytes(), 1);
        encoder.dispatch_threads(
            MTLSize {
                width: (len as u64).min(ROW),
                height: (len as u64).div_ceil(ROW),
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }

    fn copy_bytes(
        &self,
        encoder: &ComputeEncoder,
        input: &MetalBuffer,
        output: &MetalBuffer,
        len: usize,
    ) {
        self.copy_bytes_from_offset(encoder, input, 0, output, len);
    }

    fn copy_bytes_from_offset(
        &self,
        encoder: &ComputeEncoder,
        input: &MetalBuffer,
        input_offset: usize,
        output: &MetalBuffer,
        len: usize,
    ) {
        const ROW: u64 = 1 << 30;
        let chunks = len.div_ceil(16) as u64;
        encoder.set_compute_pipeline_state(&self.pipelines.copy_bytes);
        encoder.set_buffer(input, input_offset as u64, 0);
        encoder.set_buffer(output, 0, 1);
        encoder.set_bytes(&(len as u64).to_le_bytes(), 2);
        encoder.dispatch_threads(
            MTLSize {
                width: chunks.min(ROW),
                height: chunks.div_ceil(ROW),
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_embed(&self, encoder: &ComputeEncoder, batch: usize) {
        let s = self.scratch_layout;
        encoder.set_compute_pipeline_state(&self.pipelines.embed);
        encoder.set_buffer(&self.scratch, s.token as u64, 0);
        encoder.set_buffer(
            self.weight_buffer(self.embed.weight),
            self.embed.weight.offset,
            1,
        );
        encoder.set_buffer(
            self.weight_buffer(self.embed.scales),
            self.embed.scales.offset,
            2,
        );
        encoder.set_buffer(
            self.weight_buffer(self.embed.biases),
            self.embed.biases.offset,
            3,
        );
        encoder.set_buffer(&self.scratch, s.hidden as u64, 4);
        encoder.set_bytes(&(HIDDEN as u32).to_le_bytes(), 5);
        encoder.dispatch_threads(
            MTLSize {
                width: HIDDEN as u64,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_argmax(&self, encoder: &ComputeEncoder, input: usize, row: usize) {
        const TG: usize = 256;
        let partials = VOCAB.div_ceil(TG);
        let stride = partials * 2;
        let base = self.scratch_layout.argmax_partial + row * stride * 4;
        encoder.set_compute_pipeline_state(&self.pipelines.argmax);
        encoder.set_buffer(&self.scratch, input as u64, 0);
        encoder.set_buffer(&self.scratch, base as u64, 1);
        encoder.set_buffer(&self.scratch, (base + partials * 4) as u64, 2);
        encoder.set_bytes(&(VOCAB as u32).to_le_bytes(), 3);
        encoder.dispatch_thread_groups(
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
    }

    fn read_argmax(&self, row: usize) -> u32 {
        const TG: usize = 256;
        let partials = VOCAB.div_ceil(TG);
        let stride = partials * 2;
        let base = self.scratch_layout.argmax_partial + row * stride * 4;
        let maxes = read_f32(&self.scratch, base, partials);
        let idx_bytes = self.scratch.read_bytes((base + partials * 4) as u64, partials * 4);
        let mut best = f32::NEG_INFINITY;
        let mut best_index = 0u32;
        for (i, v) in maxes.iter().enumerate() {
            if *v > best {
                best = *v;
                best_index = u32::from_le_bytes(
                    idx_bytes[i * 4..i * 4 + 4].try_into().unwrap(),
                );
            }
        }
        best_index
    }

    fn encode_qmv(
        &self,
        encoder: &ComputeEncoder,
        input: usize,
        linear: &Q4Linear,
        output: usize,
        batch: usize,
    ) {
        debug_assert!(batch == 1 || batch == 3);
        encoder.set_compute_pipeline_state(if batch == 3 {
            &self.pipelines.qmv_wide3
        } else {
            &self.pipelines.qmv
        });
        encoder.set_buffer(&self.scratch, input as u64, 0);
        encoder.set_buffer(self.weight_buffer(linear.weight), linear.weight.offset, 1);
        encoder.set_buffer(self.weight_buffer(linear.scales), linear.scales.offset, 2);
        encoder.set_buffer(self.weight_buffer(linear.biases), linear.biases.offset, 3);
        encoder.set_buffer(&self.scratch, output as u64, 4);
        let mut dims = Vec::with_capacity(8);
        dims.extend_from_slice(&(linear.output as u32).to_le_bytes());
        dims.extend_from_slice(&(linear.input as u32).to_le_bytes());
        encoder.set_bytes(&dims, 5);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: linear.output.div_ceil(16) as u64,
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

    fn encode_qmv_fused(
        &self,
        encoder: &ComputeEncoder,
        input: usize,
        projections: &[(&Q4Linear, usize)],
        batch: usize,
    ) {
        debug_assert!(batch == 1 || batch == 3);
        let pipeline = match (projections.len(), batch) {
            (2, 1) => &self.pipelines.qmv_fused2,
            (3, 1) => &self.pipelines.qmv_fused3,
            (4, 1) => &self.pipelines.qmv_fused4,
            (2, 3) => &self.pipelines.qmv_fused2_wide3,
            (3, 3) => &self.pipelines.qmv_fused3_wide3,
            (4, 3) => &self.pipelines.qmv_fused4_wide3,
            _ => unreachable!("only fused2, fused3 and fused4 are supported"),
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(&self.scratch, input as u64, 0);
        let mut dims = Vec::with_capacity((projections.len() + 1) * 4);
        let mut widest = 0usize;
        for (index, (linear, output)) in projections.iter().enumerate() {
            debug_assert_eq!(linear.input, projections[0].0.input);
            let base = 1 + index as u64 * 4;
            encoder.set_buffer(
                self.weight_buffer(linear.weight),
                linear.weight.offset,
                base,
            );
            encoder.set_buffer(
                self.weight_buffer(linear.scales),
                linear.scales.offset,
                base + 1,
            );
            encoder.set_buffer(
                self.weight_buffer(linear.biases),
                linear.biases.offset,
                base + 2,
            );
            encoder.set_buffer(&self.scratch, *output as u64, base + 3);
            dims.extend_from_slice(&(linear.output as u32).to_le_bytes());
            widest = widest.max(linear.output);
        }
        dims.extend_from_slice(&(projections[0].0.input as u32).to_le_bytes());
        encoder.set_bytes(&dims, 1 + projections.len() as u64 * 4);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: widest.div_ceil(16) as u64,
                height: projections.len() as u64,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_norm(
        &self,
        encoder: &ComputeEncoder,
        input: usize,
        weight: TensorRef,
        output: usize,
        batch: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.rms);
        encoder.set_buffer(&self.scratch, input as u64, 0);
        encoder.set_buffer(self.weight_buffer(weight), weight.offset, 1);
        encoder.set_buffer(&self.scratch, output as u64, 2);
        encoder.set_bytes(&(HIDDEN as u32).to_le_bytes(), 3);
        encoder.set_bytes(&EPS.to_le_bytes(), 4);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: batch as u64,
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

    fn encode_residual(
        &self,
        encoder: &ComputeEncoder,
        input: usize,
        branch: usize,
        output: usize,
        batch: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.residual);
        encoder.set_buffer(&self.scratch, input as u64, 0);
        encoder.set_buffer(&self.scratch, branch as u64, 1);
        encoder.set_buffer(&self.scratch, output as u64, 2);
        encoder.set_bytes(&((batch * HIDDEN) as u32).to_le_bytes(), 3);
        encoder.dispatch_threads(
            MTLSize {
                width: (batch * HIDDEN) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_mlp(&self, encoder: &ComputeEncoder, weights: &MlpWeights, batch: usize) {
        let s = self.scratch_layout;
        self.encode_qmv_fused(
            encoder,
            s.post_norm,
            &[(&weights.gate, s.mlp_gate), (&weights.up, s.mlp_up)],
            batch,
        );
        encoder.set_compute_pipeline_state(&self.pipelines.silu);
        encoder.set_buffer(&self.scratch, s.mlp_gate as u64, 0);
        encoder.set_buffer(&self.scratch, s.mlp_up as u64, 1);
        encoder.set_buffer(&self.scratch, s.mlp_active as u64, 2);
        encoder.set_bytes(&((batch * INTERMEDIATE) as u32).to_le_bytes(), 3);
        encoder.dispatch_threads(
            MTLSize {
                width: (batch * INTERMEDIATE) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
        self.encode_qmv(encoder, s.mlp_active, &weights.down, s.branch_output, batch);
    }

    fn encode_gdn(
        &self,
        encoder: &ComputeEncoder,
        weights: &GdnWeights,
        batch: usize,
        capture: bool,
    ) {
        let s = self.scratch_layout;
        self.encode_qmv_fused(
            encoder,
            s.normalized,
            &[
                (&weights.qkv, s.projection),
                (&weights.z, s.aux0),
                (&weights.a, s.small0),
                (&weights.b, s.small1),
            ],
            batch,
        );

        let conv_state_offset = weights.state_index * 3 * GDN_CONV_DIM * 2;
        encoder.set_compute_pipeline_state(&self.pipelines.gdn_conv);
        encoder.set_buffer(&self.scratch, s.projection as u64, 0);
        encoder.set_buffer(&self.scratch, s.small1 as u64, 1);
        encoder.set_buffer(&self.scratch, s.small0 as u64, 2);
        encoder.set_buffer(
            self.weight_buffer(weights.conv_weight),
            weights.conv_weight.offset,
            3,
        );
        encoder.set_buffer(&self.conv_state, conv_state_offset as u64, 4);
        encoder.set_buffer(self.weight_buffer(weights.a_log), weights.a_log.offset, 5);
        encoder.set_buffer(
            self.weight_buffer(weights.dt_bias),
            weights.dt_bias.offset,
            6,
        );
        encoder.set_buffer(&self.scratch, s.conv_output as u64, 7);
        encoder.set_buffer(&self.scratch, s.q_norm as u64, 8);
        encoder.set_buffer(&self.scratch, s.k_norm as u64, 9);
        encoder.set_buffer(&self.scratch, s.beta as u64, 10);
        encoder.set_buffer(&self.scratch, s.decay as u64, 11);
        encoder.set_bytes(&u32_bytes(&[16, 48, 128, 128]), 12);
        let inv = 1.0f32 / (GDN_HEAD_DIM as f32).sqrt();
        encoder.set_bytes(&f32_bytes(&[inv * inv, inv]), 13);
        encoder.set_bytes(&(batch as u32).to_le_bytes(), 14);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 48,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );

        // The convolution must consume the old history before this update.
        encoder.set_compute_pipeline_state(&self.pipelines.gdn_update_conv);
        encoder.set_buffer(&self.scratch, s.projection as u64, 0);
        encoder.set_buffer(&self.conv_state, conv_state_offset as u64, 1);
        encoder.set_bytes(&(GDN_CONV_DIM as u32).to_le_bytes(), 2);
        encoder.set_bytes(&(batch as u32).to_le_bytes(), 3);
        if capture {
            let layer_offset = weights.state_index * 3 * GDN_CONV_DIM * 2;
            encoder.set_buffer(&self.verify_conv_slots, layer_offset as u64, 4);
            encoder.set_bytes(&((GDN_LAYERS * 3 * GDN_CONV_DIM) as u32).to_le_bytes(), 5);
        } else {
            encoder.set_buffer(&self.conv_state, conv_state_offset as u64, 4);
            encoder.set_bytes(&0u32.to_le_bytes(), 5);
        }
        encoder.dispatch_threads(
            MTLSize {
                width: GDN_CONV_DIM as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.pipelines.gdn_recurrence);
        encoder.set_buffer(&self.scratch, s.conv_output as u64, 0);
        encoder.set_buffer(&self.scratch, s.q_norm as u64, 1);
        encoder.set_buffer(&self.scratch, s.k_norm as u64, 2);
        encoder.set_buffer(&self.scratch, s.beta as u64, 3);
        encoder.set_buffer(&self.scratch, s.decay as u64, 4);
        encoder.set_buffer(
            &self.gdn_state,
            (weights.state_index * GDN_STATE_ELEMENTS * 4) as u64,
            5,
        );
        encoder.set_buffer(&self.scratch, s.state_output as u64, 6);
        encoder.set_bytes(&u32_bytes(&[48, 128, 128, 3]), 7);
        encoder.set_bytes(&(batch as u32).to_le_bytes(), 8);
        if capture {
            let layer_offset = weights.state_index * GDN_STATE_ELEMENTS * 4;
            encoder.set_buffer(&self.verify_gdn_slots, layer_offset as u64, 9);
            encoder.set_bytes(
                &((GDN_LAYERS * GDN_STATE_ELEMENTS) as u32).to_le_bytes(),
                10,
            );
        } else {
            encoder.set_buffer(
                &self.gdn_state,
                (weights.state_index * GDN_STATE_ELEMENTS * 4) as u64,
                9,
            );
            encoder.set_bytes(&0u32.to_le_bytes(), 10);
        }
        encoder.dispatch_threads(
            MTLSize {
                width: 32,
                height: 128,
                depth: 48,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.pipelines.gdn_gate);
        encoder.set_buffer(&self.scratch, s.state_output as u64, 0);
        encoder.set_buffer(&self.scratch, s.aux0 as u64, 1);
        encoder.set_buffer(self.weight_buffer(weights.norm), weights.norm.offset, 2);
        encoder.set_buffer(&self.scratch, s.gated as u64, 3);
        encoder.set_bytes(&u32_bytes(&[48, 128, batch as u32]), 4);
        encoder.set_bytes(&EPS.to_le_bytes(), 5);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 48,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        self.encode_qmv(encoder, s.gated, &weights.output, s.branch_output, batch);
    }

    fn encode_attention(&self, encoder: &ComputeEncoder, weights: &AttentionWeights, batch: usize) {
        let s = self.scratch_layout;
        self.encode_qmv_fused(
            encoder,
            s.normalized,
            &[
                (&weights.q, s.projection),
                (&weights.k, s.aux0),
                (&weights.v, s.aux1),
            ],
            batch,
        );

        encoder.set_compute_pipeline_state(if batch == 3 {
            &self.pipelines.attn_q_block3
        } else {
            &self.pipelines.attn_q
        });
        encoder.set_buffer(&self.scratch, s.projection as u64, 0);
        encoder.set_buffer(self.weight_buffer(weights.q_norm), weights.q_norm.offset, 1);
        encoder.set_buffer(&self.scratch, s.query as u64, 2);
        encoder.set_buffer(&self.scratch, s.attention_gate as u64, 3);
        encoder.set_bytes(&(self.position as u32).to_le_bytes(), 4);
        encoder.set_bytes(&EPS.to_le_bytes(), 5);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 24,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        let layer_cache_bytes = ATTN_KV_HEADS * self.capacity * ATTN_HEAD_DIM * 2;
        let cache_offset = weights.cache_index * layer_cache_bytes;
        encoder.set_compute_pipeline_state(if batch == 3 {
            &self.pipelines.attn_store_block3
        } else {
            &self.pipelines.attn_store
        });
        encoder.set_buffer(&self.scratch, s.aux0 as u64, 0);
        encoder.set_buffer(&self.scratch, s.aux1 as u64, 1);
        encoder.set_buffer(self.weight_buffer(weights.k_norm), weights.k_norm.offset, 2);
        encoder.set_buffer(&self.key_cache, cache_offset as u64, 3);
        encoder.set_buffer(&self.value_cache, cache_offset as u64, 4);
        encoder.set_bytes(&(self.position as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(self.capacity as u32).to_le_bytes(), 6);
        encoder.set_bytes(&EPS.to_le_bytes(), 7);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 4,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(if batch == 3 {
            &self.pipelines.attn_sdpa_block3
        } else {
            &self.pipelines.attn_sdpa
        });
        encoder.set_buffer(&self.scratch, s.query as u64, 0);
        encoder.set_buffer(&self.key_cache, cache_offset as u64, 1);
        encoder.set_buffer(&self.value_cache, cache_offset as u64, 2);
        encoder.set_buffer(&self.scratch, s.attention_output as u64, 3);
        let context = if batch == 3 {
            self.position
        } else {
            self.position + 1
        };
        encoder.set_bytes(&(context as u32).to_le_bytes(), 4);
        encoder.set_bytes(&(self.capacity as u32).to_le_bytes(), 5);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 24,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.pipelines.attn_gate);
        encoder.set_buffer(&self.scratch, s.attention_output as u64, 0);
        encoder.set_buffer(&self.scratch, s.attention_gate as u64, 1);
        encoder.set_buffer(&self.scratch, s.gated as u64, 2);
        encoder.set_bytes(&((batch * ATTN_OUT) as u32).to_le_bytes(), 3);
        encoder.dispatch_threads(
            MTLSize {
                width: (batch * ATTN_OUT) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        self.encode_qmv(encoder, s.gated, &weights.output, s.branch_output, batch);
    }

    // ---- Batched prefill encoders. `m` is the real row count (<= PREFILL_BATCH);
    // all buffers are sized for PREFILL_BATCH and offsets come from `layout`.

    // C[m,N] = A[m,K]^bf16 · deq(W[N,K/8]^u4, gs64)^T via the NAX tensor-core
    // GEMM. A = scratch[input] (bf16), weights from the linear's resident shards.
    fn encode_gemm(
        &self,
        encoder: &ComputeEncoder,
        scratch: &MetalBuffer,
        m: usize,
        input: usize,
        linear: &Q4Linear,
        output: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.q4_gemm);
        encoder.set_buffer(scratch, input as u64, 0);
        encoder.set_buffer(self.weight_buffer(linear.weight), linear.weight.offset, 1);
        encoder.set_buffer(self.weight_buffer(linear.scales), linear.scales.offset, 2);
        encoder.set_buffer(self.weight_buffer(linear.biases), linear.biases.offset, 3);
        encoder.set_buffer(scratch, output as u64, 4);
        // Grid is in 16-row tiles; a tail chunk (m % 16 != 0) still fills the
        // last tile because the A/C buffers are sized for PREFILL_BATCH, and
        // downstream kernels only consume the real `m` rows.
        let g = m.div_ceil(16);
        let m16 = g * 16;
        let mut nk = Vec::with_capacity(12);
        nk.extend_from_slice(&(linear.output as u32).to_le_bytes());
        nk.extend_from_slice(&(linear.input as u32).to_le_bytes());
        nk.extend_from_slice(&(m16 as u32).to_le_bytes());
        encoder.set_bytes(&nk, 5);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: g as u64,
                height: linear.output.div_ceil(32) as u64,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }

    // Layout-agnostic batch RMSNorm / residual add reading `scratch`, used by
    // both the decode (batch<=3) and the batched prefill paths.
    fn encode_norm_raw(
        &self,
        encoder: &ComputeEncoder,
        scratch: &MetalBuffer,
        input: usize,
        weight: TensorRef,
        output: usize,
        batch: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.rms);
        encoder.set_buffer(scratch, input as u64, 0);
        encoder.set_buffer(self.weight_buffer(weight), weight.offset, 1);
        encoder.set_buffer(scratch, output as u64, 2);
        encoder.set_bytes(&(HIDDEN as u32).to_le_bytes(), 3);
        encoder.set_bytes(&EPS.to_le_bytes(), 4);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: batch as u64,
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

    fn encode_residual_raw(
        &self,
        encoder: &ComputeEncoder,
        scratch: &MetalBuffer,
        input: usize,
        branch: usize,
        output: usize,
        batch: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.residual);
        encoder.set_buffer(scratch, input as u64, 0);
        encoder.set_buffer(scratch, branch as u64, 1);
        encoder.set_buffer(scratch, output as u64, 2);
        encoder.set_bytes(&((batch * HIDDEN) as u32).to_le_bytes(), 3);
        encoder.dispatch_threads(
            MTLSize {
                width: (batch * HIDDEN) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_prefill_embed(&self, encoder: &ComputeEncoder, m: usize) {
        let s = self.prefill_layout;
        encoder.set_compute_pipeline_state(&self.pipelines.embed);
        encoder.set_buffer(&self.prefill_scratch, s.token as u64, 0);
        encoder.set_buffer(
            self.weight_buffer(self.embed.weight),
            self.embed.weight.offset,
            1,
        );
        encoder.set_buffer(
            self.weight_buffer(self.embed.scales),
            self.embed.scales.offset,
            2,
        );
        encoder.set_buffer(
            self.weight_buffer(self.embed.biases),
            self.embed.biases.offset,
            3,
        );
        encoder.set_buffer(&self.prefill_scratch, s.hidden as u64, 4);
        encoder.set_bytes(&(HIDDEN as u32).to_le_bytes(), 5);
        encoder.dispatch_threads(
            MTLSize {
                width: HIDDEN as u64,
                height: m as u64,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_prefill_silu(
        &self,
        encoder: &ComputeEncoder,
        scratch: &MetalBuffer,
        batch: usize,
        gate: usize,
        up: usize,
        active: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.silu);
        encoder.set_buffer(scratch, gate as u64, 0);
        encoder.set_buffer(scratch, up as u64, 1);
        encoder.set_buffer(scratch, active as u64, 2);
        encoder.set_bytes(&((batch * INTERMEDIATE) as u32).to_le_bytes(), 3);
        encoder.dispatch_threads(
            MTLSize {
                width: (batch * INTERMEDIATE) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_prefill_mlp(
        &self,
        encoder: &ComputeEncoder,
        scratch: &MetalBuffer,
        weights: &MlpWeights,
        batch: usize,
        post_norm: usize,
        branch_output: usize,
    ) {
        let gate = self.prefill_layout.mlp_gate;
        let up = self.prefill_layout.mlp_up;
        let active = self.prefill_layout.mlp_active;
        self.encode_gemm(encoder, scratch, batch, post_norm, &weights.gate, gate);
        self.encode_gemm(encoder, scratch, batch, post_norm, &weights.up, up);
        self.encode_prefill_silu(encoder, scratch, batch, gate, up, active);
        self.encode_gemm(encoder, scratch, batch, active, &weights.down, branch_output);
    }

    fn encode_prefill_gdn(
        &self,
        encoder: &ComputeEncoder,
        scratch: &MetalBuffer,
        weights: &GdnWeights,
        batch: usize,
        normalized: usize,
        branch_output: usize,
    ) {
        let s = self.prefill_layout;
        self.encode_gemm(encoder, scratch, batch, normalized, &weights.qkv, s.projection);
        // z/a/b are the small fused GDN side projections (single GEMM each).
        // a (A_log) and b (dt_bias) are [value_heads]-wide; z is the gate input.
        self.encode_gemm(encoder, scratch, batch, normalized, &weights.z, s.aux0);
        self.encode_gemm(encoder, scratch, batch, normalized, &weights.a, s.small0);
        self.encode_gemm(encoder, scratch, batch, normalized, &weights.b, s.small1);

        let conv_state_offset = weights.state_index * 3 * GDN_CONV_DIM * 2;
        encoder.set_compute_pipeline_state(&self.pipelines.gdn_conv);
        encoder.set_buffer(scratch, s.projection as u64, 0);
        encoder.set_buffer(scratch, s.small1 as u64, 1);
        encoder.set_buffer(scratch, s.small0 as u64, 2);
        encoder.set_buffer(
            self.weight_buffer(weights.conv_weight),
            weights.conv_weight.offset,
            3,
        );
        encoder.set_buffer(&self.conv_state, conv_state_offset as u64, 4);
        encoder.set_buffer(self.weight_buffer(weights.a_log), weights.a_log.offset, 5);
        encoder.set_buffer(
            self.weight_buffer(weights.dt_bias),
            weights.dt_bias.offset,
            6,
        );
        encoder.set_buffer(scratch, s.conv_output as u64, 7);
        encoder.set_buffer(scratch, s.q_norm as u64, 8);
        encoder.set_buffer(scratch, s.k_norm as u64, 9);
        encoder.set_buffer(scratch, s.beta as u64, 10);
        encoder.set_buffer(scratch, s.decay as u64, 11);
        encoder.set_bytes(&u32_bytes(&[16, 48, 128, 128]), 12);
        let inv = 1.0f32 / (GDN_HEAD_DIM as f32).sqrt();
        encoder.set_bytes(&f32_bytes(&[inv * inv, inv]), 13);
        encoder.set_bytes(&(batch as u32).to_le_bytes(), 14);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 48,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.pipelines.gdn_update_conv);
        encoder.set_buffer(scratch, s.projection as u64, 0);
        encoder.set_buffer(&self.conv_state, conv_state_offset as u64, 1);
        encoder.set_bytes(&(GDN_CONV_DIM as u32).to_le_bytes(), 2);
        encoder.set_bytes(&(batch as u32).to_le_bytes(), 3);
        encoder.set_buffer(&self.conv_state, conv_state_offset as u64, 4);
        encoder.set_bytes(&0u32.to_le_bytes(), 5);
        encoder.dispatch_threads(
            MTLSize {
                width: GDN_CONV_DIM as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.pipelines.gdn_recurrence);
        encoder.set_buffer(scratch, s.conv_output as u64, 0);
        encoder.set_buffer(scratch, s.q_norm as u64, 1);
        encoder.set_buffer(scratch, s.k_norm as u64, 2);
        encoder.set_buffer(scratch, s.beta as u64, 3);
        encoder.set_buffer(scratch, s.decay as u64, 4);
        encoder.set_buffer(
            &self.gdn_state,
            (weights.state_index * GDN_STATE_ELEMENTS * 4) as u64,
            5,
        );
        encoder.set_buffer(scratch, s.state_output as u64, 6);
        encoder.set_bytes(&u32_bytes(&[48, 128, 128, 3]), 7);
        encoder.set_bytes(&(batch as u32).to_le_bytes(), 8);
        encoder.set_buffer(
            &self.gdn_state,
            (weights.state_index * GDN_STATE_ELEMENTS * 4) as u64,
            9,
        );
        encoder.set_bytes(&0u32.to_le_bytes(), 10);
        encoder.dispatch_threads(
            MTLSize {
                width: 32,
                height: 128,
                depth: 48,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.pipelines.gdn_gate);
        encoder.set_buffer(scratch, s.state_output as u64, 0);
        encoder.set_buffer(scratch, s.aux0 as u64, 1);
        encoder.set_buffer(self.weight_buffer(weights.norm), weights.norm.offset, 2);
        encoder.set_buffer(scratch, s.gated as u64, 3);
        encoder.set_bytes(&u32_bytes(&[48, 128, batch as u32]), 4);
        encoder.set_bytes(&EPS.to_le_bytes(), 5);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 48,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        self.encode_gemm(encoder, scratch, batch, s.gated, &weights.output, branch_output);
    }

    fn encode_prefill_attention(
        &self,
        encoder: &ComputeEncoder,
        scratch: &MetalBuffer,
        weights: &AttentionWeights,
        batch: usize,
        normalized: usize,
        branch_output: usize,
    ) {
        let s = self.prefill_layout;
        self.encode_gemm(encoder, scratch, batch, normalized, &weights.q, s.projection);
        self.encode_gemm(encoder, scratch, batch, normalized, &weights.k, s.aux0);
        self.encode_gemm(encoder, scratch, batch, normalized, &weights.v, s.aux1);

        let base = self.position;
        encoder.set_compute_pipeline_state(&self.pipelines.attn_q_prefill);
        encoder.set_buffer(scratch, s.projection as u64, 0);
        encoder.set_buffer(self.weight_buffer(weights.q_norm), weights.q_norm.offset, 1);
        encoder.set_buffer(scratch, s.query as u64, 2);
        encoder.set_buffer(scratch, s.attention_gate as u64, 3);
        encoder.set_bytes(&(base as u32).to_le_bytes(), 4);
        encoder.set_bytes(&EPS.to_le_bytes(), 5);
        encoder.set_bytes(&(batch as u32).to_le_bytes(), 6);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 24,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        let layer_cache_bytes = ATTN_KV_HEADS * self.capacity * ATTN_HEAD_DIM * 2;
        let cache_offset = weights.cache_index * layer_cache_bytes;
        encoder.set_compute_pipeline_state(&self.pipelines.attn_store_prefill);
        encoder.set_buffer(scratch, s.aux0 as u64, 0);
        encoder.set_buffer(scratch, s.aux1 as u64, 1);
        encoder.set_buffer(self.weight_buffer(weights.k_norm), weights.k_norm.offset, 2);
        encoder.set_buffer(&self.key_cache, cache_offset as u64, 3);
        encoder.set_buffer(&self.value_cache, cache_offset as u64, 4);
        encoder.set_bytes(&(base as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(self.capacity as u32).to_le_bytes(), 6);
        encoder.set_bytes(&EPS.to_le_bytes(), 7);
        encoder.set_bytes(&(batch as u32).to_le_bytes(), 8);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 4,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.pipelines.attn_sdpa_prefill);
        encoder.set_buffer(scratch, s.query as u64, 0);
        encoder.set_buffer(&self.key_cache, cache_offset as u64, 1);
        encoder.set_buffer(&self.value_cache, cache_offset as u64, 2);
        encoder.set_buffer(scratch, s.attention_output as u64, 3);
        encoder.set_bytes(&(base as u32).to_le_bytes(), 4);
        encoder.set_bytes(&(self.capacity as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(batch as u32).to_le_bytes(), 6);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 24,
                height: batch as u64,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.pipelines.attn_gate);
        encoder.set_buffer(scratch, s.attention_output as u64, 0);
        encoder.set_buffer(scratch, s.attention_gate as u64, 1);
        encoder.set_buffer(scratch, s.gated as u64, 2);
        encoder.set_bytes(&((batch * ATTN_OUT) as u32).to_le_bytes(), 3);
        encoder.dispatch_threads(
            MTLSize {
                width: (batch * ATTN_OUT) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        self.encode_gemm(encoder, scratch, batch, s.gated, &weights.output, branch_output);
    }

    fn encode_prefill_argmax_row(
        &self,
        encoder: &ComputeEncoder,
        scratch: &MetalBuffer,
        row: usize,
    ) {
        const TG: usize = 256;
        let partials = VOCAB.div_ceil(TG);
        let stride = partials * 2;
        let input = self.prefill_layout.logits + row * VOCAB * 4;
        let base = self.prefill_layout.argmax_partial + row * stride * 4;
        encoder.set_compute_pipeline_state(&self.pipelines.argmax);
        encoder.set_buffer(scratch, input as u64, 0);
        encoder.set_buffer(scratch, base as u64, 1);
        encoder.set_buffer(scratch, (base + partials * 4) as u64, 2);
        encoder.set_bytes(&(VOCAB as u32).to_le_bytes(), 3);
        encoder.dispatch_thread_groups(
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
    }

    fn read_prefill_row(&self, row: usize) -> (u32, Vec<f32>) {
        const TG: usize = 256;
        let partials = VOCAB.div_ceil(TG);
        let stride = partials * 2;
        let base = self.prefill_layout.argmax_partial + row * stride * 4;
        let maxes = read_f32(&self.prefill_scratch, base, partials);
        let idx_bytes = self
            .prefill_scratch
            .read_bytes((base + partials * 4) as u64, partials * 4);
        let mut best = f32::NEG_INFINITY;
        let mut best_index = 0u32;
        for (i, v) in maxes.iter().enumerate() {
            if *v > best {
                best = *v;
                best_index =
                    u32::from_le_bytes(idx_bytes[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        let hb = self.prefill_layout.hidden + row * HIDDEN * 2;
        let hidden = read_bf16(&self.prefill_scratch, hb, HIDDEN);
        (best_index, hidden)
    }



    /// Batched prefill of `tokens` in fused chunks of `PREFILL_BATCH` rows.
    /// Mirrors the MLX batched affine-QMM prefill (GEMM projections + batched
    /// attention + blocked GDN recurrence) instead of N sequential dispatch
    /// decodes. Recurrent state (GDN/conv/KV) carries across chunks, so the
    /// per-position outputs equal the sequential decode argmaxes. Returns
    /// `(argmax, hidden)` for every token, advancing `position` by len.
    pub fn forward_prefill(&mut self, tokens: &[u32]) -> Result<Vec<(u32, Vec<f32>)>, String> {
        if tokens.iter().any(|token| *token as usize >= VOCAB) {
            return Err("prefill token is outside vocabulary".into());
        }
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        if self.position + tokens.len() > self.capacity {
            return Err(format!(
                "KV capacity {} exhausted by prefill of {} tokens",
                self.capacity,
                tokens.len()
            ));
        }
        let mut outputs = Vec::with_capacity(tokens.len());
        let mut base = 0usize;
        while base < tokens.len() {
            let m = (tokens.len() - base).min(PREFILL_BATCH);
            let chunk = &tokens[base..base + m];
            let s = self.prefill_layout;
            // Tokens for this chunk at the head of the token scratch (offset 0).
            let token_bytes: Vec<u8> = chunk.iter().flat_map(|t| t.to_le_bytes()).collect();
            self.prefill_scratch.write_bytes(0, &token_bytes);

            let command = self.queue.new_command_buffer();
            let encoder = command.compute_compute_encoder();
            self.encode_prefill_embed(&encoder, m);
            for layer in &self.layers {
                self.encode_norm_raw(
                    &encoder,
                    &self.prefill_scratch,
                    s.hidden,
                    layer.common.input_norm,
                    s.normalized,
                    m,
                );
                match &layer.branch {
                    BranchWeights::Gdn(weights) => {
                        self.encode_prefill_gdn(&encoder, &self.prefill_scratch, weights, m, s.normalized, s.branch_output)
                    }
                    BranchWeights::Attention(weights) => {
                        self.encode_prefill_attention(&encoder, &self.prefill_scratch, weights, m, s.normalized, s.branch_output)
                    }
                }
                self.encode_residual_raw(
                    &encoder,
                    &self.prefill_scratch,
                    s.hidden,
                    s.branch_output,
                    s.residual,
                    m,
                );
                self.encode_norm_raw(
                    &encoder,
                    &self.prefill_scratch,
                    s.residual,
                    layer.common.post_norm,
                    s.post_norm,
                    m,
                );
                self.encode_prefill_mlp(&encoder, &self.prefill_scratch, &layer.common.mlp, m, s.post_norm, s.branch_output);
                self.encode_residual_raw(
                    &encoder,
                    &self.prefill_scratch,
                    s.residual,
                    s.branch_output,
                    s.hidden,
                    m,
                );
            }
            self.encode_norm_raw(
                &encoder,
                &self.prefill_scratch,
                s.hidden,
                self.final_norm,
                s.normalized,
                m,
            );
            self.encode_gemm(&encoder, &self.prefill_scratch, m, s.normalized, &self.head, s.logits);
            for row in 0..m {
                self.encode_prefill_argmax_row(&encoder, &self.prefill_scratch, row);
            }
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();

            for row in 0..m {
                outputs.push(self.read_prefill_row(row));
            }
            self.position += m;
            base += m;
        }
        Ok(outputs)
    }
}

struct MtpWeights {
    pre_fc_norm_embedding: TensorRef,
    pre_fc_norm_hidden: TensorRef,
    fc: Q4Linear,
    input_norm: TensorRef,
    attention: AttentionWeights,
    post_norm: TensorRef,
    mlp: MlpWeights,
    final_norm: TensorRef,
}

#[derive(Clone, Copy)]
struct MtpScratchLayout {
    token: usize,
    token_embed: usize,
    target_hidden: usize,
    pre_fc: usize,
    pre_fc_hidden: usize,
    fc_input: usize,
    fc_output: usize,
    hidden: usize,
    normalized: usize,
    branch_output: usize,
    residual: usize,
    post_norm: usize,
    mlp_gate: usize,
    mlp_up: usize,
    mlp_active: usize,
    projection: usize,
    key: usize,
    value: usize,
    query: usize,
    attention_gate: usize,
    attention_output: usize,
    gated: usize,
    logits: usize,
    argmax_partial: usize,
    len: usize,
}

impl MtpScratchLayout {
    fn new(batch: usize) -> Self {
        let mut next = 0usize;
        let mut take = |bytes: usize| {
            next = next.div_ceil(16) * 16;
            let offset = next;
            next += bytes;
            offset
        };
        let token = take(batch * 4);
        let token_embed = take(batch * HIDDEN * 2);
        let target_hidden = take(batch * HIDDEN * 2);
        let pre_fc = take(batch * HIDDEN * 2);
        let pre_fc_hidden = take(batch * HIDDEN * 2);
        debug_assert_eq!(pre_fc_hidden, pre_fc + batch * HIDDEN * 2);
        // fc input = concat(pre_fc, pre_fc_hidden) per-row [batch, 2*HIDDEN]
        let fc_input = take(batch * 2 * HIDDEN * 2);
        let fc_output = take(batch * HIDDEN * 4);
        let hidden = take(batch * HIDDEN * 2);
        let normalized = take(batch * HIDDEN * 2);
        let branch_output = take(batch * HIDDEN * 4);
        let residual = take(batch * HIDDEN * 2);
        let post_norm = take(batch * HIDDEN * 2);
        let mlp_gate = take(batch * INTERMEDIATE * 4);
        let mlp_up = take(batch * INTERMEDIATE * 4);
        let mlp_active = take(batch * INTERMEDIATE * 2);
        let projection = take(batch * ATTN_Q_OUT * 4);
        let key = take(batch * ATTN_KV_OUT * 4);
        let value = take(batch * ATTN_KV_OUT * 4);
        let query = take(batch * ATTN_OUT * 2);
        let attention_gate = take(batch * ATTN_OUT * 4);
        let attention_output = take(batch * ATTN_OUT * 4);
        let gated = take(batch * ATTN_OUT * 2);
        let logits = take(batch * VOCAB * 4);
        let argmax_partial = take(batch * VOCAB.div_ceil(256) * 2 * 4);
        Self {
            token,
            token_embed,
            target_hidden,
            pre_fc,
            pre_fc_hidden,
            fc_input,
            fc_output,
            hidden,
            normalized,
            branch_output,
            residual,
            post_norm,
            mlp_gate,
            mlp_up,
            mlp_active,
            projection,
            key,
            value,
            query,
            attention_gate,
            attention_output,
            gated,
            logits,
            argmax_partial,
            len: next,
        }
    }
}

/// Result of one native MTP position. `hidden` is the final normalized MTP
/// state and can condition the following draft position.
pub struct MtpOutput {
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub argmax: u32,
}

/// Native M=1 runner for the separate Qwen3.8 MTP head. Target embedding and
/// language-head buffers are retained through the target runner's shared
/// shard allocation; only the 31 MTP tensors get a new GPU copy.
pub struct MtpRunner {
    _device: MetalDevice,
    _index: WeightIndex,
    target_shards: Arc<Vec<MetalBuffer>>,
    mtp_shards: Vec<MetalBuffer>,
    queue: CommandQueue,
    pipelines: Pipelines,
    embed: Q4Linear,
    head: Q4Linear,
    weights: MtpWeights,
    scratch: MetalBuffer,
    scratch_layout: MtpScratchLayout,
    prefill_scratch: MetalBuffer,
    prefill_layout: MtpScratchLayout,
    key_cache: MetalBuffer,
    value_cache: MetalBuffer,
    capacity: usize,
    position: usize,
}

impl MtpRunner {
    pub fn load(target: &QwenRunner, snapshot: &Path, capacity: usize) -> Result<Self, String> {
        if capacity == 0 {
            return Err("MTP KV capacity must be greater than zero".into());
        }
        if capacity > u32::MAX as usize {
            return Err("MTP KV capacity exceeds the kernel index range".into());
        }
        let paths = snapshot_paths(snapshot)?;
        let index = WeightIndex::open(&paths);
        if index.len() != 31 {
            return Err(format!(
                "invalid MTP checkpoint: found {} tensors, expected 31",
                index.len()
            ));
        }
        let weights = build_mtp_weights(&index)?;
        let device = MetalDevice::default();
        let mut mtp_shards = Vec::with_capacity(index.shard_count());
        for shard in 0..index.shard_count() {
            let bytes = index
                .shard_bytes(shard)
                .ok_or_else(|| format!("missing MTP shard {shard}"))?;
            let buffer = device.new_untracked_buffer(bytes.len());
            buffer.write_bytes(0, bytes);
            mtp_shards.push(buffer);
        }
        let scratch_layout = MtpScratchLayout::new(1);
        let cache_bytes = ATTN_KV_HEADS
            .checked_mul(capacity)
            .and_then(|value| value.checked_mul(ATTN_HEAD_DIM * 2))
            .ok_or_else(|| "MTP KV capacity overflow".to_owned())?;
        let scratch = device.new_buffer(scratch_layout.len);
        let prefill_layout = MtpScratchLayout::new(PREFILL_BATCH);
        let prefill_scratch = device.new_buffer(prefill_layout.len);
        let key_cache = device.new_buffer(cache_bytes);
        let value_cache = device.new_buffer(cache_bytes);
        let queue = device.new_command_queue();
        let pipelines = Pipelines::new(&device);
        let mut runner = Self {
            _device: device,
            _index: index,
            target_shards: Arc::clone(&target.shards),
            mtp_shards,
            queue,
            pipelines,
            embed: target.embed,
            head: target.head,
            weights,
            scratch,
            scratch_layout,
            prefill_scratch,
            prefill_layout,
            key_cache,
            value_cache,
            capacity,
            position: 0,
        };
        runner.reset_state();
        Ok(runner)
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn reset_state(&mut self) {
        let cache_len = ATTN_KV_HEADS * self.capacity * ATTN_HEAD_DIM * 2;
        let command = self.queue.new_command_buffer();
        let encoder = command.compute_compute_encoder();
        self.clear(&encoder, &self.key_cache, cache_len);
        self.clear(&encoder, &self.value_cache, cache_len);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        self.position = 0;
    }

    /// Rewinds the logical cache length. Stale rows need no copy or clear and
    /// are overwritten by subsequent forwards.
    pub fn trim_state(&mut self, count: usize) -> Result<(), String> {
        if count > self.position {
            return Err(format!(
                "cannot trim {count} MTP positions from {}",
                self.position
            ));
        }
        self.position -= count;
        Ok(())
    }

    pub fn forward_position(
        &mut self,
        token: u32,
        target_hidden: &[f32],
    ) -> Result<MtpOutput, String> {
        if usize::try_from(token).map_err(|_| "token does not fit usize")? >= VOCAB {
            return Err(format!("token {token} is outside vocabulary"));
        }
        if target_hidden.len() != HIDDEN {
            return Err(format!(
                "MTP target hidden has {} values, expected {HIDDEN}",
                target_hidden.len()
            ));
        }
        if self.position >= self.capacity {
            return Err(format!("MTP KV capacity {} exhausted", self.capacity));
        }
        let s = self.scratch_layout;
        self.scratch
            .write_bytes(s.token as u64, &token.to_le_bytes());
        let hidden_bf16: Vec<u8> = target_hidden
            .iter()
            .flat_map(|value| f32_to_bf16(*value).to_le_bytes())
            .collect();
        self.scratch
            .write_bytes(s.target_hidden as u64, &hidden_bf16);

        let command = self.queue.new_command_buffer();
        let encoder = command.compute_compute_encoder();
        self.encode_embed(&encoder);
        self.encode_norm(
            &encoder,
            s.token_embed,
            self.weights.pre_fc_norm_embedding,
            s.pre_fc,
        );
        self.encode_norm(
            &encoder,
            s.target_hidden,
            self.weights.pre_fc_norm_hidden,
            s.pre_fc_hidden,
        );
        self.encode_mtp_qmv(&encoder, s.pre_fc, &self.weights.fc, s.fc_output);
        self.encode_cast(&encoder, s.fc_output, s.hidden, HIDDEN);
        self.encode_norm(&encoder, s.hidden, self.weights.input_norm, s.normalized);
        self.encode_attention(&encoder);
        self.encode_residual(&encoder, s.hidden, s.branch_output, s.residual);
        self.encode_norm(&encoder, s.residual, self.weights.post_norm, s.post_norm);
        self.encode_mlp(&encoder);
        self.encode_residual(&encoder, s.residual, s.branch_output, s.hidden);
        self.encode_norm(&encoder, s.hidden, self.weights.final_norm, s.normalized);
        self.encode_target_qmv(&encoder, s.normalized, &self.head, s.logits);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();

        let hidden = read_bf16(&self.scratch, s.normalized, HIDDEN);
        let logits = read_f32(&self.scratch, s.logits, VOCAB);
        let argmax = argmax(&logits);
        self.position += 1;
        Ok(MtpOutput {
            hidden,
            logits,
            argmax,
        })
    }

    /// Decode-only MTP forward: identical to `forward_position` but the
    /// language-head argmax is reduced on the GPU so the full-vocab logits are
    /// never read back. Returns `(argmax, hidden)` — the speculative loop only
    /// consumes the argmax and the hidden state.
    pub fn forward_position_decode(
        &mut self,
        token: u32,
        target_hidden: &[f32],
    ) -> Result<(u32, Vec<f32>), String> {
        if usize::try_from(token).map_err(|_| "token does not fit usize")? >= VOCAB {
            return Err(format!("token {token} is outside vocabulary"));
        }
        if target_hidden.len() != HIDDEN {
            return Err(format!(
                "MTP target hidden has {} values, expected {HIDDEN}",
                target_hidden.len()
            ));
        }
        if self.position >= self.capacity {
            return Err(format!("MTP KV capacity {} exhausted", self.capacity));
        }
        let s = self.scratch_layout;
        self.scratch
            .write_bytes(s.token as u64, &token.to_le_bytes());
        let hidden_bf16: Vec<u8> = target_hidden
            .iter()
            .flat_map(|value| f32_to_bf16(*value).to_le_bytes())
            .collect();
        self.scratch
            .write_bytes(s.target_hidden as u64, &hidden_bf16);

        let command = self.queue.new_command_buffer();
        let encoder = command.compute_compute_encoder();
        self.encode_embed(&encoder);
        self.encode_norm(
            &encoder,
            s.token_embed,
            self.weights.pre_fc_norm_embedding,
            s.pre_fc,
        );
        self.encode_norm(
            &encoder,
            s.target_hidden,
            self.weights.pre_fc_norm_hidden,
            s.pre_fc_hidden,
        );
        self.encode_mtp_qmv(&encoder, s.pre_fc, &self.weights.fc, s.fc_output);
        self.encode_cast(&encoder, s.fc_output, s.hidden, HIDDEN);
        self.encode_norm(&encoder, s.hidden, self.weights.input_norm, s.normalized);
        self.encode_attention(&encoder);
        self.encode_residual(&encoder, s.hidden, s.branch_output, s.residual);
        self.encode_norm(&encoder, s.residual, self.weights.post_norm, s.post_norm);
        self.encode_mlp(&encoder);
        self.encode_residual(&encoder, s.residual, s.branch_output, s.hidden);
        self.encode_norm(&encoder, s.hidden, self.weights.final_norm, s.normalized);
        self.encode_target_qmv(&encoder, s.normalized, &self.head, s.logits);
        self.encode_argmax(&encoder, s.logits);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();

        let argmax = self.read_argmax();
        let hidden = read_bf16(&self.scratch, s.normalized, HIDDEN);
        self.position += 1;
        Ok((argmax, hidden))
    }

    fn encode_argmax(&self, encoder: &ComputeEncoder, input: usize) {
        const TG: usize = 256;
        let partials = VOCAB.div_ceil(TG);
        let base = self.scratch_layout.argmax_partial;
        encoder.set_compute_pipeline_state(&self.pipelines.argmax);
        encoder.set_buffer(&self.scratch, input as u64, 0);
        encoder.set_buffer(&self.scratch, base as u64, 1);
        encoder.set_buffer(&self.scratch, (base + partials * 4) as u64, 2);
        encoder.set_bytes(&(VOCAB as u32).to_le_bytes(), 3);
        encoder.dispatch_thread_groups(
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
    }

    fn read_argmax(&self) -> u32 {
        const TG: usize = 256;
        let partials = VOCAB.div_ceil(TG);
        let base = self.scratch_layout.argmax_partial;
        let maxes = read_f32(&self.scratch, base, partials);
        let idx_bytes = self.scratch.read_bytes((base + partials * 4) as u64, partials * 4);
        let mut best = f32::NEG_INFINITY;
        let mut best_index = 0u32;
        for (i, v) in maxes.iter().enumerate() {
            if *v > best {
                best = *v;
                best_index =
                    u32::from_le_bytes(idx_bytes[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        best_index
    }

    fn mtp_weight_buffer(&self, tensor: TensorRef) -> &MetalBuffer {
        &self.mtp_shards[tensor.shard]
    }

    fn target_weight_buffer(&self, tensor: TensorRef) -> &MetalBuffer {
        &self.target_shards[tensor.shard]
    }

    fn clear(&self, encoder: &ComputeEncoder, buffer: &MetalBuffer, len: usize) {
        const ROW: u64 = 1 << 30;
        encoder.set_compute_pipeline_state(&self.pipelines.clear);
        encoder.set_buffer(buffer, 0, 0);
        encoder.set_bytes(&(len as u64).to_le_bytes(), 1);
        encoder.dispatch_threads(
            MTLSize {
                width: (len as u64).min(ROW),
                height: (len as u64).div_ceil(ROW),
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_embed(&self, encoder: &ComputeEncoder) {
        let s = self.scratch_layout;
        encoder.set_compute_pipeline_state(&self.pipelines.embed);
        encoder.set_buffer(&self.scratch, s.token as u64, 0);
        encoder.set_buffer(
            self.target_weight_buffer(self.embed.weight),
            self.embed.weight.offset,
            1,
        );
        encoder.set_buffer(
            self.target_weight_buffer(self.embed.scales),
            self.embed.scales.offset,
            2,
        );
        encoder.set_buffer(
            self.target_weight_buffer(self.embed.biases),
            self.embed.biases.offset,
            3,
        );
        encoder.set_buffer(&self.scratch, s.token_embed as u64, 4);
        encoder.set_bytes(&(HIDDEN as u32).to_le_bytes(), 5);
        encoder.dispatch_threads(
            MTLSize {
                width: HIDDEN as u64,
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

    fn encode_norm(
        &self,
        encoder: &ComputeEncoder,
        input: usize,
        weight: TensorRef,
        output: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.rms);
        encoder.set_buffer(&self.scratch, input as u64, 0);
        encoder.set_buffer(self.mtp_weight_buffer(weight), weight.offset, 1);
        encoder.set_buffer(&self.scratch, output as u64, 2);
        encoder.set_bytes(&(HIDDEN as u32).to_le_bytes(), 3);
        encoder.set_bytes(&EPS.to_le_bytes(), 4);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 1,
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

    fn encode_mtp_qmv(
        &self,
        encoder: &ComputeEncoder,
        input: usize,
        linear: &Q4Linear,
        output: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.qmv);
        encoder.set_buffer(&self.scratch, input as u64, 0);
        encoder.set_buffer(
            self.mtp_weight_buffer(linear.weight),
            linear.weight.offset,
            1,
        );
        encoder.set_buffer(
            self.mtp_weight_buffer(linear.scales),
            linear.scales.offset,
            2,
        );
        encoder.set_buffer(
            self.mtp_weight_buffer(linear.biases),
            linear.biases.offset,
            3,
        );
        self.finish_qmv(encoder, linear, output);
    }

    fn encode_target_qmv(
        &self,
        encoder: &ComputeEncoder,
        input: usize,
        linear: &Q4Linear,
        output: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.qmv);
        encoder.set_buffer(&self.scratch, input as u64, 0);
        encoder.set_buffer(
            self.target_weight_buffer(linear.weight),
            linear.weight.offset,
            1,
        );
        encoder.set_buffer(
            self.target_weight_buffer(linear.scales),
            linear.scales.offset,
            2,
        );
        encoder.set_buffer(
            self.target_weight_buffer(linear.biases),
            linear.biases.offset,
            3,
        );
        self.finish_qmv(encoder, linear, output);
    }

    fn finish_qmv(&self, encoder: &ComputeEncoder, linear: &Q4Linear, output: usize) {
        encoder.set_buffer(&self.scratch, output as u64, 4);
        encoder.set_bytes(&u32_bytes(&[linear.output as u32, linear.input as u32]), 5);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: linear.output.div_ceil(16) as u64,
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

    fn encode_mtp_qmv_fused(
        &self,
        encoder: &ComputeEncoder,
        input: usize,
        projections: &[(&Q4Linear, usize)],
    ) {
        let pipeline = match projections.len() {
            2 => &self.pipelines.qmv_fused2,
            3 => &self.pipelines.qmv_fused3,
            _ => unreachable!("MTP only uses fused2 and fused3"),
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(&self.scratch, input as u64, 0);
        let mut dims = Vec::with_capacity((projections.len() + 1) * 4);
        let mut widest = 0usize;
        for (index, (linear, output)) in projections.iter().enumerate() {
            debug_assert_eq!(linear.input, projections[0].0.input);
            let base = 1 + index as u64 * 4;
            encoder.set_buffer(
                self.mtp_weight_buffer(linear.weight),
                linear.weight.offset,
                base,
            );
            encoder.set_buffer(
                self.mtp_weight_buffer(linear.scales),
                linear.scales.offset,
                base + 1,
            );
            encoder.set_buffer(
                self.mtp_weight_buffer(linear.biases),
                linear.biases.offset,
                base + 2,
            );
            encoder.set_buffer(&self.scratch, *output as u64, base + 3);
            dims.extend_from_slice(&(linear.output as u32).to_le_bytes());
            widest = widest.max(linear.output);
        }
        dims.extend_from_slice(&(projections[0].0.input as u32).to_le_bytes());
        encoder.set_bytes(&dims, 1 + projections.len() as u64 * 4);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: widest.div_ceil(16) as u64,
                height: projections.len() as u64,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_cast(&self, encoder: &ComputeEncoder, input: usize, output: usize, len: usize) {
        encoder.set_compute_pipeline_state(&self.pipelines.cast_bf16);
        encoder.set_buffer(&self.scratch, input as u64, 0);
        encoder.set_buffer(&self.scratch, output as u64, 1);
        encoder.set_bytes(&(len as u32).to_le_bytes(), 2);
        encoder.dispatch_threads(
            MTLSize {
                width: len as u64,
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

    fn encode_residual(
        &self,
        encoder: &ComputeEncoder,
        input: usize,
        branch: usize,
        output: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.residual);
        encoder.set_buffer(&self.scratch, input as u64, 0);
        encoder.set_buffer(&self.scratch, branch as u64, 1);
        encoder.set_buffer(&self.scratch, output as u64, 2);
        encoder.set_bytes(&(HIDDEN as u32).to_le_bytes(), 3);
        encoder.dispatch_threads(
            MTLSize {
                width: HIDDEN as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_mlp(&self, encoder: &ComputeEncoder) {
        let s = self.scratch_layout;
        let weights = &self.weights.mlp;
        self.encode_mtp_qmv_fused(
            encoder,
            s.post_norm,
            &[(&weights.gate, s.mlp_gate), (&weights.up, s.mlp_up)],
        );
        encoder.set_compute_pipeline_state(&self.pipelines.silu);
        encoder.set_buffer(&self.scratch, s.mlp_gate as u64, 0);
        encoder.set_buffer(&self.scratch, s.mlp_up as u64, 1);
        encoder.set_buffer(&self.scratch, s.mlp_active as u64, 2);
        encoder.set_bytes(&(INTERMEDIATE as u32).to_le_bytes(), 3);
        encoder.dispatch_threads(
            MTLSize {
                width: INTERMEDIATE as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
        self.encode_mtp_qmv(encoder, s.mlp_active, &weights.down, s.branch_output);
    }

    fn encode_attention(&self, encoder: &ComputeEncoder) {
        let s = self.scratch_layout;
        let weights = &self.weights.attention;
        self.encode_mtp_qmv_fused(
            encoder,
            s.normalized,
            &[
                (&weights.q, s.projection),
                (&weights.k, s.key),
                (&weights.v, s.value),
            ],
        );

        encoder.set_compute_pipeline_state(&self.pipelines.attn_q);
        encoder.set_buffer(&self.scratch, s.projection as u64, 0);
        encoder.set_buffer(
            self.mtp_weight_buffer(weights.q_norm),
            weights.q_norm.offset,
            1,
        );
        encoder.set_buffer(&self.scratch, s.query as u64, 2);
        encoder.set_buffer(&self.scratch, s.attention_gate as u64, 3);
        encoder.set_bytes(&(self.position as u32).to_le_bytes(), 4);
        encoder.set_bytes(&EPS.to_le_bytes(), 5);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: ATTN_HEADS as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.pipelines.attn_store);
        encoder.set_buffer(&self.scratch, s.key as u64, 0);
        encoder.set_buffer(&self.scratch, s.value as u64, 1);
        encoder.set_buffer(
            self.mtp_weight_buffer(weights.k_norm),
            weights.k_norm.offset,
            2,
        );
        encoder.set_buffer(&self.key_cache, 0, 3);
        encoder.set_buffer(&self.value_cache, 0, 4);
        encoder.set_bytes(&(self.position as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(self.capacity as u32).to_le_bytes(), 6);
        encoder.set_bytes(&EPS.to_le_bytes(), 7);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: ATTN_KV_HEADS as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.pipelines.attn_sdpa);
        encoder.set_buffer(&self.scratch, s.query as u64, 0);
        encoder.set_buffer(&self.key_cache, 0, 1);
        encoder.set_buffer(&self.value_cache, 0, 2);
        encoder.set_buffer(&self.scratch, s.attention_output as u64, 3);
        encoder.set_bytes(&((self.position + 1) as u32).to_le_bytes(), 4);
        encoder.set_bytes(&(self.capacity as u32).to_le_bytes(), 5);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: ATTN_HEADS as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&self.pipelines.attn_gate);
        encoder.set_buffer(&self.scratch, s.attention_output as u64, 0);
        encoder.set_buffer(&self.scratch, s.attention_gate as u64, 1);
        encoder.set_buffer(&self.scratch, s.gated as u64, 2);
        encoder.set_bytes(&(ATTN_OUT as u32).to_le_bytes(), 3);
        encoder.dispatch_threads(
            MTLSize {
                width: ATTN_OUT as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        self.encode_mtp_qmv(encoder, s.gated, &weights.output, s.branch_output);
    }

    // ---- Batched MTP prefill encoders. `m` real rows <= PREFILL_BATCH; all
    // buffers come from `pf` = the prefill scratch/layout. MTP weights resolve
    // through mtp_weight_buffer, but the shared embed/head use target shards.

    fn mtp_prefill_gemm(
        &self,
        encoder: &ComputeEncoder,
        scratch: &MetalBuffer,
        m: usize,
        input: usize,
        linear: &Q4Linear,
        output: usize,
        use_mtp_weights: bool,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.q4_gemm);
        encoder.set_buffer(scratch, input as u64, 0);
        let wbuf = if use_mtp_weights {
            self.mtp_weight_buffer(linear.weight)
        } else {
            self.target_weight_buffer(linear.weight)
        };
        let sbul = if use_mtp_weights {
            self.mtp_weight_buffer(linear.scales)
        } else {
            self.target_weight_buffer(linear.scales)
        };
        let bb = if use_mtp_weights {
            self.mtp_weight_buffer(linear.biases)
        } else {
            self.target_weight_buffer(linear.biases)
        };
        encoder.set_buffer(wbuf, linear.weight.offset, 1);
        encoder.set_buffer(sbul, linear.scales.offset, 2);
        encoder.set_buffer(bb, linear.biases.offset, 3);
        encoder.set_buffer(scratch, output as u64, 4);
        let g = m.div_ceil(16);
        let m16 = g * 16;
        let mut nk = Vec::with_capacity(12);
        nk.extend_from_slice(&(linear.output as u32).to_le_bytes());
        nk.extend_from_slice(&(linear.input as u32).to_le_bytes());
        nk.extend_from_slice(&(m16 as u32).to_le_bytes());
        encoder.set_bytes(&nk, 5);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: g as u64,
                height: linear.output.div_ceil(32) as u64,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }

    fn mtp_prefill_norm(
        &self,
        encoder: &ComputeEncoder,
        scratch: &MetalBuffer,
        input: usize,
        weight: TensorRef,
        output: usize,
        m: usize,
        mtp_weight: bool,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.rms);
        encoder.set_buffer(scratch, input as u64, 0);
        encoder.set_buffer(
            if mtp_weight {
                self.mtp_weight_buffer(weight)
            } else {
                self.target_weight_buffer(weight)
            },
            weight.offset,
            1,
        );
        encoder.set_buffer(scratch, output as u64, 2);
        encoder.set_bytes(&(HIDDEN as u32).to_le_bytes(), 3);
        encoder.set_bytes(&EPS.to_le_bytes(), 4);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: m as u64,
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

    fn mtp_prefill_residual(
        &self,
        encoder: &ComputeEncoder,
        scratch: &MetalBuffer,
        input: usize,
        branch: usize,
        output: usize,
        m: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.residual);
        encoder.set_buffer(scratch, input as u64, 0);
        encoder.set_buffer(scratch, branch as u64, 1);
        encoder.set_buffer(scratch, output as u64, 2);
        encoder.set_bytes(&((m * HIDDEN) as u32).to_le_bytes(), 3);
        encoder.dispatch_threads(
            MTLSize {
                width: (m * HIDDEN) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
    }

    fn mtp_prefill_silu(
        &self,
        encoder: &ComputeEncoder,
        scratch: &MetalBuffer,
        m: usize,
        gate: usize,
        up: usize,
        active: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.silu);
        encoder.set_buffer(scratch, gate as u64, 0);
        encoder.set_buffer(scratch, up as u64, 1);
        encoder.set_buffer(scratch, active as u64, 2);
        encoder.set_bytes(&((m * INTERMEDIATE) as u32).to_le_bytes(), 3);
        encoder.dispatch_threads(
            MTLSize {
                width: (m * INTERMEDIATE) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
    }

    fn mtp_prefill_cast(
        &self,
        encoder: &ComputeEncoder,
        scratch: &MetalBuffer,
        input: usize,
        output: usize,
        len: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.cast_bf16);
        encoder.set_buffer(scratch, input as u64, 0);
        encoder.set_buffer(scratch, output as u64, 1);
        encoder.set_bytes(&(len as u32).to_le_bytes(), 2);
        encoder.dispatch_threads(
            MTLSize {
                width: len as u64,
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

    /// Write an RMSNorm row into one half (col_base = 0 or HIDDEN) of the
    /// MTP pre-fc input rows of `layout.stride = 2*HIDDEN`. Mirrors the M=1
    /// contiguous pre_fc+pre_fc_hidden layout without a separate concat pass.
    fn mtp_prefill_norm_fc_half(
        &self,
        encoder: &ComputeEncoder,
        scratch: &MetalBuffer,
        m: usize,
        input: usize,
        weight: TensorRef,
        fc_input: usize,
        col_base: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.rms_stride);
        encoder.set_buffer(scratch, input as u64, 0);
        encoder.set_buffer(self.mtp_weight_buffer(weight), weight.offset, 1);
        encoder.set_buffer(scratch, fc_input as u64, 2);
        encoder.set_bytes(&(HIDDEN as u32).to_le_bytes(), 3);
        encoder.set_bytes(&EPS.to_le_bytes(), 4);
        encoder.set_bytes(&u32_bytes(&[(2 * HIDDEN) as u32, col_base as u32]), 5);
        encoder.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
    }

    /// Batched MTP prefill of `positions`: each entry `(token, target_hidden)`
    /// is one MTP position processed in a fused multi-token pass. Mirrors the
    /// target `forward_prefill` (GEMM projections + batched causal attention)
    /// for the MTP head's single attention+MLP block. Returns `(argmax,
    /// hidden)` per position, advancing MTP `position` by the count.
    pub fn forward_position_batch(
        &mut self,
        positions: &[(u32, &[f32])],
    ) -> Result<Vec<(u32, Vec<f32>)>, String> {
        if self.position + positions.len() > self.capacity {
            return Err(format!(
                "MTP KV capacity {} exhausted by prefill of {} positions",
                self.capacity,
                positions.len()
            ));
        }
        let mut outputs = Vec::with_capacity(positions.len());
        let mut base = 0usize;
        while base < positions.len() {
            let m = (positions.len() - base).min(PREFILL_BATCH);
            let chunk = &positions[base..base + m];
            let s = self.prefill_layout;
            // Tokens + target_hidden scratch at the head.
            let token_bytes: Vec<u8> = chunk.iter().flat_map(|(t, _)| t.to_le_bytes()).collect();
            self.prefill_scratch.write_bytes(0, &token_bytes);
            let mut hidden_bf16: Vec<u8> = Vec::with_capacity(m * HIDDEN * 2);
            for (_, h) in chunk {
                hidden_bf16.extend(h.iter().flat_map(|v| f32_to_bf16(*v).to_le_bytes()));
            }
            self.prefill_scratch
                .write_bytes(s.target_hidden as u64, &hidden_bf16);

            let command = self.queue.new_command_buffer();
            let encoder = command.compute_compute_encoder();
            // embed tokens
            encoder.set_compute_pipeline_state(&self.pipelines.embed);
            encoder.set_buffer(&self.prefill_scratch, s.token as u64, 0);
            encoder.set_buffer(
                self.target_weight_buffer(self.embed.weight),
                self.embed.weight.offset,
                1,
            );
            encoder.set_buffer(
                self.target_weight_buffer(self.embed.scales),
                self.embed.scales.offset,
                2,
            );
            encoder.set_buffer(
                self.target_weight_buffer(self.embed.biases),
                self.embed.biases.offset,
                3,
            );
            encoder.set_buffer(&self.prefill_scratch, s.token_embed as u64, 4);
            encoder.set_bytes(&(HIDDEN as u32).to_le_bytes(), 5);
            encoder.dispatch_threads(
                MTLSize {
                    width: HIDDEN as u64,
                    height: m as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.mtp_prefill_norm_fc_half(&encoder, &self.prefill_scratch, m, s.token_embed, self.weights.pre_fc_norm_embedding, s.fc_input, 0);
            self.mtp_prefill_norm_fc_half(&encoder, &self.prefill_scratch, m, s.target_hidden, self.weights.pre_fc_norm_hidden, s.fc_input, HIDDEN);
            self.mtp_prefill_gemm(&encoder, &self.prefill_scratch, m, s.fc_input, &self.weights.fc, s.fc_output, true);
            self.mtp_prefill_cast(&encoder, &self.prefill_scratch, s.fc_output, s.hidden, m * HIDDEN);
            self.mtp_prefill_norm(&encoder, &self.prefill_scratch, s.hidden, self.weights.input_norm, s.normalized, m, true);
            self.mtp_prefill_attention(&encoder, m);
            self.mtp_prefill_residual(&encoder, &self.prefill_scratch, s.hidden, s.branch_output, s.residual, m);
            self.mtp_prefill_norm(&encoder, &self.prefill_scratch, s.residual, self.weights.post_norm, s.post_norm, m, true);
            self.mtp_prefill_mlp(&encoder, m);
            self.mtp_prefill_residual(&encoder, &self.prefill_scratch, s.residual, s.branch_output, s.hidden, m);
            self.mtp_prefill_norm(&encoder, &self.prefill_scratch, s.hidden, self.weights.final_norm, s.normalized, m, true);
            self.mtp_prefill_gemm(&encoder, &self.prefill_scratch, m, s.normalized, &self.head, s.logits, false);
            for row in 0..m {
                self.mtp_prefill_argmax_row(&encoder, row);
            }
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();

            for row in 0..m {
                outputs.push(self.mtp_prefill_read_row(row));
            }
            self.position += m;
            base += m;
        }
        Ok(outputs)
    }

    fn mtp_prefill_argmax_row(&self, encoder: &ComputeEncoder, row: usize) {
        const TG: usize = 256;
        let partials = VOCAB.div_ceil(TG);
        let stride = partials * 2;
        let input = self.prefill_layout.logits + row * VOCAB * 4;
        let base = self.prefill_layout.argmax_partial + row * stride * 4;
        encoder.set_compute_pipeline_state(&self.pipelines.argmax);
        encoder.set_buffer(&self.prefill_scratch, input as u64, 0);
        encoder.set_buffer(&self.prefill_scratch, base as u64, 1);
        encoder.set_buffer(&self.prefill_scratch, (base + partials * 4) as u64, 2);
        encoder.set_bytes(&(VOCAB as u32).to_le_bytes(), 3);
        encoder.dispatch_thread_groups(
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
    }

    fn mtp_prefill_read_row(&self, row: usize) -> (u32, Vec<f32>) {
        const TG: usize = 256;
        let partials = VOCAB.div_ceil(TG);
        let stride = partials * 2;
        let base = self.prefill_layout.argmax_partial + row * stride * 4;
        let maxes = read_f32(&self.prefill_scratch, base, partials);
        let idx_bytes = self
            .prefill_scratch
            .read_bytes((base + partials * 4) as u64, partials * 4);
        let mut best = f32::NEG_INFINITY;
        let mut best_index = 0u32;
        for (i, v) in maxes.iter().enumerate() {
            if *v > best {
                best = *v;
                best_index = u32::from_le_bytes(idx_bytes[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        let hb = self.prefill_layout.normalized + row * HIDDEN * 2;
        let hidden = read_bf16(&self.prefill_scratch, hb, HIDDEN);
        (best_index, hidden)
    }

    #[allow(unused)]
    fn mtp_prefill_mlp(&self, encoder: &ComputeEncoder, m: usize) {
        let s = self.prefill_layout;
        let weights = &self.weights.mlp;
        self.mtp_prefill_gemm(encoder, &self.prefill_scratch, m, s.post_norm, &weights.gate, s.mlp_gate, true);
        self.mtp_prefill_gemm(encoder, &self.prefill_scratch, m, s.post_norm, &weights.up, s.mlp_up, true);
        self.mtp_prefill_silu(encoder, &self.prefill_scratch, m, s.mlp_gate, s.mlp_up, s.mlp_active);
        self.mtp_prefill_gemm(encoder, &self.prefill_scratch, m, s.mlp_active, &weights.down, s.branch_output, true);
    }

    fn mtp_prefill_attention(&self, encoder: &ComputeEncoder, m: usize) {
        let s = self.prefill_layout;
        let weights = &self.weights.attention;
        let base = self.position;
        self.mtp_prefill_gemm(encoder, &self.prefill_scratch, m, s.normalized, &weights.q, s.projection, true);
        self.mtp_prefill_gemm(encoder, &self.prefill_scratch, m, s.normalized, &weights.k, s.key, true);
        self.mtp_prefill_gemm(encoder, &self.prefill_scratch, m, s.normalized, &weights.v, s.value, true);

        encoder.set_compute_pipeline_state(&self.pipelines.attn_q_prefill);
        encoder.set_buffer(&self.prefill_scratch, s.projection as u64, 0);
        encoder.set_buffer(self.mtp_weight_buffer(weights.q_norm), weights.q_norm.offset, 1);
        encoder.set_buffer(&self.prefill_scratch, s.query as u64, 2);
        encoder.set_buffer(&self.prefill_scratch, s.attention_gate as u64, 3);
        encoder.set_bytes(&(base as u32).to_le_bytes(), 4);
        encoder.set_bytes(&EPS.to_le_bytes(), 5);
        encoder.set_bytes(&(m as u32).to_le_bytes(), 6);
        encoder.dispatch_thread_groups(
            MTLSize { width: 24, height: m as u64, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );

        encoder.set_compute_pipeline_state(&self.pipelines.attn_store_prefill);
        encoder.set_buffer(&self.prefill_scratch, s.key as u64, 0);
        encoder.set_buffer(&self.prefill_scratch, s.value as u64, 1);
        encoder.set_buffer(self.mtp_weight_buffer(weights.k_norm), weights.k_norm.offset, 2);
        encoder.set_buffer(&self.key_cache, 0, 3);
        encoder.set_buffer(&self.value_cache, 0, 4);
        encoder.set_bytes(&(base as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(self.capacity as u32).to_le_bytes(), 6);
        encoder.set_bytes(&EPS.to_le_bytes(), 7);
        encoder.set_bytes(&(m as u32).to_le_bytes(), 8);
        encoder.dispatch_thread_groups(
            MTLSize { width: 4, height: m as u64, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );

        encoder.set_compute_pipeline_state(&self.pipelines.attn_sdpa_prefill);
        encoder.set_buffer(&self.prefill_scratch, s.query as u64, 0);
        encoder.set_buffer(&self.key_cache, 0, 1);
        encoder.set_buffer(&self.value_cache, 0, 2);
        encoder.set_buffer(&self.prefill_scratch, s.attention_output as u64, 3);
        encoder.set_bytes(&(base as u32).to_le_bytes(), 4);
        encoder.set_bytes(&(self.capacity as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(m as u32).to_le_bytes(), 6);
        encoder.dispatch_thread_groups(
            MTLSize { width: 24, height: m as u64, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );

        encoder.set_compute_pipeline_state(&self.pipelines.attn_gate);
        encoder.set_buffer(&self.prefill_scratch, s.attention_output as u64, 0);
        encoder.set_buffer(&self.prefill_scratch, s.attention_gate as u64, 1);
        encoder.set_buffer(&self.prefill_scratch, s.gated as u64, 2);
        encoder.set_bytes(&((m * ATTN_OUT) as u32).to_le_bytes(), 3);
        encoder.dispatch_threads(
            MTLSize { width: (m * ATTN_OUT) as u64, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        self.mtp_prefill_gemm(encoder, &self.prefill_scratch, m, s.gated, &weights.output, s.branch_output, true);
    }
}

impl Pipelines {
    fn new(device: &MetalDevice) -> Self {
        let q4 = device.new_library(NAX_AFFINE_U4_SHADER);
        let embed = device.new_library(EMBED_SHADER);
        let norm = device.new_library(NORM_SHADER);
        let mlp = device.new_library(MLP_SHADER);
        let gdn = device.new_library(GDN_SHADER);
        let attention = device.new_library(ATTENTION_SHADER);
        Self {
            qmv: device.new_compute_pipeline(&q4.function_named("q4_qmv_fast")),
            qmv_fused2: device.new_compute_pipeline(&q4.function_named("q4_qmv_fused2")),
            qmv_fused3: device.new_compute_pipeline(&q4.function_named("q4_qmv_fused3")),
            qmv_fused4: device.new_compute_pipeline(&q4.function_named("q4_qmv_fused4")),
            qmv_fused2_wide3: device
                .new_compute_pipeline(&q4.function_named("q4_qmv_fused2_wide3")),
            qmv_fused3_wide3: device
                .new_compute_pipeline(&q4.function_named("q4_qmv_fused3_wide3")),
            qmv_fused4_wide3: device
                .new_compute_pipeline(&q4.function_named("q4_qmv_fused4_wide3")),
            qmv_wide3: device.new_compute_pipeline(&q4.function_named("q4_qmv_wide3")),
            q4_gemm: device.new_compute_pipeline(&q4.function_named("q4_gemm_nax_coop")),
            embed: device.new_compute_pipeline(&embed.function_named("affine_u4_lookup")),
            rms: device.new_compute_pipeline(&norm.function_named("rms_norm_rows")),
            residual: device.new_compute_pipeline(&norm.function_named("residual_add")),
            clear: device.new_compute_pipeline(&norm.function_named("clear_bytes")),
            copy_bytes: device.new_compute_pipeline(&norm.function_named("copy_bytes")),
            cast_bf16: device.new_compute_pipeline(&norm.function_named("f32_to_bf16")),
            rms_stride: device.new_compute_pipeline(&norm.function_named("rms_norm_rows_stride")),
            silu: device.new_compute_pipeline(&mlp.function_named("silu_mul_up")),
            gdn_conv: device
                .new_compute_pipeline(&gdn.function_named("gdn_conv_norm_gates_bf16_weights")),
            gdn_update_conv: device
                .new_compute_pipeline(&gdn.function_named("gdn_update_conv_state")),
            gdn_recurrence: device.new_compute_pipeline(&gdn.function_named("gdn_recurrence")),
            gdn_gate: device.new_compute_pipeline(&gdn.function_named("gdn_rms_gate_bf16_weight")),
            attn_q: device
                .new_compute_pipeline(&attention.function_named("q_norm_gate_rope_decode")),
            attn_store: device
                .new_compute_pipeline(&attention.function_named("kv_cache_store_decode")),
            attn_sdpa: device
                .new_compute_pipeline(&attention.function_named("sdpa_decode_streaming")),
            attn_gate: device.new_compute_pipeline(&attention.function_named("gate_out_bf16")),
            attn_q_block3: device
                .new_compute_pipeline(&attention.function_named("q_norm_gate_rope_block3")),
            attn_store_block3: device
                .new_compute_pipeline(&attention.function_named("kv_cache_store_block3")),
            attn_sdpa_block3: device
                .new_compute_pipeline(&attention.function_named("sdpa_decode_streaming_block3")),
            attn_q_prefill: device
                .new_compute_pipeline(&attention.function_named("q_norm_gate_rope_prefill")),
            attn_store_prefill: device
                .new_compute_pipeline(&attention.function_named("kv_cache_store_prefill")),
            attn_sdpa_prefill: device
                .new_compute_pipeline(&attention.function_named("sdpa_decode_prefill")),
            argmax: device.new_compute_pipeline(&embed.function_named("argmax_f32_partial")),
        }
    }
}

fn build_layers(index: &WeightIndex) -> Result<Vec<LayerWeights>, String> {
    let mut layers = Vec::with_capacity(LAYERS);
    let mut gdn_index = 0;
    let mut attention_index = 0;
    for layer in 0..LAYERS {
        let prefix = format!("language_model.model.layers.{layer}");
        let common = CommonLayer {
            input_norm: bf16(
                index,
                &format!("{prefix}.input_layernorm.weight"),
                &[HIDDEN],
            )?,
            post_norm: bf16(
                index,
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[HIDDEN],
            )?,
            mlp: MlpWeights {
                gate: q4(
                    index,
                    &format!("{prefix}.mlp.gate_proj"),
                    HIDDEN,
                    INTERMEDIATE,
                )?,
                up: q4(
                    index,
                    &format!("{prefix}.mlp.up_proj"),
                    HIDDEN,
                    INTERMEDIATE,
                )?,
                down: q4(
                    index,
                    &format!("{prefix}.mlp.down_proj"),
                    INTERMEDIATE,
                    HIDDEN,
                )?,
            },
        };
        let branch = match layer_kind(layer) {
            LayerKind::Gdn => {
                let p = format!("{prefix}.linear_attn");
                let weights = GdnWeights {
                    qkv: q4(index, &format!("{p}.in_proj_qkv"), HIDDEN, GDN_CONV_DIM)?,
                    z: q4(index, &format!("{p}.in_proj_z"), HIDDEN, GDN_VALUE_DIM)?,
                    a: q4(index, &format!("{p}.in_proj_a"), HIDDEN, GDN_VALUE_HEADS)?,
                    b: q4(index, &format!("{p}.in_proj_b"), HIDDEN, GDN_VALUE_HEADS)?,
                    output: q4(index, &format!("{p}.out_proj"), GDN_VALUE_DIM, HIDDEN)?,
                    conv_weight: bf16(index, &format!("{p}.conv1d.weight"), &[GDN_CONV_DIM, 4, 1])?,
                    a_log: bf16(index, &format!("{p}.A_log"), &[GDN_VALUE_HEADS])?,
                    dt_bias: bf16(index, &format!("{p}.dt_bias"), &[GDN_VALUE_HEADS])?,
                    norm: bf16(index, &format!("{p}.norm.weight"), &[GDN_HEAD_DIM])?,
                    state_index: gdn_index,
                };
                gdn_index += 1;
                BranchWeights::Gdn(weights)
            }
            LayerKind::Attention => {
                let p = format!("{prefix}.self_attn");
                let weights = AttentionWeights {
                    q: q4(index, &format!("{p}.q_proj"), HIDDEN, ATTN_Q_OUT)?,
                    k: q4(index, &format!("{p}.k_proj"), HIDDEN, ATTN_KV_OUT)?,
                    v: q4(index, &format!("{p}.v_proj"), HIDDEN, ATTN_KV_OUT)?,
                    output: q4(index, &format!("{p}.o_proj"), ATTN_OUT, HIDDEN)?,
                    q_norm: bf16(index, &format!("{p}.q_norm.weight"), &[ATTN_HEAD_DIM])?,
                    k_norm: bf16(index, &format!("{p}.k_norm.weight"), &[ATTN_HEAD_DIM])?,
                    cache_index: attention_index,
                };
                attention_index += 1;
                BranchWeights::Attention(weights)
            }
        };
        layers.push(LayerWeights { common, branch });
    }
    if gdn_index != GDN_LAYERS || attention_index != ATTN_LAYERS {
        return Err("invalid hybrid layer schedule".into());
    }
    Ok(layers)
}

fn build_mtp_weights(index: &WeightIndex) -> Result<MtpWeights, String> {
    let prefix = "layers.0";
    let attention = format!("{prefix}.self_attn");
    Ok(MtpWeights {
        pre_fc_norm_embedding: bf16(index, "pre_fc_norm_embedding.weight", &[HIDDEN])?,
        pre_fc_norm_hidden: bf16(index, "pre_fc_norm_hidden.weight", &[HIDDEN])?,
        fc: q4(index, "fc", HIDDEN * 2, HIDDEN)?,
        input_norm: bf16(
            index,
            &format!("{prefix}.input_layernorm.weight"),
            &[HIDDEN],
        )?,
        attention: AttentionWeights {
            q: q4(index, &format!("{attention}.q_proj"), HIDDEN, ATTN_Q_OUT)?,
            k: q4(index, &format!("{attention}.k_proj"), HIDDEN, ATTN_KV_OUT)?,
            v: q4(index, &format!("{attention}.v_proj"), HIDDEN, ATTN_KV_OUT)?,
            output: q4(index, &format!("{attention}.o_proj"), ATTN_OUT, HIDDEN)?,
            q_norm: bf16(
                index,
                &format!("{attention}.q_norm.weight"),
                &[ATTN_HEAD_DIM],
            )?,
            k_norm: bf16(
                index,
                &format!("{attention}.k_norm.weight"),
                &[ATTN_HEAD_DIM],
            )?,
            cache_index: 0,
        },
        post_norm: bf16(
            index,
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[HIDDEN],
        )?,
        mlp: MlpWeights {
            gate: q4(
                index,
                &format!("{prefix}.mlp.gate_proj"),
                HIDDEN,
                INTERMEDIATE,
            )?,
            up: q4(
                index,
                &format!("{prefix}.mlp.up_proj"),
                HIDDEN,
                INTERMEDIATE,
            )?,
            down: q4(
                index,
                &format!("{prefix}.mlp.down_proj"),
                INTERMEDIATE,
                HIDDEN,
            )?,
        },
        final_norm: bf16(index, "norm.weight", &[HIDDEN])?,
    })
}

fn q4(index: &WeightIndex, prefix: &str, input: usize, output: usize) -> Result<Q4Linear, String> {
    let weight = slot(
        index,
        &format!("{prefix}.weight"),
        Dtype::U32,
        &[output, input / 8],
    )?;
    let scales = slot(
        index,
        &format!("{prefix}.scales"),
        Dtype::BF16,
        &[output, input / 64],
    )?;
    let biases = slot(
        index,
        &format!("{prefix}.biases"),
        Dtype::BF16,
        &[output, input / 64],
    )?;
    Ok(Q4Linear {
        weight: tensor_ref(&weight),
        scales: tensor_ref(&scales),
        biases: tensor_ref(&biases),
        input,
        output,
    })
}

fn bf16(index: &WeightIndex, name: &str, shape: &[usize]) -> Result<TensorRef, String> {
    slot(index, name, Dtype::BF16, shape).map(|slot| tensor_ref(&slot))
}

fn slot(
    index: &WeightIndex,
    name: &str,
    dtype: Dtype,
    shape: &[usize],
) -> Result<WeightSlot, String> {
    let slot = index
        .slot(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    let expected: Vec<u64> = shape.iter().map(|dimension| *dimension as u64).collect();
    if slot.dtype != dtype || slot.shape != expected {
        return Err(format!(
            "invalid tensor {name}: {:?} {:?}, expected {:?} {:?}",
            slot.dtype, slot.shape, dtype, expected
        ));
    }
    Ok(slot.clone())
}

fn tensor_ref(slot: &WeightSlot) -> TensorRef {
    TensorRef {
        shard: slot.shard,
        offset: slot.offset as u64,
    }
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn snapshot_paths(snapshot: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .map_err(|error| format!("read snapshot {}: {error}", snapshot.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("no safetensors shards in {}", snapshot.display()));
    }
    Ok(paths)
}

fn read_f32(buffer: &MetalBuffer, offset: usize, len: usize) -> Vec<f32> {
    buffer
        .read_bytes(offset as u64, len * 4)
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}

fn read_bf16(buffer: &MetalBuffer, offset: usize, len: usize) -> Vec<f32> {
    buffer
        .read_bytes(offset as u64, len * 2)
        .chunks_exact(2)
        .map(|bytes| bf16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
        .collect()
}

fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index as u32)
}
