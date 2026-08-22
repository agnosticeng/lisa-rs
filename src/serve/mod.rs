// OpenAI-compatible HTTP serving layer (axum + tokio). Exposes:
//
//   GET  /v1/models
//   POST /v1/chat/completions   (streaming SSE and non-streaming)
//   POST /v1/completions        (legacy text completion)
//
// The heavy QwenRunner/MtpRunner live in one `Model` behind a mutex; generation
// serializes on that mutex and runs on the blocking pool so the async runtime
// stays responsive.
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::model::runner::{MtpCheckpoint, MtpRunner, QwenRunner, QwenStateCheckpoint};
use crate::speculative::{
    generate_greedy_block3_prefilled_streaming, prefill_prompt_from_with_progress,
    prefill_prefix_until_with_progress,
};
use crate::tokenizer::{ChatMessage, ChatToolCall, ChatTokenizer, split_tool_calls};

pub mod metrics;
pub mod ui;
use metrics::SharedMetrics;

#[derive(Clone)]
pub struct AppState {
    model: Arc<Mutex<Model>>,
}

pub struct Model {
    target: QwenRunner,
    mtp: MtpRunner,
    tokenizer: ChatTokenizer,
    model_id: String,
    metrics: SharedMetrics,
    /// Messages-only ids of the last request (no generation-prompt suffix), used
    /// to find the reusable prefix on the next turn.
    last_msg_ids: Vec<u32>,
    /// The (target, mtp) recurrent/KV/position state at the messages-only
    /// boundary of the last request. Restoring it + prefilling only the delta
    /// skips re-prefilling the whole conversation on every turn.
    last_msgs_checkpoint: Option<(QwenStateCheckpoint, MtpCheckpoint)>,
}

pub struct SessionSnapshot {
    pub last_msg_ids: Vec<u32>,
    pub messages_checkpoint: (QwenStateCheckpoint, MtpCheckpoint),
}

impl Model {
    pub fn load(
        target_snapshot: &Path,
        mtp_snapshot: &Path,
        tokenizer_path: &Path,
        capacity: usize,
        model_id: String,
    ) -> Result<Self, String> {
        Self::load_with_metrics(
            target_snapshot,
            mtp_snapshot,
            tokenizer_path,
            capacity,
            model_id,
            metrics::Metrics::shared(),
        )
    }

    pub fn load_with_metrics(
        target_snapshot: &Path,
        mtp_snapshot: &Path,
        tokenizer_path: &Path,
        capacity: usize,
        model_id: String,
        metrics: SharedMetrics,
    ) -> Result<Self, String> {
        let target = QwenRunner::load(target_snapshot, capacity)?;
        let mtp = MtpRunner::load(&target, mtp_snapshot, capacity)?;
        let tokenizer = ChatTokenizer::load(tokenizer_path)?;
        Ok(Self {
            target,
            mtp,
            tokenizer,
            model_id,
            metrics,
            last_msg_ids: Vec::new(),
            last_msgs_checkpoint: None,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn metrics(&self) -> &SharedMetrics {
        &self.metrics
    }

    /// Run one completion. `on_token` receives each generated token id (the EOS
    /// token is excluded). Returns `(prompt_tokens, generated)`.
    pub fn complete<F>(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        enable_thinking: bool,
        reasoning_effort: Option<&str>,
        tools: &[String],
        mut on_token: F,
    ) -> Result<(usize, Vec<u32>), String>
    where
        F: FnMut(u32),
    {
        // Full prompt (messages + generation-prompt suffix).
        let prompt = self.tokenizer.apply_chat_template(
            messages,
            enable_thinking,
            reasoning_effort,
            tools,
            true,
        )?;
        // Messages-only render (no generation-prompt suffix). Because
        // apply_chat_template only appends the suffix at the very end, this is
        // exactly a token-prefix of `prompt`. `msg_len` is the boundary between
        // the reusable conversation prefix and the generation segment.
        let msg_only = self.tokenizer.apply_chat_template(
            messages,
            enable_thinking,
            reasoning_effort,
            tools,
            false,
        )?;
        let full_ids = self.tokenizer.encode(&prompt)?;
        let msg_ids = self.tokenizer.encode(&msg_only)?;
        let prompt_tokens = full_ids.len();
        let msg_len = msg_ids.len();

        // LISA_LOG_REQUESTS=1 prints a compact per-request debug line to stderr
        // (message roles, token counts, prefix-cache decision) so we can see
        // exactly what a client like pi sends and whether the full history is
        // being prefilled. Off by default; no perf cost on the hot path.
        if std::env::var("LISA_LOG_REQUESTS").is_ok() {
            let roles: Vec<String> = messages
                .iter()
                .map(|m| {
                    format!(
                        "{}/{}",
                        m.role,
                        if m.tool_calls.is_empty() { 0 } else { m.tool_calls.len() }
                    )
                })
                .collect();
            eprintln!(
                "[lisa] req n={} full_tok={} msg_tok={} roles=[{}] last={:?}",
                messages.len(),
                prompt_tokens,
                msg_len,
                roles.join(","),
                messages.last().map(|m| m.content.chars().take(120).collect::<String>()),
            );
        }

        // Record the in-flight request in telemetry.
        let preview = messages
            .last()
            .map(|m| m.content.as_str())
            .unwrap_or_default();
        let session = session_fingerprint(messages, &self.model_id);
        let _seq = if let Ok(mut metrics) = self.metrics.lock() {
            metrics.begin(self.preview(preview, 48), prompt_tokens, session)
        } else {
            0
        };

        let start = std::time::Instant::now();

        // Stream live prefill speed into telemetry as tokens get prefilled
        // (long prompts show progress instead of zeros until prefill finishes).
        let mut prefill_done = 0usize;
        let mut record_progress = |done: usize| {
            prefill_done = prefill_done.max(done);
            if let Ok(mut metrics) = self.metrics.lock() {
                metrics.prefill_tick(prefill_done);
            }
        };

        // Session-cache reuse: if this request's messages-only prefix is an
        // exact extension of the last one, restore the recurrent/KV state at the
        // last messages boundary and prefill only the delta. Otherwise fall back
        // to a full reset + re-prefill. Any error/mismatch degrades safely to a
        // full re-prefill so a poisoned cache never breaks generation.
        let reuse_prefix = self.last_msg_ids.len() < msg_len
            && msg_ids[..self.last_msg_ids.len()] == self.last_msg_ids[..]
            && self.last_msgs_checkpoint.is_some();
        let used_cache = if reuse_prefix {
            let last_len = self.last_msg_ids.len();
            let (target_ck, mtp_ck) = self.last_msgs_checkpoint.take().expect("checked above");
            if self.target.restore_state(&target_ck).is_err()
                || self.mtp.restore_state(&mtp_ck).is_err()
            {
                // Fall back: reset and re-prefill from scratch.
                self.target.reset_state();
                self.mtp.reset_state();
                false
            } else {
                let delta_ok = prefill_prefix_until_with_progress(
                    &mut self.target,
                    &mut self.mtp,
                    &full_ids,
                    last_len,
                    msg_len,
                    |done| record_progress(done),
                )
                .is_ok();
                if !delta_ok {
                    self.target.reset_state();
                    self.mtp.reset_state();
                    false
                } else {
                    true
                }
            }
        } else {
            self.target.reset_state();
            self.mtp.reset_state();
            false
        };

        if !used_cache {
            // Full (or fallback) path: prefill the messages prefix, checkpoint
            // it for the next turn, then prefill the generation segment.
            prefill_prefix_until_with_progress(
                &mut self.target,
                &mut self.mtp,
                &full_ids,
                0,
                msg_len,
                |done| record_progress(done),
            )?;
        }

        if std::env::var("LISA_LOG_REQUESTS").is_ok() {
            eprintln!(
                "[lisa] prefill cached={} prefix_tok={} delta_tok={}",
                used_cache,
                msg_len,
                prompt_tokens.saturating_sub(msg_len)
            );
        }

        // Snapshot the messages-only boundary for the NEXT turn (before
        // generation so the stored state is exactly the reusable prefix).
        let checkpoint = (self.target.checkpoint_state(), self.mtp.checkpoint_state());

        // Prefill the generation segment (gen-prompt suffix) and produce the seed.
        let (bonus, target_hidden, mtp_seed) = prefill_prompt_from_with_progress(
            &mut self.target,
            &mut self.mtp,
            &full_ids,
            msg_len,
            |done| record_progress(done),
        )?;

        // Commit the new session prefix for the next request.
        self.last_msg_ids = msg_ids;
        self.last_msgs_checkpoint = Some(checkpoint);

        let prefill_ms = start.elapsed().as_secs_f64() * 1_000.0;
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.prefill_done(prefill_ms);
        }

        let mut generated = Vec::new();
        let eos = self.tokenizer.eos();
        let decode_start = std::time::Instant::now();
        let result = generate_greedy_block3_prefilled_streaming(
            &mut self.target,
            &mut self.mtp,
            bonus,
            target_hidden,
            mtp_seed,
            max_tokens,
            Some(eos),
            &mut |token| {
                generated.push(token);
                on_token(token);
                if let Ok(mut metrics) = self.metrics.lock() {
                    metrics.tick(generated.len());
                }
            },
        )?;
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1_000.0;

        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.finish(
                prefill_ms,
                decode_ms,
                result.target_forwards,
                result.drafted_tokens,
                result.accepted_drafts,
            );
        }

        Ok((prompt_tokens, generated))
    }

    /// Preview the first content line of a chat/prompt, truncated for UI display.
    pub fn preview(&self, text: &str, max_len: usize) -> String {
        let mut text = text.trim().replace('\n', " ");
        if text.chars().count() > max_len {
            text = text.chars().take(max_len).collect::<String>() + "…";
        }
        text
    }
}

pub fn build_router(model: Model) -> Router {
    let state = AppState {
        model: Arc::new(Mutex::new(model)),
    };
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<ToolDef>>,
}

#[derive(Deserialize)]
pub struct OpenAiMessage {
    pub role: String,
    #[serde(default)]
    pub content: serde_json::Value,
    #[serde(default)]
    pub tool_calls: Option<Vec<ClientToolCall>>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

#[derive(Deserialize)]
pub struct ClientToolCall {
    pub function: FunctionCall,
}

#[derive(Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ToolDef {
    #[serde(default)]
    pub r#type: Option<String>,
    pub function: FunctionDef,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: serde_json::Value,
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: String,
    pub prompt: serde_json::Value,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
}

#[derive(Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: ResponseMessage,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct ResponseMessage {
    pub role: &'static str,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ResponseToolCall>>,
}

#[derive(Serialize)]
pub struct ResponseToolCall {
    pub id: String,
    pub r#type: &'static str,
    pub function: FunctionCallOut,
}

#[derive(Serialize)]
pub struct FunctionCallOut {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Serialize, Default)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ResponseToolCall>>,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn request_id() -> String {
    format!("chatcmpl-{}", now_secs())
}

fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    let body = serde_json::json!({ "error": { "message": message.into(), "type": "server_error" } });
    (status, Json(body)).into_response()
}

/// Stable fingerprint of a conversation, used to group telemetry rows into
/// sessions. Two queries from the same chat share the message history up to
/// their last message, so hash the full history minus the final message +
/// model id. Falls back to hashing everything if there's a single message.
fn session_fingerprint(messages: &[ChatMessage], model_id: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for byte in model_id.bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    let history = messages.get(..messages.len().saturating_sub(1)).unwrap_or(messages);
    for m in history {
        for byte in m.role.bytes().chain(m.content.bytes()) {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("{h:06x}")
}

fn content_to_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn to_chat_messages(messages: &[OpenAiMessage]) -> Vec<ChatMessage> {
    messages
        .iter()
        .map(|message| ChatMessage {
            role: message.role.clone(),
            content: content_to_text(&message.content),
            reasoning_content: message.reasoning_content.clone(),
            tool_calls: message
                .tool_calls
                .iter()
                .flatten()
                .map(|call| ChatToolCall {
                    name: call.function.name.clone(),
                    arguments: coerce_arguments(&call.function.arguments),
                })
                .collect(),
            name: None,
        })
        .collect()
}

/// The client sends `arguments` as a JSON string; parse it to a Value (fall
/// back to a string if invalid).
fn coerce_arguments(arguments: &str) -> serde_json::Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| arguments.to_owned().into())
}

/// Serialize the OpenAI `tools` array into the per-tool JSON blobs the chat
/// template expects (one per tool).
fn tools_to_template(tools: &[ToolDef]) -> Vec<String> {
    tools
        .iter()
        .filter_map(|tool| serde_json::to_string(tool).ok())
        .collect()
}

/// Build OpenAI `tool_calls` response entries from the model's structured calls.
fn tool_calls_to_response(calls: &[ChatToolCall]) -> Vec<ResponseToolCall> {
    calls
        .iter()
        .enumerate()
        .map(|(i, call)| ResponseToolCall {
            id: format!("call_{}", i + 1),
            r#type: "function",
            function: FunctionCallOut {
                name: call.name.clone(),
                arguments: call.arguments.to_string(),
            },
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let model = state.model.lock().unwrap();
    Json(ModelList {
        object: "list",
        data: vec![ModelInfo {
            id: model.model_id().to_string(),
            object: "model",
            created: now_secs(),
            owned_by: "lisa-rs".to_string(),
        }],
    })
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let tools = tools_to_template(req.tools.as_deref().unwrap_or(&[]));
    let enable_thinking = req
        .enable_thinking
        .unwrap_or_else(|| tools.is_empty());
    let max_tokens = req
        .max_tokens
        .or(req.max_completion_tokens)
        .unwrap_or(4096)
        .min(8192);
    let messages = to_chat_messages(&req.messages);
    let tools = tools_to_template(req.tools.as_deref().unwrap_or(&[]));
    let model_id = state
        .model
        .lock()
        .map(|m| m.model_id().to_string())
        .unwrap_or_default();

    if req.stream {
        stream_response(state, model_id, messages, max_tokens, enable_thinking, tools, &req)
            .await
            .into_response()
    } else {
        let effort = req.reasoning_effort.clone();
        let result = tokio::task::spawn_blocking(move || {
            let mut model = state.model.lock().unwrap();
            let mut tokens = Vec::new();
            let (prompt_tokens, generated) = model.complete(
                &messages,
                max_tokens,
                enable_thinking,
                effort.as_deref(),
                &tools,
                |token| tokens.push(token),
            )?;
            let full_text = model.tokenizer.decode(&tokens)?;
            let (reasoning, answer) =
                model.tokenizer.split_thinking(&full_text, enable_thinking);
            let (reasoning, content, tool_calls) = split_tool_calls(reasoning, answer);
            let finish_reason = if !tool_calls.is_empty() {
                "tool_calls"
            } else if generated.len() < max_tokens {
                "stop"
            } else {
                "length"
            };
            Ok::<_, String>((
                prompt_tokens,
                generated.len(),
                reasoning,
                content,
                tool_calls,
                finish_reason.to_string(),
            ))
        })
        .await;

        match result {
            Ok(Ok((
                prompt_tokens,
                completion_tokens,
                reasoning,
                content,
                tool_calls,
                finish_reason,
            ))) => {
                let response = ChatCompletionResponse {
                    id: request_id(),
                    object: "chat.completion",
                    created: now_secs(),
                    model: model_id,
                    choices: vec![Choice {
                        index: 0,
                        message: ResponseMessage {
                            role: "assistant",
                            content,
                            reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
                            tool_calls: if tool_calls.is_empty() {
                                None
                            } else {
                                Some(tool_calls_to_response(&tool_calls))
                            },
                        },
                        finish_reason,
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Ok(Err(e)) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
            Err(e) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("generation task failed: {e}"),
            ),
        }
    }
}

async fn stream_response(
    state: AppState,
    model_id: String,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    enable_thinking: bool,
    tools: Vec<String>,
    req: &ChatCompletionRequest,
) -> Response {
    let (tx, rx) = mpsc::channel::<Result<String, String>>(64);
    let model = state.model.clone();
    let effort = req.reasoning_effort.clone();
    let id = request_id();
    let created = now_secs();

    tokio::task::spawn_blocking(move || {
        let mut m = model.lock().unwrap();
        let tokenizer = m.tokenizer.clone();

        let mut ids = Vec::new();
        let mut last_reasoning_len = 0usize;
        let mut last_answer_len = 0usize;
        let mut sent_role = false;
        let mut tool_boundary: Option<usize> = None;

        let result = m.complete(
            &messages,
            max_tokens,
            enable_thinking,
            effort.as_deref(),
            &tools,
            |token| {
                ids.push(token);
                let full_text = tokenizer.decode(&ids).unwrap_or_default();
                let (reasoning, answer) =
                    tokenizer.split_thinking(&full_text, enable_thinking);

                if reasoning.len() > last_reasoning_len {
                    let delta = reasoning[last_reasoning_len..].to_string();
                    last_reasoning_len = reasoning.len();
                    let chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_id.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: None,
                                reasoning_content: Some(delta),
                                ..Default::default()
                            },
                            finish_reason: None,
                        }],
                    };
                    let _ = tx.blocking_send(Ok(serde_json::to_string(&chunk).unwrap()));
                }

                // Once a <tool_call> block begins, stop streaming raw content.
                if tool_boundary.is_none() {
                    if let Some(p) = answer.find("<tool_call") {
                        tool_boundary = Some(p);
                    }
                }
                let boundary = tool_boundary.unwrap_or(answer.len());
                let boundary = boundary.min(answer.len());
                if answer.len() > last_answer_len && boundary > last_answer_len {
                    let delta = answer[last_answer_len..boundary].to_string();
                    last_answer_len = boundary;
                    let role = if sent_role {
                        None
                    } else {
                        sent_role = true;
                        Some("assistant")
                    };
                    let chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_id.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role,
                                content: Some(delta),
                                reasoning_content: None,
                                ..Default::default()
                            },
                            finish_reason: None,
                        }],
                    };
                    let _ = tx.blocking_send(Ok(serde_json::to_string(&chunk).unwrap()));
                }
            },
        );

        if let Err(e) = result {
            let _ = tx.blocking_send(Err(e));
        }
        // Final chunk carries finish_reason, then the SSE done sentinel.
        let full_text = tokenizer.decode(&ids).unwrap_or_default();
        let (reasoning, answer) = tokenizer.split_thinking(&full_text, enable_thinking);
        let (_reasoning, _content, tool_calls) = split_tool_calls(reasoning, answer);
        let tool_calls = tool_calls_to_response(&tool_calls);
        let finish_reason = if !tool_calls.is_empty() {
            "tool_calls"
        } else if ids.len() < max_tokens {
            "stop"
        } else {
            "length"
        };
        let finish = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_id.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    ..Default::default()
                },
                finish_reason: Some(finish_reason.to_string()),
            }],
        };
        let _ = tx.blocking_send(Ok(serde_json::to_string(&finish).unwrap()));
        let _ = tx.blocking_send(Ok("[DONE]".to_string()));
    });

    let stream = ReceiverStream::new(rx).map(|item| match item {
        Ok(text) if text == "[DONE]" => {
            Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
        }
        Ok(text) => Ok::<Event, std::convert::Infallible>(Event::default().data(text)),
        Err(e) => Ok::<Event, std::convert::Infallible>(Event::default().data(
            serde_json::json!({ "error": { "message": e } }).to_string(),
        )),
    });

    Sse::new(stream).into_response()
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let chat = ChatCompletionRequest {
        model: req.model,
        messages: vec![OpenAiMessage {
            role: "user".to_string(),
            content: req.prompt,
            tool_calls: None,
            reasoning_content: None,
        }],
        stream: req.stream,
        max_tokens: req.max_tokens,
        max_completion_tokens: req.max_completion_tokens,
        temperature: req.temperature,
        enable_thinking: Some(false),
        reasoning_effort: None,
        tools: None,
    };
    chat_completions(State(state), Json(chat)).await
}
