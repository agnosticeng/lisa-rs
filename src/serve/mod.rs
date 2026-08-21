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

use crate::model::runner::{MtpRunner, QwenRunner};
use crate::speculative::{generate_greedy_block3_prefilled_streaming, prefill_prompt};
use crate::tokenizer::{ChatMessage, ChatTokenizer};

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
        mut on_token: F,
    ) -> Result<(usize, Vec<u32>), String>
    where
        F: FnMut(u32),
    {
        let prompt = self.tokenizer.apply_chat_template(
            messages,
            enable_thinking,
            reasoning_effort,
        )?;
        let prompt_ids = self.tokenizer.encode(&prompt)?;
        let prompt_tokens = prompt_ids.len();

        // Record the in-flight request in telemetry.
        let preview = messages
            .last()
            .map(|m| m.content.as_str())
            .unwrap_or_default();
        let _seq = if let Ok(mut metrics) = self.metrics.lock() {
            metrics.begin(self.preview(preview, 48), prompt_tokens)
        } else {
            0
        };

        // Each request starts from a clean recurrent state.
        self.target.reset_state();
        self.mtp.reset_state();

        let start = std::time::Instant::now();
        let (bonus, target_hidden, mtp_seed) =
            prefill_prompt(&mut self.target, &mut self.mtp, &prompt_ids)?;
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
}

#[derive(Deserialize)]
pub struct OpenAiMessage {
    pub role: String,
    #[serde(default)]
    pub content: serde_json::Value,
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
    let enable_thinking = req.enable_thinking.unwrap_or(true);
    let max_tokens = req
        .max_tokens
        .or(req.max_completion_tokens)
        .unwrap_or(512)
        .min(8192);
    let messages = to_chat_messages(&req.messages);
    let model_id = state
        .model
        .lock()
        .map(|m| m.model_id().to_string())
        .unwrap_or_default();

    if req.stream {
        stream_response(state, model_id, messages, max_tokens, enable_thinking, &req)
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
                |token| tokens.push(token),
            )?;
            let full_text = model.tokenizer.decode(&tokens)?;
            let (reasoning, answer) =
                model.tokenizer.split_thinking(&full_text, enable_thinking);
            let finish_reason = if generated.len() < max_tokens {
                "stop"
            } else {
                "length"
            };
            Ok::<_, String>((
                prompt_tokens,
                generated.len(),
                reasoning,
                answer,
                finish_reason.to_string(),
            ))
        })
        .await;

        match result {
            Ok(Ok((prompt_tokens, completion_tokens, reasoning, answer, finish_reason))) => {
                let response = ChatCompletionResponse {
                    id: request_id(),
                    object: "chat.completion",
                    created: now_secs(),
                    model: model_id,
                    choices: vec![Choice {
                        index: 0,
                        message: ResponseMessage {
                            role: "assistant",
                            content: answer,
                            reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
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

        let result = m.complete(
            &messages,
            max_tokens,
            enable_thinking,
            effort.as_deref(),
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
                            },
                            finish_reason: None,
                        }],
                    };
                    let _ = tx.blocking_send(Ok(serde_json::to_string(&chunk).unwrap()));
                }

                if answer.len() > last_answer_len {
                    let delta = answer[last_answer_len..].to_string();
                    last_answer_len = answer.len();
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
        let finish_reason = if ids.len() < max_tokens { "stop" } else { "length" };
        let finish = ChatCompletionChunk {
            id,
            object: "chat.completion.chunk",
            created,
            model: model_id,
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta::default(),
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
        }],
        stream: req.stream,
        max_tokens: req.max_tokens,
        max_completion_tokens: req.max_completion_tokens,
        temperature: req.temperature,
        enable_thinking: Some(false),
        reasoning_effort: None,
    };
    chat_completions(State(state), Json(chat)).await
}
