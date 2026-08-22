// Qwen3 chat tokenizer + chat template (text-only path).
//
// The tokenizer is loaded from the HF `tokenizer.json` (byte-level BPE, Qwen2
// backend). The chat template is the Qwen3 `chat_template.jinja` text path,
// ported to Rust: it reproduces the default `enable_thinking=true` /
// `reasoning_effort="xhigh"` behavior (reasoning instructions injected as a
// system prefix) and the `<|im_start|>...<|im_end|>` turn framing.
use std::path::Path;

use tokenizers::Tokenizer as HfTokenizer;

/// One parsed/rendered tool call. `arguments` is a JSON object (OpenAI style).
#[derive(Clone, Debug)]
pub struct ChatToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// A chat message. `content`/`name` are text; assistant messages may carry
/// `tool_calls` and `reasoning_content`; tool-response messages use `"tool"`.
#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ChatToolCall>,
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            name: None,
        }
    }

    /// Assistant message that issued tool calls (content is the pre-call text).
    pub fn assistant_with_tool_calls(content: impl Into<String>, tool_calls: Vec<ChatToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            reasoning_content: None,
            tool_calls,
            name: None,
        }
    }
}

pub struct ChatTokenizer {
    inner: HfTokenizer,
    eos: u32,
    im_start: u32,
    im_end: u32,
}

impl Clone for ChatTokenizer {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            eos: self.eos,
            im_start: self.im_start,
            im_end: self.im_end,
        }
    }
}

fn id_of(tok: &HfTokenizer, text: &str) -> Result<u32, String> {
    tok.token_to_id(text)
        .ok_or_else(|| format!("tokenizer missing special token {text:?}"))
}

impl ChatTokenizer {
    pub fn load(path: &Path) -> Result<Self, String> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| format!("failed to load tokenizer {}: {e}", path.display()))?;
        let eos = id_of(&inner, "<|im_end|>")?;
        let im_start = id_of(&inner, "<|im_start|>")?;
        let im_end = id_of(&inner, "<|im_end|>")?;
        Ok(Self {
            inner,
            eos,
            im_start,
            im_end,
        })
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }

    pub fn im_start(&self) -> u32 {
        self.im_start
    }

    pub fn im_end(&self) -> u32 {
        self.im_end
    }

    /// Tokenize text (special tokens in the text are recognized as-is).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| format!("tokenize failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode a token sequence back to text (special tokens skipped).
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| format!("detokenize failed: {e}"))
    }

    /// Qwen3 chat template (text path). `enable_thinking` defaults to true and
    /// mirrors the jinja default; `reasoning_effort` is one of xhigh/medium/low.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        enable_thinking: bool,
        reasoning_effort: Option<&str>,
        tools: &[String],
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        if messages.is_empty() {
            return Err("no messages provided".to_string());
        }
        let effort = reasoning_effort.unwrap_or("xhigh");
        let instructions = match effort {
            "xhigh" => Some(
                "Reasoning effort is set to xhigh. Please think carefully through the task, \
                 validate key assumptions, consider plausible alternatives, and prioritize \
                 correctness, consistency, and clarity in the final answer.",
            ),
            "low" => Some(
                "Reasoning effort is set to low. Keep your thinking brief and focused, moving \
                 directly to the conclusion without unnecessary elaboration.",
            ),
            "medium" => None,
            other => return Err(format!("unsupported reasoning effort {other:?}")),
        };

        let mut prompt = String::new();

        // System message (with tools, the system turn carries the tool
        // definitions + call format, then the real system content).
        let has_tools = !tools.is_empty();
        if has_tools {
            prompt.push_str("<|im_start|>system\n");
            if let Some(instructions) = instructions {
                prompt.push_str(instructions);
                prompt.push_str("\n\n");
            }
            prompt.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
            for tool in tools {
                prompt.push('\n');
                prompt.push_str(tool);
            }
            prompt.push_str(
                "\n</tools>\n\nIf you choose to call a function ONLY reply in the following \
                 format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n\
                 <parameter=example_parameter_1>\nvalue_1\n</parameter>\n\
                 <parameter=example_parameter_2>\nThis is the value for the second parameter\n\
                 that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n\
                 <IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: \
                 an inner <function=...></function> block must be nested within \
                 <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n\
                 - You may provide optional reasoning for your function call in natural \
                 language BEFORE the function call, but NOT after\n- If there is no function \
                 call available, answer the question like normal with your current knowledge \
                 and do not tell the user about function calls\n</IMPORTANT>",
            );
            if messages[0].role == "system" && !messages[0].content.trim().is_empty() {
                prompt.push_str("\n\n");
                prompt.push_str(messages[0].content.trim());
            }
            prompt.push_str("<|im_end|>\n");
        } else if messages[0].role == "system" {
            let content = messages[0].content.trim();
            if !content.is_empty() {
                prompt.push_str("<|im_start|>system\n");
                if enable_thinking && let Some(instructions) = instructions {
                    prompt.push_str(instructions);
                    prompt.push_str("\n\n");
                }
                prompt.push_str(content);
                prompt.push_str("<|im_end|>\n");
            } else if enable_thinking && let Some(instructions) = instructions {
                prompt.push_str("<|im_start|>system\n");
                prompt.push_str(instructions);
                prompt.push_str("<|im_end|>\n");
            }
        } else if enable_thinking && let Some(instructions) = instructions {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(instructions);
            prompt.push_str("<|im_end|>\n");
        }

        for (index, message) in messages.iter().enumerate() {
            match message.role.as_str() {
                "system" => {}
                "user" => {
                    prompt.push_str("<|im_start|>user\n");
                    prompt.push_str(message.content.trim());
                    prompt.push_str("<|im_end|>\n");
                }
                "assistant" => {
                    prompt.push_str("<|im_start|>assistant\n");
                    if let Some(reasoning) = message.reasoning_content.as_deref() {
                        if !reasoning.trim().is_empty() {
                            prompt.push_str(" thinking\n");
                            prompt.push_str(reasoning.trim());
                            prompt.push_str("\n response\n\n");
                        }
                    } else if !message.content.trim().is_empty() || !message.tool_calls.is_empty() {
                        prompt.push_str(" thinking\n\n response\n\n");
                    }
                    prompt.push_str(message.content.trim());
                    if !message.tool_calls.is_empty() {
                        for (ci, call) in message.tool_calls.iter().enumerate() {
                            if ci == 0 {
                                if !message.content.trim().is_empty() {
                                    prompt.push_str("\n\n<tool_call>\n<function=");
                                } else {
                                    prompt.push_str("<tool_call>\n<function=");
                                }
                            } else {
                                prompt.push_str("\n<tool_call>\n<function=");
                            }
                            prompt.push_str(&call.name);
                            prompt.push_str(">\n");
                            if let serde_json::Value::Object(map) = &call.arguments {
                                for (key, value) in map {
                                    prompt.push_str("<parameter=");
                                    prompt.push_str(key);
                                    prompt.push_str(">\n");
                                    if let serde_json::Value::String(ss) = value {
                                        prompt.push_str(ss);
                                    } else {
                                        prompt.push_str(&value.to_string());
                                    }
                                    prompt.push_str("\n</parameter>\n");
                                }
                            }
                            prompt.push_str("</function>\n</tool_call>");
                        }
                    }
                    prompt.push_str("<|im_end|>\n");
                }
                "tool" => {
                    let prev_is_tool = index > 0 && messages[index - 1].role == "tool";
                    let next_is_tool =
                        index + 1 < messages.len() && messages[index + 1].role == "tool";
                    if !prev_is_tool {
                        prompt.push_str("<|im_start|>user");
                    }
                    prompt.push_str("\n<tool_response>\n");
                    prompt.push_str(message.content.trim());
                    prompt.push_str("\n</tool_response>");
                    if !next_is_tool || (index + 1 == messages.len()) {
                        prompt.push_str("<|im_end|>\n");
                    }
                }
                other => return Err(format!("unexpected message role {other:?}")),
            }
        }

        // Generation prompt (add_generation_prompt). When false, only the
        // conversation prefix is rendered (used for prefix-cache matching).
        if add_generation_prompt {
            prompt.push_str("<|im_start|>assistant\n");
            if enable_thinking {
                prompt.push_str(" thinking\n");
            } else {
                prompt.push_str(" thinking\n\n response\n\n");
            }
        }
        Ok(prompt)
    }

    /// Split a generated turn into `(reasoning, answer)`. The generation prompt
    /// already emits the opening ` thinking` tag, so the generated text is
    /// `{reasoning}\n response\n\n{answer}` when thinking is enabled (the answer
    /// is everything after ` response`); without thinking the whole text is the
    /// answer.
    pub fn split_thinking(&self, text: &str, enable_thinking: bool) -> (String, String) {
        if enable_thinking {
            if let Some(end) = text.find("</think>") {
                let reasoning = text[..end].trim().to_string();
                let mut answer = text[end + "</think>".len()..].to_string();
                while answer.starts_with('\n') {
                    answer.remove(0);
                }
                (reasoning, answer)
            } else {
                (text.trim().to_string(), String::new())
            }
        } else {
            (String::new(), text.to_string())
        }
    }
}

/// Split a text on a pair of XML-like tags, returning the inner bodies.
fn between_all<'a>(text: &'a str, open: &str, close: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut rest = text;
    loop {
        let Some(s) = rest.find(open) else { break };
        let s = s + open.len();
        let Some(mut e) = rest[s..].find(close) else { break };
        e += s;
        out.push(&rest[s..e]);
        rest = &rest[e + close.len()..];
    }
    out
}

fn between<'a>(text: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let s = text.find(open)? + open.len();
    let e = text[s..].find(close)? + s;
    Some(&text[s..e])
}

/// Try to parse `value` as JSON; a bare non-JSON scalar is returned as a
/// string, matching the way the template renders arg values.
fn coerce_arg(value: &str) -> serde_json::Value {
    let trimmed = value.trim();
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        v
    } else {
        serde_json::Value::String(trimmed.to_string())
    }
}

/// Parse the `<tool_call>` blocks the model emits into structured tool calls.
/// Format (Qwen tools template): `<tool_call>\n<function=NAME>\n
/// <parameter=K>\nVALUE\n</parameter>\n...</function>\n</tool_call>`.
pub fn parse_tool_calls(text: &str) -> Vec<ChatToolCall> {
    let mut calls = Vec::new();
    for block in between_all(text, "<tool_call>", "</tool_call>") {
        let Some(fn_block) = between(block, "<function=", "</function>") else {
            continue;
        };
        let name = fn_block.split('>').next().unwrap_or("").trim().to_string();
        let body = fn_block.splitn(2, '>').nth(1).unwrap_or("");
        let mut arguments = serde_json::Map::new();
        for (key, value) in extract_parameters(body) {
            arguments.insert(key, coerce_arg(&value));
        }
        calls.push(ChatToolCall {
            name,
            arguments: serde_json::Value::Object(arguments),
        });
    }
    calls
}

fn extract_parameters<'a>(body: &'a str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for block in between_all(body, "<parameter=", "</parameter>") {
        let key = block.split('>').next().unwrap_or("").trim().to_string();
        let value = block.splitn(2, '>').nth(1).unwrap_or("").trim().to_string();
        if !key.is_empty() {
            out.push((key, value));
        }
    }
    out
}

/// Split generated text into `(reasoning, answer, tool_calls)`: the answer is
/// the copy of the generated text with any trailing `<tool_call>` XML removed
/// (tool-call blocks are surfaced separately as structured calls).
pub fn split_tool_calls(reasoning: String, answer: String) -> (String, String, Vec<ChatToolCall>) {
    let calls = parse_tool_calls(&answer);
    if calls.is_empty() {
        return (reasoning, answer, calls);
    }
    if let Some(pos) = answer.find("<tool_call>") {
        let content = answer[..pos].trim().to_string();
        (reasoning, content, calls)
    } else {
        (reasoning, String::new(), calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load() -> Option<ChatTokenizer> {
        let root = std::env::home_dir()?.join(
            ".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots",
        );
        let dir = std::fs::read_dir(root).ok()?.filter_map(Result::ok).map(|e| e.path()).find(|p| p.is_dir())?;
        ChatTokenizer::load(&dir.join("tokenizer.json")).ok()
    }

    #[test]
    fn round_trips_text() {
        let Some(tok) = load() else {
            eprintln!("skip: tokenizer not cached");
            return;
        };
        let text = "Hello, world! This is a test.";
        let ids = tok.encode(text).unwrap();
        let back = tok.decode(&ids).unwrap();
        assert_eq!(back.trim(), text);
    }

    #[test]
    fn special_tokens_are_single_ids() {
        let Some(tok) = load() else {
            eprintln!("skip: tokenizer not cached");
            return;
        };
        let ids = tok.encode("<|im_start|>system\nhi<|im_end|>").unwrap();
        assert_eq!(ids[0], tok.im_start());
        assert_eq!(*ids.last().unwrap(), tok.im_end());
    }

    #[test]
    fn split_tool_calls_extracts_structured_calls() {
        let text = "Let me check the README.\n\n<tool_call>\n<function=read>\n<parameter=file_path>\n/Users/x/README.md\n</parameter>\n</function>\n</tool_call>";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(
            calls[0].arguments["file_path"],
            serde_json::Value::String("/Users/x/README.md".to_string())
        );

        // split_tool_calls strips the XML from the answer content.
        let (reasoning, content, calls2) = split_tool_calls(String::new(), text.to_string());
        assert_eq!(reasoning, "");
        assert_eq!(content, "Let me check the README.");
        assert_eq!(calls2.len(), 1);

        // Multiple calls.
        let multi = "<tool_call>\n<function=a>\n<parameter=k>\n1\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n<parameter=k>\n2\n</parameter>\n</function>\n</tool_call>";
        let calls3 = parse_tool_calls(multi);
        assert_eq!(calls3.len(), 2);
        assert_eq!(calls3[0].name, "a");
        assert_eq!(calls3[1].name, "b");

        // Non-JSON scalar args are coerced to strings, numbers to numbers.
        let num = "<tool_call>\n<function=f>\n<parameter=n>\n42\n</parameter>\n</function>\n</tool_call>";
        let calls4 = parse_tool_calls(num);
        assert_eq!(calls4[0].arguments["n"], serde_json::json!(42));
    }

    #[test]
    fn splits_thinking_block() {
        let Some(tok) = load() else {
            eprintln!("skip: tokenizer not cached");
            return;
        };
        let (reasoning, answer) = tok.split_thinking("I should think.\n</think>\n\n42", true);
        assert_eq!(reasoning, "I should think.");
        assert_eq!(answer, "42");

        let (reasoning, answer) = tok.split_thinking("just the answer", false);
        assert_eq!(reasoning, "");
        assert_eq!(answer, "just the answer");
    }

    #[test]
    fn chat_template_matches_transformers_reference() {
        let Some(tok) = load() else {
            eprintln!("skip: tokenizer not cached");
            return;
        };
        let messages = vec![
            ChatMessage::new("system", "You are a helpful assistant."),
            ChatMessage::new("user", "What is 2+2?"),
        ];
        let prompt = tok.apply_chat_template(&messages, true, None, &[], true).unwrap();
        let ids = tok.encode(&prompt).unwrap();

        // Reference produced by `transformers.AutoTokenizer.apply_chat_template`
        // on the same checkpoint (enable_thinking=true, reasoning_effort=xhigh).
        let expected: Vec<u32> = vec![
            248045, 8678, 198, 24342, 286, 4879, 369, 716, 310, 830, 11553, 13, 5044, 1683, 15060,
            1472, 279, 3274, 11, 9307, 1328, 30800, 11, 2814, 47675, 25605, 11, 321, 60445, 55404,
            11, 27224, 11, 321, 30246, 303, 279, 1534, 4087, 13, 271, 2523, 513, 264, 10631, 17313,
            13, 248046, 198, 248045, 846, 198, 3710, 369, 220, 17, 10, 17, 30, 248046, 198, 248045,
            74455, 198, 248068, 198,
        ];
        // The tokenizers crate may resolve the generated ` thinking` tag to
        // either of two equivalent added tokens (7047 / 248068, both decode to
        // " thinking"); accept either while keeping every other token exact.
        // The ambiguous id is second-to-last (followed by a trailing newline).
        if ids.len() >= 2 && expected.len() >= 2 {
            let a = ids[ids.len() - 2];
            let e = expected[expected.len() - 2];
            let is_thinking_tag = a == 7047 || a == 248068;
            if a != e && is_thinking_tag {
                assert_eq!(
                    &ids[..ids.len() - 2],
                    &expected[..expected.len() - 2],
                    "chat template token ids must match transformers (thinking tag)"
                );
                return;
            }
        }
        assert_eq!(ids, expected, "chat template token ids must match transformers");
    }
}
