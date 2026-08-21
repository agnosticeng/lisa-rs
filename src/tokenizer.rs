// Qwen3 chat tokenizer + chat template (text-only path).
//
// The tokenizer is loaded from the HF `tokenizer.json` (byte-level BPE, Qwen2
// backend). The chat template is the Qwen3 `chat_template.jinja` text path,
// ported to Rust: it reproduces the default `enable_thinking=true` /
// `reasoning_effort="xhigh"` behavior (reasoning instructions injected as a
// system prefix) and the `<|im_start|>...<|im_end|>` turn framing.
use std::path::Path;

use tokenizers::Tokenizer as HfTokenizer;

/// A chat role's text content (we handle the text-only path).
#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
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

        // System message (only ever the first message) with optional reasoning
        // instructions prepended.
        if messages[0].role == "system" {
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

        for message in messages {
            match message.role.as_str() {
                "system" => {}
                "user" => {
                    prompt.push_str("<|im_start|>user\n");
                    prompt.push_str(message.content.trim());
                    prompt.push_str("<|im_end|>\n");
                }
                "assistant" => {
                    prompt.push_str("<|im_start|>assistant\n");
                    prompt.push_str(message.content.trim());
                    prompt.push_str("<|im_end|>\n");
                }
                "tool" => {
                    return Err("tool messages are not supported yet".to_string());
                }
                other => return Err(format!("unexpected message role {other:?}")),
            }
        }

        // Generation prompt (add_generation_prompt).
        prompt.push_str("<|im_start|>assistant\n");
        if enable_thinking {
            prompt.push_str("<think>\n");
        } else {
            prompt.push_str("<think>\n\n</think>\n\n");
        }
        Ok(prompt)
    }

    /// Split a generated turn into `(reasoning, answer)`. The generation prompt
    /// already emits the opening `<think>` tag, so the generated text is
    /// `{reasoning}\n</think>\n\n{answer}` when thinking is enabled (the answer
    /// is everything after `</think>`); without thinking the whole text is the
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
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "What is 2+2?".to_string(),
            },
        ];
        let prompt = tok.apply_chat_template(&messages, true, None).unwrap();
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
        assert_eq!(ids, expected, "chat template token ids must match transformers");
    }
}
