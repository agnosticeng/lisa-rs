pub mod cache; // HF cache snapshot resolution
pub mod cli; // CLI subcommands (serve / run)
pub mod device; // Metal host bridge (P0)
pub mod format; // safetensors container + dtype (P0)
pub mod kernels; // .metal shaders + host dispatch (P1/P2)
pub mod model; // hybrid GDN + attention layers, KV cache (P3)
pub mod speculative; // MTP speculative decode (P4)
pub mod tokenizer; // Qwen3 chat tokenizer + template
pub mod serve; // OpenAI-compatible HTTP server
