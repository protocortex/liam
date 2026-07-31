//! Generative completion: turn a prompt into text. The store never does this;
//! the daemon uses it for synthesis (M2) and extraction (M3).

use async_trait::async_trait;

use crate::error::Result;

#[async_trait]
pub trait Llm: Send + Sync {
    /// Generate a completion for `prompt` under `system` guidance.
    async fn complete(&self, system: &str, prompt: &str) -> Result<String>;
}

/// Deterministic echo LLM for the base build and tests: no model, stable output.
pub struct MockLlm;

#[async_trait]
impl Llm for MockLlm {
    async fn complete(&self, system: &str, prompt: &str) -> Result<String> {
        Ok(format!("[mock] system={system} prompt={prompt}"))
    }
}

/// In-process chat model over candle (feature `local`). Loads a quantized
/// instruct model by Hugging Face id; no server, offline once cached. The sync
/// generate loop runs on a blocking thread so the async runtime stays free.
///
/// VERSION CHECK: the candle-transformers generation API (model constructor,
/// `forward`, logits processing) moves across releases. Confirm the
/// quantized-model surface against the candle line pinned by candle-core 0.10
/// before relying on this in production.
#[cfg(feature = "local")]
pub struct CandleLlm {
    inner: std::sync::Arc<std::sync::Mutex<candle_chat::Session>>,
}

#[cfg(feature = "local")]
impl CandleLlm {
    /// Load a quantized instruct model by HF id and GGUF filename (e.g.
    /// `Qwen/Qwen2.5-0.5B-Instruct-GGUF`, `qwen2.5-0.5b-instruct-q4_k_m.gguf`),
    /// caching weights under `cache_dir`.
    pub fn load(model_id: &str, gguf_file: &str, cache_dir: &str) -> Result<Self> {
        let session = candle_chat::Session::load(model_id, gguf_file, cache_dir)
            .map_err(|e| crate::error::ModelError::Llm(e.to_string()))?;
        Ok(Self { inner: std::sync::Arc::new(std::sync::Mutex::new(session)) })
    }
}

#[cfg(feature = "local")]
#[async_trait]
impl Llm for CandleLlm {
    async fn complete(&self, system: &str, prompt: &str) -> Result<String> {
        let inner = self.inner.clone();
        let system = system.to_string();
        let prompt = prompt.to_string();
        let out = tokio::task::spawn_blocking(move || {
            let mut s = inner
                .lock()
                .map_err(|_| crate::error::ModelError::Llm("model lock poisoned".into()))?;
            s.complete(&system, &prompt)
                .map_err(|e| crate::error::ModelError::Llm(e.to_string()))
        })
        .await
        .map_err(|e| crate::error::ModelError::Llm(e.to_string()))??;
        Ok(out)
    }
}

/// Version-fragile candle glue, isolated from the `Llm` trait boundary above.
/// Holds the quantized-model construction, tokenizer, and a greedy/argmax
/// decode loop over candle-transformers' `quantized_qwen2` model.
#[cfg(feature = "local")]
mod candle_chat {
    use candle_core::quantized::gguf_file;
    use candle_core::{Device, Tensor};
    use candle_transformers::generation::{LogitsProcessor, Sampling};
    use candle_transformers::models::quantized_qwen2::ModelWeights;

    /// Hard cap on generated tokens so a missing EOS can't hang the caller.
    const MAX_NEW_TOKENS: usize = 512;

    pub struct Session {
        model: ModelWeights,
        tokenizer: tokenizers::Tokenizer,
        device: Device,
        eos_token: u32,
    }

    impl Session {
        /// Download (if needed) and load a quantized GGUF chat model plus its
        /// tokenizer, mirroring `FastEmbedEmbedder::load`'s explicit
        /// download-then-construct pattern so the cache lives under
        /// `cache_dir` rather than the default HF cache.
        pub fn load(model_id: &str, gguf_file: &str, cache_dir: &str) -> anyhow::Result<Self> {
            let api = hf_hub::api::sync::ApiBuilder::new()
                .with_cache_dir(std::path::PathBuf::from(cache_dir))
                .build()?;
            let repo = api.model(model_id.to_string());

            let weights_path = repo.get(gguf_file)?;
            let tokenizer_path = repo.get("tokenizer.json")?;

            let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
                .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

            let device = Device::Cpu;
            let mut file = std::fs::File::open(&weights_path)?;
            let content = gguf_file::Content::read(&mut file)
                .map_err(|e| e.with_path(weights_path.display().to_string()))?;
            let model = ModelWeights::from_gguf(content, &mut file, &device)?;

            let eos_token = tokenizer
                .token_to_id("<|im_end|>")
                .or_else(|| tokenizer.token_to_id("</s>"))
                .ok_or_else(|| anyhow::anyhow!("tokenizer has no recognizable EOS token"))?;

            Ok(Self { model, tokenizer, device, eos_token })
        }

        /// Greedy (argmax) generation: encode the prompt, run the prefill
        /// forward pass, then decode one token at a time until EOS or the
        /// max-new-tokens cap.
        pub fn complete(&mut self, system: &str, prompt: &str) -> anyhow::Result<String> {
            // ChatML-style template; matches the tokenizer's <|im_start|>/
            // <|im_end|> special tokens used by Qwen2.5-Instruct GGUF builds.
            let text = format!(
                "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            );

            let encoding = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
            let mut tokens = encoding.get_ids().to_vec();
            if tokens.is_empty() {
                anyhow::bail!("empty prompt encoding");
            }

            let mut logits_processor = LogitsProcessor::from_sampling(42, Sampling::ArgMax);
            let mut generated: Vec<u32> = Vec::new();
            let mut pos = 0usize;

            // Prefill: feed the whole prompt, then decode one token at a time.
            let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?;
            pos += tokens.len();
            let mut next_token = logits_processor.sample(&logits)?;

            for _ in 0..MAX_NEW_TOKENS {
                if next_token == self.eos_token {
                    break;
                }
                generated.push(next_token);
                tokens.push(next_token);

                let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
                let logits = self.model.forward(&input, pos)?;
                let logits = logits.squeeze(0)?;
                pos += 1;
                next_token = logits_processor.sample(&logits)?;
            }

            let text = self
                .tokenizer
                .decode(&generated, true)
                .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;
            Ok(text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_llm_is_deterministic_and_echoes_prompt() {
        let llm = MockLlm;
        let a = llm.complete("be terse", "hello").await.unwrap();
        let b = llm.complete("be terse", "hello").await.unwrap();
        assert_eq!(a, b, "same input yields same output");
        assert!(a.contains("hello"), "output reflects the prompt");
    }
}
