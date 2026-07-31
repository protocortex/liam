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
