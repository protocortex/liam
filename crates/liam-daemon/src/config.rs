//! All daemon configuration in one TOML file. Missing file or key falls back to
//! defaults; unknown keys are rejected so typos fail loudly.

use std::path::Path;
use std::time::Duration;

use liam_store::{Millis, RetentionPolicy};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub database_path: String,
    pub log_filter: String,
    pub embedding_dims: usize,
    pub gc: GcConfig,
    pub embedder: EmbedderConfig,
    pub llm: LlmConfig,
    /// Wall-clock cap on `ask` synthesis before falling back to ranked evidence.
    pub ask_timeout_secs: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GcConfig {
    pub episode_retention_days: i64,
    pub interval_hours: u64,
    pub reclaim: bool,
    pub run_on_start: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmbedderConfig {
    /// "mock" (dev) or "local" (in-process fastembed; needs the `local` feature).
    pub provider: String,
    /// Hugging Face model id for the local provider.
    pub model: String,
    /// Where model files live. The installer pre-populates this so first run is
    /// offline. Sets FASTEMBED_CACHE_DIR.
    pub cache_dir: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmConfig {
    /// "mock" (dev) or "local" (in-process candle; needs the `local` feature).
    pub provider: String,
    /// Hugging Face model id (GGUF repo) for the local provider.
    pub model: String,
    /// GGUF filename within the repo (GGUF repos host multiple quant variants,
    /// so the file must be named explicitly). Consumed by `CandleLlm::load`.
    pub gguf_file: String,
    /// Where model files live (offline after first fetch).
    pub cache_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            database_path: "liam.db".into(),
            log_filter: "info,liam=debug".into(),
            embedding_dims: 768,
            gc: GcConfig::default(),
            embedder: EmbedderConfig::default(),
            llm: LlmConfig::default(),
            ask_timeout_secs: 30,
        }
    }
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            episode_retention_days: 30,
            interval_hours: 6,
            reclaim: true,
            run_on_start: false,
        }
    }
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            provider: "mock".into(),
            model: "Qwen/Qwen3-Embedding-0.6B".into(),
            cache_dir: "~/.liam/models".into(),
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "mock".into(),
            model: "Qwen/Qwen2.5-0.5B-Instruct-GGUF".into(),
            gguf_file: "qwen2.5-0.5b-instruct-q4_k_m.gguf".into(),
            cache_dir: "~/.liam/models".into(),
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        if path.exists() {
            let text = std::fs::read_to_string(path)?;
            Ok(toml::from_str(&text)?)
        } else {
            Ok(Self::default())
        }
    }

    pub fn gc_policy(&self) -> RetentionPolicy {
        RetentionPolicy::keep("episode", Millis::days(self.gc.episode_retention_days))
            .without_reclaim_unless(self.gc.reclaim)
    }

    pub fn gc_interval(&self) -> Duration {
        Duration::from_secs(self.gc.interval_hours * 3600)
    }
}

// Small ergonomic shim so config reads naturally.
trait ReclaimExt {
    fn without_reclaim_unless(self, reclaim: bool) -> Self;
}
impl ReclaimExt for RetentionPolicy {
    fn without_reclaim_unless(self, reclaim: bool) -> Self {
        if reclaim {
            self
        } else {
            self.without_reclaim()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_defaults_to_mock() {
        let c = Config::default();
        assert_eq!(c.llm.provider, "mock");
    }
}
