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
    /// Whether `ask` runs a yes/no sufficiency pre-pass before synthesizing. On
    /// by default: without it no local model tested will decline to answer a
    /// question its evidence cannot answer. Costs one extra short model call per
    /// `ask`, so turn it off if latency matters more than refusals.
    pub ask_sufficiency_check: bool,
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
    /// "mock" (dev) or "llama-cpp" (in-process llama.cpp; needs the `llama`
    /// feature). "local" was the retired candle provider.
    pub provider: String,
    /// Hugging Face model id (GGUF repo) for the local provider.
    pub model: String,
    /// GGUF filename within the repo (GGUF repos host multiple quant variants,
    /// so the file must be named explicitly). Consumed by the llama.cpp
    /// provider's loader.
    pub gguf_file: String,
    /// Where model files live (offline after first fetch).
    pub cache_dir: String,
    /// Compute backend: "auto" (default), "metal", "cuda", or "cpu". `auto` takes
    /// the fastest backend compiled into this binary and falls back to CPU. macOS
    /// builds include Metal automatically; CUDA needs `--features cuda` at build
    /// time. See `liam_model::llm::DevicePreference`.
    pub device: String,
    /// Generate one throwaway token at startup so the backend's first-call cost
    /// (on Metal, ~10s of GPU kernel compilation) is paid before a user waits.
    pub warmup: bool,
    /// The model's context window. Sizes both the llama.cpp context and the
    /// evidence trimming in `ask`, because those two have to agree: if they
    /// disagreed a full-size prompt would overflow the context and decode
    /// would fail.
    pub context_tokens: usize,
    /// Generation is serialized by default: each concurrent context costs about
    /// 110MB of KV cache, while measured parallel throughput on a saturated GPU
    /// gains only 1.13x. Operators with headroom can raise this.
    pub max_concurrent_generations: usize,
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
            ask_sufficiency_check: true,
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
            // Chosen by measurement, not by size: on the grounding eval
            // (crates/liam-daemon/src/eval.rs) 1.5B scored 3/4 judged cases at
            // ~4.8s per answer, against 2/4 for 0.5B, 3/4 for Gemma-3-1B at ~9.3s,
            // and 2/4 for Qwen3-1.7B. Apache-2.0 at this size (the 3B is not).
            model: "Qwen/Qwen2.5-1.5B-Instruct-GGUF".into(),
            gguf_file: "qwen2.5-1.5b-instruct-q4_k_m.gguf".into(),
            cache_dir: "~/.liam/models".into(),
            device: "auto".into(),
            warmup: true,
            context_tokens: 8192,
            max_concurrent_generations: 1,
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path)?;

        // Parse into a permissive `toml::Value` first so we can name a removed
        // key with a helpful error before the strict parse below rejects it as
        // just another unknown field. We check the parsed value rather than
        // matching on toml's error string: error text is not part of the
        // crate's stability contract, so a string match would silently stop
        // catching this the next time the `toml` dependency is upgraded.
        let value: toml::Value = toml::from_str(&text)?;
        if value
            .get("llm")
            .and_then(|llm| llm.get("tokenizer_model"))
            .is_some()
        {
            anyhow::bail!(
                "llm.tokenizer_model was removed: llama.cpp reads the tokenizer \
                 from the GGUF, delete this line from your liam.toml"
            );
        }

        Ok(toml::from_str(&text)?)
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

    /// Writes `contents` to a fresh, uniquely-named temp file so tests exercise
    /// the real `Config::load` file-reading path without touching the repo's
    /// `liam.toml`. The counter plus pid keeps filenames unique across the
    /// parallel threads `cargo test` runs within one process.
    fn write_temp_toml(contents: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "liam-daemon-config-test-{}-{unique}.toml",
            std::process::id()
        ));
        std::fs::write(&path, contents).expect("write temp config fixture");
        path
    }

    #[test]
    fn llm_defaults_to_mock() {
        let c = Config::default();
        assert_eq!(c.llm.provider, "mock");
    }

    #[test]
    fn shipped_liam_toml_parses() {
        // WHY: `deny_unknown_fields` plus TOML's table scoping (a top-level key
        // written after a `[table]` header belongs to that table) make it easy to
        // ship a config the daemon rejects at startup. Parsing the real file in
        // CI catches that before an operator does.
        let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../liam.toml"));
        let c = Config::load(path).expect("shipped liam.toml must parse");
        assert_eq!(c.ask_timeout_secs, 30);
        assert!(c.ask_sufficiency_check);
        assert_eq!(c.llm.context_tokens, 8192);
        assert_eq!(c.llm.max_concurrent_generations, 1);
    }

    #[test]
    fn llm_device_default_is_a_value_the_model_crate_accepts() {
        // The daemon rejects an unparseable device at startup, so the shipped
        // default must parse. This catches a typo in our own default.
        let c = Config::default();
        assert_eq!(
            liam_model::llm::DevicePreference::parse(&c.llm.device),
            Some(liam_model::llm::DevicePreference::Auto)
        );
        assert!(c.llm.warmup, "warmup should be on by default");
    }

    #[test]
    fn stale_tokenizer_model_key_names_itself_and_the_fix() {
        let path = write_temp_toml("[llm]\ntokenizer_model = \"Qwen/Qwen2.5-1.5B-Instruct\"\n");
        let err = Config::load(&path).expect_err("stale tokenizer_model must be rejected");
        let message = err.to_string();
        assert!(
            message.contains("llm.tokenizer_model"),
            "error should name the removed key: {message}"
        );
        assert!(
            message.contains("delete"),
            "error should say to delete the line: {message}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn llm_context_and_concurrency_default_when_absent() {
        let path = write_temp_toml("[llm]\nprovider = \"mock\"\n");
        let c = Config::load(&path).expect("config without the new llm keys must still parse");
        assert_eq!(c.llm.context_tokens, 8192);
        assert_eq!(c.llm.max_concurrent_generations, 1);
        let _ = std::fs::remove_file(&path);
    }
}
