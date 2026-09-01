// SPDX-License-Identifier: Apache-2.0
//! All daemon configuration in one TOML file. Missing file or key falls back to
//! defaults; unknown keys are rejected so typos fail loudly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
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
    /// Unix socket path for the multi-client listener (WU-6). `~` is not
    /// meaningful to the socket API, so resolve it with `expand_tilde`
    /// before binding rather than passing this literally.
    pub socket_path: String,
    /// Independent connections `liam-store` holds open for reads; passed
    /// straight through to `GraphConfig::read_pool_size`. See that field's
    /// doc for why an in-memory database ignores this regardless of what is
    /// configured here.
    pub read_pool_size: usize,
    /// Ceiling on concurrent Unix-socket connections `transport::socket`'s
    /// accept loop holds open at once (WU-6). Without a cap, an unbounded
    /// accept loop can exhaust file descriptors, and each session that ends
    /// up generating holds its own KV cache (measured around 110MB), so
    /// this is a real resource bound, not a nicety. `0` is floored to 1 by
    /// the accept loop rather than silently wedging it: see
    /// `transport::socket::floor_max_connections`.
    pub max_connections: usize,
    /// MCP client name to canonical producer id, plus the fallback id for a
    /// client this map does not name (WU-7).
    pub producers: ProducersConfig,
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
    /// Where both models' files live, each under its own `models--Org--Name`
    /// subdirectory (reranker via `FASTEMBED_CACHE_DIR`, embedder via `with_cache_dir`).
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

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProducersConfig {
    /// Producer id recorded for a connecting MCP client whose name is not a
    /// key in `clients`. Defaults to "unknown", the same fallback
    /// `nodes.producer` uses for a database predating producer tracking, so
    /// an unrecognised client looks the same as a pre-existing row.
    pub unknown_id: String,
    /// MCP client name to canonical producer id (WU-7 resolves each
    /// connection through this). Kept as its own field rather than flattened
    /// into `ProducersConfig` itself: a free-form map of arbitrary keys
    /// cannot carry `deny_unknown_fields`, since every key is unknown by
    /// design, so nesting it under a named key is what lets this struct, and
    /// every other section, keep that strictness.
    pub clients: HashMap<String, String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            database_path: "~/.liam/liam.db".into(),
            log_filter: "info,liam=debug".into(),
            embedding_dims: 768,
            gc: GcConfig::default(),
            embedder: EmbedderConfig::default(),
            llm: LlmConfig::default(),
            ask_timeout_secs: 30,
            ask_sufficiency_check: true,
            socket_path: "~/.liam/liamd.sock".into(),
            read_pool_size: 4,
            max_connections: 16,
            producers: ProducersConfig::default(),
        }
    }
}

impl Default for ProducersConfig {
    fn default() -> Self {
        Self {
            unknown_id: "unknown".into(),
            clients: HashMap::new(),
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

/// Expands a leading `~` to `home`. A bare `~` becomes `home`; `~/rest`
/// becomes `{home}/rest`; anything else, including a path with no leading
/// `~`, is returned unchanged. Takes `home` explicitly rather than reading
/// `$HOME` itself, so this stays a pure function tests can exercise without
/// touching the process environment.
///
/// `socket_path` defaults to `~/.liam/liamd.sock`, and the socket API has no
/// notion of `~`, so a caller must resolve it through this (or an
/// equivalent) before using it as a filesystem path. Callers go through
/// `models::resolve_path_with_home`, which adds the missing-`HOME` check;
/// this stays the pure string half of that pair.
pub(crate) fn expand_tilde(path: &str, home: &str) -> String {
    if path == "~" {
        return home.to_string();
    }
    match path.strip_prefix("~/") {
        Some(rest) => format!("{home}/{rest}"),
        None => path.to_string(),
    }
}

/// The config file read when neither `--config` nor `LIAM_CONFIG` says
/// otherwise.
pub const DEFAULT_CONFIG: &str = "liam.toml";

/// Which config file to read: an explicit `--config` beats `LIAM_CONFIG`,
/// which beats `liam.toml` in the working directory.
///
/// Lives here, rather than in either binary's argument parser, because
/// `liamd` and `liam` have to pick the SAME file. A CLI that resolved this
/// differently would fetch the models one config asks for while the daemon
/// booted from another, and the only symptom would be a download that
/// appears to have done nothing.
///
/// Takes the environment value as an argument rather than reading it here,
/// so the precedence stays a pure function. Environment variables are
/// process-global and `cargo test` runs a binary's tests in one process, so
/// a test that set `LIAM_CONFIG` would race every other test beside it.
pub fn resolve_config_source(flag: Option<&Path>, env_config: Option<&str>) -> PathBuf {
    if let Some(path) = flag {
        return path.to_path_buf();
    }
    match env_config {
        // An unset shell variable expands to the empty string in a launchd
        // plist or a wrapper script, so blank means absent, not a path to "".
        Some(path) if !path.is_empty() => PathBuf::from(path),
        _ => PathBuf::from(DEFAULT_CONFIG),
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
        assert_eq!(c.socket_path, "~/.liam/liamd.sock");
        // The SHIPPED liam.toml is the dev config and keeps a relative
        // database next to the checkout, which is what `cargo run` wants.
        // The built-in default below is the one an installed daemon uses.
        assert_eq!(c.database_path, "liam.db");
        assert_eq!(c.read_pool_size, 4);
        assert_eq!(c.max_connections, 16);
        assert_eq!(c.producers.unknown_id, "unknown");
        assert!(c.producers.clients.is_empty());
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

    #[test]
    fn producers_table_maps_client_names_to_producer_ids() {
        // Given a config with `[producers]` entries
        let path = write_temp_toml(
            "[producers]\n\
             unknown_id = \"guest\"\n\
             \n\
             [producers.clients]\n\
             claude-code = \"claude\"\n\
             ai-notetaker = \"notetaker\"\n",
        );

        // When loaded
        let c = Config::load(&path).expect("config with producers must parse");

        // Then the mapping is available and maps the names given
        assert_eq!(c.producers.unknown_id, "guest");
        assert_eq!(
            c.producers.clients.get("claude-code").map(String::as_str),
            Some("claude")
        );
        assert_eq!(
            c.producers.clients.get("ai-notetaker").map(String::as_str),
            Some("notetaker")
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn unknown_key_under_producers_still_fails_loudly() {
        // Given an unknown key directly under `[producers]` (not nested
        // under `producers.clients`, whose keys are arbitrary by design).
        // Proves `deny_unknown_fields` still guards `ProducersConfig` even
        // though `clients` itself accepts any key.
        let path = write_temp_toml("[producers]\nbogus = \"x\"\n");

        // When loaded
        let err = Config::load(&path).expect_err("unknown key under [producers] must be rejected");

        // Then it still fails loudly
        let message = err.to_string();
        assert!(message.contains("bogus"), "message: {message}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn unknown_top_level_key_still_fails_loudly() {
        // Given an unexpected top-level key
        let path = write_temp_toml("bogus_top_level_key = true\n");

        // When loaded
        let err = Config::load(&path).expect_err("unknown top-level key must be rejected");

        // Then it still fails loudly, confirming adding `producers` did not
        // weaken `deny_unknown_fields` on the rest of the config
        let message = err.to_string();
        assert!(
            message.contains("bogus_top_level_key"),
            "message: {message}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_built_in_database_default_lives_under_the_home_directory() {
        // Given no config file at all, which is what an installed daemon
        // gets before anyone writes one
        let c = Config::default();

        // Then the database lands somewhere stable rather than relative to
        // whatever directory the process happened to start in. A relative
        // default scatters a separate store per working directory, and under
        // a supervisor it resolves against `/`, which is read-only.
        assert_eq!(c.database_path, "~/.liam/liam.db");
        assert!(
            c.database_path.starts_with('~'),
            "the default must be home-relative so it survives any working directory"
        );
    }

    #[test]
    fn tilde_prefixed_socket_path_expands_to_the_given_home_directory() {
        // Given a socket path containing `~`
        // When resolved (against a home this test controls, not $HOME)
        // Then it expands to that home directory
        assert_eq!(
            expand_tilde("~/.liam/liamd.sock", "/home/alice"),
            "/home/alice/.liam/liamd.sock"
        );
    }

    #[test]
    fn a_bare_tilde_expands_to_home_itself() {
        assert_eq!(expand_tilde("~", "/home/alice"), "/home/alice");
    }

    #[test]
    fn a_path_without_a_leading_tilde_is_unchanged() {
        assert_eq!(
            expand_tilde("/var/run/liamd.sock", "/home/alice"),
            "/var/run/liamd.sock"
        );
    }
}
