// SPDX-License-Identifier: Apache-2.0
//! Config path resolution and model construction, shared by `liamd` and `liam`.
//!
//! The daemon calls these at startup to build what it serves with. The CLI
//! calls the same functions from `liam fetch-models` to download and verify
//! those models ahead of time. Sharing them is the point: a fetch that
//! resolved a cache directory, a model id, or a device differently from the
//! daemon would report success and still leave the daemon unable to start.
//!
//! Every loader here is synchronous, so the CLI needs no async runtime to
//! drive them.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use liam_model::llm::DevicePreference;
use liam_model::{ArcticEmbedder, Embedder, Llm, MockEmbedder, MockLlm, NoticeCaller, TextKind};
use liam_store::{ContentEmbedder, EmbedError};

use crate::config::{self, Config};
use crate::lazy_llm::{LazyLlm, Loader};

/// Expands `~` in a configured path. Nothing below the config layer
/// understands a home directory: the socket API would create a directory
/// literally named `~`, libSQL would create a database inside one, and the
/// model loaders would download gigabytes into one.
///
/// A tilde path with no `HOME` set is an error rather than a default,
/// because the silent version is worse than useless: an empty home turns
/// `~/.liam/liamd.sock` into `/.liam/liamd.sock`, at the filesystem root,
/// which fails later with a permission error naming a path the operator
/// never configured. Sparse environments are a real case here, since a
/// launchd job only has the variables its plist declares.
///
/// `key` names the config field so the error tells the operator which line
/// to fix rather than making them guess which path was at fault.
pub fn resolve_config_path(key: &str, value: &str) -> anyhow::Result<std::path::PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    Ok(std::path::PathBuf::from(resolve_path_with_home(
        key, value, &home,
    )?))
}

/// The same rule as `resolve_config_path`, yielding a `String` because the
/// model loaders take `&str` paths rather than `Path`, and taking `home` as an
/// argument so it is testable without mutating the process environment (which
/// would race the other tests in this binary).
pub fn resolve_path_with_home(key: &str, value: &str, home: &str) -> anyhow::Result<String> {
    if home.is_empty() && value.starts_with('~') {
        anyhow::bail!(
            "{key} is {value:?} but HOME is not set, so `~` cannot be expanded. \
             Set HOME, or write an absolute {key} in your config."
        );
    }
    Ok(config::expand_tilde(value, home))
}

/// The directory of the event log for the database at `database`: `[log] dir`
/// when set, else the default beside the database. The configured one has `~`
/// expanded and, when relative, is taken from the database's directory so the
/// log does not move with the working directory. `liamd` and `liam rebuild`
/// both go through this so they open the same log.
pub fn resolve_log_dir(config: &Config, database: &std::path::Path) -> anyhow::Result<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    resolve_log_dir_with_home(config, database, &home)
}

/// `resolve_log_dir` with `home` passed in, so it is testable without
/// mutating the process environment.
fn resolve_log_dir_with_home(
    config: &Config,
    database: &std::path::Path,
    home: &str,
) -> anyhow::Result<PathBuf> {
    match &config.log.dir {
        Some(dir) => {
            let dir = PathBuf::from(resolve_path_with_home("log.dir", dir, home)?);
            Ok(match database.parent() {
                Some(parent) if dir.is_relative() => parent.join(dir),
                _ => dir,
            })
        }
        None => Ok(config::default_log_dir(database)),
    }
}

/// The directory consolidation runs keep their snapshots in: `[consolidation]
/// snapshot_dir` when set, else the default beside the database. Resolved like
/// the log directory, so `liamd` and `liam consolidate` agree.
pub fn resolve_snapshot_dir(
    config: &Config,
    database: &std::path::Path,
) -> anyhow::Result<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    resolve_snapshot_dir_with_home(config, database, &home)
}

fn resolve_snapshot_dir_with_home(
    config: &Config,
    database: &std::path::Path,
    home: &str,
) -> anyhow::Result<PathBuf> {
    match &config.consolidation.snapshot_dir {
        Some(dir) => {
            let dir = PathBuf::from(resolve_path_with_home(
                "consolidation.snapshot_dir",
                dir,
                home,
            )?);
            Ok(match database.parent() {
                Some(parent) if dir.is_relative() => parent.join(dir),
                _ => dir,
            })
        }
        None => Ok(config::default_snapshot_dir(database)),
    }
}

/// Lets the store embed the content of nodes a replay restored, with the same
/// embedder the daemon serves with.
pub struct StoreEmbedder(pub Arc<dyn Embedder>);

#[async_trait]
impl ContentEmbedder for StoreEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(self.0.embed(text, TextKind::Document).await?)
    }
}

/// The `embedder.provider` value that selects the in-process models.
pub const LOCAL_PROVIDER: &str = "local";

/// The provider value that selects the mock embedder or LLM, for tests and dev.
pub const MOCK_PROVIDER: &str = "mock";

/// The `llm.provider` value that selects in-process llama.cpp.
pub const LLAMA_PROVIDER: &str = "llama-cpp";

/// Choose the embedder from config. `mock` is for tests and dev; the `local`
/// provider loads the arctic ONNX model in-process. Any other name is an error,
/// so a typo cannot quietly select the mock.
pub fn build_embedder(config: &Config) -> anyhow::Result<Arc<dyn Embedder>> {
    build_embedder_for(config, NoticeCaller::Daemon)
}

/// `build_embedder`, where `caller` decides what the download notice says.
pub fn build_embedder_for(
    config: &Config,
    caller: NoticeCaller,
) -> anyhow::Result<Arc<dyn Embedder>> {
    if config.embedder.provider == LOCAL_PROVIDER {
        return build_local_embedder(config, caller);
    }
    if config.embedder.provider == MOCK_PROVIDER {
        return Ok(Arc::new(MockEmbedder::new(config.embedding_dims)));
    }
    anyhow::bail!(
        "embedder.provider = {:?} is not one of {LOCAL_PROVIDER:?}, {MOCK_PROVIDER:?}",
        config.embedder.provider
    )
}

fn build_local_embedder(
    config: &Config,
    caller: NoticeCaller,
) -> anyhow::Result<Arc<dyn Embedder>> {
    let home = std::env::var("HOME").unwrap_or_default();
    let cache_dir =
        resolve_path_with_home("embedder.cache_dir", &config.embedder.cache_dir, &home)?;
    Ok(Arc::new(ArcticEmbedder::load(
        config.embedding_dims,
        &cache_dir,
        caller,
    )?))
}

/// Choose the LLM from config. Mock is for tests and dev; `llama-cpp`
/// loads llama.cpp in-process. `local` named the
/// retired candle provider: an operator who still has it in their liam.toml
/// gets an actionable error instead of a silent downgrade to mock.
pub fn build_llm(config: &Config) -> anyhow::Result<Arc<dyn Llm>> {
    build_llm_for(config, NoticeCaller::Daemon)
}

/// `build_llm`, where `caller` decides what the download notice says.
pub fn build_llm_for(config: &Config, caller: NoticeCaller) -> anyhow::Result<Arc<dyn Llm>> {
    let llm: Arc<dyn Llm> = if config.llm.provider == LLAMA_PROVIDER {
        build_llama_llm(config, caller)?
    } else if config.llm.provider == "local" {
        anyhow::bail!(
            "llm.provider = \"local\" was removed: the candle provider is gone, \
             generation now runs on llama.cpp, set llm.provider = \"llama-cpp\" in your liam.toml"
        );
    } else if config.llm.provider == MOCK_PROVIDER {
        Arc::new(MockLlm)
    } else {
        anyhow::bail!(
            "llm.provider = {:?} is not one of {LLAMA_PROVIDER:?}, {MOCK_PROVIDER:?}",
            config.llm.provider
        );
    };

    // An operator debugging latency needs to see which backend actually
    // came up, not just which provider was configured: `auto` can silently
    // resolve to a slower one.
    tracing::info!(backend = llm.backend(), "llm ready");

    // A backend that resolved to CPU on macOS is roughly a 5x slowdown
    // nobody would notice until a user complains, so it fails startup
    // instead of serving degraded. `macos_backend_error` exempts an
    // explicit `device = "cpu"`: that is not a fallback, it is what was
    // asked for. Any provider that already validated `llm.device` above
    // guarantees it parses here too, so `unwrap_or_default` only matters
    // for the mock provider, whose "mock" label never trips the check.
    let device = DevicePreference::parse(&config.llm.device).unwrap_or_default();
    reject_unexpected_cpu_backend(llm.backend(), device)?;

    Ok(llm)
}

/// The llama.cpp settings both the eager and the lazy path read, resolved and
/// checked once.
struct LlamaSettings {
    device: DevicePreference,
    cache_dir: String,
}

fn llama_settings(config: &Config) -> anyhow::Result<LlamaSettings> {
    // Reject an unknown device rather than quietly running 5x slower on CPU.
    let device = DevicePreference::parse(&config.llm.device).ok_or_else(|| {
        anyhow::anyhow!(
            "llm.device = {:?} is not one of auto, metal, cuda, cpu",
            config.llm.device
        )
    })?;
    // Same `~` problem as the embedder cache dir above: hf-hub joins this
    // string as a plain relative path, so an unexpanded `~/.liam/models`
    // downloads a multi-gigabyte GGUF into a directory named `~`.
    let cache_dir = resolve_path_with_home(
        "llm.cache_dir",
        &config.llm.cache_dir,
        &std::env::var("HOME").unwrap_or_default(),
    )?;
    Ok(LlamaSettings { device, cache_dir })
}

fn load_llama(
    config: &Config,
    settings: &LlamaSettings,
    caller: NoticeCaller,
) -> liam_model::Result<Arc<dyn Llm>> {
    use liam_model::LlamaCppLlm;
    Ok(Arc::new(LlamaCppLlm::load_from_hub(
        &config.llm.model,
        &config.llm.gguf_file,
        &settings.cache_dir,
        config.llm.context_tokens as u32,
        settings.device,
        caller,
    )?))
}

fn build_llama_llm(config: &Config, caller: NoticeCaller) -> anyhow::Result<Arc<dyn Llm>> {
    let settings = llama_settings(config)?;
    Ok(load_llama(config, &settings, caller)?)
}

/// The LLM the daemon serves with, plus the handle that unloads it when idle.
pub struct ServingLlm {
    pub llm: Arc<dyn Llm>,
    /// `Some` for a provider that loads on first use, `None` for the mock.
    pub lazy: Option<Arc<LazyLlm>>,
}

/// Like `build_llm`, but a llama.cpp model is not loaded here: the returned
/// handle loads it on the first call and gives it back after an idle period.
/// Config errors and the macOS backend check still fail startup, because the
/// backend label is read without loading weights. `liam fetch-models` keeps
/// `build_llm`, which loads, because it exists to prove the model loads.
pub fn build_serving_llm(config: &Config) -> anyhow::Result<ServingLlm> {
    if config.llm.provider != LLAMA_PROVIDER {
        return Ok(ServingLlm {
            llm: build_llm(config)?,
            lazy: None,
        });
    }
    let settings = llama_settings(config)?;
    let backend = liam_model::llama::llama_runtime_backend()?;
    reject_unexpected_cpu_backend(backend, settings.device)?;
    let loader: Loader = {
        let config = config.clone();
        Arc::new(move || load_llama(&config, &settings, NoticeCaller::Daemon))
    };
    let lazy = Arc::new(LazyLlm::new(loader, backend));
    tracing::info!(backend, "llm ready, loads on first use");
    Ok(ServingLlm {
        llm: lazy.clone(),
        lazy: Some(lazy),
    })
}

/// The warning for a configured llama.cpp model that is not in the cache, or
/// `None` when it is there or the provider is not llama.cpp. A file check
/// only: it loads and hashes nothing.
pub fn missing_model_warning(config: &Config) -> Option<String> {
    if config.llm.provider != LLAMA_PROVIDER {
        return None;
    }
    let settings = llama_settings(config).ok()?;
    let cached = liam_model::LlamaCppLlm::cached_gguf_path(
        &config.llm.model,
        &config.llm.gguf_file,
        &settings.cache_dir,
    );
    cached.is_none().then(|| {
        format!(
            "llm model {} file {} is not in the cache {}; entity pages cannot be written until `liam fetch-models` runs",
            config.llm.model, config.llm.gguf_file, settings.cache_dir
        )
    })
}

/// What the daemon does with the LLM at start. With `warmup` it loads the
/// model and stops the daemon when the load fails; the warmup run after a
/// good load only warns. Without it nothing loads, and a model that is
/// missing from the cache is a single warning.
pub async fn prepare_llm_at_start(config: &Config, llm: &dyn Llm) -> anyhow::Result<()> {
    if !config.llm.warmup {
        if let Some(warning) = missing_model_warning(config) {
            tracing::warn!("{warning}");
        }
        return Ok(());
    }
    let started = std::time::Instant::now();
    llm.ensure_loaded()
        .await
        .map_err(|e| anyhow::anyhow!("llm.warmup is true and the model did not load: {e}"))?;
    match llm.warmup().await {
        Ok(()) => tracing::info!(elapsed = ?started.elapsed(), "llm warmed up"),
        Err(e) => tracing::warn!(error = %e, "llm warmup failed"),
    }
    Ok(())
}

/// Applies `macos_backend_error` on macOS and does nothing elsewhere.
fn reject_unexpected_cpu_backend(backend: &str, device: DevicePreference) -> anyhow::Result<()> {
    #[cfg(target_os = "macos")]
    if let Some(message) = macos_backend_error(backend, device) {
        anyhow::bail!(message);
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (backend, device);
    Ok(())
}

/// Whether a resolved backend label is a macOS startup error. Pure and
/// platform-independent by design, so this branch is testable on any host;
/// `build_llm` applies it only under `cfg(target_os = "macos")`.
///
/// Matches the label by CONTAINS rather than equality: `runtime_backend` in
/// `liam_model::llama` appends a parenthetical suffix
/// ("llama.cpp/cpu (metal unavailable)") when Metal was compiled in but no
/// device came up at runtime, and that case has to trip this the same as a
/// plain "cpu" label.
///
/// Returns `None` whenever `device` is `DevicePreference::Cpu`: an operator
/// who asked for CPU explicitly gets exactly what they asked for, silently.
/// The error exists only to catch `auto` or `metal` RESOLVING to CPU, which
/// on Apple Silicon means Metal did not come up as expected.
///
/// Deliberately compiled on every platform so this branch stays unit-tested
/// everywhere; only the macOS build actually calls it, so a non-macOS build
/// sees it as unused without the `allow` below.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn macos_backend_error(backend: &str, device: DevicePreference) -> Option<String> {
    if device == DevicePreference::Cpu || !backend.contains("cpu") {
        return None;
    }
    Some(format!(
        "llm backend resolved to {backend:?} but Metal was expected on macOS. \
         If running on CPU is intentional, set llm.device = \"cpu\" in liam.toml \
         to silence this error."
    ))
}

#[cfg(test)]
mod serving_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// The arctic model's own output width.
    const ARCTIC_NATIVE_DIMS: usize = 768;

    fn llama_config(cache_dir: &Path, warmup: bool) -> Config {
        let mut config = Config::default();
        config.llm.provider = LLAMA_PROVIDER.to_string();
        config.llm.cache_dir = cache_dir.to_string_lossy().into_owned();
        config.llm.warmup = warmup;
        config
    }

    #[test]
    fn a_llama_model_missing_from_the_cache_gets_one_warning_naming_it() {
        // Arrange
        let cache = tempfile::tempdir().unwrap();
        let config = llama_config(cache.path(), false);

        // Act
        let warning = missing_model_warning(&config).expect("an empty cache has no model");

        // Assert
        assert!(warning.contains(&config.llm.model), "{warning}");
        assert!(warning.contains(&config.llm.gguf_file), "{warning}");
        assert!(warning.contains("liam fetch-models"), "{warning}");
    }

    #[test]
    fn the_mock_provider_has_no_model_to_miss() {
        // Arrange
        let config = Config::default();

        // Act / Assert
        assert!(missing_model_warning(&config).is_none());
    }

    struct FailingLoadLlm;

    #[async_trait]
    impl Llm for FailingLoadLlm {
        async fn ensure_loaded(&self) -> liam_model::Result<()> {
            Err(liam_model::ModelError::Llm("no weights".into()))
        }

        async fn complete(&self, _system: &str, _prompt: &str) -> liam_model::Result<String> {
            unreachable!("start never generates")
        }
    }

    #[tokio::test]
    async fn warmup_stops_the_start_when_the_model_does_not_load() {
        // Arrange
        let config = llama_config(Path::new("/unused"), true);

        // Act
        let result = prepare_llm_at_start(&config, &FailingLoadLlm).await;

        // Assert
        let message = result.unwrap_err().to_string();
        assert!(message.contains("no weights"), "{message}");
    }

    #[tokio::test]
    async fn without_warmup_a_model_that_cannot_load_does_not_stop_the_start() {
        // Arrange
        let cache = tempfile::tempdir().unwrap();
        let config = llama_config(cache.path(), false);

        // Act
        let result = prepare_llm_at_start(&config, &FailingLoadLlm).await;

        // Assert
        assert!(result.is_ok());
    }

    /// The model cache dirs are the two paths that were NOT being expanded,
    /// while `socket_path` and `database_path` always were. Left raw, hf-hub
    /// treats `~/.liam/models` as a relative path and downloads
    /// gigabytes into a directory named `~`, under the launchd job's
    /// `WorkingDirectory`. The shipped mock defaults hid this because a mock
    /// embedder never opens the cache dir.
    #[test]
    fn a_tilde_model_cache_dir_expands_to_the_home_directory() {
        assert_eq!(
            resolve_path_with_home("embedder.cache_dir", "~/.liam/models", "/home/alice").unwrap(),
            "/home/alice/.liam/models"
        );
        assert_eq!(
            resolve_path_with_home("llm.cache_dir", "~/.liam/models", "/home/alice").unwrap(),
            "/home/alice/.liam/models"
        );
    }

    /// The local provider must reach the arctic loader: its width check fails
    /// before any download, so this runs with no network and leaves the cache
    /// empty.
    #[test]
    fn the_local_provider_loads_the_arctic_embedder() {
        // Arrange
        let cache = tempfile::tempdir().unwrap();
        let mut config = Config::default();
        config.embedder.provider = LOCAL_PROVIDER.to_string();
        config.embedder.cache_dir = cache.path().to_string_lossy().into_owned();
        config.embedding_dims = ARCTIC_NATIVE_DIMS + 1;

        // Act
        let error = match build_embedder(&config) {
            Ok(_) => panic!("a width beyond the model's own must be refused"),
            Err(error) => error.to_string(),
        };

        // Assert
        assert!(error.contains(liam_model::EMBEDDER_MODEL_ID), "{error}");
        assert_eq!(std::fs::read_dir(cache.path()).unwrap().count(), 0);
    }

    /// An absolute cache dir is already correct and must be passed through
    /// untouched, so an operator can point at a shared model directory.
    #[test]
    fn an_absolute_model_cache_dir_is_left_alone() {
        assert_eq!(
            resolve_path_with_home("llm.cache_dir", "/opt/liam/models", "/home/alice").unwrap(),
            "/opt/liam/models"
        );
    }

    /// A launchd job only gets the variables its plist declares, so an empty
    /// HOME is a real case here. Failing names the offending config field,
    /// because the silent version resolves to `/.liam/models` at the
    /// filesystem root and fails later with a path the operator never wrote.
    #[test]
    fn a_tilde_model_cache_dir_without_home_is_an_error_naming_the_field() {
        let error = resolve_path_with_home("llm.cache_dir", "~/.liam/models", "")
            .expect_err("a tilde path with no HOME must not be silently resolved");
        let message = error.to_string();
        assert!(
            message.contains("llm.cache_dir"),
            "error should name the config field: {message}"
        );
        assert!(
            message.contains("HOME"),
            "error should say what is missing: {message}"
        );
    }

    struct FailingEmbedder;

    #[async_trait]
    impl Embedder for FailingEmbedder {
        fn dims(&self) -> usize {
            8
        }

        async fn embed(&self, _text: &str, _kind: TextKind) -> liam_model::error::Result<Vec<f32>> {
            Err(liam_model::error::ModelError::Embed("no model".into()))
        }
    }

    #[tokio::test]
    async fn the_store_embedder_returns_the_vector_its_embedder_made() {
        // Arrange
        let embedder = Arc::new(MockEmbedder::new(8));
        let expected = embedder.embed("alpha", TextKind::Document).await.unwrap();

        // Act
        let vector = StoreEmbedder(embedder).embed("alpha").await.unwrap();

        // Assert
        assert_eq!(vector, expected);
        assert_eq!(vector.len(), 8);
    }

    #[tokio::test]
    async fn the_store_embedder_embeds_as_a_document() {
        // Arrange
        struct KindRecorder(std::sync::Mutex<Vec<TextKind>>);

        #[async_trait]
        impl Embedder for KindRecorder {
            fn dims(&self) -> usize {
                8
            }

            async fn embed(
                &self,
                _text: &str,
                kind: TextKind,
            ) -> liam_model::error::Result<Vec<f32>> {
                self.0.lock().unwrap().push(kind);
                Ok(vec![0.0; 8])
            }
        }
        let recorder = Arc::new(KindRecorder(std::sync::Mutex::new(Vec::new())));

        // Act
        StoreEmbedder(recorder.clone())
            .embed("alpha")
            .await
            .unwrap();

        // Assert
        assert_eq!(*recorder.0.lock().unwrap(), vec![TextKind::Document]);
    }

    #[tokio::test]
    async fn the_store_embedder_passes_an_embed_failure_on_as_an_error() {
        // Act
        let error = StoreEmbedder(Arc::new(FailingEmbedder))
            .embed("alpha")
            .await
            .expect_err("a failing embedder must not yield a vector");

        // Assert
        assert!(error.to_string().contains("no model"), "{error}");
    }

    #[test]
    fn the_default_config_builds_only_an_embedder() {
        // Arrange
        let config = Config::default();

        // Act
        let embedder: Arc<dyn Embedder> = build_embedder(&config).unwrap();

        // Assert
        assert_eq!(embedder.dims(), config.embedding_dims);
    }

    #[test]
    fn an_unknown_embedder_provider_is_an_error_not_a_mock() {
        let mut config = Config::default();
        config.embedder.provider = "loca".to_string();

        let message = build_embedder(&config).err().unwrap().to_string();

        assert!(
            message.contains("loca") && message.contains("local"),
            "{message}"
        );
    }

    #[test]
    fn an_unknown_llm_provider_is_an_error_not_a_mock() {
        let mut config = Config::default();
        config.llm.provider = "llama".to_string();

        let message = build_llm(&config).err().unwrap().to_string();

        assert!(
            message.contains("llama-cpp") && message.contains("mock"),
            "{message}"
        );
    }

    #[test]
    fn llm_provider_local_is_rejected_with_an_actionable_error() {
        // Arrange: an old liam.toml still naming the retired candle provider.
        let mut config = Config::default();
        config.llm.provider = "local".to_string();

        // Act
        let result = build_llm(&config);

        // Assert: the error names the removed provider and the fix, so an
        // operator upgrading with a stale config is told what to change
        // instead of silently getting a mock that answers nothing useful.
        // `dyn Llm` is not `Debug`, so this matches instead of `expect_err`.
        let message = match result {
            Ok(_) => panic!("removed provider must error, not fall back"),
            Err(e) => e.to_string(),
        };
        assert!(message.contains("local"), "message: {message}");
        assert!(message.contains("llama-cpp"), "message: {message}");
    }

    #[test]
    fn auto_resolving_to_cpu_on_macos_is_a_startup_error() {
        // Arrange: the backend label reports cpu and the operator asked
        // for auto, so a CPU result was not what was requested.
        let backend = "llama.cpp/cpu (metal unavailable)";
        let device = DevicePreference::Auto;

        // Act
        let result = macos_backend_error(backend, device);

        // Assert: the error names Metal and the device override that
        // silences it, so an operator knows what happened and what to do.
        let message = result.expect("auto resolving to cpu must error");
        assert!(message.contains("Metal"), "message: {message}");
        assert!(message.contains("device"), "message: {message}");
    }

    #[test]
    fn an_explicit_cpu_choice_is_honoured() {
        // Arrange: the backend label reports cpu and the operator asked
        // for cpu explicitly.
        let backend = "llama.cpp/cpu (metal unavailable)";
        let device = DevicePreference::Cpu;

        // Act
        let result = macos_backend_error(backend, device);

        // Assert: an explicit choice is not a silent fallback, so it must
        // not error.
        assert_eq!(result, None);
    }

    #[test]
    fn a_metal_backend_passes() {
        // Arrange: the backend label reports metal and the operator asked
        // for auto.
        let backend = "llama.cpp/metal";
        let device = DevicePreference::Auto;

        // Act
        let result = macos_backend_error(backend, device);

        // Assert: Metal came up as expected, nothing to report.
        assert_eq!(result, None);
    }

    #[test]
    fn an_explicit_metal_request_that_resolved_to_cpu_is_an_error() {
        // Arrange: the backend label reports cpu even though the operator
        // asked for metal explicitly, so something is broken.
        let backend = "llama.cpp/cpu (metal unavailable)";
        let device = DevicePreference::Metal;

        // Act
        let result = macos_backend_error(backend, device);

        // Assert: only an explicit cpu choice is exempt; an explicit metal
        // request that fell back to cpu must still error.
        assert!(result.is_some());
    }

    // ---- the snapshot directory ----

    fn config_with_snapshot_dir(dir: Option<&str>) -> Config {
        let mut config = Config::default();
        config.consolidation.snapshot_dir = dir.map(str::to_string);
        config
    }

    #[test]
    fn an_unset_snapshot_dir_is_the_default_beside_the_database() {
        let config = config_with_snapshot_dir(None);

        let dir =
            resolve_snapshot_dir_with_home(&config, Path::new("/data/work.db"), "/h").unwrap();

        assert_eq!(dir, PathBuf::from("/data/work.snapshots"));
    }

    #[test]
    fn a_snapshot_dir_expands_the_tilde_and_resolves_a_relative_path_from_the_database() {
        for (configured, expected) in [
            ("~/snaps", "/home/alice/snaps"),
            ("snaps", "/data/snaps"),
            ("/var/snaps", "/var/snaps"),
        ] {
            let config = config_with_snapshot_dir(Some(configured));

            let dir =
                resolve_snapshot_dir_with_home(&config, Path::new("/data/work.db"), "/home/alice")
                    .unwrap();

            assert_eq!(dir, PathBuf::from(expected), "configured {configured}");
        }
    }

    // ---- the log directory ----

    fn config_with_log_dir(log_dir: Option<&str>) -> Config {
        let mut config = Config::default();
        config.log.dir = log_dir.map(str::to_string);
        config
    }

    #[test]
    fn without_a_configured_dir_the_log_sits_beside_the_database() {
        // Arrange
        let config = config_with_log_dir(None);

        // Act
        let dir =
            resolve_log_dir_with_home(&config, Path::new("/data/work.db"), "/home/alice").unwrap();

        // Assert
        assert_eq!(dir, PathBuf::from("/data/work.log"));
    }

    #[test]
    fn a_configured_dir_beats_the_default() {
        // Arrange
        let config = config_with_log_dir(Some("/var/liam/events"));

        // Act
        let dir =
            resolve_log_dir_with_home(&config, Path::new("/data/work.db"), "/home/alice").unwrap();

        // Assert
        assert_eq!(dir, PathBuf::from("/var/liam/events"));
    }

    #[test]
    fn a_tilde_in_the_configured_dir_expands_to_the_home_directory() {
        // Arrange
        let config = config_with_log_dir(Some("~/events"));

        // Act
        let dir =
            resolve_log_dir_with_home(&config, Path::new("/data/work.db"), "/home/alice").unwrap();

        // Assert
        assert_eq!(dir, PathBuf::from("/home/alice/events"));
    }

    #[test]
    fn a_tilde_dir_without_home_is_an_error_naming_the_key() {
        // Arrange
        let config = config_with_log_dir(Some("~/events"));

        // Act
        let error = resolve_log_dir_with_home(&config, Path::new("/data/work.db"), "")
            .expect_err("a tilde dir with no HOME must not be silently resolved");

        // Assert
        assert!(error.to_string().contains("log.dir"), "{error}");
    }

    #[test]
    fn a_relative_dir_is_taken_from_the_database_directory() {
        // Arrange
        let config = config_with_log_dir(Some("events"));

        // Act
        let dir =
            resolve_log_dir_with_home(&config, Path::new("/data/work.db"), "/home/alice").unwrap();

        // Assert
        assert_eq!(dir, PathBuf::from("/data/events"));
    }

    const BATCH_TEXT_COUNT: usize = 20;
    const EMBED_EPSILON: f32 = 1e-4;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Loads the real model through `build_embedder`. It reads the pinned files
    /// from the directory `LIAM_MODEL_CACHE` names and downloads the missing
    /// ones (311 MB). Run with
    /// `LIAM_MODEL_CACHE=/path/to/cache cargo test -p liam-daemon --lib -- --ignored arctic_ranks`.
    #[tokio::test]
    #[ignore = "needs the arctic weights (311 MB) in LIAM_MODEL_CACHE or a download"]
    async fn arctic_ranks_the_relevant_document_first() {
        // Arrange
        let mut config = Config::default();
        config.embedder.provider = LOCAL_PROVIDER.to_string();
        config.embedder.cache_dir =
            std::env::var("LIAM_MODEL_CACHE").expect("set LIAM_MODEL_CACHE");
        let embedder = build_embedder(&config).expect("load arctic");
        let documents = [
            "The mitochondria is the powerhouse of the cell".to_string(),
            "I live in Lisbon, Portugal".to_string(),
            "The stock market closed higher on Tuesday".to_string(),
        ];
        let many: Vec<String> = (0..BATCH_TEXT_COUNT)
            .map(|i| format!("{} {i}", documents[i % documents.len()]))
            .collect();

        // Act
        let query = embedder
            .embed("where do I live", TextKind::Query)
            .await
            .expect("embed query");
        let vectors = embedder
            .embed_batch(&documents, TextKind::Document)
            .await
            .expect("embed documents");
        let many_vectors = embedder
            .embed_batch(&many, TextKind::Document)
            .await
            .expect("embed more documents than one call");
        let alone = embedder
            .embed(&many[0], TextKind::Document)
            .await
            .expect("embed one document");

        // Assert
        let scores: Vec<f32> = vectors.iter().map(|v| cosine(&query, v)).collect();
        assert_eq!(query.len(), config.embedding_dims);
        assert!(scores[1] > scores[0] && scores[1] > scores[2], "{scores:?}");
        assert_eq!(many_vectors.len(), BATCH_TEXT_COUNT);
        let drift = many_vectors[0]
            .iter()
            .zip(&alone)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(drift < EMBED_EPSILON, "{drift}");
    }
}
