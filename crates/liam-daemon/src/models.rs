// SPDX-License-Identifier: AGPL-3.0-only
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

use std::sync::Arc;

use liam_model::llm::DevicePreference;
use liam_model::{Embedder, IdentityReranker, Llm, MockEmbedder, MockLlm, Reranker};

use crate::config::{self, Config};

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

/// Choose the embedder and reranker from config. The mock pair keeps the base
/// build runnable; the `local` provider (with the `local` feature) loads
/// fastembed in-process.
///
/// Callers must set `FASTEMBED_CACHE_DIR` before this runs, while the process
/// is still single-threaded. `liamd` does it in `main`, `liam` does it in
/// `fetch_models`. That variable steers the reranker only; the embedder
/// weights are pinned to the hf-hub default by fastembed, as measured and
/// explained on `EmbedderConfig::cache_dir`.
pub fn build_models(config: &Config) -> anyhow::Result<(Arc<dyn Embedder>, Arc<dyn Reranker>)> {
    if config.embedder.provider == "local" {
        return build_local(config);
    }
    Ok((
        Arc::new(MockEmbedder::new(config.embedding_dims)),
        Arc::new(IdentityReranker),
    ))
}

#[cfg(feature = "local")]
fn build_local(config: &Config) -> anyhow::Result<(Arc<dyn Embedder>, Arc<dyn Reranker>)> {
    use liam_model::{FastEmbedEmbedder, FastEmbedReranker};
    let embedder = Arc::new(FastEmbedEmbedder::load(
        &config.embedder.model,
        config.embedding_dims,
    )?);
    let reranker = Arc::new(FastEmbedReranker::load()?);
    Ok((embedder, reranker))
}

/// Refuses to start rather than substituting the mock embedder.
///
/// Silently downgrading here is the worst possible outcome: mock embeddings
/// are random, so the vector channel returns noise and `recall` quality
/// collapses with nothing to show for it. The old warning went to stderr,
/// which an MCP client never surfaces, so a user would have seen only bad
/// answers. Failing at startup with the actual fix is what makes a
/// misconfigured install obvious in the one second it takes to notice.
#[cfg(not(feature = "local"))]
fn build_local(_config: &Config) -> anyhow::Result<(Arc<dyn Embedder>, Arc<dyn Reranker>)> {
    anyhow::bail!(
        "embedder.provider = \"local\" needs a binary built with the `local` feature, \
         and this one was not. Either install a release build (they ship with it), \
         rebuild with `--features local`, or set embedder.provider = \"mock\" if you \
         actually want the dev embedder. Mock embeddings are random, so recall would \
         be meaningless."
    )
}

/// Choose the LLM from config. Mock keeps the base build runnable; `llama-cpp`
/// (with the `llama` feature) loads llama.cpp in-process. `local` named the
/// retired candle provider: an operator who still has it in their liam.toml
/// gets an actionable error instead of a silent downgrade to mock.
pub fn build_llm(config: &Config) -> anyhow::Result<Arc<dyn Llm>> {
    let llm: Arc<dyn Llm> = if config.llm.provider == "llama-cpp" {
        build_llama_llm(config)?
    } else if config.llm.provider == "local" {
        anyhow::bail!(
            "llm.provider = \"local\" was removed: the candle provider is gone, \
             generation now runs on llama.cpp, set llm.provider = \"llama-cpp\" in your liam.toml"
        );
    } else {
        Arc::new(MockLlm)
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
    #[cfg(target_os = "macos")]
    {
        let device = DevicePreference::parse(&config.llm.device).unwrap_or_default();
        if let Some(message) = macos_backend_error(llm.backend(), device) {
            anyhow::bail!(message);
        }
    }

    Ok(llm)
}

#[cfg(feature = "llama")]
fn build_llama_llm(config: &Config) -> anyhow::Result<Arc<dyn Llm>> {
    use liam_model::LlamaCppLlm;
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
    Ok(Arc::new(LlamaCppLlm::load_from_hub(
        &config.llm.model,
        &config.llm.gguf_file,
        &cache_dir,
        config.llm.context_tokens as u32,
        device,
    )?))
}

/// Refuses to start rather than substituting the mock LLM.
///
/// `ask` synthesizes an answer from retrieved evidence, so a mock LLM makes
/// it produce confident nonsense. The old warning went to stderr, invisible
/// to an MCP client, which meant a user asking a question got a fabricated
/// answer with no indication anything was wrong. That is the single worst
/// failure mode this daemon has, so it is now fatal at startup.
#[cfg(not(feature = "llama"))]
fn build_llama_llm(_config: &Config) -> anyhow::Result<Arc<dyn Llm>> {
    anyhow::bail!(
        "llm.provider = \"llama-cpp\" needs a binary built with the `llama` feature, \
         and this one was not. Either install a release build (they ship with it), \
         rebuild with `--features llama`, or set llm.provider = \"mock\" if you \
         actually want the dev model. The mock LLM invents answers, so `ask` would \
         be worse than useless."
    )
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
mod tests {
    use super::*;

    /// The model cache dirs are the two paths that were NOT being expanded,
    /// while `socket_path` and `database_path` always were. Left raw, fastembed
    /// and hf-hub treat `~/.liam/models` as a relative path and download
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

    /// A build without `local` must refuse `embedder.provider = "local"`
    /// rather than quietly swapping in the mock. Mock embeddings are random,
    /// so the substitution used to destroy recall quality while logging only
    /// to stderr, where an MCP client never sees it.
    #[cfg(not(feature = "local"))]
    #[test]
    fn local_embedder_without_the_feature_fails_instead_of_using_mock() {
        let mut config = Config::default();
        config.embedder.provider = "local".to_string();

        let error = build_models(&config)
            .err()
            .expect("a local embedder without the feature must not fall back to mock");
        let message = error.to_string();
        assert!(
            message.contains("--features local") || message.contains("`local` feature"),
            "error should name the feature: {message}"
        );
        assert!(
            message.contains("mock"),
            "error should say what the alternative is: {message}"
        );
    }

    /// Same rule for generation, and it matters more: a mock LLM makes `ask`
    /// return confident fiction.
    #[cfg(not(feature = "llama"))]
    #[test]
    fn llama_provider_without_the_feature_fails_instead_of_using_mock() {
        let mut config = Config::default();
        config.llm.provider = "llama-cpp".to_string();

        let error = build_llm(&config)
            .err()
            .expect("llama-cpp without the feature must not fall back to mock");
        let message = error.to_string();
        assert!(
            message.contains("`llama` feature") || message.contains("--features llama"),
            "error should name the feature: {message}"
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
}
