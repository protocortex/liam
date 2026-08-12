// SPDX-License-Identifier: AGPL-3.0-only
//! liam-daemon: serves liam memory to agents over MCP.
//!
//! A thin shell wiring `liam-store` (retrieval) and `liam-model`
//! (embedding, reranking). The embedder is Mock by default; the `local` feature
//! plus `provider = "local"` loads fastembed in-process (Qwen3 embedder,
//! cross-encoder reranker), no server.

mod ask;
mod config;
/// Grounding eval for `ask`; test-only, see the module docs to run it.
#[cfg(test)]
mod eval;
mod mcp;
mod storelock;
mod telemetry;
mod transport;

use std::path::Path;
use std::sync::Arc;

use liam_model::llm::DevicePreference;
use liam_model::{Embedder, IdentityReranker, Llm, MockEmbedder, MockLlm, Reranker};
use liam_store::{DefaultGraph, GraphConfig};

use config::Config;
use mcp::MemoryServer;

fn main() -> anyhow::Result<()> {
    let config = Config::load(config_path().as_ref())?;
    // Set fastembed's cache dir before the async runtime starts. Mutating the
    // environment once worker threads exist is a data race on POSIX (and
    // `unsafe` on edition 2024), so it must happen while single-threaded.
    if config.embedder.provider == "local" {
        std::env::set_var("FASTEMBED_CACHE_DIR", &config.embedder.cache_dir);
    }
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run(config))
}

async fn run(config: Config) -> anyhow::Result<()> {
    telemetry::init(&config.log_filter);

    // Exclusive per-process lock, taken once, before the first store open.
    // `spawn_gc` below opens a second CONNECTION to the same database on
    // purpose; that is not a second process, so the lock is not retaken for
    // it. Bound to a named variable so it lives for the rest of the process:
    // `let _ = ...` would drop it immediately and release the lock right
    // away. See `storelock` for why this is a real advisory `flock` and not
    // a PID file, and for the contract the future `liamd proxy` mode (which
    // opens no store) must follow.
    let _lock = storelock::StoreLock::acquire(Path::new(&config.database_path))?;

    let store = DefaultGraph::open(
        &config.database_path,
        GraphConfig::new(config.embedding_dims).with_read_pool_size(config.read_pool_size),
    )
    .await?;
    let store = Arc::new(store);

    let (embedder, reranker) = build_models(&config)?;
    let llm = build_llm(&config)?;
    if config.llm.warmup {
        let started = std::time::Instant::now();
        match llm.warmup().await {
            Ok(()) => tracing::info!(elapsed = ?started.elapsed(), "llm warmed up"),
            // A failed warmup is not fatal: the first real call will simply pay
            // the cost, or fail with a better message than this one would.
            Err(e) => tracing::warn!(error = %e, "llm warmup failed"),
        }
    }

    spawn_gc(&config).await?;

    let server = MemoryServer::new(
        store,
        embedder,
        reranker,
        llm,
        config.ask_timeout_secs,
        config.ask_sufficiency_check,
        config.llm.context_tokens,
        config.llm.max_concurrent_generations,
    );

    // rmcp stdio serve. Confirm against your pinned rmcp version.
    use rmcp::ServiceExt;
    let running = server.serve(rmcp::transport::stdio()).await?;
    running.waiting().await?;
    Ok(())
}

fn config_path() -> std::path::PathBuf {
    std::env::var("LIAM_CONFIG")
        .unwrap_or_else(|_| "liam.toml".to_string())
        .into()
}

/// Choose the embedder and reranker from config. The mock pair keeps the base
/// build runnable; the `local` provider (with the `local` feature) loads
/// fastembed in-process.
fn build_models(config: &Config) -> anyhow::Result<(Arc<dyn Embedder>, Arc<dyn Reranker>)> {
    if config.embedder.provider == "local" {
        // FASTEMBED_CACHE_DIR is set in `main` before the runtime starts.
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

#[cfg(not(feature = "local"))]
fn build_local(config: &Config) -> anyhow::Result<(Arc<dyn Embedder>, Arc<dyn Reranker>)> {
    tracing::warn!("embedder.provider is 'local' but the daemon was built without the `local` feature; using mock");
    Ok((
        Arc::new(MockEmbedder::new(config.embedding_dims)),
        Arc::new(IdentityReranker),
    ))
}

/// Choose the LLM from config. Mock keeps the base build runnable; `llama-cpp`
/// (with the `llama` feature) loads llama.cpp in-process. `local` named the
/// retired candle provider: an operator who still has it in their liam.toml
/// gets an actionable error instead of a silent downgrade to mock.
fn build_llm(config: &Config) -> anyhow::Result<Arc<dyn Llm>> {
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
    Ok(Arc::new(LlamaCppLlm::load_from_hub(
        &config.llm.model,
        &config.llm.gguf_file,
        &config.llm.cache_dir,
        config.llm.context_tokens as u32,
        device,
    )?))
}

#[cfg(not(feature = "llama"))]
fn build_llama_llm(_config: &Config) -> anyhow::Result<Arc<dyn Llm>> {
    tracing::warn!(
        "llm.provider is 'llama-cpp' but the daemon was built without the `llama` feature; using mock"
    );
    Ok(Arc::new(MockLlm))
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
/// everywhere; only the macOS build actually calls it, so a non-macOS bin
/// build sees it as unused without the `allow` below.
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

/// GC runs on its own store connection so it never contends with requests.
async fn spawn_gc(config: &Config) -> anyhow::Result<()> {
    let store = DefaultGraph::open(
        &config.database_path,
        GraphConfig::new(config.embedding_dims).with_read_pool_size(config.read_pool_size),
    )
    .await?;
    let policy = config.gc_policy();
    let interval = config.gc_interval();
    let run_on_start = config.gc.run_on_start;
    tokio::spawn(async move {
        if run_on_start {
            sweep(&store, &policy).await;
        }
        let mut tick = tokio::time::interval(interval);
        tick.tick().await; // drop the immediate first tick
        loop {
            tick.tick().await;
            sweep(&store, &policy).await;
        }
    });
    Ok(())
}

async fn sweep(store: &DefaultGraph, policy: &liam_store::RetentionPolicy) {
    match store.gc(policy).await {
        Ok(report) => tracing::info!(?report, "gc completed"),
        Err(e) => tracing::warn!(error = %e, "gc failed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
