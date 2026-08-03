//! liam-daemon: serves liam memory to agents over MCP.
//!
//! A thin shell wiring `liam-store` (retrieval) and `liam-model`
//! (embedding, reranking). The embedder is Mock by default; the `local` feature
//! plus `provider = "local"` loads fastembed in-process (Qwen3 embedder,
//! cross-encoder reranker), no server.

mod ask;
mod config;
mod mcp;
mod telemetry;

use std::sync::Arc;

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

    let store = DefaultGraph::open(
        &config.database_path,
        GraphConfig::new(config.embedding_dims),
    )
    .await?;
    let store = Arc::new(store);

    let (embedder, reranker) = build_models(&config)?;
    let llm = build_llm(&config)?;

    spawn_gc(&config).await?;

    let server = MemoryServer::new(store, embedder, reranker, llm, config.ask_timeout_secs);

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

/// Choose the LLM from config. Mock keeps the base build runnable; the `local`
/// provider (with the `local` feature) loads a candle chat model in-process.
fn build_llm(config: &Config) -> anyhow::Result<Arc<dyn Llm>> {
    if config.llm.provider == "local" {
        return build_local_llm(config);
    }
    Ok(Arc::new(MockLlm))
}

#[cfg(feature = "local")]
fn build_local_llm(config: &Config) -> anyhow::Result<Arc<dyn Llm>> {
    use liam_model::CandleLlm;
    Ok(Arc::new(CandleLlm::load(
        &config.llm.model,
        &config.llm.gguf_file,
        &config.llm.cache_dir,
    )?))
}

#[cfg(not(feature = "local"))]
fn build_local_llm(_config: &Config) -> anyhow::Result<Arc<dyn Llm>> {
    tracing::warn!(
        "llm.provider is 'local' but the daemon was built without the `local` feature; using mock"
    );
    Ok(Arc::new(MockLlm))
}

/// GC runs on its own store connection so it never contends with requests.
async fn spawn_gc(config: &Config) -> anyhow::Result<()> {
    let store = DefaultGraph::open(
        &config.database_path,
        GraphConfig::new(config.embedding_dims),
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
