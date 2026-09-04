// SPDX-License-Identifier: Apache-2.0
//! liam-daemon: serves liam memory to agents over MCP.
//!
//! A thin shell wiring `liam-store` (retrieval) and `liam-model`
//! (embedding, reranking). The embedder is Mock by default; the `local` feature
//! plus `provider = "local"` loads fastembed in-process (Qwen3 embedder,
//! cross-encoder reranker), no server.

mod ask;
mod cli;
mod clusters;
/// Grounding eval for `ask`; test-only, see the module docs to run it.
#[cfg(test)]
mod eval;
mod mcp;
/// Retrieval-quality benchmark for `Graph::query`; test-only, see the module
/// docs to run each tier.
#[cfg(test)]
mod retrieval_eval;
mod storelock;
mod synthesis;
mod telemetry;
/// Grounding eval for remember/recall/relate; test-only, see the module docs to run each tier.
#[cfg(test)]
mod tool_eval;
mod transport;
mod tuning;

use std::sync::Arc;

use liam_store::{DefaultGraph, GraphConfig};

// Re-exported at this crate's root, not merely imported, so the existing
// `crate::config::...` paths keep resolving now that the module itself lives
// in the library both binaries share. The call sites are in the submodules,
// not the files above them: `mcp/producer.rs`, `transport/activation.rs`,
// `transport/socket.rs`, and `eval.rs`.
pub use liam_daemon::config;

use config::Config;
use liam_daemon::models::{build_llm, build_models, resolve_config_path, resolve_path_with_home};
use mcp::MemoryServer;

fn main() -> anyhow::Result<()> {
    // Parse BEFORE anything else touches the filesystem or the environment.
    // A usage error must exit 2 without having opened the store or taken the
    // store lock, since a typo like `liamd serv` would otherwise break the
    // real daemon's next start. `parse` exits the process itself on a usage
    // error or on --help/--version.
    let cli = <cli::Cli as clap::Parser>::parse();
    let mode = cli.mode();
    let config_path = cli.config_path(std::env::var("LIAM_CONFIG").ok().as_deref());

    let config = Config::load(config_path.as_ref())?;
    // Set fastembed's cache dir before the async runtime starts. Mutating the
    // environment once worker threads exist is a data race on POSIX (and
    // `unsafe` on edition 2024), so it must happen while single-threaded.
    // Skipped for the proxy, which loads no model.
    if mode != cli::Mode::Proxy && config.embedder.provider == "local" {
        // fastembed does not expand `~`, and no std path API does either, so
        // passing the configured value through raw creates a directory
        // literally named `~` under the process's working directory. Under
        // the launchd job that is `WorkingDirectory`, so models the user
        // already fetched are invisible and get re-downloaded to the wrong
        // place. `socket_path` and `database_path` were always expanded; the
        // two model cache dirs were missed, and the shipped mock defaults hid
        // it because a mock embedder never reads the cache dir at all.
        //
        // This sets the reranker's cache dir via the env var; the embedder
        // lands under the same directory too, via its own explicit parameter.
        let home = std::env::var("HOME").unwrap_or_default();
        let cache_dir =
            resolve_path_with_home("embedder.cache_dir", &config.embedder.cache_dir, &home)?;
        std::env::set_var("FASTEMBED_CACHE_DIR", cache_dir);
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(run(mode, config));

    if mode == cli::Mode::Proxy {
        // The proxy's stdin reader sits in a blocking read that cannot be
        // cancelled, and dropping a runtime WAITS for blocking tasks that
        // have already started. So when the daemon closes the session first,
        // a plain drop here hangs forever on a read for input nobody will
        // ever consume, which is the whole failure `transport::proxy` works
        // to avoid. Measured: without this the proxy never exits after the
        // daemon goes away.
        //
        // Safe to drop on the floor precisely because the proxy owns no
        // state worth unwinding: no store, no lock, no socket of its own.
        runtime.shutdown_background();
    }

    result
}

/// Dispatches to the selected mode. The proxy branch returns before any
/// store or model setup on purpose: it must not take the per-process store
/// lock the daemon it forwards to already holds.
async fn run(mode: cli::Mode, config: Config) -> anyhow::Result<()> {
    telemetry::init(&config.log_filter);

    if mode == cli::Mode::Proxy {
        let socket_path = resolve_config_path("socket_path", &config.socket_path)?;
        return transport::proxy::run(&socket_path).await;
    }

    serve_with_store(mode, config).await
}

/// Everything that needs the store and the models: the stdio server and the
/// socket daemon. Both take the per-process store lock.
async fn serve_with_store(mode: cli::Mode, config: Config) -> anyhow::Result<()> {
    // Exclusive per-process lock, taken once, before the first store open.
    // `spawn_gc` below shares this same `Graph`, so there is no second
    // connection to reason about; the lock is not retaken for it regardless.
    // Bound to a named variable so it lives for the rest of the process:
    // `let _ = ...` would drop it immediately and release the lock right
    // away. See `storelock` for why this is a real advisory `flock` and not
    // a PID file, and for the contract the future `liamd proxy` mode (which
    // opens no store) must follow.
    let database_path = resolve_config_path("database_path", &config.database_path)?;
    // A fresh install has no ~/.liam yet, and libSQL will not create a parent
    // directory for the database the way `socket::bind` does for the socket.
    if let Some(parent) = database_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|source| {
                anyhow::anyhow!(
                    "failed to create the database directory {}: {source}",
                    parent.display()
                )
            })?;
        }
    }
    let _lock = storelock::StoreLock::acquire(&database_path)?;

    let store = DefaultGraph::open(
        database_path.to_str().unwrap_or(&config.database_path),
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

    spawn_gc(&config, Arc::clone(&store));

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

    match mode {
        cli::Mode::Serve => serve_socket(&config, server).await,
        // `run` returns on `Proxy` before this function is ever called, and
        // it must stay that way: reaching here would mean the proxy had
        // already opened the store and taken the lock held by the daemon it
        // exists to forward to. Panicking is the point. Folding it in with
        // `Stdio` would instead give a future refactor a silently wrong
        // proxy that serves its own stdio session.
        cli::Mode::Proxy => unreachable!(
            "proxy mode returns in run() before any store setup; \
             reaching serve_with_store means that guard was removed"
        ),
        cli::Mode::Stdio => {
            // rmcp stdio serve. Confirm against your pinned rmcp version.
            use rmcp::ServiceExt;
            let running = server.serve(rmcp::transport::stdio()).await?;
            running.waiting().await?;
            Ok(())
        }
    }
}

/// The socket daemon: resolve the listener (activated by launchd, or bound
/// here), serve it, and stop on SIGTERM or SIGINT through the ordered
/// shutdown in `transport::shutdown`.
async fn serve_socket(config: &Config, server: MemoryServer) -> anyhow::Result<()> {
    use tokio_util::sync::CancellationToken;

    let socket_path = resolve_config_path("socket_path", &config.socket_path)?;
    let source = transport::activation::resolve(&socket_path).await?;
    let cancel = CancellationToken::new();

    // Signals are watched on their own task so the accept loop owns the
    // main flow. Cancelling the token is all this does; the drain and the
    // unlink belong to the accept loop's shutdown path.
    let signal_cancel = cancel.clone();
    tokio::spawn(async move {
        match transport::shutdown::signal().await {
            Ok(trigger) => {
                tracing::info!(signal = trigger.as_str(), "shutting down");
                signal_cancel.cancel();
            }
            // Without a handler the process would be killed outright on
            // SIGTERM and lose in-flight work, so this is worth surfacing
            // rather than logging at debug and moving on.
            Err(error) => {
                tracing::error!(error = %error, "failed to install signal handlers; shutdown will not be graceful")
            }
        }
    });

    transport::socket::accept_loop(
        source,
        server,
        config.max_connections,
        std::sync::Arc::new(config.producers.clone()),
        cancel,
        transport::shutdown::DEFAULT_DRAIN_DEADLINE,
    )
    .await
}

/// GC and the cluster refresh, on the SAME `Graph` every request handler
/// uses, not a second connection: a second connection only traded an
/// in-process wait for an opaque SQLite lock timeout. See ADR-0002
/// Amendment 4.
fn spawn_gc(config: &Config, store: Arc<DefaultGraph>) {
    let policy = config.gc_policy();
    let interval = config.gc_interval();
    let run_on_start = config.gc.run_on_start;
    tokio::spawn(async move {
        if run_on_start {
            maintenance_tick(&store, &policy).await;
        }
        let mut tick = tokio::time::interval(interval);
        tick.tick().await; // drop the immediate first tick
        loop {
            tick.tick().await;
            maintenance_tick(&store, &policy).await;
        }
    });
}

/// Sweep, then refresh clusters if anything moved the edge fingerprint. Runs
/// the refresh even after a partial sweep, since `gc` is independent
/// statements rather than one transaction.
async fn maintenance_tick(store: &DefaultGraph, policy: &liam_store::RetentionPolicy) {
    sweep(store, policy).await;
    refresh_clusters(store).await;
}

async fn sweep(store: &DefaultGraph, policy: &liam_store::RetentionPolicy) {
    match store.gc(policy).await {
        Ok(report) => tracing::info!(?report, "gc completed"),
        Err(e) => tracing::warn!(error = %e, "gc failed"),
    }
}

/// Not serving stale: the tick serves no one. A skipped or failed refresh
/// just means the next read re-runs the check and recomputes then.
async fn refresh_clusters(store: &DefaultGraph) -> bool {
    match store.refresh_communities().await {
        Ok(true) => {
            tracing::info!("clusters refreshed");
            true
        }
        Ok(false) => {
            tracing::debug!("clusters already current");
            false
        }
        Err(e) => {
            tracing::warn!(error = %e, "cluster refresh failed");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use liam_store::{FixedClock, GraphConfig, Millis, NewEdge, NewNode, RetentionPolicy};

    async fn seeded_pair(t0: Millis) -> (DefaultGraph, std::sync::Arc<FixedClock>) {
        let clock = std::sync::Arc::new(FixedClock::new(t0));
        let store = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
        let a = store.insert(NewNode::now("fact", "a", "x")).await.unwrap();
        let b = store.insert(NewNode::now("fact", "b", "x")).await.unwrap();
        store.link(NewEdge::new(&a, &b, "mentions")).await.unwrap();
        (store, clock)
    }

    #[tokio::test]
    async fn the_tick_refreshes_clusters_after_the_sweep() {
        let (store, _clock) = seeded_pair(Millis(1000)).await;
        // Nothing is old enough to sweep under this policy, which isolates the
        // refresh half of `maintenance_tick` from the sweep half.
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));

        maintenance_tick(&store, &policy).await;

        assert!(
            !store.refresh_communities().await.unwrap(),
            "the tick must have already refreshed; a second call finds nothing to do"
        );
    }

    #[tokio::test]
    async fn the_refresh_runs_after_the_sweep_not_before() {
        // If the refresh ran BEFORE the sweep, it would capture a fingerprint
        // of an edge set that still includes the edge the sweep is about to
        // delete. The next check would then see the swept, edge-free live
        // state as a MISMATCH against that stale fingerprint and find more
        // work to do. Refreshing after the sweep, the fingerprint it captures
        // already reflects the deletion, so a follow-up check finds nothing
        // left to do.
        let t0 = Millis(1_000_000);
        let (store, clock) = seeded_pair(t0).await;
        clock.set(Millis(t0.0 + 10_000));
        let policy = RetentionPolicy::keep("fact", Millis(1));

        maintenance_tick(&store, &policy).await;

        assert!(
            !store.refresh_communities().await.unwrap(),
            "a refresh that ran before the sweep would leave more work behind"
        );
    }

    #[tokio::test]
    async fn refresh_clusters_reports_no_work_on_an_idle_store() {
        let (store, _clock) = seeded_pair(Millis(1000)).await;
        assert!(refresh_clusters(&store).await, "first call always has work");

        let did_work = refresh_clusters(&store).await;

        assert!(!did_work, "an unchanged store must report no work");
    }
}
