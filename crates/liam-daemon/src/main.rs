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
use liam_model::{Embedder, Llm, Reranker};
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

    let server = if config.llm.max_concurrent_generations == 0 {
        let cache_dir = resolve_config_path("llm.cache_dir", &config.llm.cache_dir)?
            .to_string_lossy()
            .into_owned();
        let model_fingerprint = format!("{}/{}", config.llm.model, config.llm.gguf_file);
        let backend = llm.backend().to_string();
        build_autotuned_server(
            store,
            embedder,
            reranker,
            llm,
            config.ask_timeout_secs,
            config.ask_sufficiency_check,
            config.llm.context_tokens,
            cache_dir,
            model_fingerprint,
            backend,
        )
    } else {
        MemoryServer::new(
            store,
            embedder,
            reranker,
            llm,
            config.ask_timeout_secs,
            config.ask_sufficiency_check,
            config.llm.context_tokens,
            config.llm.max_concurrent_generations,
        )
    };

    spawn_gc(&config, server.clone());

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

/// Builds the server for the `max_concurrent_generations == 0` sentinel: a
/// cache hit constructs at that value; a miss floors to 1, then grows it later.
#[allow(clippy::too_many_arguments)]
fn build_autotuned_server(
    store: Arc<DefaultGraph>,
    embedder: Arc<dyn Embedder>,
    reranker: Arc<dyn Reranker>,
    llm: Arc<dyn Llm>,
    ask_timeout_secs: u64,
    ask_sufficiency_check: bool,
    ask_context_tokens: usize,
    cache_dir: String,
    model_fingerprint: String,
    backend: String,
) -> MemoryServer {
    let ceiling = tuning::memory_ceiling(ask_context_tokens);
    let cached = tuning::load_cached(&cache_dir, &model_fingerprint, &backend, ask_context_tokens);
    let server = MemoryServer::new(
        store,
        embedder,
        reranker,
        llm.clone(),
        ask_timeout_secs,
        ask_sufficiency_check,
        ask_context_tokens,
        cached.unwrap_or(ceiling),
    );

    if cached.is_none() {
        let handle = server.generation_permits_handle();
        let granted_capacity = server.granted_capacity_handle();
        tokio::spawn(async move {
            let result = tuning::cold_start_benchmark(&*llm, ceiling, &handle).await;
            tuning::save_cache(
                &cache_dir,
                &model_fingerprint,
                &backend,
                result,
                ask_context_tokens,
            );
            tuning::reconcile_capacity(&handle, &granted_capacity, ceiling, result);
        });
    }

    server
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
/// Amendment 4. Takes a `MemoryServer` clone, not just its store, so the tick
/// can also reach the LLM and entity-resynthesis path a later change adds.
fn spawn_gc(config: &Config, server: MemoryServer) {
    let policy = config.gc_policy();
    let interval = config.gc_interval();
    let run_on_start = config.gc.run_on_start;
    let max_resynth_per_tick = config.gc.max_resynth_per_tick;
    let full_synthesis_mention_threshold = config.gc.full_synthesis_mention_threshold;
    let full_synthesis_max_new_tokens = config.gc.full_synthesis_max_new_tokens;
    let ask_timeout_secs = config.ask_timeout_secs;
    // An autotuned server (`max_concurrent_generations == 0`) is still
    // calibrating its generation-permit capacity when the run-on-start tick
    // fires, so that one tick alone skips resynthesis to avoid competing with
    // the calibration benchmark for the same permits. Every later tick,
    // periodic or otherwise, resynthesizes normally.
    let skip_first_resynth = run_on_start && config.llm.max_concurrent_generations == 0;
    tokio::spawn(async move {
        if run_on_start {
            maintenance_tick(
                &server,
                &policy,
                max_resynth_per_tick,
                ask_timeout_secs,
                skip_first_resynth,
                full_synthesis_mention_threshold,
                full_synthesis_max_new_tokens,
            )
            .await;
        }
        let mut tick = tokio::time::interval(interval);
        tick.tick().await; // drop the immediate first tick
        loop {
            tick.tick().await;
            maintenance_tick(
                &server,
                &policy,
                max_resynth_per_tick,
                ask_timeout_secs,
                false,
                full_synthesis_mention_threshold,
                full_synthesis_max_new_tokens,
            )
            .await;
        }
    });
}

/// Sweep, repair, refresh clusters, then resynthesize stale entities unless
/// `skip_resynthesis` is set (the run-on-start tick of an autotuning server,
/// which cannot yet spare generation permits for it).
async fn maintenance_tick(
    server: &MemoryServer,
    policy: &liam_store::RetentionPolicy,
    max_resynth_per_tick: usize,
    ask_timeout_secs: u64,
    skip_resynthesis: bool,
    full_synthesis_mention_threshold: usize,
    full_synthesis_max_new_tokens: usize,
) {
    let store = server.store_handle();
    sweep(&store, policy).await;
    repair_mentions(&store).await;
    refresh_clusters(&store).await;
    if !skip_resynthesis {
        resynthesize_stale(
            server,
            &store,
            max_resynth_per_tick,
            ask_timeout_secs,
            full_synthesis_mention_threshold,
            full_synthesis_max_new_tokens,
        )
        .await;
    }
}

/// Resynthesizes up to `max_resynth_per_tick` stale entities, oldest first.
/// Entities are processed one at a time, never concurrently: `resynthesize_entity`
/// draws from the same shared generation-permit pool `ask` uses, so firing them
/// all at once would starve concurrent user requests instead of yielding to them
/// between entities.
async fn resynthesize_stale(
    server: &MemoryServer,
    store: &DefaultGraph,
    max_resynth_per_tick: usize,
    ask_timeout_secs: u64,
    full_synthesis_mention_threshold: usize,
    full_synthesis_max_new_tokens: usize,
) {
    let now = liam_store::Millis::now();
    let stale = match store.stale_entities(max_resynth_per_tick, now).await {
        Ok(stale) => stale,
        Err(e) => {
            tracing::warn!(error = %e, "stale entity lookup failed");
            return;
        }
    };

    for (node_id, subject, scope) in stale {
        // Deliberate duplicate query: stale_entities already computed a fingerprint
        // internally and discarded it; widening its return type would touch ~10 more
        // test call sites elsewhere, out of scope here.
        let tier_fingerprint = store
            .entity_mentions_fingerprint(&subject, scope.as_deref(), now)
            .await;
        if let Err(e) = &tier_fingerprint {
            tracing::warn!(subject, error = %e, "tier fingerprint lookup failed, falling back to light tier");
        }
        let max_new_tokens = resynthesis_budget(
            tier_fingerprint.as_ref().map(|f| *f),
            full_synthesis_mention_threshold,
            full_synthesis_max_new_tokens,
            mcp::ENTITY_SYNTHESIS_MAX_NEW_TOKENS,
        );
        let deadline =
            tokio::time::Instant::now() + std::time::Duration::from_secs(ask_timeout_secs.max(1));
        match server
            .resynthesize_entity(node_id, deadline, max_new_tokens)
            .await
        {
            Ok(()) => {
                // `resynthesize_entity` supersedes using the real wall clock, so the
                // node it just closed and the one it just opened straddle an instant
                // strictly later than the loop-level `now` captured above (which is
                // for the initial stale-entities query, not this fingerprint). Reusing
                // that stale `now` here would still resolve the just-superseded old
                // node, recording its mention count against the new node's key and
                // making the entity look stale again on every following tick.
                let post_resynthesis_now = liam_store::Millis::now();
                match store
                    .entity_mentions_fingerprint(&subject, scope.as_deref(), post_resynthesis_now)
                    .await
                {
                    Ok(fingerprint) => {
                        match store
                            .mark_entity_synthesized(
                                &subject,
                                scope.as_deref(),
                                fingerprint,
                                post_resynthesis_now,
                            )
                            .await
                        {
                            Ok(()) => tracing::info!(subject, "entity resynthesized"),
                            Err(e) => {
                                tracing::warn!(subject, error = %e, "failed to mark entity synthesized")
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(subject, error = %e, "failed to recompute entity mentions fingerprint after resynthesis")
                    }
                }
            }
            Err(e) => {
                tracing::warn!(subject, error = %e, "entity resynthesis failed");
            }
        }
    }
}

/// Picks the resynthesis token budget for a fingerprint: full tier at or above the
/// mention threshold, light tier below it or when the fingerprint lookup failed.
fn resynthesis_budget(
    fingerprint: Result<liam_store::Fingerprint, &liam_store::Error>,
    threshold: usize,
    full_budget: usize,
    light_budget: usize,
) -> usize {
    match fingerprint {
        Ok(f) if f.edge_count as usize >= threshold => full_budget,
        _ => light_budget,
    }
}

/// Repairs `mentions` edges left dangling by a supersession, so citing
/// entities keep pointing at the current fact. A failure here must not block
/// the rest of the tick, matching `sweep`'s and `refresh_clusters`'s own
/// error-doesn't-abort-the-tick pattern.
async fn repair_mentions(store: &DefaultGraph) {
    match store.repair_superseded_mentions().await {
        Ok(repaired) => tracing::info!(repaired, "provenance repair completed"),
        Err(e) => tracing::warn!(error = %e, "provenance repair failed"),
    }
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
    use liam_store::{
        relation, Fingerprint, FixedClock, GraphConfig, Millis, NewEdge, NewNode, RetentionPolicy,
    };

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

    /// A `MemoryServer` wrapping `store`, backed by mock model doubles that
    /// never touch a real embedder, reranker, or LLM: enough for
    /// `maintenance_tick` to reach the store it holds, nothing else.
    fn test_server(store: Arc<DefaultGraph>) -> MemoryServer {
        MemoryServer::new(
            store,
            Arc::new(liam_model::MockEmbedder::new(8)),
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
            30,
            false,
            8192,
            1,
        )
    }

    #[tokio::test]
    async fn the_tick_refreshes_clusters_after_the_sweep() {
        let (store, _clock) = seeded_pair(Millis(1000)).await;
        // Nothing is old enough to sweep under this policy, which isolates the
        // refresh half of `maintenance_tick` from the sweep half.
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let server = test_server(Arc::clone(&store));

        maintenance_tick(&server, &policy, 0, 30, true, 8, 512).await;

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
        let store = Arc::new(store);
        let server = test_server(Arc::clone(&store));

        maintenance_tick(&server, &policy, 0, 30, true, 8, 512).await;

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

    /// `Llm` double counting `complete_capped` calls, blocked on a `Notify`
    /// the test never fires, so a benchmark using it starts but never finishes.
    struct GatedCountingLlm {
        release: Arc<tokio::sync::Notify>,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl GatedCountingLlm {
        fn new(release: Arc<tokio::sync::Notify>) -> Self {
            Self {
                release,
                calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl Llm for GatedCountingLlm {
        async fn complete(&self, system: &str, prompt: &str) -> liam_model::Result<String> {
            self.complete_capped(system, prompt, usize::MAX).await
        }

        async fn complete_capped(
            &self,
            _system: &str,
            _prompt: &str,
            _max_new_tokens: usize,
        ) -> liam_model::Result<String> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.release.notified().await;
            Ok("gated".to_string())
        }
    }

    async fn autotune_store() -> Arc<DefaultGraph> {
        Arc::new(
            DefaultGraph::open(":memory:", GraphConfig::new(8))
                .await
                .expect("open in-memory store"),
        )
    }

    #[tokio::test]
    async fn autotuned_server_accepts_requests_up_to_the_memory_ceiling_before_the_benchmark_completes(
    ) {
        // Given no cache file, so a benchmark runs, gated so it never
        // resolves in this test
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().to_str().expect("utf8 path").to_string();
        let release = Arc::new(tokio::sync::Notify::new());
        let llm_double = Arc::new(GatedCountingLlm::new(release));
        let llm: Arc<dyn Llm> = llm_double.clone();
        let ceiling = tuning::memory_ceiling(8192);

        // When the server is built
        let server = build_autotuned_server(
            autotune_store().await,
            Arc::new(liam_model::MockEmbedder::new(8)),
            Arc::new(liam_model::IdentityReranker),
            llm,
            30,
            false,
            8192,
            cache_dir,
            "model/quant".to_string(),
            "cpu".to_string(),
        );

        // Then it is already usable up to the memory-safe ceiling before the
        // benchmark resolves, and the benchmark has genuinely started in the
        // background, holding one of those same permits for its own probe
        assert_eq!(
            server.generation_permits_handle().available_permits(),
            ceiling
        );
        tokio::task::yield_now().await;
        assert_eq!(
            llm_double.calls(),
            1,
            "the background benchmark must have started"
        );
        assert_eq!(
            server.generation_permits_handle().available_permits(),
            ceiling - 1,
            "the benchmark's own probe must hold a real permit from the shared semaphore"
        );
    }

    #[tokio::test]
    async fn autotuned_server_skips_the_benchmark_on_a_cache_hit() {
        // Given a cache entry matching the fingerprint used below
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().to_str().expect("utf8 path").to_string();
        tuning::save_cache(&cache_dir, "model/quant", "cpu", 4, 8192);
        let release = Arc::new(tokio::sync::Notify::new());
        let llm_double = Arc::new(GatedCountingLlm::new(release));
        let llm: Arc<dyn Llm> = llm_double.clone();

        // When the server is built with the same cache dir/model/backend
        let server = build_autotuned_server(
            autotune_store().await,
            Arc::new(liam_model::MockEmbedder::new(8)),
            Arc::new(liam_model::IdentityReranker),
            llm,
            30,
            false,
            8192,
            cache_dir,
            "model/quant".to_string(),
            "cpu".to_string(),
        );

        // Then the semaphore is sized from the cache immediately, and no
        // benchmark ever runs
        assert_eq!(server.generation_permits_handle().available_permits(), 4);
        tokio::task::yield_now().await;
        assert_eq!(
            llm_double.calls(),
            0,
            "a cache hit must spawn no benchmark at all"
        );
    }

    /// Captures everything written to it, so a test can assert on log text
    /// without a real terminal.
    #[derive(Clone, Default)]
    struct CapturedLogs(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for CapturedLogs {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("captured logs lock").extend(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLogs {
        type Writer = CapturedLogs;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Installs `captured` as the default subscriber for the calling thread
    /// and returns the guard that must stay in scope for the duration of the
    /// awaited call being observed. A plain `with_default` closure cannot
    /// wrap an `.await`, so this uses the guard form instead; `#[tokio::test]`
    /// defaults to a current-thread runtime, so the test body never migrates
    /// to another OS thread mid-await and the thread-local guard stays valid
    /// across the awaits it wraps.
    fn install_captured_logs(captured: CapturedLogs) -> tracing::subscriber::DefaultGuard {
        // Without `with_ansi(false)` the formatter wraps field separators in
        // color codes (e.g. `repaired\x1b[2m=\x1b[0m1`), which breaks a plain
        // substring match like `repaired=1` even though the field is there.
        let subscriber = tracing_subscriber::fmt()
            .with_writer(captured)
            .with_ansi(false)
            .finish();
        tracing::subscriber::set_default(subscriber)
    }

    #[tokio::test]
    async fn the_tick_repairs_a_superseded_facts_mentions_edge() {
        // Given a fact superseded since the watermark, cited by an entity
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let store = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
        let entity = store
            .insert(NewNode::entity("person", "Ada"))
            .await
            .unwrap();
        let fact = store
            .insert(NewNode::now("fact", "first", "x"))
            .await
            .unwrap();
        store
            .relate(&entity, &fact, relation::MENTIONS)
            .await
            .unwrap();
        clock.set(Millis(2000));
        let fact2 = store
            .supersede(&fact, NewNode::now("fact", "second", "x"))
            .await
            .unwrap();
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let server = test_server(Arc::clone(&store));
        let captured = CapturedLogs::default();
        let _guard = install_captured_logs(captured.clone());

        // When the tick fires
        maintenance_tick(&server, &policy, 0, 30, true, 8, 512).await;

        // Then the citing entity gains a mentions edge to the new fact
        let neighbors = store.neighbors(&entity, Millis(2000)).await.unwrap();
        assert!(
            neighbors.contains(&fact2),
            "the entity must gain a mentions edge to the fact that superseded the original"
        );

        // And the repaired count is logged
        let log = String::from_utf8(captured.0.lock().expect("captured logs lock").clone())
            .expect("log output is utf8");
        assert!(
            log.contains("repaired=1"),
            "expected the repaired count to be logged: {log}"
        );
    }

    #[tokio::test]
    async fn the_tick_logs_zero_repairs_when_nothing_was_superseded() {
        // Given nothing superseded since the watermark
        let (store, _clock) = seeded_pair(Millis(1000)).await;
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let server = test_server(Arc::clone(&store));
        let captured = CapturedLogs::default();
        let _guard = install_captured_logs(captured.clone());

        // When the tick fires
        maintenance_tick(&server, &policy, 0, 30, true, 8, 512).await;

        // Then repair is a no-op and logs zero repairs
        let log = String::from_utf8(captured.0.lock().expect("captured logs lock").clone())
            .expect("log output is utf8");
        assert!(
            log.contains("repaired=0"),
            "expected a zero repaired count to be logged: {log}"
        );
    }

    /// As `test_server`, but with a caller-supplied `Llm` instead of the fixed
    /// `MockLlm`, for the resynthesis tests below that need a double which
    /// either succeeds with a grounded reply or fails outright.
    fn test_server_with_llm(store: Arc<DefaultGraph>, llm: Arc<dyn Llm>) -> MemoryServer {
        MemoryServer::new(
            store,
            Arc::new(liam_model::MockEmbedder::new(8)),
            Arc::new(liam_model::IdentityReranker),
            llm,
            30,
            false,
            8192,
            1,
        )
    }

    /// Always errors, so a resynthesis attempt driven by it must leave the
    /// entity stale rather than succeed (mirror mcp.rs's `FailingLlm`).
    struct FailingLlm;
    #[async_trait::async_trait]
    impl Llm for FailingLlm {
        async fn complete(&self, _s: &str, _p: &str) -> liam_model::Result<String> {
            Err(liam_model::ModelError::Llm("boom".into()))
        }
    }

    /// Extracts "(kind) label" from a synthesis prompt's "Entity: (kind) label"
    /// line and echoes it back, which entity synthesis's grounding check always
    /// accepts since the reply is drawn entirely from its own vocabulary seed
    /// (mirror mcp.rs's `grounded_entity_reply`).
    fn grounded_entity_reply(prompt: &str) -> String {
        let line = prompt
            .lines()
            .find(|l| l.starts_with("Entity: ("))
            .expect("synthesis prompt must have an Entity line");
        let rest = line.trim_start_matches("Entity: (");
        let close = rest
            .find(')')
            .expect("Entity line must have a closing paren");
        format!("{} {}", &rest[..close], rest[close + 1..].trim())
    }

    /// Always succeeds with a reply grounded in the entity's own kind/label,
    /// for resynthesis tests that need a working generation path.
    struct GroundedLlm;
    #[async_trait::async_trait]
    impl Llm for GroundedLlm {
        async fn complete(&self, _s: &str, prompt: &str) -> liam_model::Result<String> {
            Ok(grounded_entity_reply(prompt))
        }
    }

    /// As `GroundedLlm`, but sleeps on the real clock first, so the wall-clock
    /// instant `resynthesize_entity`'s supersede runs at is measurably later
    /// than the one `resynthesize_stale` captured before its loop began: the
    /// gap a fresh-fingerprint regression test needs to be reproducible
    /// rather than a coin flip on how fast the test happens to run.
    struct DelayedGroundedLlm;
    #[async_trait::async_trait]
    impl Llm for DelayedGroundedLlm {
        async fn complete(&self, _s: &str, prompt: &str) -> liam_model::Result<String> {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            Ok(grounded_entity_reply(prompt))
        }
    }

    /// Blocks inside `complete()` on a `Notify` the test controls, tracking the
    /// PEAK number of concurrent `complete()` calls it has ever seen: a final
    /// count of 0 proves nothing, since every call reaches 0 eventually
    /// whether or not two of them were ever in flight together (mirror
    /// mcp.rs's `GatedLlm`).
    struct GatedLlm {
        release: Arc<tokio::sync::Notify>,
        in_flight: std::sync::atomic::AtomicUsize,
        peak: std::sync::atomic::AtomicUsize,
    }

    impl GatedLlm {
        fn new(release: Arc<tokio::sync::Notify>) -> Self {
            Self {
                release,
                in_flight: std::sync::atomic::AtomicUsize::new(0),
                peak: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn in_flight(&self) -> usize {
            self.in_flight.load(std::sync::atomic::Ordering::SeqCst)
        }

        fn peak(&self) -> usize {
            self.peak.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl Llm for GatedLlm {
        async fn complete(&self, _s: &str, _prompt: &str) -> liam_model::Result<String> {
            let now = self
                .in_flight
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;
            self.peak
                .fetch_max(now, std::sync::atomic::Ordering::SeqCst);
            self.release.notified().await;
            self.in_flight
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            Ok("gated reply".to_string())
        }
    }

    /// Cooperatively yields until `condition()` is true, bounded so a real bug
    /// panics the test instead of hanging the suite (mirror mcp.rs's
    /// `wait_until`).
    async fn wait_until(mut condition: impl FnMut() -> bool) {
        for _ in 0..10_000 {
            if condition() {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("condition never became true; the two tasks likely deadlocked");
    }

    #[tokio::test]
    async fn the_tick_resynthesizes_at_most_the_configured_cap_oldest_stale_first() {
        // Given 2 stale (never-synthesized) entities and a cap of 1
        let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        let capped = store
            .insert(NewNode::entity("person", "First Stale"))
            .await
            .unwrap();
        let leftover = store
            .insert(NewNode::entity("person", "Second Stale"))
            .await
            .unwrap();
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let llm: Arc<dyn Llm> = Arc::new(GroundedLlm);
        let server = test_server_with_llm(Arc::clone(&store), llm);

        // When the tick fires with max_resynth_per_tick = 1
        maintenance_tick(&server, &policy, 1, 30, false, 8, 512).await;

        // Then exactly the cap's worth were resynthesized: the capped
        // entity's old id no longer resolves, superseded by a fresh one
        assert!(
            store.resolve_handle(capped.as_str()).await.is_err(),
            "the entity within the cap must have been superseded"
        );
        // And the entity beyond the cap is untouched and still reported stale
        assert!(
            store.resolve_handle(leftover.as_str()).await.is_ok(),
            "the entity beyond the cap must remain live under its old id"
        );
        let stale = store.stale_entities(10, Millis::now()).await.unwrap();
        assert!(
            stale.iter().any(|(id, _, _)| id == &leftover),
            "the entity beyond the cap must still be reported stale for the next tick"
        );
    }

    #[tokio::test]
    async fn a_successful_resynthesis_clears_the_entitys_stale_state() {
        // Given a single stale (never-synthesized) entity
        let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        let old_id = store
            .insert(NewNode::entity("person", "Solo Stale"))
            .await
            .unwrap();
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let llm: Arc<dyn Llm> = Arc::new(GroundedLlm);
        let server = test_server_with_llm(Arc::clone(&store), llm);

        // When the tick fires
        maintenance_tick(&server, &policy, 5, 30, false, 8, 512).await;

        // Then the entity's id changed, proving a real resynthesis ran
        assert!(
            store.resolve_handle(old_id.as_str()).await.is_err(),
            "a successful resynthesis must supersede the old id"
        );
        // And a follow-up check at the same as_of no longer reports it stale,
        // proving the (subject, scope) staleness state was updated even
        // though the NodeId it is keyed against changed underneath it
        let stale = store.stale_entities(10, Millis::now()).await.unwrap();
        assert!(
            stale.is_empty(),
            "a freshly resynthesized entity must not still be reported stale: {stale:?}"
        );
    }

    #[tokio::test]
    async fn a_resynthesized_entity_with_a_live_mention_is_not_reported_stale_again_next_tick() {
        // Given an entity with one live mentions edge of its own, so the old
        // and new node fingerprint differently (1 mention vs. 0, since a
        // supersede never carries a node's own edges over on its own), and a
        // slow-but-successful llm so the real wall clock has visibly moved
        // by the time the resynthesis it drives actually supersedes the
        // entity, past the `now` `resynthesize_stale` captured before its
        // loop began
        let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        let entity = store
            .insert(NewNode::entity("person", "Mentioned Stale"))
            .await
            .unwrap();
        let fact = store
            .insert(NewNode::now("fact", "some fact", "x"))
            .await
            .unwrap();
        store
            .relate(&entity, &fact, relation::MENTIONS)
            .await
            .unwrap();
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let llm: Arc<dyn Llm> = Arc::new(DelayedGroundedLlm);
        let server = test_server_with_llm(Arc::clone(&store), llm);

        // When the tick fires and resynthesizes it
        maintenance_tick(&server, &policy, 5, 30, false, 8, 512).await;
        assert!(
            store.resolve_handle(entity.as_str()).await.is_err(),
            "a successful resynthesis must supersede the old id"
        );

        // Then a later check, at a genuinely fresh wall-clock instant, must
        // not find it stale again. A fingerprint recorded against the
        // just-superseded old node's mention count (this entity's 1, rather
        // than the new node's real 0) would mismatch the new node's actual
        // fingerprint forever after, reporting the entity stale on every
        // following tick regardless of whether anything had changed
        let stale = store.stale_entities(10, Millis::now()).await.unwrap();
        assert!(
            stale.is_empty(),
            "the recorded fingerprint must reflect the new node, not the just-superseded \
             old one: {stale:?}"
        );
    }

    #[tokio::test]
    async fn a_failed_resynthesis_leaves_the_entity_stale_and_logs_it_without_a_panic() {
        // Given a stale entity whose resynthesis will fail
        let failing_store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        let failing_entity = failing_store
            .insert(NewNode::entity("person", "Fails Guy"))
            .await
            .unwrap();
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let failing_store = Arc::new(failing_store);
        let failing_llm: Arc<dyn Llm> = Arc::new(FailingLlm);
        let failing_server = test_server_with_llm(Arc::clone(&failing_store), failing_llm);
        let captured = CapturedLogs::default();
        let _guard = install_captured_logs(captured.clone());

        // When its tick fires
        maintenance_tick(&failing_server, &policy, 5, 30, false, 8, 512).await;

        // Then the tick completed without panicking (reaching here proves
        // that), the entity remains live under its old id and still stale
        assert!(
            failing_store
                .resolve_handle(failing_entity.as_str())
                .await
                .is_ok(),
            "a failed resynthesis must leave the entity's old id live"
        );
        let stale = failing_store
            .stale_entities(10, Millis::now())
            .await
            .unwrap();
        assert!(
            stale.iter().any(|(id, _, _)| id == &failing_entity),
            "a failed resynthesis must leave the entity reported stale"
        );
        // And the failure was logged
        let log = String::from_utf8(captured.0.lock().expect("captured logs lock").clone())
            .expect("log output is utf8");
        assert!(
            log.contains("entity resynthesis failed"),
            "expected the failure to be logged: {log}"
        );

        // And a separate entity's own tick, with a working llm, still
        // succeeds independently: the failure path above never touched
        // shared state a second entity's resynthesis would also need
        let working_store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        let working_entity = working_store
            .insert(NewNode::entity("person", "Works Guy"))
            .await
            .unwrap();
        let working_store = Arc::new(working_store);
        let working_llm: Arc<dyn Llm> = Arc::new(GroundedLlm);
        let working_server = test_server_with_llm(Arc::clone(&working_store), working_llm);

        maintenance_tick(&working_server, &policy, 5, 30, false, 8, 512).await;

        assert!(
            working_store
                .resolve_handle(working_entity.as_str())
                .await
                .is_err(),
            "the other entity's own resynthesis must still succeed independently"
        );
    }

    #[tokio::test]
    async fn resynthesize_stale_processes_entities_one_at_a_time_not_concurrently() {
        // Given 2 stale entities, a cap covering both, and an llm gated so a
        // call blocks until this test decides to release it
        let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        store
            .insert(NewNode::entity("person", "Gated One"))
            .await
            .unwrap();
        store
            .insert(NewNode::entity("person", "Gated Two"))
            .await
            .unwrap();
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let release = Arc::new(tokio::sync::Notify::new());
        let llm = Arc::new(GatedLlm::new(release.clone()));
        let server = test_server_with_llm(Arc::clone(&store), llm.clone());

        // When the tick runs, spawned since the gated llm blocks until released
        let handle = tokio::spawn(async move {
            maintenance_tick(&server, &policy, 2, 30, false, 8, 512).await;
        });

        // Then only one entity's generation is ever in flight: release the
        // first, wait for the second to start, release it too
        wait_until(|| llm.in_flight() == 1).await;
        release.notify_one();
        wait_until(|| llm.in_flight() == 1).await;
        release.notify_one();
        handle.await.expect("maintenance tick task panicked");

        // The PEAK proves the two generations were never in flight together,
        // not merely that the count returned to normal afterward
        assert_eq!(
            llm.peak(),
            1,
            "resynthesize_stale must process entities sequentially, \
             never two generations in flight together"
        );
    }

    #[tokio::test]
    async fn skip_resynthesis_skips_only_resynthesis_not_the_rest_of_the_tick() {
        // Given a stale entity, and a superseded fact so repair has work to log
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let store = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
        let stale_entity = store
            .insert(NewNode::entity("person", "Autotune Guy"))
            .await
            .unwrap();
        let citing_entity = store
            .insert(NewNode::entity("person", "Cited By"))
            .await
            .unwrap();
        let fact = store
            .insert(NewNode::now("fact", "first", "x"))
            .await
            .unwrap();
        store
            .relate(&citing_entity, &fact, relation::MENTIONS)
            .await
            .unwrap();
        clock.set(Millis(2000));
        store
            .supersede(&fact, NewNode::now("fact", "second", "x"))
            .await
            .unwrap();
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let llm: Arc<dyn Llm> = Arc::new(GroundedLlm);
        let server = test_server_with_llm(Arc::clone(&store), llm);
        let captured = CapturedLogs::default();
        let _guard = install_captured_logs(captured.clone());

        // When the autotuning run-on-start tick fires with resynthesis skipped
        maintenance_tick(&server, &policy, 5, 30, true, 8, 512).await;

        // Then repair still ran (its own log line is still present) ...
        let log = String::from_utf8(captured.0.lock().expect("captured logs lock").clone())
            .expect("log output is utf8");
        assert!(
            log.contains("repaired=1"),
            "repair must still run when resynthesis is skipped: {log}"
        );
        // ... but resynthesis never touched the stale entity
        assert!(
            store.resolve_handle(stale_entity.as_str()).await.is_ok(),
            "a skipped resynthesis pass must leave the stale entity untouched"
        );

        // When a later tick fires with resynthesis no longer skipped
        maintenance_tick(&server, &policy, 5, 30, false, 8, 512).await;

        // Then that later tick resynthesizes the entity normally
        assert!(
            store.resolve_handle(stale_entity.as_str()).await.is_err(),
            "a later, non-skipped tick must resynthesize the entity"
        );
    }

    #[test]
    fn resynthesis_budget_returns_full_budget_at_exact_threshold() {
        // Given a fingerprint whose edge_count sits exactly at the threshold
        let fingerprint = Ok(Fingerprint {
            edge_count: 8,
            max_tx_from: Millis(0),
        });

        // When the resynthesis budget is computed
        let budget = resynthesis_budget(fingerprint, 8, 512, 256);

        // Then the full budget applies: the boundary is inclusive, not exclusive
        assert_eq!(budget, 512);
    }

    #[test]
    fn resynthesis_budget_returns_light_budget_below_threshold() {
        // Given a fingerprint one edge short of the threshold
        let fingerprint = Ok(Fingerprint {
            edge_count: 7,
            max_tx_from: Millis(0),
        });

        // When the resynthesis budget is computed
        let budget = resynthesis_budget(fingerprint, 8, 512, 256);

        // Then the light budget applies
        assert_eq!(budget, 256);
    }

    #[test]
    fn resynthesis_budget_returns_full_budget_comfortably_above_threshold() {
        // Given a fingerprint comfortably above the threshold
        let fingerprint = Ok(Fingerprint {
            edge_count: 20,
            max_tx_from: Millis(0),
        });

        // When the resynthesis budget is computed
        let budget = resynthesis_budget(fingerprint, 8, 512, 256);

        // Then the full budget applies
        assert_eq!(budget, 512);
    }

    #[test]
    fn resynthesis_budget_returns_light_budget_on_fingerprint_error() {
        // Given a failed fingerprint lookup
        let err = liam_store::Error::Backend("test".into());
        let fingerprint = Err(&err);

        // When the resynthesis budget is computed
        let budget = resynthesis_budget(fingerprint, 8, 512, 256);

        // Then it falls back to the light budget rather than aborting
        assert_eq!(budget, 256);
    }

    /// Captures the actual `max_new_tokens` value the resynthesis path passed
    /// to `complete_capped`, so a test can assert on what really reached the
    /// LLM call rather than inferring it from an isolated threshold check.
    struct RecordingMaxTokensLlm {
        recorded_max_new_tokens: std::sync::Mutex<Option<usize>>,
    }

    impl RecordingMaxTokensLlm {
        fn new() -> Self {
            Self {
                recorded_max_new_tokens: std::sync::Mutex::new(None),
            }
        }

        fn recorded(&self) -> Option<usize> {
            *self.recorded_max_new_tokens.lock().unwrap()
        }
    }

    #[async_trait::async_trait]
    impl Llm for RecordingMaxTokensLlm {
        async fn complete(&self, system: &str, prompt: &str) -> liam_model::Result<String> {
            self.complete_capped(system, prompt, usize::MAX).await
        }

        async fn complete_capped(
            &self,
            _system: &str,
            _prompt: &str,
            max_new_tokens: usize,
        ) -> liam_model::Result<String> {
            *self.recorded_max_new_tokens.lock().unwrap() = Some(max_new_tokens);
            Ok("recorded".to_string())
        }
    }

    #[tokio::test]
    async fn the_tick_gives_a_frequently_mentioned_entity_the_full_tier_budget() {
        // Given a stale entity with 20 live mentions, comfortably at or above
        // a full-synthesis mention threshold of 8
        let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        let entity = store
            .insert(NewNode::entity("person", "Frequently Mentioned"))
            .await
            .unwrap();
        for i in 0..20 {
            let fact = store
                .insert(NewNode::now("fact", format!("fact {i}"), "x"))
                .await
                .unwrap();
            store
                .relate(&entity, &fact, relation::MENTIONS)
                .await
                .unwrap();
        }
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let llm = Arc::new(RecordingMaxTokensLlm::new());
        let server = test_server_with_llm(Arc::clone(&store), llm.clone());

        // When the tick fires with a full-synthesis threshold of 8 and a
        // full-tier budget of 512
        maintenance_tick(&server, &policy, 5, 30, false, 8, 512).await;

        // Then the actual max_new_tokens value reaching the LLM call is the
        // full-tier budget, not the light-tier default
        assert_eq!(
            llm.recorded(),
            Some(512),
            "an entity at or above the mention threshold must get the full-tier budget"
        );
    }

    #[tokio::test]
    async fn the_tick_gives_a_rarely_mentioned_entity_the_baseline_budget() {
        // Given a stale entity with no live mentions, below a
        // full-synthesis mention threshold of 8
        let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        store
            .insert(NewNode::entity("person", "Rarely Mentioned"))
            .await
            .unwrap();
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let llm = Arc::new(RecordingMaxTokensLlm::new());
        let server = test_server_with_llm(Arc::clone(&store), llm.clone());

        // When the tick fires with the same full-synthesis threshold of 8
        // and full-tier budget of 512
        maintenance_tick(&server, &policy, 5, 30, false, 8, 512).await;

        // Then the actual max_new_tokens value reaching the LLM call is the
        // baseline light-tier budget
        assert_eq!(
            llm.recorded(),
            Some(mcp::ENTITY_SYNTHESIS_MAX_NEW_TOKENS),
            "an entity below the mention threshold must get the baseline budget"
        );
    }
}
