// SPDX-License-Identifier: Apache-2.0
//! liam-daemon: serves liam memory to agents over MCP.
//!
//! A thin shell wiring `liam-store` (retrieval) and `liam-model`
//! (embedding). `provider = "local"` loads the embedder in-process, no server.

mod cli;
mod clusters;
mod consolidation_job;
mod degraded;
mod deletion;
mod entity_page;
mod extraction;
mod hosted;
mod log_startup;
mod mcp;
mod metrics;
mod page_retry;
mod prompt;
mod recall;
/// Recall latency and nDCG@10 benchmark; test-only, see the module docs to run it.
#[cfg(test)]
mod recall_bench;
mod render;
mod shutdown;
mod summary;
mod sync;
#[cfg(test)]
mod test_logs;
mod tiering;
/// Grounding eval for remember/recall/relate; test-only, see the module docs to run each tier.
#[cfg(test)]
mod tool_eval;
mod transport;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use liam_store::DefaultGraph;

// Re-exported at this crate's root, not merely imported, so the existing
// `crate::config::...` paths keep resolving now that the module itself lives
// in the library both binaries share. The call sites are in the submodules,
// not the files above them: `mcp/producer.rs`, `transport/activation.rs`,
// and `transport/socket.rs`.
pub use liam_daemon::config;

use config::Config;
use consolidation_job::{ConsolidationJob, JobSetup};
use liam_daemon::consolidation::maintenance::standard_steps;
use liam_daemon::models::{
    build_embedder, resolve_config_path, resolve_log_dir, resolve_snapshot_dir,
};
use liam_daemon::{embedder_fingerprint, state_dir, storelock, telemetry};
use liam_log::compactor::CompactorTask;
use log_startup::StartedStore;
use mcp::MemoryServer;
use shutdown::Shutdown;
use tiering::usage::{UsageFlush, UsageFlusher, UsageRecorder};

/// How long the maintenance loop gets to stop once the daemon is stopping, and
/// then how long the compactor gets to finish what is queued. With the session
/// drain they stay below the supervisor's grace period.
const MAINTENANCE_STOP_DEADLINE: Duration = Duration::from_secs(2);
const COMPACTOR_DRAIN_DEADLINE: Duration = Duration::from_secs(3);
/// How long the last usage flush gets once the transports have drained.
const USAGE_FLUSH_DEADLINE: Duration = Duration::from_secs(2);
/// What a directory readable by others exposes, for the start-up warning.
const STORED_DATA_RISK: &str = "other local users may be able to read the memory stored in it";
const SOCKET_RISK: &str =
    "the socket is owner-only from creation, but others can still see its name in this directory";

fn main() -> anyhow::Result<()> {
    // Before any thread starts: the mask is process-wide, and it closes the
    // window in which a file the store creates is wider than its owner.
    liam_log::private_fs::restrict_umask();
    // Parse BEFORE anything else touches the filesystem or the environment.
    // A usage error must exit 2 without having opened the store or taken the
    // store lock, since a typo like `liamd serv` would otherwise break the
    // real daemon's next start. `parse` exits the process itself on a usage
    // error or on --help/--version.
    let cli = <cli::Cli as clap::Parser>::parse();
    let mode = cli.mode();
    let config_path = cli.config_path(std::env::var("LIAM_CONFIG").ok().as_deref());

    let config = Config::load(config_path.as_ref())?;
    let is_proxy = mode == cli::Mode::Proxy;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(run(mode, config));

    if is_proxy {
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

    if mode == cli::Mode::EncryptLog {
        let result = deletion::encrypt_log(&config).await;
        secure_state_files_after_command(&config).await;
        return result;
    }

    if let cli::Mode::Delete(action) = mode {
        let result = deletion::delete_command(&config, action, &mut std::io::stdout()).await;
        secure_state_files_after_command(&config).await;
        return result;
    }

    serve_with_store(mode, config).await
}

/// A command that opens the store may leave files wider than their owner,
/// whatever its outcome.
async fn secure_state_files_after_command(config: &Config) {
    if let Ok(database) = resolve_config_path("database_path", &config.database_path) {
        secure_state_files_off_runtime(config, database).await;
    }
}

/// The walk reads every directory entry, which can take a while for a large log
/// directory, so it runs on a blocking thread and leaves the async workers free.
async fn secure_state_files_off_runtime(config: &Config, database: std::path::PathBuf) {
    let config = config.clone();
    let walk = tokio::task::spawn_blocking(move || {
        state_dir::secure_state_files_for(&config, &database, None);
    });
    if let Err(error) = walk.await {
        tracing::warn!(%error, "the check of the state files did not finish");
    }
}

/// Tightens the state directories left over from an earlier start, before the
/// first file is created in them. New ones are created private.
fn secure_existing_state_dirs(
    database_path: &std::path::Path,
    log_dir: &std::path::Path,
    snapshot_dir: &std::path::Path,
) {
    let Some(state_root) = state_dir::state_root_from_home(std::env::var_os("HOME")) else {
        return;
    };
    let database_dir = database_path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty());
    let dirs: Vec<&std::path::Path> = database_dir
        .into_iter()
        .chain([log_dir, snapshot_dir])
        .collect();
    state_dir::secure_dirs(&dirs, &state_root, STORED_DATA_RISK);
}

/// Everything that needs the store and the models: the stdio server and the
/// socket daemon. Both take the per-process store lock.
async fn serve_with_store(mode: cli::Mode, config: Config) -> anyhow::Result<()> {
    serve_until(mode, config, Shutdown::default()).await
}

/// `serve_with_store` that stops when `shutdown`'s token is cancelled, which a
/// signal or a poisoned event log does.
async fn serve_until(mode: cli::Mode, config: Config, shutdown: Shutdown) -> anyhow::Result<()> {
    // Exclusive per-process lock, taken once, before the first store open.
    // `spawn_gc` below shares this same `Graph`, so there is no second
    // connection to reason about; the lock is not retaken for it regardless.
    // Bound to a named variable so it lives for the rest of the process:
    // `let _ = ...` would drop it immediately and release the lock right
    // away. See `storelock` for why this is a real advisory `flock` and not
    // a PID file.
    let database_path = resolve_config_path("database_path", &config.database_path)?;
    let log_dir = resolve_log_dir(&config, &database_path)?;
    secure_existing_state_dirs(
        &database_path,
        &log_dir,
        &resolve_snapshot_dir(&config, &database_path)?,
    );
    let _lock = storelock::StoreLock::acquire(&database_path).context(
        "liamd needs the store to itself; if the holder is another liamd, `liamd proxy` \
         shuttles to it and opens no store",
    )?;

    // A store that predates both the log and the fingerprint is refused below,
    // and `liam rebuild` needs a log to clear that, so the log is made first.
    if embedder_fingerprint::predates_fingerprints(&database_path)? {
        log_startup::migrate_unlogged_rows(&config, &database_path, &log_dir).await?;
    }

    // Verified before the store opens, which creates `node_vectors` at the
    // configured width, and recorded only once it has opened.
    let fingerprint = embedder_fingerprint::verify(
        &database_path,
        &embedder_fingerprint::EmbedderFingerprint::from_config(&config),
    )?;

    if let Some(warning) = config.removed_reranker_warning() {
        tracing::warn!("{warning}");
    }
    if let Some(warning) = config.embedder.removed_device_warning() {
        tracing::warn!("{warning}");
    }
    if let Some(warning) = config.removed_ask_sufficiency_check_warning() {
        tracing::warn!("{warning}");
    }
    for warning in config.removed_gc_page_warnings() {
        tracing::warn!("{warning}");
    }
    if let Some(warning) = config.missing_config_warning() {
        tracing::warn!("{warning}");
    }
    let embedder = build_embedder(&config)?;
    let StartedStore {
        store,
        compactor,
        log,
        keys,
    } = log_startup::open_store_with_log(
        &config,
        &database_path,
        &log_dir,
        embedder.clone(),
        &shutdown,
    )
    .await?;
    // The store has created its files by now, and an older start left others wider.
    secure_state_files_off_runtime(&config, database_path.clone()).await;
    if let Some(fingerprint) = fingerprint {
        fingerprint.commit()?;
    }
    let store = Arc::new(store);

    let job_embedder = embedder.clone();
    let server = MemoryServer::new(store, embedder, config.ask_timeout_secs);

    let usage = config
        .tiering
        .enabled
        .then(|| Arc::new(UsageRecorder::new(config.tiering.max_tracked_items)));
    let server = match &usage {
        Some(recorder) => server.with_usage_recorder(recorder.clone()).with_feedback(
            config.tiering.feedback_per_producer_per_hour,
            config.tiering.max_tracked_items,
        ),
        None => server,
    };
    let metrics = config.metrics.enabled.then(|| {
        metrics::build(
            &config.metrics,
            &database_path,
            config.deletion.deadline_days,
        )
    });
    let server = match &metrics {
        Some(metrics) => server.with_metrics(metrics.clone()),
        None => server,
    };
    let server = server
        .with_shutdown(shutdown.token())
        .with_recall_timeout(config.recall_timeout_secs)
        .with_vocabulary(config::resolve_vocabulary(&config.extraction))
        .with_max_basis_ids_rendered(config.summary.max_basis_ids_rendered)
        .with_min_summary_similarity(config.summary.min_summary_similarity);
    let job = Arc::new(ConsolidationJob::new(JobSetup {
        config: &config,
        database_path: &database_path,
        log_dir: &log_dir,
        log,
        keys: keys.clone(),
        embedder: job_embedder,
        server: &server,
        shutdown: &shutdown,
        metrics: metrics.clone(),
    })?);
    let run_now = job
        .start(config.gc.run_on_start)
        .await
        .context("could not finish the consolidation rollback a crash interrupted")?;
    let node_states: Vec<Arc<dyn deletion::NodeState>> = usage
        .iter()
        .map(|recorder| recorder.clone() as Arc<dyn deletion::NodeState>)
        .collect();
    let server = server
        .with_admin(config.admin.clone())
        .with_consolidator(job.clone());
    let server = match &keys {
        Some(keys) => server.with_delete_preview(Arc::new(deletion::DeletePreview::new(
            &config,
            &database_path,
            liam_daemon::event_log::log_reader(&log_dir, Some(keys))?,
            Arc::clone(keys),
            node_states.clone(),
        )?)),
        None => server,
    };
    // After the crash recovery above and before anything is served, written or
    // consolidated, so no reader or writer meets a half-deleted subject. The
    // slot stays locked for the whole resume, and the service gives its handle
    // back when it ends, which a later rollback needs.
    let shared = server.shared_store();
    let slot = shared.exclusive().await;
    let resumed = slot.as_ref().cloned().context("the store is unavailable")?;
    deletion::resume_pending(
        &config,
        &database_path,
        &log_dir,
        resumed,
        keys,
        node_states,
        &_lock,
    )
    .await?;
    drop(slot);
    let maintenance = spawn_gc(&config, job, run_now);
    let usage_flush = usage.map(|recorder| {
        UsageFlush::start(UsageFlusher::new(
            recorder,
            server.shared_store(),
            Duration::from_secs(config.tiering.usage_flush_secs),
        ))
    });

    let served = match mode {
        cli::Mode::Serve => serve_socket(&config, server, shutdown.token()).await,
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
        cli::Mode::EncryptLog => unreachable!(
            "encrypt-log returns in run() before any store setup; \
             reaching serve_with_store means that guard was removed"
        ),
        cli::Mode::Delete(_) => unreachable!(
            "delete returns in run() before any store setup; \
             reaching serve_with_store means that guard was removed"
        ),
        cli::Mode::Stdio => serve_stdio(server, shutdown.token()).await,
    };
    stop_background_tasks(&shutdown, usage_flush, maintenance, compactor).await;
    match (served, shutdown.failure()) {
        (Err(error), _) | (Ok(()), Some(error)) => Err(error),
        (Ok(()), None) => Ok(()),
    }
}

async fn serve_stdio(server: MemoryServer, shutdown: CancellationToken) -> anyhow::Result<()> {
    // rmcp stdio serve. Confirm against your pinned rmcp version.
    use rmcp::ServiceExt;
    let running = server.serve(rmcp::transport::stdio()).await?;
    let end_session = running.cancellation_token();
    let watcher = tokio::spawn(async move {
        shutdown.cancelled().await;
        end_session.cancel();
    });
    let finished = running.waiting().await;
    watcher.abort();
    finished?;
    Ok(())
}

/// Flushes the usage counts once the transports have drained, then stops the
/// maintenance loop, whose server handle is the last one that keeps
/// the store alive, then asks the compactor to stop and waits for the compaction it is running.
/// Both are bounded: a queued or interrupted compaction is redone by the next
/// start.
async fn stop_background_tasks(
    shutdown: &Shutdown,
    usage_flush: Option<UsageFlush>,
    maintenance: JoinHandle<()>,
    mut compactor: CompactorTask,
) {
    // The transports have drained by now, and the store is still open.
    if let Some(usage_flush) = usage_flush {
        usage_flush.stop(USAGE_FLUSH_DEADLINE).await;
    }
    shutdown.token().cancel();
    match tokio::time::timeout(MAINTENANCE_STOP_DEADLINE, maintenance).await {
        Ok(joined) => log_task_failure("maintenance loop", &joined),
        Err(_) => tracing::warn!("the maintenance loop is still running; leaving it to the exit"),
    }
    // Segments still queued stay on disk as WAL files and are compacted by
    // the next start, so only the compaction already running is waited for.
    compactor.request_stop();
    match tokio::time::timeout(COMPACTOR_DRAIN_DEADLINE, &mut compactor.join).await {
        Ok(Ok(())) => tracing::info!("compactor stopped"),
        Ok(joined) => log_task_failure("compactor", &joined),
        Err(_) => {
            compactor.join.abort();
            tracing::warn!("a compaction was still running at exit; the next start redoes it");
        }
    }
}

/// The message to log when a joined background task ended in a panic or was
/// cancelled, or `None` when it finished cleanly.
fn task_failure_message(task: &str, joined: &Result<(), tokio::task::JoinError>) -> Option<String> {
    joined
        .as_ref()
        .err()
        .map(|error| format!("the {task} ended abnormally: {error}"))
}

fn log_task_failure(task: &str, joined: &Result<(), tokio::task::JoinError>) {
    if let Some(message) = task_failure_message(task, joined) {
        tracing::error!("{message}");
    }
}

/// The socket daemon: resolve the listener (activated by launchd, or bound
/// here), serve it, and stop on SIGTERM or SIGINT through the ordered
/// shutdown in `transport::shutdown`.
async fn serve_socket(
    config: &Config,
    server: MemoryServer,
    cancel: CancellationToken,
) -> anyhow::Result<()> {
    let socket_path = resolve_config_path("socket_path", &config.socket_path)?;
    if let (Some(parent), Some(state_root)) = (
        socket_path.parent(),
        state_dir::state_root_from_home(std::env::var_os("HOME")),
    ) {
        state_dir::secure_dirs(&[parent], &state_root, SOCKET_RISK);
    }
    let (signals, source) =
        transport::shutdown::watch_signals_before(|| transport::activation::resolve(&socket_path))
            .await?;

    // Signals are watched on their own task so the accept loop owns the
    // main flow. Cancelling the token is all this does; the drain and the
    // unlink belong to the accept loop's shutdown path.
    let signal_cancel = cancel.clone();
    tokio::spawn(async move {
        match signals {
            Ok(signals) => {
                let trigger = signals.wait().await;
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

/// The consolidation job on its schedule, on the SAME `Graph` every request
/// handler uses, not a second connection: a second connection only traded an
/// in-process wait for an opaque SQLite lock timeout. See ADR-0002
/// Amendment 4. The job's maintenance step keeps the tick's own steps, so GC,
/// the cluster refresh and resynthesis now run inside its snapshot and probe
/// gate. A run left unfinished by a crash resumes at once instead of waiting
/// out an interval.
fn spawn_gc(config: &Config, job: Arc<ConsolidationJob>, run_now: bool) -> JoinHandle<()> {
    let interval = config.gc_interval();
    let stopping = job.stopping();
    tokio::spawn(async move {
        if run_now {
            job.tick().await;
        }
        let mut tick = tokio::time::interval(interval);
        tick.tick().await; // drop the immediate first tick
        loop {
            tokio::select! {
                _ = stopping.cancelled() => return,
                _ = tick.tick() => {}
            }
            job.tick().await;
        }
    })
}

/// The tick's steps without the consolidation envelope, for tests of a single
/// step. Replay the log, sweep, repair, refresh clusters, then resynthesize stale
/// entities unless `skip_resynthesis` is set. The
/// tick stops after any step that left the daemon stopping, as a poisoned log
/// does: the later steps would only be refused, and resynthesis would spend a
/// model call on a write that cannot land.
#[cfg(test)]
async fn maintenance_tick(
    server: &MemoryServer,
    policy: &liam_store::RetentionPolicy,
    max_resynth_per_tick: usize,
    synthesis_timeout_secs: u64,
    skip_resynthesis: bool,
) {
    let resynthesis = (!skip_resynthesis).then_some(Resynthesis {
        max_per_tick: max_resynth_per_tick,
        synthesis_timeout_secs,
    });
    match server.store().await {
        Ok(store) => {
            maintenance_steps(server, &store, policy, resynthesis, None).await;
        }
        Err(error) => tracing::warn!(%error, "the maintenance tick found no store"),
    }
}

/// How much resynthesis a tick does.
#[derive(Clone, Copy)]
pub(crate) struct Resynthesis {
    max_per_tick: usize,
    synthesis_timeout_secs: u64,
}

impl Resynthesis {
    pub(crate) fn from_config(config: &Config) -> Self {
        Self {
            max_per_tick: config.gc.max_resynth_per_tick,
            synthesis_timeout_secs: config.ask_timeout_secs,
        }
    }
}

/// The tick's steps. With a `write_source`, every log write they make is
/// stamped with it, so a consolidation run that calls them can have those
/// writes cancelled by a rollback; without one they keep their own sources.
/// Cluster assignments are derived rows kept out of the log, which a restore
/// puts back with the rest of the database.
pub(crate) async fn maintenance_steps(
    server: &MemoryServer,
    store: &DefaultGraph,
    policy: &liam_store::RetentionPolicy,
    resynthesis: Option<Resynthesis>,
    write_source: Option<&str>,
) -> bool {
    let stopping = server.shutdown_token();
    if !standard_steps(store, policy, write_source, &stopping).await {
        return false;
    }
    if let Some(resynthesis) = resynthesis {
        resynthesize_stale(server, store, &resynthesis, write_source).await;
    }
    !stopping.is_cancelled()
}

/// Writing a page supersedes the entity row, and the entity's own mentions stay
/// on the closed row until repair moves them. Running repair here, before the
/// fingerprint is recorded, lets the next tick see the mentions the page was
/// built from instead of finding the entity stale again.
async fn carry_mentions_to_the_new_page(store: &DefaultGraph, write_source: Option<&str>) {
    let repaired = match write_source {
        Some(source) => store.repair_superseded_mentions_from(source).await,
        None => store.repair_superseded_mentions().await,
    };
    if let Err(e) = repaired {
        tracing::warn!(error = %e, "carrying the mentions to the new entity page failed");
    }
}

/// Records the mention count a fresh page was built from. It reads the store's
/// clock again because the page write superseded the entity at a later instant
/// than the one the tick began; resolving the entity at the earlier instant
/// would find the closed row and the entity would look stale again on every
/// following tick.
async fn record_fingerprint(
    store: &DefaultGraph,
    subject: &str,
    scope: Option<&str>,
) -> liam_store::Result<()> {
    let now = store.now();
    let fingerprint = store
        .entity_mentions_fingerprint(subject, scope, now)
        .await?;
    store
        .mark_entity_synthesized(subject, scope, fingerprint, now)
        .await
}

/// Up to `limit` entities whose page is due: the stale ones, oldest first, then
/// entities whose page an earlier version wrote, which `timeline` does not show
/// until a new page replaces it. `marker` is the attribute that tells the new
/// pages from the old. A failed lookup of the old pages is logged and leaves the
/// stale ones due.
async fn due_for_rebuild(
    store: &DefaultGraph,
    limit: usize,
    now: liam_store::Millis,
    marker: &str,
) -> liam_store::Result<Vec<(liam_store::NodeId, String, Option<String>)>> {
    let mut due = store.stale_entities(limit, now).await?;
    let room = limit.saturating_sub(due.len());
    if room > 0 {
        let legacy = match store
            .pages_without_marker(marker, room + due.len(), now)
            .await
        {
            Ok(legacy) => legacy,
            Err(e) => {
                tracing::warn!(error = %e, "lookup of the pages an earlier version wrote failed");
                Vec::new()
            }
        };
        let already: Vec<liam_store::NodeId> = due.iter().map(|(id, _, _)| id.clone()).collect();
        due.extend(
            legacy
                .into_iter()
                .filter(|(id, _, _)| !already.contains(id))
                .take(room),
        );
    }
    Ok(due)
}

/// Resynthesizes up to `max_resynth_per_tick` stale entities, oldest first.
/// Entities are processed one at a time, never concurrently, so a tick's page
/// builds do not crowd out the embedder calls of concurrent user requests.
async fn resynthesize_stale(
    server: &MemoryServer,
    store: &DefaultGraph,
    resynthesis: &Resynthesis,
    write_source: Option<&str>,
) {
    let now = store.now();
    forget_retries_with_new_mentions(server, store, now).await;
    let paused = server.page_retries().paused(now);
    let stale = match due_for_rebuild(
        store,
        resynthesis.max_per_tick + paused.len(),
        now,
        entity_page::PAGE_MARKER_KEY,
    )
    .await
    {
        Ok(stale) => stale,
        Err(e) => {
            tracing::warn!(error = %e, "stale entity lookup failed");
            return;
        }
    };
    let stale = stale
        .into_iter()
        .filter(|(id, _, _)| !paused.contains(id))
        .take(resynthesis.max_per_tick);

    for (node_id, subject, scope) in stale {
        let deadline = tokio::time::Instant::now()
            + std::time::Duration::from_secs(resynthesis.synthesis_timeout_secs.max(1));
        match server
            .resynthesize_entity_as(store, node_id.clone(), deadline, write_source)
            .await
        {
            Ok(()) => {
                server.page_retries().clear(&node_id);
                carry_mentions_to_the_new_page(store, write_source).await;
                match record_fingerprint(store, &subject, scope.as_deref()).await {
                    Ok(()) => tracing::info!(subject, "entity resynthesized"),
                    Err(e) => {
                        tracing::warn!(subject, error = %e, "failed to mark entity synthesized")
                    }
                }
            }
            Err(e) => {
                note_failed_build(server, store, &node_id, &subject, scope.as_deref(), &e, now)
                    .await;
            }
        }
    }
}

/// Drops the failures of an entity that has had a mention added or removed
/// since, so new input gets a fresh set of attempts. An entity whose mentions
/// cannot be read keeps its failures.
async fn forget_retries_with_new_mentions(
    server: &MemoryServer,
    store: &DefaultGraph,
    now: liam_store::Millis,
) {
    for (id, subject, scope, seen) in server.page_retries().recorded() {
        let current = store
            .entity_mentions_fingerprint(&subject, scope.as_deref(), now)
            .await;
        if current.as_ref().is_ok_and(|current| *current != seen) {
            server.page_retries().clear(&id);
        }
    }
}

/// Logs a failed page build and remembers it: a warning the first time and
/// again when the entity is given up on for now, a debug line between.
async fn note_failed_build(
    server: &MemoryServer,
    store: &DefaultGraph,
    id: &liam_store::NodeId,
    subject: &str,
    scope: Option<&str>,
    error: &str,
    now: liam_store::Millis,
) {
    let failure = match store.entity_mentions_fingerprint(subject, scope, now).await {
        Ok(seen) => server
            .page_retries()
            .record_failure(id, subject, scope, seen, now),
        // Nothing to compare a later mention set with, so it is not recorded.
        Err(_) => page_retry::Failure::First,
    };
    match failure {
        page_retry::Failure::First => {
            tracing::warn!(subject, %error, "entity resynthesis failed");
        }
        page_retry::Failure::Again => {
            tracing::debug!(subject, %error, "entity resynthesis failed again");
        }
        page_retry::Failure::GivenUp => tracing::warn!(
            subject, %error,
            "entity resynthesis failed again; giving up on it until it changes or the pause ends"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use liam_daemon::consolidation::maintenance::refresh_clusters;
    use liam_store::{
        relation, FixedClock, GraphConfig, Millis, NewEdge, NewNode, RetentionPolicy,
    };

    /// A compactor with nothing queued whose sender is still alive, as at a
    /// real shutdown: its channel never closes on its own.
    fn idle_compactor() -> (liam_log::compactor::ChannelSink, CompactorTask) {
        let dir = tempfile::tempdir().unwrap();
        let (sink, receiver) = liam_log::compactor::ChannelSink::channel();
        let compactor = liam_log::compactor::Compactor::local(dir.path()).unwrap();
        (sink, compactor.spawn_stoppable(receiver))
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_does_not_wait_out_the_compactor_deadline_when_nothing_is_running() {
        // Arrange
        let (_sink, compactor) = idle_compactor();
        let maintenance = tokio::spawn(async {});
        let started = tokio::time::Instant::now();

        // Act
        stop_background_tasks(&Shutdown::default(), None, maintenance, compactor).await;

        // Assert: the paused clock only advances when every task is idle, so a
        // wait for the deadline would show here.
        assert!(started.elapsed() < COMPACTOR_DRAIN_DEADLINE);
    }

    #[tokio::test]
    async fn a_panicked_task_yields_an_error_message_naming_it() {
        // Arrange
        let joined = tokio::spawn(async { panic!("boom") }).await;

        // Act
        let message = task_failure_message("maintenance loop", &joined);

        // Assert
        let message = message.expect("a panic must be reported");
        assert!(message.contains("maintenance loop"), "{message}");
    }

    #[tokio::test]
    async fn a_cancelled_task_yields_an_error_message_naming_it() {
        // Arrange
        let handle = tokio::spawn(std::future::pending::<()>());
        handle.abort();
        let joined = handle.await;

        // Act
        let message = task_failure_message("maintenance loop", &joined);

        // Assert
        let message = message.expect("a cancellation must be reported");
        assert!(message.contains("maintenance loop"), "{message}");
    }

    #[tokio::test]
    async fn a_task_that_finished_cleanly_yields_no_message() {
        // Arrange
        let joined = tokio::spawn(async {}).await;

        // Act
        let message = task_failure_message("maintenance loop", &joined);

        // Assert
        assert_eq!(message, None);
    }

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
    /// never touch a real embedder or LLM: enough for
    /// `maintenance_tick` to reach the store it holds, nothing else.
    fn test_server(store: Arc<DefaultGraph>) -> MemoryServer {
        MemoryServer::new(store, Arc::new(liam_model::MockEmbedder::new(8)), 30)
    }

    #[tokio::test]
    async fn a_mismatched_or_corrupt_fingerprint_stops_startup_before_the_store_opens() {
        for contents in [
            r#"{"version":1,"provider":"mock","model":"other-model","dims":768,"prompt_scheme":0}"#,
            "{not json",
        ] {
            // Arrange
            let dir = tempfile::tempdir().unwrap();
            let database = dir.path().join("liam.db");
            let config = Config {
                database_path: database.to_string_lossy().into_owned(),
                ..Config::default()
            };
            std::fs::write(
                liam_daemon::embedder_fingerprint::fingerprint_path_for(&database),
                contents,
            )
            .unwrap();

            // Act
            let result = serve_until(cli::Mode::Stdio, config, Shutdown::default()).await;

            // Assert
            let error = result.expect_err("startup must fail").to_string();
            assert!(
                error.contains("embedder fingerprint"),
                "wrong error for {contents}: {error}"
            );
            assert!(!database.exists(), "the store was opened for {contents}");
        }
    }

    #[tokio::test]
    async fn a_start_that_fails_to_open_the_store_records_no_fingerprint() {
        // Arrange: a log directory that is a file, which the store open refuses
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("liam.db");
        let not_a_log = dir.path().join("not-a-log");
        std::fs::write(&not_a_log, b"data").unwrap();
        let mut config = Config {
            database_path: database.to_string_lossy().into_owned(),
            ..Config::default()
        };
        config.log.dir = Some(not_a_log.to_string_lossy().into_owned());

        // Act
        let result = serve_until(cli::Mode::Stdio, config, Shutdown::default()).await;

        // Assert
        assert!(result.is_err());
        assert!(!liam_daemon::embedder_fingerprint::fingerprint_path_for(&database).exists());
    }

    #[tokio::test]
    async fn the_tick_refreshes_clusters_after_the_sweep() {
        let (store, _clock) = seeded_pair(Millis(1000)).await;
        // Nothing is old enough to sweep under this policy, which isolates the
        // refresh half of `maintenance_tick` from the sweep half.
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let server = test_server(Arc::clone(&store));

        maintenance_tick(&server, &policy, 0, 30, true).await;

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

        maintenance_tick(&server, &policy, 0, 30, true).await;

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
        maintenance_tick(&server, &policy, 0, 30, true).await;

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
        maintenance_tick(&server, &policy, 0, 30, true).await;

        // Then repair is a no-op and logs zero repairs
        let log = String::from_utf8(captured.0.lock().expect("captured logs lock").clone())
            .expect("log output is utf8");
        assert!(
            log.contains("repaired=0"),
            "expected a zero repaired count to be logged: {log}"
        );
    }

    /// As `test_server`, over a caller-supplied embedder, for the page
    /// rebuild tests below that need one which is slow, gated or failing.
    fn test_server_with_embedder(
        store: Arc<DefaultGraph>,
        embedder: Arc<dyn liam_model::Embedder>,
    ) -> MemoryServer {
        MemoryServer::new(store, embedder, 30)
    }

    /// An embedder that moves the store's clock when it is called, so the
    /// instant a page write supersedes at is later than the one
    /// `resynthesize_stale` captured before its loop began. A fresh-fingerprint
    /// regression test needs that gap, and a fake clock supplies it without
    /// waiting.
    struct ClockAdvancingEmbedder {
        inner: liam_model::MockEmbedder,
        clock: Arc<FixedClock>,
        to: Millis,
    }

    #[async_trait::async_trait]
    impl liam_model::Embedder for ClockAdvancingEmbedder {
        fn dims(&self) -> usize {
            self.inner.dims()
        }
        async fn embed(
            &self,
            text: &str,
            kind: liam_model::TextKind,
        ) -> liam_model::Result<Vec<f32>> {
            self.clock.set(self.to);
            self.inner.embed(text, kind).await
        }
    }

    /// An embedder whose every call fails, so a page build driven by it must
    /// leave the entity stale rather than succeed.
    struct FailingEmbedder;

    #[async_trait::async_trait]
    impl liam_model::Embedder for FailingEmbedder {
        fn dims(&self) -> usize {
            8
        }
        async fn embed(
            &self,
            _text: &str,
            _kind: liam_model::TextKind,
        ) -> liam_model::Result<Vec<f32>> {
            Err(liam_model::ModelError::embed_failed("embed", "refused"))
        }
    }

    /// Blocks inside `embed()` on a `Notify` the test controls, tracking the
    /// PEAK number of concurrent calls it has ever seen: a final count of 0
    /// proves nothing, since every call reaches 0 eventually whether or not two
    /// of them were ever in flight together.
    struct GatedEmbedder {
        release: Arc<tokio::sync::Notify>,
        in_flight: std::sync::atomic::AtomicUsize,
        peak: std::sync::atomic::AtomicUsize,
    }

    impl GatedEmbedder {
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
    impl liam_model::Embedder for GatedEmbedder {
        fn dims(&self) -> usize {
            8
        }
        async fn embed(
            &self,
            _text: &str,
            _kind: liam_model::TextKind,
        ) -> liam_model::Result<Vec<f32>> {
            let now = self
                .in_flight
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;
            self.peak
                .fetch_max(now, std::sync::atomic::Ordering::SeqCst);
            self.release.notified().await;
            self.in_flight
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            Ok(vec![1.0; 8])
        }
    }

    #[tokio::test]
    async fn the_test_embedders_report_the_width_of_the_vectors_they_return() {
        // Arrange
        let release = Arc::new(tokio::sync::Notify::new());
        release.notify_one();
        let gated = GatedEmbedder::new(release);
        let clock = ClockAdvancingEmbedder {
            inner: liam_model::MockEmbedder::new(8),
            clock: Arc::new(FixedClock::new(Millis(1))),
            to: Millis(2),
        };

        // Act
        let gated_vector = liam_model::Embedder::embed(&gated, "x", liam_model::TextKind::Document)
            .await
            .unwrap();
        let clock_vector = liam_model::Embedder::embed(&clock, "x", liam_model::TextKind::Document)
            .await
            .unwrap();

        // Assert
        assert_eq!(gated_vector.len(), liam_model::Embedder::dims(&gated));
        assert_eq!(clock_vector.len(), liam_model::Embedder::dims(&clock));
        assert_eq!(liam_model::Embedder::dims(&FailingEmbedder), 8);
    }

    /// Cooperatively yields until `condition()` is true, bounded so a real bug
    /// panics the test instead of hanging the suite.
    async fn wait_until(mut condition: impl FnMut() -> bool) {
        for _ in 0..10_000 {
            if condition() {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("condition never became true; the two tasks likely deadlocked");
    }

    /// Inserts `entity` with one mention that has no vector, so a page build for
    /// it makes a single embed call, for the mention.
    async fn entity_with_a_mention(store: &DefaultGraph, name: &str) -> liam_store::NodeId {
        let entity = store.insert(NewNode::entity("person", name)).await.unwrap();
        let fact = store
            .insert(NewNode::now(
                "fact",
                "some fact",
                format!("{name} likes tea"),
            ))
            .await
            .unwrap();
        store
            .relate(&entity, &fact, relation::MENTIONS)
            .await
            .unwrap();
        entity
    }

    #[tokio::test]
    async fn a_failed_lookup_of_old_pages_keeps_the_stale_entities_due_and_is_logged() {
        // Given a stale entity, a page an earlier version wrote, and a marker
        // the store cannot read as a JSON path, so the lookup of old pages fails
        let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        entity_with_a_mention(&store, "Stale One").await;
        let mut legacy = NewNode::entity("person", "Old Page");
        legacy.content = "a profile a model wrote".to_string();
        store.insert(legacy).await.unwrap();
        let captured = CapturedLogs::default();
        let _guard = install_captured_logs(captured.clone());

        // When the due entities are listed
        let due = due_for_rebuild(&store, 5, Millis::now(), "[")
            .await
            .expect("the stale list survives the failed lookup");

        // Then the stale entity is still due, and the failure was logged
        let subjects: Vec<&str> = due.iter().map(|(_, subject, _)| subject.as_str()).collect();
        assert_eq!(subjects, ["stale one"]);
        let log = String::from_utf8(captured.0.lock().expect("captured logs lock").clone())
            .expect("log output is utf8");
        assert!(
            log.contains("lookup of the pages an earlier version wrote failed"),
            "{log}"
        );
    }

    #[tokio::test]
    async fn the_tick_rebuilds_at_most_the_configured_cap_oldest_stale_first() {
        // Given 2 stale (never-built) entities, each with a mention, and a cap of 1
        let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        let capped = entity_with_a_mention(&store, "First Stale").await;
        let leftover = entity_with_a_mention(&store, "Second Stale").await;
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let server = test_server(Arc::clone(&store));

        // When the tick fires with max_resynth_per_tick = 1
        maintenance_tick(&server, &policy, 1, 30, false).await;

        // Then exactly the cap's worth were rebuilt: the capped entity's old
        // id no longer resolves, superseded by the row that holds its page
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
    async fn a_successful_page_build_clears_the_entitys_stale_state() {
        // Given a single stale (never-built) entity with a mention
        let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        let old_id = entity_with_a_mention(&store, "Solo Stale").await;
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let server = test_server(Arc::clone(&store));

        // When the tick fires
        maintenance_tick(&server, &policy, 5, 30, false).await;

        // Then the entity's id changed, proving a real page was written
        assert!(
            store.resolve_handle(old_id.as_str()).await.is_err(),
            "a successful build must supersede the old id"
        );
        // And a follow-up check at the same as_of no longer reports it stale,
        // proving the (subject, scope) staleness state was updated even
        // though the NodeId it is keyed against changed underneath it
        let stale = store.stale_entities(10, Millis::now()).await.unwrap();
        assert!(
            stale.is_empty(),
            "a freshly built entity must not still be reported stale: {stale:?}"
        );
    }

    #[tokio::test]
    async fn an_entity_with_no_mention_is_marked_built_and_not_picked_again() {
        // Given a stale entity nothing mentions yet
        let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        let lonely = store
            .insert(NewNode::entity("person", "Lonely"))
            .await
            .unwrap();
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let server = test_server(Arc::clone(&store));

        // When the tick fires
        maintenance_tick(&server, &policy, 5, 30, false).await;

        // Then it gets no page, keeps its row, and is not reported stale again
        assert!(store.resolve_handle(lonely.as_str()).await.is_ok());
        let stale = store.stale_entities(10, Millis::now()).await.unwrap();
        assert!(stale.is_empty(), "{stale:?}");
    }

    #[tokio::test]
    async fn a_new_mention_makes_a_built_page_stale_and_the_next_tick_rebuilds_it() {
        // Given an entity whose page the first tick built
        let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        entity_with_a_mention(&store, "Grows").await;
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let server = test_server(Arc::clone(&store));
        maintenance_tick(&server, &policy, 5, 30, false).await;
        assert!(store
            .stale_entities(10, Millis::now())
            .await
            .unwrap()
            .is_empty());
        let page = store
            .live_by_subject("grows", None, Millis::now())
            .await
            .unwrap()
            .expect("the first tick built a page");
        assert!(!page.retrieved.as_text().contains("moved to Porto"));

        // When a new fact mentions the entity
        let fact = store
            .insert(
                NewNode::now("fact", "news", "Grows moved to Porto")
                    .with_embedding(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
            .await
            .unwrap();
        store
            .relate(&page.id, &fact, relation::MENTIONS)
            .await
            .unwrap();

        // Then the entity is reported stale, and the next tick rebuilds the page
        let stale = store.stale_entities(10, Millis::now()).await.unwrap();
        assert_eq!(stale.len(), 1, "{stale:?}");
        maintenance_tick(&server, &policy, 5, 30, false).await;
        let rebuilt = store
            .live_by_subject("grows", None, Millis::now())
            .await
            .unwrap()
            .expect("the page is still there");
        assert!(rebuilt.retrieved.as_text().contains("moved to Porto"));
        assert!(store
            .stale_entities(10, Millis::now())
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn a_rebuilt_entity_with_a_live_mention_is_not_reported_stale_again_next_tick() {
        // Given an entity with one live mentions edge of its own, so the old
        // and new node fingerprint differently (1 mention vs. 0, since a
        // supersede never carries a node's own edges over on its own), and an
        // embedder that moves the store's clock forward, so the page write
        // supersedes the entity later than the `now` `resynthesize_stale`
        // captured before its loop began
        let clock = Arc::new(FixedClock::new(Millis(5_000)));
        let store = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .expect("open in-memory store");
        let entity = entity_with_a_mention(&store, "Mentioned Stale").await;
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let server = test_server_with_embedder(
            Arc::clone(&store),
            Arc::new(ClockAdvancingEmbedder {
                inner: liam_model::MockEmbedder::new(8),
                clock,
                to: Millis(6_000),
            }),
        );

        // When the tick fires and rebuilds it
        maintenance_tick(&server, &policy, 5, 30, false).await;
        assert!(
            store.resolve_handle(entity.as_str()).await.is_err(),
            "a successful build must supersede the old id"
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
    async fn a_failed_page_build_leaves_the_entity_stale_and_logs_it_without_a_panic() {
        // Given a stale entity whose page build will fail
        let failing_store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        let failing_entity = entity_with_a_mention(&failing_store, "Fails Guy").await;
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let failing_store = Arc::new(failing_store);
        let failing_server =
            test_server_with_embedder(Arc::clone(&failing_store), Arc::new(FailingEmbedder));
        let captured = CapturedLogs::default();
        let _guard = install_captured_logs(captured.clone());

        // When its tick fires
        maintenance_tick(&failing_server, &policy, 5, 30, false).await;

        // Then the tick completed without panicking (reaching here proves
        // that), the entity remains live under its old id and still stale
        assert!(
            failing_store
                .resolve_handle(failing_entity.as_str())
                .await
                .is_ok(),
            "a failed build must leave the entity's old id live"
        );
        let stale = failing_store
            .stale_entities(10, Millis::now())
            .await
            .unwrap();
        assert!(
            stale.iter().any(|(id, _, _)| id == &failing_entity),
            "a failed build must leave the entity reported stale"
        );
        // And the failure was logged
        let log = String::from_utf8(captured.0.lock().expect("captured logs lock").clone())
            .expect("log output is utf8");
        assert!(
            log.contains("entity resynthesis failed"),
            "expected the failure to be logged: {log}"
        );

        // And a separate entity's own tick, with a working embedder, still
        // succeeds independently: the failure path above never touched
        // shared state a second entity's build would also need
        let working_store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        let working_entity = entity_with_a_mention(&working_store, "Works Guy").await;
        let working_store = Arc::new(working_store);
        let working_server = test_server(Arc::clone(&working_store));

        maintenance_tick(&working_server, &policy, 5, 30, false).await;

        assert!(
            working_store
                .resolve_handle(working_entity.as_str())
                .await
                .is_err(),
            "the other entity's own build must still succeed independently"
        );
    }

    #[tokio::test]
    async fn a_build_that_keeps_failing_is_retried_with_backoff_then_given_up_until_a_mention_arrives(
    ) {
        const MINUTE: i64 = 60_000;
        // Given a stale entity whose page build always fails
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let store = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
        let entity = entity_with_a_mention(&store, "Broken Guy").await;
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let server = test_server_with_embedder(Arc::clone(&store), Arc::new(FailingEmbedder));
        let captured = CapturedLogs::default();
        let _guard = install_captured_logs(captured.clone());
        let warnings = || {
            let log = captured.0.lock().expect("captured logs lock").clone();
            String::from_utf8(log)
                .unwrap()
                .matches("entity resynthesis failed")
                .count()
        };
        let tick_at = |minutes: i64| {
            clock.set(Millis(1000 + minutes * MINUTE));
            maintenance_tick(&server, &policy, 5, 30, false)
        };

        // When it fails the first time, then a tick comes before the delay is over
        tick_at(0).await;
        tick_at(1).await;

        // Then one warning, and the entity waits
        assert_eq!(warnings(), 1);
        assert_eq!(
            server.page_retries().paused(store.now()),
            std::slice::from_ref(&entity)
        );

        // When the delays pass, the later failures stay quiet until the last
        tick_at(11).await;
        assert_eq!(warnings(), 1);
        tick_at(40).await;

        // Then the entity is given up on, with a second warning, and left alone
        assert_eq!(warnings(), 2);
        tick_at(100).await;
        assert_eq!(warnings(), 2);

        // When a new mention arrives
        let fact = store
            .insert(NewNode::now("fact", "more", "Broken Guy likes coffee"))
            .await
            .unwrap();
        store
            .relate(&entity, &fact, relation::MENTIONS)
            .await
            .unwrap();
        tick_at(101).await;

        // Then the count starts again with a fresh first warning
        assert_eq!(warnings(), 3);
    }

    #[tokio::test]
    async fn resynthesize_stale_processes_entities_one_at_a_time_not_concurrently() {
        // Given 2 stale entities, a cap covering both, and an embedder gated so
        // a call blocks until this test decides to release it
        let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        entity_with_a_mention(&store, "Gated One").await;
        entity_with_a_mention(&store, "Gated Two").await;
        let policy = RetentionPolicy::keep("nonexistent-kind", Millis(1));
        let store = Arc::new(store);
        let release = Arc::new(tokio::sync::Notify::new());
        let embedder = Arc::new(GatedEmbedder::new(release.clone()));
        let server = test_server_with_embedder(Arc::clone(&store), embedder.clone());

        // When the tick runs, spawned since the gated embedder blocks until released
        let handle = tokio::spawn(async move {
            maintenance_tick(&server, &policy, 2, 30, false).await;
        });

        // Then only one entity's page embed is ever in flight: release the
        // first, wait for the second to start, release it too
        wait_until(|| embedder.in_flight() == 1).await;
        release.notify_one();
        wait_until(|| embedder.in_flight() == 1).await;
        release.notify_one();
        handle.await.expect("maintenance tick task panicked");

        // The PEAK proves the two embeds were never in flight together, not
        // merely that the count returned to normal afterward
        assert_eq!(
            embedder.peak(),
            1,
            "resynthesize_stale must process entities sequentially, \
             never two page embeds in flight together"
        );
    }

    #[tokio::test]
    async fn skip_resynthesis_skips_only_resynthesis_not_the_rest_of_the_tick() {
        // Given a stale entity, and a superseded fact so repair has work to log
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let store = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
        let stale_entity = entity_with_a_mention(&store, "Skipped Guy").await;
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
        let server = test_server(Arc::clone(&store));
        let captured = CapturedLogs::default();
        let _guard = install_captured_logs(captured.clone());

        // When a tick fires with resynthesis skipped
        maintenance_tick(&server, &policy, 5, 30, true).await;

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
        maintenance_tick(&server, &policy, 5, 30, false).await;

        // Then that later tick resynthesizes the entity normally
        assert!(
            store.resolve_handle(stale_entity.as_str()).await.is_err(),
            "a later, non-skipped tick must resynthesize the entity"
        );
    }

    // ---- the tick over a store with an event log ----

    mod with_a_log {
        use super::*;
        use crate::log_startup::test_support::{
            append_unapplied_node, database_in, log_dir_in, log_over, logged_events,
            standalone_log, FailingReader, ToggledWriter, DIMS,
        };
        use crate::shutdown::Shutdown;
        use liam_log::event::{LogPayload, TombstoneTable};
        use liam_log::reader::SequentialScanReader;
        use liam_store::NodeId;
        use liam_store::SharedLog;
        use std::sync::atomic::{AtomicBool, Ordering};

        const DAY: i64 = 86_400_000;

        /// The store over the standalone log, with no catch-up: whatever the
        /// log holds past the cursor stays unapplied until something replays it.
        async fn reopen(
            dir: &std::path::Path,
            clock: Arc<FixedClock>,
        ) -> (Arc<DefaultGraph>, MemoryServer) {
            store_over(dir, clock, standalone_log(&log_dir_in(dir))).await
        }

        async fn store_over(
            dir: &std::path::Path,
            clock: Arc<FixedClock>,
            log: SharedLog,
        ) -> (Arc<DefaultGraph>, MemoryServer) {
            let store = DefaultGraph::open_with_clock(
                database_in(dir).to_str().unwrap(),
                GraphConfig::new(DIMS),
                clock,
            )
            .await
            .unwrap()
            .with_log(log)
            .await
            .unwrap();
            let store = Arc::new(store);
            let server = test_server(Arc::clone(&store));
            (store, server)
        }

        async fn link_two_facts(store: &DefaultGraph) {
            let a = store.insert(NewNode::now("fact", "a", "x")).await.unwrap();
            let b = store.insert(NewNode::now("fact", "b", "x")).await.unwrap();
            store.link(NewEdge::new(&a, &b, "mentions")).await.unwrap();
        }

        /// A store whose log meets poison only once `armed` is set: the log
        /// first goes bad quietly, as after a failed fsync, so what the tick
        /// itself reports can be told from what already happened. Returns the
        /// shutdown the log's hook drives.
        async fn store_with_a_poisoned_log(
            dir: &std::path::Path,
        ) -> (Arc<DefaultGraph>, MemoryServer, Shutdown, Arc<AtomicBool>) {
            let log_dir = log_dir_in(dir);
            let shutdown = Shutdown::default();
            let armed = Arc::new(AtomicBool::new(false));
            let hook = shutdown.poison_hook();
            let armed_hook = Arc::clone(&armed);
            let (writer, bad) = ToggledWriter::open(&log_dir, false);
            let log = log_over(
                writer,
                Arc::new(SequentialScanReader::local(&log_dir).unwrap()),
                Some(Arc::new(move || {
                    if armed_hook.load(Ordering::SeqCst) {
                        hook();
                    }
                })),
            );
            let (store, server) =
                store_over(dir, Arc::new(FixedClock::new(Millis(1000))), log).await;
            link_two_facts(&store).await;
            bad.store(true, Ordering::SeqCst);
            let refused = store.insert(NewNode::now("fact", "c", "x")).await;
            assert!(refused.is_err(), "precondition: the write must be refused");
            let server = server.with_shutdown(shutdown.token());
            (store, server, shutdown, armed)
        }

        async fn is_live(store: &DefaultGraph, id: &str) -> bool {
            store
                .get(&NodeId::from_raw(id), Millis::now())
                .await
                .unwrap()
                .is_some()
        }

        fn nothing_to_sweep() -> RetentionPolicy {
            RetentionPolicy::keep("nonexistent-kind", Millis(1))
        }

        fn sweep_old_facts() -> RetentionPolicy {
            RetentionPolicy::keep("fact", Millis::days(30)).without_reclaim()
        }

        #[tokio::test]
        async fn the_tick_applies_an_event_the_log_holds_past_the_cursor() {
            // Arrange: a store that attached its log, then an event that a
            // double fault left in the log without a projection
            let dir = tempfile::tempdir().unwrap();
            let clock = Arc::new(FixedClock::new(Millis(2000)));
            drop(reopen(dir.path(), clock.clone()).await);
            append_unapplied_node(&log_dir_in(dir.path()), "leftover", "unapplied");
            let (store, server) = reopen(dir.path(), clock).await;
            assert!(!is_live(&store, "leftover").await, "precondition");

            // Act
            maintenance_tick(&server, &nothing_to_sweep(), 0, 30, true).await;

            // Assert
            assert!(
                is_live(&store, "leftover").await,
                "the tick must replay what the log holds past the cursor"
            );
        }

        #[tokio::test]
        async fn the_tick_catches_up_before_the_sweep() {
            // Arrange: the leftover is old enough to be swept. Replayed first,
            // it is swept with the rest; replayed after, it would outlive the sweep.
            let dir = tempfile::tempdir().unwrap();
            let clock = Arc::new(FixedClock::new(Millis(100 * DAY)));
            drop(reopen(dir.path(), clock.clone()).await);
            append_unapplied_node(&log_dir_in(dir.path()), "stale-leftover", "unapplied");
            let (store, server) = reopen(dir.path(), clock).await;

            // Act
            maintenance_tick(&server, &sweep_old_facts(), 0, 30, true).await;

            // Assert: gone from the store, and the log tombstones it
            assert!(!is_live(&store, "stale-leftover").await);
            let tombstoned = logged_events(&log_dir_in(dir.path()))
                .await
                .into_iter()
                .any(|event| match event.payload {
                    LogPayload::Tombstone(targets) => targets.iter().any(|target| {
                        target.table == TombstoneTable::Nodes && target.id == "stale-leftover"
                    }),
                    _ => false,
                });
            assert!(
                tombstoned,
                "the sweep must have tombstoned the replayed row"
            );
        }

        #[tokio::test]
        async fn a_poisoned_log_in_the_tick_stops_the_daemon_and_the_rest_of_the_tick() {
            // Arrange: the log is poisoned, and the tick is the first to say so
            let dir = tempfile::tempdir().unwrap();
            let (store, server, shutdown, armed) = store_with_a_poisoned_log(dir.path()).await;
            armed.store(true, Ordering::SeqCst);
            assert!(!shutdown.token().is_cancelled(), "precondition");
            crate::test_logs::capture();

            // Act
            maintenance_tick(&server, &nothing_to_sweep(), 0, 30, true).await;

            // Assert: the daemon is stopping with a failure, the sweep after
            // the poisoned replay was not even tried, and the cluster refresh
            // that follows it never ran
            assert!(shutdown.token().is_cancelled());
            let text = crate::test_logs::captured();
            assert!(!text.contains("gc failed"), "{text}");
            assert!(shutdown.failure().is_some());
            assert!(
                store.refresh_communities().await.unwrap(),
                "the tick must not have refreshed the clusters"
            );
        }

        #[tokio::test]
        async fn a_failing_replay_that_is_not_poison_leaves_the_rest_of_the_tick_running() {
            // Arrange: an event past the cursor and a reader that cannot read it
            let dir = tempfile::tempdir().unwrap();
            let clock = Arc::new(FixedClock::new(Millis(1000)));
            let (store, first_server) = reopen(dir.path(), clock.clone()).await;
            link_two_facts(&store).await;
            drop((store, first_server));
            append_unapplied_node(&log_dir_in(dir.path()), "leftover", "unapplied");
            let log_dir = log_dir_in(dir.path());
            let log = log_over(
                liam_daemon::event_log::open_writer(
                    &log_dir,
                    liam_log::wal::WalConfig {
                        segment_max_bytes: 1 << 20,
                        rotate_interval_secs: 3600,
                    },
                )
                .unwrap(),
                Arc::new(FailingReader),
                None,
            );
            let (store, server) = store_over(dir.path(), clock, log).await;
            assert!(store.catch_up().await.is_err(), "precondition");

            // Act
            maintenance_tick(&server, &nothing_to_sweep(), 0, 30, true).await;

            // Assert: the failure was only logged, and the refresh still ran
            assert!(!server.shutdown_token().is_cancelled());
            assert!(
                !store.refresh_communities().await.unwrap(),
                "the tick must have gone on to refresh the clusters"
            );
        }

        #[tokio::test]
        async fn a_tool_write_that_meets_a_poisoned_log_stops_the_daemon() {
            // Arrange
            let dir = tempfile::tempdir().unwrap();
            let (_store, server, shutdown, armed) = store_with_a_poisoned_log(dir.path()).await;
            armed.store(true, Ordering::SeqCst);
            let args = mcp::RememberArgs {
                kind: "fact".into(),
                label: "label".into(),
                content: "after the poison".into(),
                scope: None,
                subject: None,
                attributes: None,
                valid_from: None,
                confidence: None,
                episode: None,
            };

            // Act
            let reply = server
                .remember(rmcp::handler::server::wrapper::Parameters(args))
                .await;

            // Assert
            assert!(reply.starts_with("remember failed"), "{reply}");
            assert!(shutdown.token().is_cancelled());
            assert!(shutdown.failure().is_some());
        }

        const RUN_SOURCE: &str = "consolidation:9";

        /// The source of every logged record whose payload `pick` keeps.
        async fn sources_of(dir: &std::path::Path, pick: fn(&LogPayload) -> bool) -> Vec<String> {
            logged_events(&log_dir_in(dir))
                .await
                .into_iter()
                .filter(|event| pick(&event.payload))
                .map(|event| event.source)
                .collect()
        }

        #[tokio::test]
        async fn a_sweep_in_a_run_is_stamped_with_the_runs_source() {
            // Arrange
            let dir = tempfile::tempdir().unwrap();
            let clock = Arc::new(FixedClock::new(Millis(100 * DAY)));
            let (store, server) = reopen(dir.path(), clock).await;
            let old = NewNode::now("fact", "label", "old").with_valid_from(Millis(10 * DAY));
            store.insert(old).await.unwrap();

            // Act
            maintenance_steps(&server, &store, &sweep_old_facts(), None, Some(RUN_SOURCE)).await;

            // Assert
            let sources = sources_of(dir.path(), |p| matches!(p, LogPayload::Tombstone(_))).await;
            assert_eq!(sources, [RUN_SOURCE]);
        }

        #[tokio::test]
        async fn a_provenance_repair_in_a_run_is_stamped_with_the_runs_source() {
            // Arrange: a fact mentions an entity that was then superseded
            let dir = tempfile::tempdir().unwrap();
            let clock = Arc::new(FixedClock::new(Millis(1000)));
            let (store, server) = reopen(dir.path(), clock).await;
            let entity = store
                .upsert_by(NewNode::entity("person", "Kim"))
                .await
                .unwrap();
            let fact = store
                .insert(NewNode::now("fact", "a", "Kim"))
                .await
                .unwrap();
            store
                .relate(&fact, &entity, liam_store::relation::MENTIONS)
                .await
                .unwrap();
            let mut renewed = NewNode::entity("person", "Kim");
            renewed.content = "renewed".to_owned();
            store.upsert_by(renewed).await.unwrap();

            // Act
            maintenance_steps(&server, &store, &nothing_to_sweep(), None, Some(RUN_SOURCE)).await;

            // Assert
            let sources = sources_of(dir.path(), |p| matches!(p, LogPayload::EdgeWrite(_))).await;
            assert_eq!(sources.iter().filter(|s| *s == RUN_SOURCE).count(), 1);
        }

        #[tokio::test]
        async fn a_sweep_in_the_tick_is_logged_as_tombstones() {
            // Arrange: two facts written through the log, then aged past retention
            let dir = tempfile::tempdir().unwrap();
            let clock = Arc::new(FixedClock::new(Millis(100 * DAY)));
            let (store, server) = reopen(dir.path(), clock).await;
            let old = |content: &str| {
                NewNode::now("fact", "label", content).with_valid_from(Millis(10 * DAY))
            };
            let first = store.insert(old("first")).await.unwrap();
            let second = store.insert(old("second")).await.unwrap();

            // Act
            maintenance_tick(&server, &sweep_old_facts(), 0, 30, true).await;

            // Assert
            let tombstoned: Vec<String> = logged_events(&log_dir_in(dir.path()))
                .await
                .into_iter()
                .filter_map(|event| match event.payload {
                    LogPayload::Tombstone(targets) => Some(targets),
                    _ => None,
                })
                .flatten()
                .map(|target| target.id)
                .collect();
            assert!(
                tombstoned.contains(&first.as_str().to_string()),
                "{tombstoned:?}"
            );
            assert!(
                tombstoned.contains(&second.as_str().to_string()),
                "{tombstoned:?}"
            );
        }
    }
}
