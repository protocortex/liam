// SPDX-License-Identifier: Apache-2.0
//! The job over real database files and a real log: the tick's regression
//! tests now run through `consolidate()`, a rolled back run is undone while
//! the server keeps serving, and a crash's leftovers are reconciled.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use liam_daemon::config::Config;
use liam_daemon::consolidation::maintenance::refresh_clusters;
use liam_daemon::consolidation::manifest::{manifest_path_for, RunManifest, RunOutcome};
use liam_daemon::consolidation::probes::{run_probes, Probe, ProbeError, ProbeReport};
use liam_daemon::consolidation::run::{ProbeRunner, StoreReopener};
use liam_daemon::consolidation::settings::identity;
use liam_daemon::store_open::{open_database, open_store_on_log};
use liam_model::Llm;
use liam_store::{DefaultGraph, FixedClock, GraphConfig, Millis, NewEdge, NewNode};

use super::*;
use crate::log_startup::test_support::{
    config_in, database_in, embedder, log_dir_in, log_over, standalone_log, ToggledWriter, DIMS,
};

const POLL: std::time::Duration = std::time::Duration::from_millis(5);
const WAIT_POLLS: usize = 2_000;
const DAY: i64 = 86_400_000;
/// The step name a manifest records once a rollback is committed to.
const ROLLBACK_PENDING: &str = "rollback:pending";

struct Harness {
    /// Owns the directory when the harness made it.
    _dir: Option<tempfile::TempDir>,
    root: PathBuf,
    config: Config,
    log: SharedLog,
    shutdown: Shutdown,
}

impl Harness {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let log = standalone_log(&log_dir_in(&root));
        let mut harness = Self::over(&root, log);
        harness._dir = Some(dir);
        harness
    }

    /// A harness over files that already exist, with the given log.
    fn over(root: &Path, log: SharedLog) -> Self {
        let mut config = config_in(root);
        config.gc.episode_retention_days = 30;
        Self {
            _dir: None,
            root: root.to_path_buf(),
            config,
            log,
            shutdown: Shutdown::default(),
        }
    }

    async fn store_at(&self, now: Millis) -> (DefaultGraph, Arc<FixedClock>) {
        let clock = Arc::new(FixedClock::new(now));
        let store = DefaultGraph::open_with_clock(
            database_in(&self.root).to_str().unwrap(),
            GraphConfig::new(DIMS),
            clock.clone(),
        )
        .await
        .unwrap()
        .with_log(self.log.clone())
        .await
        .unwrap();
        (store, clock)
    }

    fn server(&self, store: DefaultGraph, llm: Arc<dyn Llm>) -> MemoryServer {
        self.server_arc(Arc::new(store), llm)
    }

    /// A server over a store the test also holds, which a rollback cannot take.
    fn server_arc(&self, store: Arc<DefaultGraph>, llm: Arc<dyn Llm>) -> MemoryServer {
        MemoryServer::new(
            store,
            embedder(),
            Arc::new(liam_model::IdentityReranker),
            llm,
            30,
            false,
            8192,
            1,
        )
        .with_shutdown(self.shutdown.token())
    }

    fn job(&self, server: &MemoryServer, probes: Arc<dyn ProbeRunner>) -> ConsolidationJob {
        self.job_reopening(server, probes, None)
    }

    fn job_reopening(
        &self,
        server: &MemoryServer,
        probes: Arc<dyn ProbeRunner>,
        reopener: Option<Arc<dyn StoreReopener>>,
    ) -> ConsolidationJob {
        ConsolidationJob::with_parts(
            JobSetup {
                config: &self.config,
                database_path: &database_in(&self.root),
                log_dir: &log_dir_in(&self.root),
                log: self.log.clone(),
                embedder: embedder(),
                server,
                shutdown: &self.shutdown,
            },
            probes,
            reopener,
        )
        .unwrap()
    }

    fn manifest_path(&self) -> std::path::PathBuf {
        manifest_path_for(&database_in(&self.root))
    }
}

fn honest() -> Arc<dyn ProbeRunner> {
    Arc::new(StoreProbes)
}

/// Scores the baseline honestly and every later run as a total loss.
struct RegressesAfterBaseline {
    calls: AtomicUsize,
}

#[async_trait]
impl ProbeRunner for RegressesAfterBaseline {
    async fn run(&self, store: &DefaultGraph, probes: &[Probe]) -> Result<ProbeReport, ProbeError> {
        let mut report = run_probes(store, probes).await?;
        if self.calls.fetch_add(1, Ordering::SeqCst) > 0 {
            report.scores.values_mut().for_each(|s| s.recall = 0.0);
        }
        Ok(report)
    }
}

fn regresses() -> Arc<dyn ProbeRunner> {
    Arc::new(RegressesAfterBaseline {
        calls: AtomicUsize::new(0),
    })
}

/// A node the retention rule leaves alone, so the run still has a probe.
async fn keep_one_fact(store: &DefaultGraph) {
    store
        .insert(NewNode::now("fact", "kept", "a fact that stays"))
        .await
        .unwrap();
}

async fn linked_pair(store: &DefaultGraph, kind: &str) {
    let a = store.insert(NewNode::now(kind, "a", "x")).await.unwrap();
    let b = store.insert(NewNode::now(kind, "b", "x")).await.unwrap();
    store.link(NewEdge::new(&a, &b, "mentions")).await.unwrap();
}

#[tokio::test]
async fn the_tick_refreshes_clusters_after_the_sweep_through_a_consolidation_run() {
    // Arrange: nothing is old enough to sweep, which isolates the refresh
    let harness = Harness::new();
    let (store, _clock) = harness.store_at(Millis(1000)).await;
    linked_pair(&store, "fact").await;
    let store = Arc::new(store);
    let server = harness.server_arc(store.clone(), Arc::new(liam_model::MockLlm));
    let job = harness.job(&server, honest());

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert!(
        !store.refresh_communities().await.unwrap(),
        "the run must have already refreshed; a second call finds nothing to do"
    );
}

#[tokio::test]
async fn the_refresh_runs_after_the_sweep_not_before_through_a_consolidation_run() {
    // Arrange: a refresh before the sweep would fingerprint an edge the sweep
    // is about to delete, and leave work behind
    let harness = Harness::new();
    let (store, clock) = harness.store_at(Millis(10 * DAY)).await;
    linked_pair(&store, "episode").await;
    keep_one_fact(&store).await;
    clock.set(Millis(100 * DAY));
    let store = Arc::new(store);
    let server = harness.server_arc(store.clone(), Arc::new(liam_model::MockLlm));
    let job = harness.job(&server, honest());

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert!(
        !store.refresh_communities().await.unwrap(),
        "a refresh that ran before the sweep would leave more work behind"
    );
}

#[tokio::test]
async fn refresh_clusters_reports_no_work_on_an_idle_store_through_a_consolidation_run() {
    // Arrange
    let harness = Harness::new();
    let (store, _clock) = harness.store_at(Millis(1000)).await;
    linked_pair(&store, "fact").await;
    let store = Arc::new(store);
    let server = harness.server_arc(store.clone(), Arc::new(liam_model::MockLlm));
    let job = harness.job(&server, honest());
    assert!(refresh_clusters(&store).await, "first call always has work");
    store
        .insert(NewNode::now("fact", "c", "a fact the run must refresh for"))
        .await
        .unwrap();

    // Act
    let outcome = job.tick().await;
    let did_work = refresh_clusters(&store).await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert!(!did_work, "the run left the store with nothing to refresh");
}

/// Counts the calls resynthesis makes and blocks each until released.
struct GatedLlm {
    release: Arc<tokio::sync::Notify>,
    in_flight: AtomicUsize,
    peak: AtomicUsize,
}

#[async_trait]
impl Llm for GatedLlm {
    async fn complete(&self, _s: &str, _p: &str) -> liam_model::Result<String> {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        self.release.notified().await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok("gated reply".to_string())
    }
}

/// Polls until `condition` holds, long enough for the run's file work.
async fn until(mut condition: impl FnMut() -> bool) {
    for _ in 0..WAIT_POLLS {
        if condition() {
            return;
        }
        tokio::time::sleep(POLL).await;
    }
    panic!("the condition never held");
}

#[tokio::test]
async fn resynthesis_processes_entities_one_at_a_time_through_a_consolidation_run() {
    // Arrange: two stale entities, a cap covering both, and a gated llm
    let mut harness = Harness::new();
    harness.config.gc.max_resynth_per_tick = 2;
    let (store, _clock) = harness.store_at(Millis(1000)).await;
    store
        .insert(NewNode::entity("person", "Gated One"))
        .await
        .unwrap();
    store
        .insert(NewNode::entity("person", "Gated Two"))
        .await
        .unwrap();
    let release = Arc::new(tokio::sync::Notify::new());
    let llm = Arc::new(GatedLlm {
        release: release.clone(),
        in_flight: AtomicUsize::new(0),
        peak: AtomicUsize::new(0),
    });
    let server = harness.server(store, llm.clone());
    let job = Arc::new(harness.job(&server, honest()));

    // Act: spawned, since the gated llm blocks until released
    let running = tokio::spawn({
        let job = job.clone();
        async move { job.tick().await }
    });
    until(|| llm.in_flight.load(Ordering::SeqCst) == 1).await;
    release.notify_one();
    until(|| llm.in_flight.load(Ordering::SeqCst) == 1).await;
    release.notify_one();
    let outcome = running.await.unwrap();

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert_eq!(llm.peak.load(Ordering::SeqCst), 1);
}

/// Counts calls to `complete`.
#[derive(Default)]
struct CountingLlm(AtomicUsize);

#[async_trait]
impl Llm for CountingLlm {
    async fn complete(&self, _s: &str, _p: &str) -> liam_model::Result<String> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok("reply".to_string())
    }
}

/// A harness whose server is autotuning, with a stale entity for resynthesis.
async fn autotuning_harness() -> (Harness, MemoryServer, Arc<CountingLlm>) {
    let mut harness = Harness::new();
    harness.config.llm.max_concurrent_generations = 0;
    let (store, _clock) = harness.store_at(Millis(1000)).await;
    store
        .insert(NewNode::entity("person", "Kim"))
        .await
        .unwrap();
    let llm = Arc::new(CountingLlm::default());
    let server = harness.server(store, llm.clone());
    (harness, server, llm)
}

#[tokio::test]
async fn resynthesis_is_skipped_for_the_run_made_at_start_of_an_autotuning_server_only() {
    // Arrange
    let (harness, server, llm) = autotuning_harness().await;
    let job = harness.job(&server, honest());

    // Act
    let run_now = job.start(true).await.unwrap();
    job.tick().await;
    let during_first = llm.0.load(Ordering::SeqCst);
    job.tick().await;

    // Assert
    assert!(run_now);
    assert_eq!(
        during_first, 0,
        "the first run leaves the permits to calibration"
    );
    assert!(
        llm.0.load(Ordering::SeqCst) > 0,
        "the next run resynthesizes"
    );
}

#[tokio::test]
async fn the_run_that_resumes_a_crashed_one_at_start_also_skips_resynthesis_while_autotuning() {
    // Arrange: nothing asks for a run on start, but a crashed run is waiting
    let (harness, server, llm) = autotuning_harness().await;
    let job = harness.job(&server, honest());
    interrupted_manifest(&harness, &[]);

    // Act
    let run_now = job.start(false).await.unwrap();
    job.tick().await;

    // Assert
    assert!(run_now);
    assert_eq!(llm.0.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_start_with_nothing_to_do_runs_nothing_and_leaves_resynthesis_on() {
    // Arrange
    let (harness, server, llm) = autotuning_harness().await;
    let job = harness.job(&server, honest());

    // Act
    let run_now = job.start(false).await.unwrap();
    job.tick().await;

    // Assert
    assert!(!run_now);
    assert!(
        llm.0.load(Ordering::SeqCst) > 0,
        "a later scheduled run resynthesizes"
    );
}

#[tokio::test]
async fn a_rolled_back_run_undoes_its_sweep_and_the_server_keeps_serving() {
    // Arrange: an old episode the run's sweep would remove, and probes that
    // fail the run after the sweep
    let harness = Harness::new();
    let (store, clock) = harness.store_at(Millis(100 * DAY)).await;
    let old =
        NewNode::now("episode", "old", "kept by the rollback").with_valid_from(Millis(10 * DAY));
    let old = store.insert(old).await.unwrap();
    store
        .insert(NewNode::now("fact", "f", "a live fact"))
        .await
        .unwrap();
    clock.set(Millis(100 * DAY));
    let server = harness.server(store, Arc::new(liam_model::MockLlm));
    let job = harness.job(&server, regresses());

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::RolledBack));
    let store = server
        .store()
        .await
        .expect("the slot holds the reopened store");
    let restored = store.get(&old, Millis::now()).await.unwrap();
    assert!(restored.is_some(), "the sweep was cancelled with the run");
}

#[tokio::test]
async fn a_committed_run_keeps_its_sweep() {
    // Arrange
    let harness = Harness::new();
    let (store, _clock) = harness.store_at(Millis(100 * DAY)).await;
    let old = NewNode::now("episode", "old", "past retention").with_valid_from(Millis(10 * DAY));
    let old = store.insert(old).await.unwrap();
    store
        .insert(NewNode::now("fact", "f", "a live fact"))
        .await
        .unwrap();
    let server = harness.server(store, Arc::new(liam_model::MockLlm));
    let job = harness.job(&server, honest());

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    let store = server.store().await.unwrap();
    assert!(store.get(&old, Millis::now()).await.unwrap().is_none());
}

fn interrupted_manifest(harness: &Harness, steps: &[&str]) {
    let mut manifest = RunManifest::new(
        "01HZZZZZZZZZZZZZZZZZZZZZZZ".to_owned(),
        7,
        identity(&harness.config).embedder,
        identity(&harness.config).llm_model,
        Millis(1),
    );
    manifest.save(&harness.manifest_path()).unwrap();
    for step in steps {
        manifest
            .mark_step_complete(step, &harness.manifest_path())
            .unwrap();
    }
}

/// Scores the baseline honestly, then fails the run and makes the log refuse
/// writes, so the rollback's marker cannot be appended.
struct FailsThenBreaksTheLog {
    log_refuses: Arc<AtomicBool>,
    calls: AtomicUsize,
}

#[async_trait]
impl ProbeRunner for FailsThenBreaksTheLog {
    async fn run(&self, store: &DefaultGraph, probes: &[Probe]) -> Result<ProbeReport, ProbeError> {
        let mut report = run_probes(store, probes).await?;
        if self.calls.fetch_add(1, Ordering::SeqCst) > 0 {
            report.scores.values_mut().for_each(|s| s.recall = 0.0);
            self.log_refuses.store(true, Ordering::SeqCst);
        }
        Ok(report)
    }
}

#[tokio::test]
async fn startup_finishes_a_rollback_a_real_crash_left_pending() {
    // Arrange: a run decides to roll back, commits to it, then cannot append
    // its marker, as a crash at that point would leave it
    let dir = tempfile::tempdir().unwrap();
    let log_dir = log_dir_in(dir.path());
    let edges_before_the_run;
    {
        let (writer, refuses) = ToggledWriter::open(&log_dir, false);
        let reader = Arc::new(liam_log::reader::SequentialScanReader::local(&log_dir).unwrap());
        let harness = Harness::over(dir.path(), log_over(writer, reader, None));
        let (store, _clock) = harness.store_at(Millis(1000)).await;
        linked_pair(&store, "fact").await;
        edges_before_the_run = store.edge_fingerprint().await.unwrap().edge_count;
        let server = harness.server(store, Arc::new(liam_model::MockLlm));
        let probes = Arc::new(FailsThenBreaksTheLog {
            log_refuses: refuses,
            calls: AtomicUsize::new(0),
        });
        let job = harness.job(&server, probes);

        let outcome = job.tick().await;

        assert_eq!(outcome, None, "the run died before a verdict");
        let left = RunManifest::load(&harness.manifest_path())
            .unwrap()
            .unwrap();
        assert!(left.completed_steps.iter().any(|s| s == ROLLBACK_PENDING));
        assert!(left.outcome.is_none());
        let linked = server.store().await.unwrap();
        assert!(
            linked.edge_fingerprint().await.unwrap().edge_count > edges_before_the_run,
            "the run's merge is still in the store"
        );
    }
    let restarted = Harness::over(dir.path(), standalone_log(&log_dir));
    let store = open_database(&restarted.config, &database_in(dir.path()))
        .await
        .unwrap();
    let store = open_store_on_log(
        store,
        &database_in(dir.path()),
        &log_dir,
        restarted.log.clone(),
        embedder(),
    )
    .await
    .unwrap();
    let server = restarted.server(store, Arc::new(liam_model::MockLlm));
    let job = restarted.job(&server, honest());

    // Act
    let run_now = job.start(false).await.unwrap();

    // Assert
    assert!(!run_now, "nothing is left to resume");
    let ended = RunManifest::load(&restarted.manifest_path())
        .unwrap()
        .unwrap();
    assert_eq!(ended.outcome, Some(RunOutcome::RolledBack));
    let store = server.store().await.unwrap();
    assert_eq!(
        store.edge_fingerprint().await.unwrap().edge_count,
        edges_before_the_run,
        "the run's merge was rolled back"
    );
}

#[tokio::test]
async fn startup_leaves_a_forward_run_for_the_first_tick_to_resume() {
    // Arrange
    let harness = Harness::new();
    let (store, _clock) = harness.store_at(Millis(1000)).await;
    store
        .insert(NewNode::now("fact", "f", "a live fact"))
        .await
        .unwrap();
    let server = harness.server(store, Arc::new(liam_model::MockLlm));
    let job = harness.job(&server, honest());
    interrupted_manifest(&harness, &[]);

    // Act
    let resumed = job.start(false).await.unwrap();
    let outcome = job.tick().await;

    // Assert
    assert!(resumed, "the forward run was not touched at startup");
    assert_eq!(outcome, Some(RunOutcome::Committed));
}

#[tokio::test]
async fn a_tick_after_the_daemon_began_stopping_runs_nothing() {
    // Arrange
    let harness = Harness::new();
    let (store, _clock) = harness.store_at(Millis(1000)).await;
    let server = harness.server(store, Arc::new(liam_model::MockLlm));
    let job = harness.job(&server, honest());
    job.stopping().cancel();

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, None);
    assert!(!harness.manifest_path().exists(), "no run was started");
}

/// Two old episodes linked by an edge, with the clock moved past retention.
async fn swept_pair(harness: &Harness) -> (Arc<DefaultGraph>, MemoryServer, ConsolidationJob) {
    let (store, clock) = harness.store_at(Millis(10 * DAY)).await;
    linked_pair(&store, "episode").await;
    keep_one_fact(&store).await;
    clock.set(Millis(100 * DAY));
    let store = Arc::new(store);
    let server = harness.server_arc(store.clone(), Arc::new(liam_model::MockLlm));
    let job = harness.job(&server, honest());
    (store, server, job)
}

#[tokio::test]
async fn a_run_sweeps_a_node_that_still_has_an_edge_pointing_at_it() {
    // Arrange: libSQL enforces foreign keys, so a sweep must remove the edge first
    let harness = Harness::new();
    let (store, _server, job) = swept_pair(&harness).await;

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert_eq!(store.edge_fingerprint().await.unwrap().edge_count, 0);
    let live = store.changes_since(liam_store::BEGINNING).await.unwrap();
    assert_eq!(
        live.iter().filter(|change| !change.closed).count(),
        1,
        "only the kept fact is left"
    );
}

#[tokio::test]
async fn a_run_sweeps_a_node_that_still_has_a_community_assignment() {
    // Arrange
    let harness = Harness::new();
    let (store, _server, job) = swept_pair(&harness).await;
    store.recompute_communities().await.unwrap();
    assert!(
        !store.communities().await.unwrap().is_empty(),
        "assignment seeded"
    );

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert!(store.communities().await.unwrap().is_empty());
}

#[tokio::test]
async fn the_fingerprint_falls_when_a_run_sweeps_an_edge() {
    // Arrange
    let harness = Harness::new();
    let (store, _server, job) = swept_pair(&harness).await;
    let before = store.edge_fingerprint().await.unwrap();
    assert_eq!(before.edge_count, 1);

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    let after = store.edge_fingerprint().await.unwrap();
    assert_eq!(after.edge_count, 0, "a swept edge must lower the count");
    assert_ne!(before, after, "the fingerprint must notice a deletion");
}

/// A reopener that cannot open the database, as after a lost file.
struct BrokenReopener;

#[async_trait]
impl StoreReopener for BrokenReopener {
    async fn reopen(&self) -> anyhow::Result<DefaultGraph> {
        anyhow::bail!("injected reopen failure")
    }
}

#[tokio::test]
async fn a_rollback_that_cannot_reopen_the_store_stops_the_daemon_with_a_failure() {
    // Arrange
    let harness = Harness::new();
    let (store, _clock) = harness.store_at(Millis(1000)).await;
    store
        .insert(NewNode::now("fact", "f", "a live fact"))
        .await
        .unwrap();
    let server = harness.server(store, Arc::new(liam_model::MockLlm));
    let job = harness.job_reopening(&server, regresses(), Some(Arc::new(BrokenReopener)));

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, None);
    assert!(server.store().await.is_err(), "the slot is empty");
    assert!(
        harness.shutdown.token().is_cancelled(),
        "the supervisor must start the daemon again"
    );
    assert!(harness.shutdown.failure().is_some());
}

#[tokio::test]
async fn a_run_that_fails_with_the_store_in_place_does_not_stop_the_daemon() {
    // Arrange: a failure that leaves the slot full, here a refused new run
    let harness = Harness::new();
    let (store, _clock) = harness.store_at(Millis(1000)).await;
    let server = harness.server(store, Arc::new(liam_model::MockLlm));
    let job = harness.job(&server, honest());
    restore_failed_manifest(&harness);

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, None);
    assert!(!harness.shutdown.token().is_cancelled());
}

fn restore_failed_manifest(harness: &Harness) -> RunManifest {
    let mut manifest = RunManifest::new(
        "01HZZZZZZZZZZZZZZZZZZZZZZZ".to_owned(),
        7,
        identity(&harness.config).embedder,
        identity(&harness.config).llm_model,
        Millis(1),
    );
    manifest.outcome = Some(RunOutcome::RestoreFailed);
    manifest.save(&harness.manifest_path()).unwrap();
    manifest
}

#[tokio::test]
async fn a_new_run_is_refused_while_the_last_one_ended_restore_failed_and_its_snapshot_stays() {
    // Arrange
    let harness = Harness::new();
    let (store, _clock) = harness.store_at(Millis(1000)).await;
    let server = harness.server(store, Arc::new(liam_model::MockLlm));
    let job = harness.job(&server, honest());
    let failed = restore_failed_manifest(&harness);
    let kept = harness
        .config
        .consolidation
        .snapshot_dir
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(|| liam_daemon_snapshot_dir(&harness))
        .join(&failed.run_id);
    std::fs::create_dir_all(&kept).unwrap();
    std::fs::write(kept.join("snapshot.db"), b"kept for inspection").unwrap();

    // Act
    let refused = job.consolidator.consolidate(&job.store).await.unwrap_err();
    let outcome = job.tick().await;

    // Assert
    let message = format!("{refused:#}");
    assert!(message.contains("failed restore"), "{message}");
    assert!(message.contains("manual inspection"), "{message}");
    assert_eq!(outcome, None);
    assert!(kept.join("snapshot.db").exists(), "the snapshot is kept");
    let after = RunManifest::load(&harness.manifest_path())
        .unwrap()
        .unwrap();
    assert_eq!(after, failed, "no new run replaced the record");
}

fn liam_daemon_snapshot_dir(harness: &Harness) -> PathBuf {
    liam_daemon::models::resolve_snapshot_dir(&harness.config, &database_in(&harness.root)).unwrap()
}

#[tokio::test]
async fn the_maintenance_step_reports_a_stopping_daemon_so_it_is_not_recorded_done() {
    // Arrange
    let harness = Harness::new();
    let (store, _clock) = harness.store_at(Millis(1000)).await;
    let server = harness.server(store, Arc::new(liam_model::MockLlm));
    let maintenance = DaemonMaintenance {
        server: server.clone(),
        policy: harness.config.gc_policy(),
        resynthesis: Resynthesis::from_config(&harness.config),
        skip_resynthesis_once: Arc::new(AtomicBool::new(false)),
    };
    harness.shutdown.token().cancel();

    // Act
    let result = maintenance
        .run(&server.store().await.unwrap(), "consolidation:test")
        .await;

    // Assert
    let error = result.unwrap_err();
    assert!(error.to_string().contains("stopping"), "{error}");
}

#[tokio::test]
async fn the_maintenance_step_is_done_when_the_daemon_keeps_running() {
    let harness = Harness::new();
    let (store, _clock) = harness.store_at(Millis(1000)).await;
    let server = harness.server(store, Arc::new(liam_model::MockLlm));
    let maintenance = DaemonMaintenance {
        server: server.clone(),
        policy: harness.config.gc_policy(),
        resynthesis: Resynthesis::from_config(&harness.config),
        skip_resynthesis_once: Arc::new(AtomicBool::new(false)),
    };

    let result = maintenance
        .run(&server.store().await.unwrap(), "consolidation:test")
        .await;

    assert!(result.is_ok());
}
