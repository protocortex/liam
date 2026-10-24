// SPDX-License-Identifier: Apache-2.0
//! Whole runs over a real database file and a real WAL in a temp directory.
//! A crash is simulated by failing the run right after a named step, and a
//! resume by running again over the same files.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use liam_log::dedup::BloomConfig;
use liam_log::event::{LogEvent, LogPayload};
use liam_log::wal::{SystemClock, WalConfig, WalError, WalWriter};
use liam_log::{LogOffset, LogWriter};
use liam_store::{EmbedError, GraphConfig, NewNode, SharedLog, BEGINNING};

use super::*;
use crate::consolidation::hooks::{ExtractionHook, TierCtx, TierHook, TierReport};
use crate::event_log;

pub(super) const DIMS: usize = 8;
pub(super) const MARGIN: f64 = 0.02;
pub(super) const THRESHOLD: f32 = 0.9;
pub(super) const PROBE_COUNT: usize = 4;
pub(super) const SEED: u64 = 11;
pub(super) const TOPICS: usize = 6;
pub(super) const DUPLICATE: &str = "topic0 detail again";
const MANY_NODES: usize = 520;
const NEARLY_EXACT: f32 = 0.9999;
pub(super) const WAL: WalConfig = WalConfig {
    segment_max_bytes: 1 << 20,
    rotate_interval_secs: 3600,
};
/// How far the near-duplicate's vector sits from its original's.
pub(super) const NUDGE: f32 = 0.05;

pub(super) fn topic(i: usize) -> String {
    format!("topic{i} detail{i}")
}

/// One axis per topic, and the duplicate on the first topic's axis with a nudge.
pub(super) struct TopicEmbedder;

#[async_trait]
impl ContentEmbedder for TopicEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let axis = text
            .strip_prefix("topic")
            .and_then(|rest| rest[..1].parse::<usize>().ok())
            .unwrap_or(DIMS - 1);
        let mut vector = vec![0.0; DIMS];
        vector[axis] = 1.0;
        if text.ends_with("again") {
            vector[DIMS - 2] = NUDGE;
        }
        Ok(vector)
    }
}

/// "x7" and "y7" share one pseudo-random vector, and no other pair is close.
struct ManyEmbedder;

#[async_trait]
impl ContentEmbedder for ManyEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let n: u64 = text[1..]
            .parse()
            .map_err(|_| -> EmbedError { "not a number".into() })?;
        Ok((0..DIMS as u64)
            .map(|j| {
                let mut z =
                    n.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ j.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 30)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 27;
                (z % 1000) as f32 / 1000.0 - 0.5
            })
            .collect())
    }
}

/// Counts its runs and keeps the source each was given.
#[derive(Default)]
pub(super) struct CountingMaintenance {
    pub(super) runs: AtomicUsize,
    pub(super) sources: Mutex<Vec<String>>,
}

#[async_trait]
impl Maintenance for CountingMaintenance {
    async fn run(&self, _store: &DefaultGraph, source: &str) -> anyhow::Result<()> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        self.sources.lock().unwrap().push(source.to_owned());
        Ok(())
    }
}

/// Scores honestly, except that once the probe set file exists (the baseline is
/// taken) it scores zero, in any process.
pub(super) struct Probes {
    pub(super) regress_once_present: Option<PathBuf>,
    pub(super) first_queries: Mutex<Vec<String>>,
}

#[async_trait]
impl ProbeRunner for Probes {
    async fn run(&self, store: &DefaultGraph, probes: &[Probe]) -> Result<ProbeReport, ProbeError> {
        {
            let mut first = self.first_queries.lock().unwrap();
            if first.is_empty() {
                first.extend(probes.iter().map(|probe| probe.query.clone()));
            }
        }
        let mut report = run_probes(store, probes).await?;
        if self
            .regress_once_present
            .as_ref()
            .is_some_and(|p| p.exists())
        {
            report
                .scores
                .values_mut()
                .for_each(|score| score.recall = 0.0);
        }
        Ok(report)
    }
}

pub(super) fn honest() -> Arc<Probes> {
    Arc::new(Probes {
        regress_once_present: None,
        first_queries: Default::default(),
    })
}

pub(super) struct FailingTier;

#[async_trait]
impl TierHook for FailingTier {
    async fn tier(&self, _ctx: &TierCtx) -> anyhow::Result<TierReport> {
        anyhow::bail!("tier part failed")
    }
}

/// Reports it did only part of its work.
struct PartialExtraction;

#[async_trait]
impl ExtractionHook for PartialExtraction {
    async fn extract(
        &self,
        _ctx: &ExtractCtx<'_>,
    ) -> anyhow::Result<crate::consolidation::hooks::ExtractionReport> {
        Ok(crate::consolidation::hooks::ExtractionReport { partial: true })
    }
}

/// Keeps the source the run gave it.
struct RecordingExtraction(Arc<Mutex<Vec<String>>>);

#[async_trait]
impl ExtractionHook for RecordingExtraction {
    async fn extract(
        &self,
        ctx: &ExtractCtx<'_>,
    ) -> anyhow::Result<crate::consolidation::hooks::ExtractionReport> {
        self.0.lock().unwrap().push(ctx.source.clone());
        Ok(Default::default())
    }
}

/// A writer that refuses to append the rollback marker while `marker_fails` is
/// set, and reports itself poisoned on any append while `poisoned` is set.
struct FlakyWriter {
    inner: WalWriter<SystemClock>,
    marker_fails: Arc<AtomicBool>,
    poisoned: Arc<AtomicBool>,
}

impl LogWriter for FlakyWriter {
    fn append(&mut self, event: &LogEvent) -> Result<LogOffset, WalError> {
        if self.poisoned.load(Ordering::SeqCst) {
            return Err(WalError::Poisoned);
        }
        let marker = matches!(event.payload, LogPayload::RunRolledBack { .. });
        if marker && self.marker_fails.load(Ordering::SeqCst) {
            return Err(WalError::Io(std::io::Error::other(
                "injected append failure",
            )));
        }
        self.inner.append(event)
    }

    fn log_id(&self) -> uuid::Uuid {
        self.inner.log_id()
    }

    fn head(&self) -> Option<LogOffset> {
        self.inner.head()
    }
}

/// Reopens the store, failing first, or poisoning the log afterwards, when told to.
pub(super) struct Reopen {
    database_path: PathBuf,
    pub(super) log: SharedLog,
    embedder: Arc<dyn ContentEmbedder>,
    reopen_fails: Arc<AtomicBool>,
    poison_on_reopen: Arc<AtomicBool>,
    writer_poisoned: Arc<AtomicBool>,
}

#[async_trait]
impl StoreReopener for Reopen {
    async fn reopen(&self) -> anyhow::Result<DefaultGraph> {
        anyhow::ensure!(
            !self.reopen_fails.load(Ordering::SeqCst),
            "injected reopen failure"
        );
        let store = open_on(&self.database_path, &self.log, &self.embedder).await?;
        if self.poison_on_reopen.load(Ordering::SeqCst) {
            self.writer_poisoned.store(true, Ordering::SeqCst);
            // The failed append leaves the log poisoned, which a replay refuses.
            store
                .insert(note("written to poison the log"))
                .await
                .unwrap_err();
        }
        Ok(store)
    }
}

pub(super) async fn open_on(
    database_path: &Path,
    log: &SharedLog,
    embedder: &Arc<dyn ContentEmbedder>,
) -> anyhow::Result<DefaultGraph> {
    let store = DefaultGraph::open(database_path.to_str().unwrap(), GraphConfig::new(DIMS))
        .await?
        .with_log(log.clone())
        .await?
        .with_embedder(embedder.clone());
    if store.needs_backfill().await? {
        store.backfill_log_from_projection().await?;
    }
    Ok(store)
}

pub(super) struct Env {
    pub(super) dir: tempfile::TempDir,
    pub(super) log: SharedLog,
    pub(super) embedder: Arc<dyn ContentEmbedder>,
    pub(super) marker_fails: Arc<AtomicBool>,
    pub(super) reopen_fails: Arc<AtomicBool>,
    pub(super) poison_on_reopen: Arc<AtomicBool>,
    writer_poisoned: Arc<AtomicBool>,
}

impl Env {
    pub(super) fn new() -> Self {
        Self::with_embedder(Arc::new(TopicEmbedder))
    }

    pub(super) fn with_embedder(embedder: Arc<dyn ContentEmbedder>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let log_dir = dir.path().join("events");
        let marker_fails = Arc::new(AtomicBool::new(false));
        let writer_poisoned = Arc::new(AtomicBool::new(false));
        let writer = FlakyWriter {
            inner: event_log::open_writer(&log_dir, WAL).unwrap(),
            marker_fails: marker_fails.clone(),
            poisoned: writer_poisoned.clone(),
        };
        let log = event_log::shared_log(writer, &log_dir, BloomConfig::default(), None).unwrap();
        Self {
            dir,
            log,
            embedder,
            marker_fails,
            reopen_fails: Default::default(),
            poison_on_reopen: Default::default(),
            writer_poisoned,
        }
    }

    pub(super) fn database_path(&self) -> PathBuf {
        self.dir.path().join("liam.db")
    }

    /// The store as the daemon opens it: attached to the log and caught up.
    pub(super) async fn open(&self) -> SharedStore {
        let store = open_on(&self.database_path(), &self.log, &self.embedder)
            .await
            .unwrap();
        store.catch_up().await.unwrap();
        SharedStore::new(store)
    }

    pub(super) async fn seeded(&self) -> SharedStore {
        let shared = self.open().await;
        let store = shared.get().await.unwrap();
        for i in 0..TOPICS {
            store.insert(note(&topic(i))).await.unwrap();
        }
        store.insert(note(DUPLICATE)).await.unwrap();
        drop(store);
        shared
    }

    /// A probe runner that fails the run once the baseline is taken.
    pub(super) fn regressing(&self) -> Arc<Probes> {
        Arc::new(Probes {
            regress_once_present: Some(probe_set_path_for(&self.database_path())),
            first_queries: Default::default(),
        })
    }

    pub(super) fn consolidator(
        &self,
        maintenance: Arc<dyn Maintenance>,
        probes: Arc<dyn ProbeRunner>,
    ) -> Consolidator {
        self.consolidator_with(maintenance, probes, ConsolidationHookSet::default())
    }

    pub(super) fn consolidator_with(
        &self,
        maintenance: Arc<dyn Maintenance>,
        probes: Arc<dyn ProbeRunner>,
        hooks: ConsolidationHookSet,
    ) -> Consolidator {
        self.consolidator_drawing(maintenance, probes, hooks, PROBE_COUNT)
    }

    /// A consolidator that draws up to `probe_count` probes.
    pub(super) fn consolidator_drawing(
        &self,
        maintenance: Arc<dyn Maintenance>,
        probes: Arc<dyn ProbeRunner>,
        hooks: ConsolidationHookSet,
        probe_count: usize,
    ) -> Consolidator {
        let settings = RunSettings {
            probe_recall_margin: MARGIN,
            dedup_similarity_threshold: THRESHOLD,
            snapshot_dir: self.dir.path().join("snapshots"),
            probe_count,
        };
        let identity = Identity {
            embedder: fingerprint(),
            llm_model: "llm/model.gguf".into(),
        };
        let parts = Parts {
            embedder: self.embedder.clone(),
            hooks,
            maintenance,
            reopener: Arc::new(Reopen {
                database_path: self.database_path(),
                log: self.log.clone(),
                embedder: self.embedder.clone(),
                reopen_fails: self.reopen_fails.clone(),
                poison_on_reopen: self.poison_on_reopen.clone(),
                writer_poisoned: self.writer_poisoned.clone(),
            }),
            probes,
        };
        Consolidator::new(self.database_path(), settings, identity, parts).with_seed(SEED)
    }

    pub(super) fn manifest(&self) -> RunManifest {
        RunManifest::load(&manifest_path_for(&self.database_path()))
            .unwrap()
            .unwrap()
    }

    /// Files left under the snapshot directory, in any run's subdirectory.
    pub(super) fn snapshot_files(&self) -> usize {
        fn count(dir: &Path) -> usize {
            std::fs::read_dir(dir)
                .map(|entries| {
                    entries
                        .flatten()
                        .map(|entry| {
                            if entry.path().is_dir() {
                                count(&entry.path())
                            } else {
                                1
                            }
                        })
                        .sum()
                })
                .unwrap_or(0)
        }
        count(&self.dir.path().join("snapshots"))
    }
}

pub(super) fn note(content: &str) -> NewNode {
    NewNode::now("fact", content, content).with_producer("client:mcp")
}

pub(super) fn fingerprint() -> EmbedderFingerprint {
    EmbedderFingerprint {
        provider: "mock".into(),
        model: "topics".into(),
        dims: DIMS,
        prompt_scheme: 1,
    }
}

pub(super) type Links = BTreeSet<(String, String)>;

/// What the store holds, by content: its live nodes and the pairs the edges link.
pub(super) async fn state(shared: &SharedStore) -> (BTreeSet<String>, Links) {
    let store = shared.get().await.unwrap();
    let mut contents = BTreeSet::new();
    let mut links = BTreeSet::new();
    let changes = store.changes_since(BEGINNING).await.unwrap();
    for change in changes.iter().filter(|change| !change.closed) {
        let Some(node) = store.get(&change.id, ALL_LIVE).await.unwrap() else {
            continue;
        };
        contents.insert(node.content.clone());
        for other in store.neighbors(&change.id, ALL_LIVE).await.unwrap() {
            let Some(other) = store.get(&other, ALL_LIVE).await.unwrap() else {
                continue;
            };
            links.insert((node.content.clone(), other.content));
        }
    }
    (contents, links)
}

pub(super) fn maintenance() -> Arc<CountingMaintenance> {
    Arc::new(CountingMaintenance::default())
}

pub(super) fn duplicate_links() -> Links {
    BTreeSet::from([
        (DUPLICATE.to_owned(), topic(0)),
        (topic(0), DUPLICATE.to_owned()),
    ])
}

#[test]
fn the_forward_steps_run_in_this_order() {
    // Assert
    assert_eq!(
        STEPS,
        [
            "snapshot",
            "baseline",
            "dedup",
            "maintenance",
            "extract",
            "summarize",
            "tier",
            "probes"
        ]
    );
}

#[tokio::test]
async fn a_run_whose_probes_hold_is_committed() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let steps = maintenance();
    let run = env.consolidator(steps.clone(), honest());

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::Committed));
    assert_eq!(steps.runs.load(Ordering::SeqCst), 1);
    let finished: Vec<&str> = manifest
        .completed_steps
        .iter()
        .map(String::as_str)
        .filter(|step| !step.ends_with(":skipped"))
        .collect();
    assert_eq!(finished, STEPS);
    let (contents, links) = state(&shared).await;
    assert_eq!(contents.len(), TOPICS + 1);
    assert_eq!(links, duplicate_links());
    assert_eq!(env.snapshot_files(), 0);
    assert_eq!(env.manifest(), manifest);
}

#[tokio::test]
async fn maintenance_and_hooks_are_given_the_runs_write_source() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let steps = maintenance();
    let extracted = Arc::new(Mutex::new(Vec::new()));
    let hooks = ConsolidationHookSet {
        extraction: Some(Arc::new(RecordingExtraction(extracted.clone()))),
        ..Default::default()
    };
    let run = env.consolidator_with(steps.clone(), honest(), hooks);

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    let source = manifest.write_source();
    assert_eq!(
        *steps.sources.lock().unwrap(),
        std::slice::from_ref(&source)
    );
    assert_eq!(*extracted.lock().unwrap(), [source]);
}

#[tokio::test]
async fn a_store_with_no_live_node_is_committed_without_the_gate() {
    // Arrange
    let env = Env::new();
    let shared = env.open().await;
    let run = env.consolidator(maintenance(), env.regressing());

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::Committed));
}

#[tokio::test]
async fn a_probe_count_of_zero_is_refused_before_anything_is_written() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let mut run = env.consolidator(maintenance(), honest());
    run.settings.probe_count = 0;

    // Act
    let result = run.consolidate(&shared).await;

    // Assert
    assert!(result.is_err());
    assert!(!manifest_path_for(&env.database_path()).exists());
    assert_eq!(env.snapshot_files(), 0);
}

#[tokio::test]
async fn hook_steps_record_skipped_when_unconfigured_and_partial_when_they_fail() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let hooks = ConsolidationHookSet {
        tiering: Some(Arc::new(FailingTier)),
        ..Default::default()
    };
    let run = env.consolidator_with(maintenance(), honest(), hooks);

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::Committed));
    assert!(manifest.is_complete("extract:skipped"));
    assert!(manifest.is_complete("summarize:skipped"));
    assert!(manifest.is_complete("tier:partial"));
    assert!(!manifest.is_complete("tier:skipped"));
}

#[tokio::test]
async fn an_extraction_part_that_reports_partial_work_is_recorded_as_partial() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let hooks = ConsolidationHookSet {
        extraction: Some(Arc::new(PartialExtraction)),
        ..Default::default()
    };
    let run = env.consolidator_with(maintenance(), honest(), hooks);

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::Committed));
    assert!(manifest.is_complete("extract:partial"));
    assert!(!manifest.is_complete("extract:skipped"));
}

#[tokio::test]
async fn a_crash_after_any_forward_step_resumes_to_the_same_committed_store() {
    // Arrange
    let reference = Env::new();
    let seeded = reference.seeded().await;
    reference
        .consolidator(maintenance(), honest())
        .consolidate(&seeded)
        .await
        .unwrap();
    let expected = state(&seeded).await;

    for point in STEPS {
        let env = Env::new();
        let steps = maintenance();
        let mut crashing = env.consolidator(steps.clone(), honest());
        crashing.crash_after = Some(point);

        // Act
        let crashed = crashing.consolidate(&env.seeded().await).await;
        let reopened = env.open().await;
        let resumed = env
            .consolidator(steps.clone(), honest())
            .consolidate(&reopened)
            .await
            .unwrap_or_else(|error| panic!("after {point}: {error:#}"));

        // Assert
        assert!(crashed.is_err(), "crash after {point}");
        assert_eq!(
            resumed.outcome,
            Some(RunOutcome::Committed),
            "after {point}"
        );
        assert_eq!(state(&reopened).await, expected, "after {point}");
        assert_eq!(steps.runs.load(Ordering::SeqCst), 1, "after {point}");
    }
}

#[tokio::test]
async fn a_crash_after_the_snapshot_is_published_leaves_nothing_behind_once_resumed() {
    // Arrange
    let env = Env::new();
    let mut crashing = env.consolidator(maintenance(), honest());
    crashing.crash_after = Some("snapshot:published");
    assert!(crashing.consolidate(&env.seeded().await).await.is_err());
    let orphaned = env.snapshot_files();
    let reopened = env.open().await;

    // Act
    let resumed = env
        .consolidator(maintenance(), honest())
        .consolidate(&reopened)
        .await
        .unwrap();

    // Assert
    assert!(orphaned > 0, "the crash left a published snapshot");
    assert_eq!(resumed.outcome, Some(RunOutcome::Committed));
    assert_eq!(env.snapshot_files(), 0);
}

#[tokio::test]
async fn a_fresh_run_removes_the_snapshots_of_runs_that_can_no_longer_resume() {
    // Arrange
    let env = Env::new();
    let dead = env.dir.path().join("snapshots").join("dead-run");
    std::fs::create_dir_all(&dead).unwrap();
    std::fs::write(dead.join("old.db"), b"left by a run nothing can resume").unwrap();
    let shared = env.seeded().await;

    // Act
    env.consolidator(maintenance(), honest())
        .consolidate(&shared)
        .await
        .unwrap();

    // Assert
    assert_eq!(env.snapshot_files(), 0);
}

#[tokio::test]
async fn the_dedup_pass_continues_where_the_last_committed_run_stopped() {
    // Arrange: more nodes than one pass scans, with the only duplicate pair last.
    let env = Env::with_embedder(Arc::new(ManyEmbedder));
    let shared = env.open().await;
    {
        let store = shared.get().await.unwrap();
        for i in 0..MANY_NODES {
            store.insert(note(&format!("x{i}"))).await.unwrap();
        }
        store
            .insert(note(&format!("y{}", MANY_NODES - 1)))
            .await
            .unwrap();
    }
    let linked = || async { state(&shared).await.1 };
    let mut first = env.consolidator(maintenance(), honest());
    first.settings.dedup_similarity_threshold = NEARLY_EXACT;
    let mut second = env.consolidator(maintenance(), honest());
    second.settings.dedup_similarity_threshold = NEARLY_EXACT;

    // Act
    first.consolidate(&shared).await.unwrap();
    let after_first = linked().await;
    second.consolidate(&shared).await.unwrap();
    let after_second = linked().await;

    // Assert
    assert!(after_first.is_empty(), "the pair is past the first pass");
    let last = format!("x{}", MANY_NODES - 1);
    let twin = format!("y{}", MANY_NODES - 1);
    assert_eq!(
        after_second,
        BTreeSet::from([(last.clone(), twin.clone()), (twin, last)])
    );
}
