// SPDX-License-Identifier: Apache-2.0
//! Whole runs over a real database file and a real WAL in a temp directory.
//! A crash is simulated by failing the run right after a named step, and a
//! resume by running again over the same files.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use liam_log::dedup::BloomConfig;
use liam_log::wal::WalConfig;
use liam_store::{EmbedError, GraphConfig, NewNode, SharedLog, BEGINNING};

use super::*;
use crate::consolidation::hooks::{TierCtx, TierHook, TierReport};
use crate::event_log;

pub(super) const DIMS: usize = 8;
pub(super) const MARGIN: f64 = 0.02;
pub(super) const THRESHOLD: f32 = 0.9;
pub(super) const PROBE_COUNT: usize = 4;
pub(super) const SEED: u64 = 11;
pub(super) const TOPICS: usize = 6;
pub(super) const DUPLICATE: &str = "topic0 detail again";
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

#[derive(Default)]
pub(super) struct CountingMaintenance(AtomicUsize);

#[async_trait]
impl Maintenance for CountingMaintenance {
    async fn run(&self, _store: &DefaultGraph) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// Scores honestly, except that once the probe set file exists (the baseline is
/// taken) it scores zero, in any process.
pub(super) struct Probes {
    pub(super) regress_once_present: Option<PathBuf>,
    pub(super) first_queries: std::sync::Mutex<Vec<String>>,
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

pub(super) struct Reopen {
    database_path: PathBuf,
    pub(super) log: SharedLog,
}

#[async_trait]
impl StoreReopener for Reopen {
    async fn reopen(&self) -> anyhow::Result<DefaultGraph> {
        open_on(&self.database_path, &self.log).await
    }
}

pub(super) async fn open_on(database_path: &Path, log: &SharedLog) -> anyhow::Result<DefaultGraph> {
    let store = DefaultGraph::open(database_path.to_str().unwrap(), GraphConfig::new(DIMS))
        .await?
        .with_log(log.clone())
        .await?
        .with_embedder(Arc::new(TopicEmbedder));
    if store.needs_backfill().await? {
        store.backfill_log_from_projection().await?;
    }
    Ok(store)
}

pub(super) struct Env {
    pub(super) dir: tempfile::TempDir,
    pub(super) log: SharedLog,
}

impl Env {
    pub(super) fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let log_dir = dir.path().join("events");
        let writer = event_log::open_writer(&log_dir, WAL).unwrap();
        let log = event_log::shared_log(writer, &log_dir, BloomConfig::default(), None).unwrap();
        Self { dir, log }
    }

    pub(super) fn database_path(&self) -> PathBuf {
        self.dir.path().join("liam.db")
    }

    /// The store as the daemon opens it: attached to the log and caught up.
    pub(super) async fn open(&self) -> DefaultGraph {
        let store = open_on(&self.database_path(), &self.log).await.unwrap();
        store.catch_up().await.unwrap();
        store
    }

    pub(super) async fn seeded(&self) -> DefaultGraph {
        let store = self.open().await;
        for i in 0..TOPICS {
            store.insert(note(&topic(i))).await.unwrap();
        }
        store.insert(note(DUPLICATE)).await.unwrap();
        store
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
        let settings = RunSettings {
            probe_recall_margin: MARGIN,
            dedup_similarity_threshold: THRESHOLD,
            snapshot_dir: self.dir.path().join("snapshots"),
            probe_count: PROBE_COUNT,
        };
        let identity = Identity {
            embedder: fingerprint(),
            llm_model: "llm/model.gguf".into(),
        };
        let parts = Parts {
            embedder: Arc::new(TopicEmbedder),
            hooks,
            maintenance,
            reopener: Arc::new(Reopen {
                database_path: self.database_path(),
                log: self.log.clone(),
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

    pub(super) fn snapshot_files(&self) -> usize {
        std::fs::read_dir(self.dir.path().join("snapshots"))
            .map(|entries| entries.count())
            .unwrap_or(0)
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
pub(super) async fn state(store: &DefaultGraph) -> (BTreeSet<String>, Links) {
    let mut contents = BTreeSet::new();
    let mut links = BTreeSet::new();
    let changes = store.changes_since(BEGINNING).await.unwrap();
    for change in changes.iter().filter(|change| !change.closed) {
        let Some(node) = store.get(&change.id, ALL_LIVE).await.unwrap() else {
            continue;
        };
        contents.insert(node.content.clone());
        for other in store.neighbors(&change.id, ALL_LIVE).await.unwrap() {
            let other = store.get(&other, ALL_LIVE).await.unwrap().unwrap();
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

#[tokio::test]
async fn a_run_whose_probes_hold_is_committed() {
    // Arrange
    let env = Env::new();
    let store = env.seeded().await;
    let steps = maintenance();
    let run = env.consolidator(steps.clone(), honest());

    // Act
    let finished = run.consolidate(store).await.unwrap();

    // Assert
    assert_eq!(finished.manifest.outcome, Some(RunOutcome::Committed));
    assert_eq!(steps.0.load(Ordering::SeqCst), 1);
    assert_eq!(finished.manifest.completed_steps, STEPS);
    let (contents, links) = state(&finished.store).await;
    assert_eq!(contents.len(), TOPICS + 1);
    assert_eq!(links, duplicate_links());
    assert_eq!(env.snapshot_files(), 0);
    assert_eq!(env.manifest(), finished.manifest);
}

#[tokio::test]
async fn a_store_with_nothing_to_probe_is_committed_without_the_gate() {
    // Arrange
    let env = Env::new();
    let store = env.open().await;
    let run = env.consolidator(maintenance(), env.regressing());

    // Act
    let finished = run.consolidate(store).await.unwrap();

    // Assert
    assert_eq!(finished.manifest.outcome, Some(RunOutcome::Committed));
}

#[tokio::test]
async fn a_failing_hook_is_recorded_and_the_run_still_commits() {
    // Arrange
    let env = Env::new();
    let store = env.seeded().await;
    let hooks = ConsolidationHookSet {
        tiering: Some(Arc::new(FailingTier)),
        ..Default::default()
    };
    let run = env.consolidator_with(maintenance(), honest(), hooks);

    // Act
    let finished = run.consolidate(store).await.unwrap();

    // Assert
    assert_eq!(finished.manifest.outcome, Some(RunOutcome::Committed));
    assert!(finished.manifest.is_complete("tier:failed"));
}

#[tokio::test]
async fn a_crash_after_any_forward_step_resumes_to_the_same_committed_store() {
    // Arrange
    let reference = Env::new();
    let uninterrupted = reference
        .consolidator(maintenance(), honest())
        .consolidate(reference.seeded().await)
        .await
        .unwrap();
    let expected = state(&uninterrupted.store).await;

    for point in STEPS {
        let env = Env::new();
        let steps = maintenance();
        let mut crashing = env.consolidator(steps.clone(), honest());
        crashing.crash_after = Some(point);

        // Act
        let crashed = crashing.consolidate(env.seeded().await).await;
        let resumed = env
            .consolidator(steps.clone(), honest())
            .consolidate(env.open().await)
            .await
            .unwrap_or_else(|error| panic!("after {point}: {error:#}"));

        // Assert
        assert!(crashed.is_err(), "crash after {point}");
        let outcome = resumed.manifest.outcome;
        assert_eq!(outcome, Some(RunOutcome::Committed), "after {point}");
        assert_eq!(state(&resumed.store).await, expected, "after {point}");
        assert_eq!(steps.0.load(Ordering::SeqCst), 1, "after {point}");
    }
}
