// SPDX-License-Identifier: Apache-2.0
//! Doubles and builders shared by the tiering tests.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use liam_daemon::config::{ExtractionConfig, TieringConfig};
use liam_daemon::consolidation::hooks::TierCtx;
use liam_daemon::consolidation::tier_history::TierHistory;
use liam_log::dedup::BloomConfig;
use liam_model::{Embedder, ModelError, TextKind};
use liam_store::{
    DefaultGraph, FixedClock, GraphConfig, LineageKey, Millis, NewNode, NodeId, OriginClaim,
    OriginVerdict, SharedLog, Tier, TierMove, UsageEntry,
};
use tokio::sync::Semaphore;

use super::hooks::TierHooks;
use super::verify::TierVerifier;
use crate::config::LogConfig;
use crate::extraction::sidecar::FreeRam;

/// An in-memory store on an event log under `dir`, read through `clock`.
pub async fn logged_store(dir: &Path, clock: Arc<FixedClock>, dims: usize) -> DefaultGraph {
    logged_store_with_log(dir, clock, dims).await.0
}

/// An event log under `dir`.
pub fn shared_test_log(dir: &Path) -> SharedLog {
    let log_dir = dir.join("events");
    let writer = liam_daemon::event_log::open_writer(&log_dir, LogConfig::default().wal()).unwrap();
    liam_daemon::event_log::shared_log(writer, &log_dir, BloomConfig::default(), None, None)
        .unwrap()
}

/// Like `logged_store`, and the log handle beside it, for tests that rebuild.
pub async fn logged_store_with_log(
    dir: &Path,
    clock: Arc<FixedClock>,
    dims: usize,
) -> (DefaultGraph, SharedLog) {
    let log = shared_test_log(dir);
    let store = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(dims), clock)
        .await
        .unwrap()
        .with_log(log.clone())
        .await
        .unwrap();
    (store, log)
}

/// A vector whose component `j` is positive when bit `j` of `pattern` is set,
/// so its sign signature is `pattern`. `MockEmbedder` gives all-positive
/// components, which would make every signature the same.
pub fn signed(pattern: u8, dims: usize) -> Vec<f32> {
    (0..dims)
        .map(|bit| if pattern >> bit & 1 == 1 { 1.0 } else { -1.0 })
        .collect()
}

/// Maps a text to the vector it embeds to; any other text is an error.
pub struct ScriptedEmbedder {
    dims: usize,
    vectors: HashMap<String, Vec<f32>>,
    calls: AtomicUsize,
}

impl ScriptedEmbedder {
    pub fn new(dims: usize, vectors: &[(&str, Vec<f32>)]) -> Self {
        Self {
            dims,
            vectors: vectors
                .iter()
                .map(|(text, vector)| ((*text).to_string(), vector.clone()))
                .collect(),
            calls: AtomicUsize::new(0),
        }
    }

    /// How many texts it was asked to embed.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Embedder for ScriptedEmbedder {
    fn dims(&self) -> usize {
        self.dims
    }

    async fn embed(&self, text: &str, _kind: TextKind) -> liam_model::Result<Vec<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.vectors
            .get(text)
            .cloned()
            .ok_or_else(|| ModelError::Embed(format!("no scripted vector for {text:?}")))
    }
}

/// Counts the calls of an embedder it wraps and records the kind of each.
pub struct CountingEmbedder<E> {
    inner: E,
    kinds: Mutex<Vec<TextKind>>,
}

impl<E> CountingEmbedder<E> {
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            kinds: Mutex::new(Vec::new()),
        }
    }

    pub fn kinds(&self) -> Vec<TextKind> {
        self.kinds.lock().unwrap().clone()
    }
}

#[async_trait]
impl<E: Embedder> Embedder for CountingEmbedder<E> {
    fn dims(&self) -> usize {
        self.inner.dims()
    }

    async fn embed(&self, text: &str, kind: TextKind) -> liam_model::Result<Vec<f32>> {
        self.kinds.lock().unwrap().push(kind);
        self.inner.embed(text, kind).await
    }
}

/// Fails its `nth` call (1-based) and, when made slow, waits that long in every
/// call, so a deadline can be tripped between moves.
pub struct FailOnNthEmbedder<E> {
    inner: E,
    nth: usize,
    delay: Option<Duration>,
    calls: AtomicUsize,
}

impl<E> FailOnNthEmbedder<E> {
    pub fn new(inner: E, nth: usize) -> Self {
        Self {
            inner,
            nth,
            delay: None,
            calls: AtomicUsize::new(0),
        }
    }

    pub fn slow(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }
}

#[async_trait]
impl<E: Embedder> Embedder for FailOnNthEmbedder<E> {
    fn dims(&self) -> usize {
        self.inner.dims()
    }

    async fn embed(&self, text: &str, kind: TextKind) -> liam_model::Result<Vec<f32>> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(delay) = self.delay {
            tokio::time::sleep(delay).await;
        }
        if call == self.nth {
            return Err(ModelError::Embed("injected failure".to_string()));
        }
        self.inner.embed(text, kind).await
    }
}

/// Judges every node `Verified` unless told otherwise, and counts its calls.
#[derive(Default)]
pub struct StubVerifier {
    verdicts: Mutex<HashMap<NodeId, OriginVerdict>>,
    calls: AtomicUsize,
    claims_seen: AtomicUsize,
}

impl StubVerifier {
    pub fn judging(&self, node: &NodeId, verdict: OriginVerdict) {
        self.verdicts.lock().unwrap().insert(node.clone(), verdict);
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    pub fn claims_seen(&self) -> usize {
        self.claims_seen.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl TierVerifier for StubVerifier {
    async fn verify(
        &self,
        claims: &[OriginClaim],
    ) -> liam_store::Result<HashMap<NodeId, OriginVerdict>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.claims_seen.fetch_add(claims.len(), Ordering::SeqCst);
        let set = self.verdicts.lock().unwrap();
        Ok(claims
            .iter()
            .map(|claim| {
                let verdict = set
                    .get(&claim.node_id)
                    .copied()
                    .unwrap_or(OriginVerdict::Verified);
                (claim.node_id.clone(), verdict)
            })
            .collect())
    }
}

/// Free memory a test sets.
pub struct FixedRam(pub AtomicU64);

impl FixedRam {
    pub fn plenty() -> Self {
        Self(AtomicU64::new(u64::MAX))
    }
}

impl FreeRam for FixedRam {
    fn free_mb(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

// ---- a store observed for 45 days, and a hook over it ----

pub const DIMS: usize = 8;
pub const DAY_MS: i64 = 86_400_000;
pub const START: Millis = Millis(10 * DAY_MS);
pub const WINDOW_DAYS: u32 = 30;
pub const HITS_CAP: u32 = 2;
pub const SOURCE: &str = "consolidation:run-1";
pub const RUN: &str = "run-1";

pub struct Fixture {
    pub store: Arc<DefaultGraph>,
    pub log: SharedLog,
    pub clock: Arc<FixedClock>,
    _dir: tempfile::TempDir,
}

impl Fixture {
    /// A fixture over a store and log the test opened itself under `dir`.
    pub fn over(
        store: Arc<DefaultGraph>,
        log: SharedLog,
        clock: Arc<FixedClock>,
        dir: tempfile::TempDir,
    ) -> Self {
        Self {
            store,
            log,
            clock,
            _dir: dir,
        }
    }
}

/// A store whose counting began at `START`, read 45 days later: long enough for
/// every item to be judged.
pub async fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(FixedClock::new(START));
    let (store, log) = logged_store_with_log(dir.path(), clock.clone(), DIMS).await;
    Fixture {
        store: Arc::new(store),
        log,
        clock,
        _dir: dir,
    }
}

pub fn config() -> TieringConfig {
    TieringConfig {
        enabled: true,
        w1: 1.0,
        w2: 0.0,
        w3: 0.0,
        w4: 0.0,
        hits_cap: HITS_CAP,
        window_days: WINDOW_DAYS,
        max_demotions_per_run: 10,
        max_promotions_per_run: 10,
        ..TieringConfig::default()
    }
}

pub fn text_of(label: &str) -> String {
    format!("text of {label}")
}

pub async fn add(fixture: &Fixture, label: &str, pattern: u8) -> (NodeId, LineageKey) {
    let id = fixture
        .store
        .insert(
            NewNode::now("fact", label, text_of(label))
                .with_subject(label)
                .with_producer("agent-a")
                .with_embedding(signed(pattern, DIMS)),
        )
        .await
        .unwrap();
    let lineage = fixture.store.lineage_of(&id).await.unwrap().unwrap();
    (id, lineage)
}

pub async fn demote(fixture: &Fixture, lineage: &LineageKey) {
    let change = TierMove {
        lineage_key: lineage.clone(),
        to: Tier::Cold,
        run_id: "earlier".to_string(),
        reason: "test".to_string(),
        changed_at: START,
        residency_until: None,
    };
    fixture
        .store
        .demote_lineage(&change, "consolidation:earlier")
        .await
        .unwrap();
}

/// Starts counting at `START`, then moves the clock 45 days on, where the items
/// in `used` have been hit enough to count fully.
pub async fn observed(fixture: &Fixture, used: &[&LineageKey]) {
    fixture
        .store
        .put_usage_logged(START.day_number(), &[])
        .await
        .unwrap();
    let later = Millis(START.0 + 45 * DAY_MS);
    fixture.clock.set(later);
    let entries: Vec<UsageEntry> = used
        .iter()
        .map(|lineage| UsageEntry {
            lineage_key: (*lineage).clone(),
            hits: HITS_CAP,
            last_access_at: later,
        })
        .collect();
    fixture
        .store
        .put_usage_logged(later.day_number(), &entries)
        .await
        .unwrap();
}

pub fn hooks(fixture: &Fixture, config: TieringConfig, embedder: Arc<dyn Embedder>) -> TierHooks {
    TierHooks {
        config,
        extraction: ExtractionConfig::default(),
        embedder,
        permits: Arc::new(Semaphore::new(1)),
        free_ram: Arc::new(FixedRam::plenty()),
        clock: fixture.clock.clone(),
    }
}

pub fn ctx<'a>(fixture: &'a Fixture, probed: &[&NodeId], history: TierHistory) -> TierCtx<'a> {
    TierCtx {
        store: &fixture.store,
        source: SOURCE.to_string(),
        run_id: RUN.to_string(),
        probed_nodes: probed.iter().map(|id| (*id).clone()).collect(),
        history,
    }
}

pub fn scripted(labels: &[&str]) -> Arc<ScriptedEmbedder> {
    Arc::new(scripted_owned(labels))
}

pub fn scripted_owned(labels: &[&str]) -> ScriptedEmbedder {
    let vectors: Vec<(String, Vec<f32>)> = labels
        .iter()
        .map(|label| (text_of(label), signed(0b0101_0101, DIMS)))
        .collect();
    let named: Vec<(&str, Vec<f32>)> = vectors
        .iter()
        .map(|(text, vector)| (text.as_str(), vector.clone()))
        .collect();
    ScriptedEmbedder::new(DIMS, &named)
}

pub async fn tier_of(fixture: &Fixture, lineage: &LineageKey) -> Tier {
    fixture.store.tier_of(lineage).await.unwrap()
}

pub async fn holds_vector(fixture: &Fixture, lineage: &LineageKey) -> bool {
    let texts = fixture.store.lineage_texts(lineage).await.unwrap();
    texts.iter().all(|text| text.has_vector)
}
