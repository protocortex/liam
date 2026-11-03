// SPDX-License-Identifier: Apache-2.0
//! A sealed store with a canary planted in every store and file class that can
//! hold content, for alice, and a second canary for bob that must survive.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use liam_daemon::consolidation::manifest::{manifest_path_for, RunManifest};
use liam_daemon::consolidation::tier_history::{self, history_path_for, Cooldown, TierHistory};
use liam_daemon::embedder_fingerprint::EmbedderFingerprint;
use liam_daemon::event_log;
use liam_daemon::models::resolve_snapshot_dir;
use liam_store::{
    ArbitrationCaps, ChildKind, EventRef, EventRole, ExtractorName, FixedClock, LineageKey, Millis,
    NewEdge, NewFact, NewNode, NewSummary, NodeId, PredicateSpec, Tier, TierMove, UsageEntry,
    FOREVER, RUN_SOURCE_PREFIX,
};
use liam_store::{Clock, SummaryChild};
use tempfile::TempDir;

use crate::deletion::batch_lineage::BatchCanceller;
use crate::deletion::batch_store::{batch_store_path_for, BatchEntry, BatchState, BatchStore};
use crate::deletion::memory::NodeState;
use crate::deletion::send_ledger::{send_ledger_path_for, SendLedger, SendRow};
use crate::deletion::service::{DeleteService, StoreParts};
use crate::deletion::target::DeleteTarget;
use crate::deletion::test_support::{sealed_subjects_with, Subjects};
use crate::extraction::cache::{cache_path, CacheStore, NewEntry};
use crate::log_startup::test_support::{database_in, log_dir_in, DIMS};
use crate::metrics::alerts::state_path_for;
use crate::storelock::StoreLock;
use crate::tiering::usage::UsageRecorder;

/// One alphanumeric token, so a text index keeps it whole.
pub const CANARY: &str = "zqxv7413canary";
/// In the attribute of an edge from alice's note to bob's.
pub const EDGE_CANARY: &str = "yqnc3307edgecanary";
/// In what a run wrote before it was rolled back.
pub const ROLLED_BACK_CANARY: &str = "hsvt9042rolledcanary";
/// Bob's content, which a delete of alice must leave readable.
pub const BOB_CANARY: &str = "kdwm2958survivor";
const ROLLED_BACK_RUN: &str = "run-rolled-back";
pub const HOSTED_PROVIDER: &str = "anthropic";
pub const HOSTED_BATCH: &str = "msgbatch_pending";
pub const ALICE_REQUEST: &str = "request-alice";
pub const BOB_REQUEST: &str = "request-bob";
const NOW: Millis = Millis(1_000);
const DAY_NUMBER: i64 = 1;
const USAGE_CAPACITY: usize = 16;
const TRUST: f64 = 0.9;
const COOLDOWN_RUNS: u32 = 3;
const SEED: u64 = 7;
const SMALL_SEGMENT_BYTES: u64 = 1;

/// A directory with the planted store and the paths around it.
pub struct Fixture {
    pub dir: TempDir,
    pub subjects: Subjects,
    pub lock: StoreLock,
    pub clock: Arc<FixedClock>,
    pub usage: Arc<UsageRecorder>,
    pub alice_note: NodeId,
    pub bob_note: NodeId,
    /// Alice's lineage key, the text the state files hold.
    pub alice_lineage: LineageKey,
    /// The event that wrote alice's note, which the hosted batch files name.
    pub alice_event: String,
}

pub fn alice() -> DeleteTarget {
    DeleteTarget::Subject {
        subject: "alice".into(),
        scope: None,
        include_children: false,
    }
}

impl Fixture {
    /// Plants alice's canary everywhere and bob's survivor beside it.
    pub async fn new() -> Self {
        Self::with_segment_bytes(None).await
    }

    /// As `new`, with every log event closing its segment, so a later archive
    /// pass has closed partitions to move.
    pub async fn with_closed_segments() -> Self {
        Self::with_segment_bytes(Some(SMALL_SEGMENT_BYTES)).await
    }

    async fn with_segment_bytes(segment_bytes: Option<u64>) -> Self {
        let dir = TempDir::new().unwrap();
        let subjects = sealed_subjects_with(dir.path(), |config| {
            if let Some(bytes) = segment_bytes {
                config.log.wal_segment_max_bytes = bytes;
            }
        })
        .await;
        let lock = StoreLock::acquire(&database_in(dir.path())).unwrap();
        let clock = Arc::new(FixedClock::new(NOW));
        let usage = Arc::new(UsageRecorder::new(USAGE_CAPACITY));
        let mut fixture = Self {
            dir,
            lock,
            clock,
            usage,
            alice_note: subjects.alice.clone(),
            bob_note: subjects.alice.clone(),
            alice_lineage: LineageKey::from_raw(""),
            alice_event: String::new(),
            subjects,
        };
        fixture.plant().await;
        fixture
    }

    pub fn database(&self) -> PathBuf {
        database_in(self.dir.path())
    }

    pub fn log_dir(&self) -> PathBuf {
        log_dir_in(self.dir.path())
    }

    pub fn snapshot_dir(&self) -> PathBuf {
        resolve_snapshot_dir(&self.subjects.config, &self.database()).unwrap()
    }

    pub fn archive_dir(&self) -> PathBuf {
        self.dir.path().join("archive")
    }

    pub fn cache_file(&self) -> PathBuf {
        cache_path(&self.database())
    }

    pub fn manifest_file(&self) -> PathBuf {
        manifest_path_for(&self.database())
    }

    pub fn cooldown_file(&self) -> PathBuf {
        history_path_for(&self.database())
    }

    pub fn metrics_state_file(&self) -> PathBuf {
        state_path_for(&self.database())
    }

    pub fn service(&self) -> DeleteService {
        self.service_with(None)
    }

    /// The service with `canceller` reaching the hosted providers, or the one
    /// the config gives when there is none.
    pub fn service_with(&self, canceller: Option<Arc<dyn BatchCanceller>>) -> DeleteService {
        let reader = event_log::log_reader(&self.log_dir(), Some(&self.subjects.keys)).unwrap();
        let states: Vec<Arc<dyn NodeState>> = vec![self.usage.clone()];
        let parts = StoreParts {
            store: Arc::clone(&self.subjects.store),
            keys: Arc::clone(&self.subjects.keys),
            reader,
            clock: Arc::clone(&self.clock) as Arc<dyn Clock>,
        };
        match canceller {
            Some(canceller) => DeleteService::open_with(
                &self.subjects.config,
                &self.database(),
                parts,
                states,
                canceller,
                &self.lock,
            ),
            None => DeleteService::open(
                &self.subjects.config,
                &self.database(),
                parts,
                states,
                &self.lock,
            ),
        }
        .unwrap()
    }

    /// The ids of the log events that wrote `node`.
    pub async fn origin_events(&self, node: &NodeId) -> Vec<String> {
        self.subjects
            .store
            .node_event_ids(node, EventRole::Origin)
            .await
            .unwrap()
            .into_iter()
            .map(|event| event.event_id)
            .collect()
    }

    async fn plant(&mut self) {
        let store = Arc::clone(&self.subjects.store);
        let note = NewNode::now("fact", "note", format!("note {CANARY}")).with_subject("alice");
        self.alice_note = store.insert(note).await.unwrap();
        let episode =
            NewNode::now("episode", "meeting", format!("episode {CANARY}")).with_subject("alice");
        store.insert(episode).await.unwrap();
        let survivor =
            NewNode::now("fact", "kept", format!("kept {BOB_CANARY}")).with_subject("bob");
        self.bob_note = store.insert(survivor).await.unwrap();
        let edge = NewEdge::new(&self.alice_note, &self.bob_note, "mentions")
            .with_attributes(serde_json::json!({ "body": EDGE_CANARY }));
        store.link(edge).await.unwrap();
        self.alice_lineage = store.lineage_of(&self.alice_note).await.unwrap().unwrap();

        let alice_events = self.origin_events(&self.alice_note).await;
        let bob_events = self.origin_events(&self.bob_note).await;
        self.alice_event = alice_events[0].clone();
        self.plant_hosted_batch(&alice_events, &bob_events);
        self.plant_derived(&alice_events, &bob_events).await;
        self.plant_rolled_back_run(&alice_events, &bob_events).await;
        self.plant_rows_beside_the_database(&alice_events).await;
        store.snapshot(&self.snapshot_dir()).await.unwrap();
    }

    /// A batch still with its provider, holding a request for alice and one for
    /// bob, and the ledger rows of both sends. Neither file holds text.
    fn plant_hosted_batch(&self, alice_events: &[String], bob_events: &[String]) {
        let entry = |cache_key: &str, events: &[String]| BatchEntry {
            cache_key: cache_key.to_owned(),
            provider: HOSTED_PROVIDER.to_owned(),
            batch_id: HOSTED_BATCH.to_owned(),
            state: BatchState::Submitted,
            submitted_at: NOW.0,
            input_hash: "hash".to_owned(),
            source_event_ids: events.to_vec(),
            attempts: 1,
            failure: None,
        };
        let row = |cache_key: &str, events: &[String]| SendRow {
            provider: HOSTED_PROVIDER.to_owned(),
            cache_key: cache_key.to_owned(),
            source_event_ids: events.to_vec(),
            sent_at: NOW.0,
        };
        BatchStore::new(batch_store_path_for(&self.database()))
            .replace(vec![
                entry(ALICE_REQUEST, alice_events),
                entry(BOB_REQUEST, bob_events),
            ])
            .unwrap();
        let ledger = SendLedger::new(send_ledger_path_for(&self.database()));
        ledger.append(&row(ALICE_REQUEST, alice_events)).unwrap();
        ledger.append(&row(BOB_REQUEST, bob_events)).unwrap();
    }

    async fn plant_derived(&self, alice_events: &[String], bob_events: &[String]) {
        let store = &self.subjects.store;
        let fact = NewFact {
            object: CANARY.to_owned(),
            ..fact_of("alice", alice_events)
        };
        store
            .assert_fact(fact, &PredicateSpec::default(), "client:test", caps())
            .await
            .unwrap();
        let survivor = NewFact {
            object: BOB_CANARY.to_owned(),
            ..fact_of("bob", bob_events)
        };
        store
            .assert_fact(survivor, &PredicateSpec::default(), "client:test", caps())
            .await
            .unwrap();
        let level_one = self
            .put_summary(
                1,
                &[node_child(&self.alice_note)],
                alice_events,
                "client:test",
                CANARY,
            )
            .await;
        let ancestor = SummaryChild {
            id: level_one,
            kind: ChildKind::Summary,
        };
        self.put_summary(2, &[ancestor], alice_events, "client:test", CANARY)
            .await;
    }

    /// A summary and a fact written by a run that was then rolled back; the
    /// fact is bob's and cites an event of alice.
    async fn plant_rolled_back_run(&self, alice_events: &[String], bob_events: &[String]) {
        let store = &self.subjects.store;
        let source = format!("{RUN_SOURCE_PREFIX}{ROLLED_BACK_RUN}");
        self.put_summary(
            1,
            &[node_child(&self.bob_note)],
            alice_events,
            &source,
            ROLLED_BACK_CANARY,
        )
        .await;
        let cited: Vec<String> = alice_events.iter().chain(bob_events).cloned().collect();
        let cross = NewFact {
            predicate: "rolled".to_owned(),
            object: ROLLED_BACK_CANARY.to_owned(),
            ..fact_of("bob", &cited)
        };
        store
            .assert_fact(cross, &PredicateSpec::default(), &source, caps())
            .await
            .unwrap();
        store.mark_run_rolled_back(ROLLED_BACK_RUN).await.unwrap();
    }

    async fn put_summary(
        &self,
        level: u8,
        children: &[SummaryChild],
        events: &[String],
        source: &str,
        token: &str,
    ) -> String {
        let summary = NewSummary {
            level,
            scope: String::new(),
            text: format!("level {level} {token}"),
            events_fed: u32::try_from(events.len()).unwrap(),
            valid_min: Millis(0),
            valid_max: NOW,
            model_id: "model".to_owned(),
            prompt_version: "prompt-1".to_owned(),
        };
        let sources: Vec<EventRef> = events
            .iter()
            .map(|event_id| EventRef {
                event_id: event_id.clone(),
                synthetic: false,
            })
            .collect();
        match self
            .subjects
            .store
            .put_summary(summary, children, &sources, &[0.5; DIMS], &[], source)
            .await
            .unwrap()
        {
            liam_store::PutOutcome::Inserted(id) | liam_store::PutOutcome::Unchanged(id) => id,
        }
    }

    /// The cache entry, the usage and tier rows, and the state files.
    async fn plant_rows_beside_the_database(&self, alice_events: &[String]) {
        let cache = CacheStore::open(self.cache_file()).unwrap();
        cache
            .put(NewEntry {
                key: "alice-response",
                response: &format!("{{\"object\":\"{CANARY}\"}}"),
                model_id: "model",
                model_sha256: "sha",
                prompt_version: "prompt-1",
                source_event_ids: alice_events,
            })
            .unwrap();
        let store = &self.subjects.store;
        store
            .put_usage_logged(
                DAY_NUMBER,
                &[UsageEntry {
                    lineage_key: self.alice_lineage.clone(),
                    producer: "client:test".to_owned(),
                    hits: 2,
                    last_access_at: NOW,
                }],
            )
            .await
            .unwrap();
        let moved = TierMove {
            lineage_key: self.alice_lineage.clone(),
            to: Tier::Cold,
            run_id: "run-tier".to_owned(),
            reason: "unused".to_owned(),
            changed_at: NOW,
            residency_until: None,
        };
        store
            .put_tier_logged(&moved, "consolidation:run-tier")
            .await
            .unwrap();
        self.usage.record("client:test", [&self.alice_note], NOW);

        let embedder = EmbedderFingerprint {
            provider: "mock".to_owned(),
            model: "mock-model".to_owned(),
            dims: DIMS,
            prompt_scheme: 1,
        };
        let mut manifest =
            RunManifest::new("run-tier".to_owned(), SEED, embedder, "llm".to_owned(), NOW);
        manifest.tier_moved_lineages = vec![self.alice_lineage.as_str().to_owned()];
        manifest.save(&self.manifest_file()).unwrap();
        let history = TierHistory {
            consecutive_rollbacks: 0,
            cooldown: vec![Cooldown {
                lineage_key: self.alice_lineage.as_str().to_owned(),
                runs_left: COOLDOWN_RUNS,
            }],
        };
        tier_history::save(&self.cooldown_file(), &history).unwrap();
        std::fs::write(
            self.metrics_state_file(),
            b"{\"version\":1,\"previous\":[]}",
        )
        .unwrap();
    }
}

fn caps() -> ArbitrationCaps {
    ArbitrationCaps::default()
}

fn node_child(node: &NodeId) -> SummaryChild {
    SummaryChild {
        id: node.as_str().to_owned(),
        kind: ChildKind::Node,
    }
}

fn fact_of(subject: &str, sources: &[String]) -> NewFact {
    NewFact {
        subject: subject.to_owned(),
        predicate: "said".to_owned(),
        object: String::new(),
        scope: None,
        valid_from: Millis(0),
        valid_until: FOREVER,
        evidence_at: NOW,
        confidence: TRUST,
        trust_score: TRUST,
        extractor: ExtractorName::Rules,
        extractor_version: "1".to_owned(),
        source_event_ids: sources.to_vec(),
    }
}

/// Where `dir`'s files are, for the byte scans.
pub fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(files_under(&path));
        } else {
            found.push(path);
        }
    }
    found.sort();
    found
}
