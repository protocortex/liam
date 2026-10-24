// SPDX-License-Identifier: Apache-2.0
//! The extraction part inside whole runs: a model double on real database and
//! log files, driven through `consolidate()`.

use std::collections::BTreeSet;
use std::time::Duration;

use liam_daemon::consolidation::hooks::ConsolidationHookSet;
use liam_log::event::LogPayload;
use liam_log::reader::{LogReader, SequentialScanReader};
use liam_log::LogOffset;
use liam_store::{ExtractorName, Fact, RebuildMode, FOREVER};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::config::ExtractionConfig;
use crate::extraction::cache::{cache_path, CacheStore};
use crate::extraction::hooks::ExtractionHooks;
use crate::extraction::model_double::{employment, ProviderDouble, MODEL_ID, MODEL_SHA};
use crate::extraction::prompt::PROMPT_VERSION;
use crate::extraction::sidecar::FreeRam;

const CLIENT: &str = "claude-code";
const PEOPLE: [&str; 5] = ["ada", "bob", "cy", "di", "ed"];
const ORG: &str = "Acme";
const SHORT_EVENT_CHARS: usize = 5;
const FACT_LIMIT: usize = 10;
/// Far past any record the tests write.
const BUMPED_INDEX: u64 = 1_000_000;
const ALL_LIVE: Millis = Millis(FOREVER.0 - 1);
/// A deadlock must fail a test, not hang it.
const DEADLINE: Duration = Duration::from_secs(60);

struct PlentyOfRam;

impl FreeRam for PlentyOfRam {
    fn free_mb(&self) -> u64 {
        u64::MAX
    }
}

/// Probes honestly, except that after the baseline it writes a client note and
/// scores a total loss, so the run is rolled back while a client wrote.
struct ClientWritesThenRegresses {
    calls: AtomicUsize,
}

#[async_trait]
impl ProbeRunner for ClientWritesThenRegresses {
    async fn run(&self, store: &DefaultGraph, probes: &[Probe]) -> Result<ProbeReport, ProbeError> {
        let mut report = run_probes(store, probes).await?;
        if self.calls.fetch_add(1, Ordering::SeqCst) > 0 {
            store
                .insert(client_note("written during the run"))
                .await
                .unwrap();
            report.scores.values_mut().for_each(|s| s.recall = 0.0);
        }
        Ok(report)
    }
}

fn done(manifest: &RunManifest, step: &str) -> bool {
    manifest.completed_steps.iter().any(|done| done == step)
}

fn client_note(text: &str) -> NewNode {
    NewNode::now("fact", text, "").with_producer(CLIENT)
}

fn employment_note(person: &str) -> NewNode {
    client_note(&employment(person, ORG))
}

/// The extraction part over the harness's files and the given model.
fn extraction_part(
    harness: &Harness,
    provider: &Arc<ProviderDouble>,
    config: ExtractionConfig,
) -> ExtractionHooks {
    let log_dir = log_dir_in(&harness.root);
    ExtractionHooks {
        config: ExtractionConfig {
            enabled: true,
            min_event_chars: SHORT_EVENT_CHARS,
            ..config
        },
        reader: Arc::new(SequentialScanReader::local(&log_dir).unwrap()),
        provider: provider.clone(),
        identity: Some(crate::extraction::model_double::identity()),
        cache: Arc::new(CacheStore::open(cache_path(&database_in(&harness.root))).unwrap()),
        permits: Arc::new(Semaphore::new(1)),
        free_ram: Arc::new(PlentyOfRam),
        stopping: harness.shutdown.token(),
    }
}

fn hooks_of(part: ExtractionHooks) -> ConsolidationHookSet {
    ConsolidationHookSet {
        extraction: Some(Arc::new(part)),
        ..Default::default()
    }
}

fn config() -> ExtractionConfig {
    ExtractionConfig {
        // Host memory must never decide a test.
        ram_budget_mb: 1,
        ..ExtractionConfig::default()
    }
}

/// A store with `PEOPLE` employment notes and one note the double reads nothing
/// from, so the run has a probe.
async fn seeded(harness: &Harness, people: usize) -> MemoryServer {
    harness.server(filled(harness, people).await, Arc::new(liam_model::MockLlm))
}

/// Like `seeded`, over a store the test also holds, which a rollback cannot take.
async fn seeded_shared(harness: &Harness, people: usize) -> (MemoryServer, Arc<DefaultGraph>) {
    let store = Arc::new(filled(harness, people).await);
    let server = harness.server_arc(store.clone(), Arc::new(liam_model::MockLlm));
    (server, store)
}

async fn filled(harness: &Harness, people: usize) -> DefaultGraph {
    let (store, _clock) = harness.store_at(Millis(1000)).await;
    let probed = NewNode::now("fact", "kept", "a fact that stays").with_producer(CLIENT);
    store.insert(probed).await.unwrap();
    for person in &PEOPLE[..people] {
        store.insert(employment_note(person)).await.unwrap();
    }
    store
}

async fn facts_of(store: &DefaultGraph) -> Vec<Fact> {
    let mut facts = Vec::new();
    for person in PEOPLE {
        let page = store
            .facts_valid_at(person, None, ALL_LIVE, FACT_LIMIT)
            .await
            .unwrap();
        facts.extend(page.facts);
    }
    facts
}

async fn cursor_of(store: &DefaultGraph) -> Option<LogOffset> {
    store.extraction_cursor(ExtractorName::Llm).await.unwrap()
}

async fn rules_cursor_of(store: &DefaultGraph) -> Option<LogOffset> {
    store.extraction_cursor(ExtractorName::Rules).await.unwrap()
}

fn cache_keys(harness: &Harness) -> BTreeSet<String> {
    let text = std::fs::read_to_string(cache_path(&database_in(&harness.root))).unwrap_or_default();
    text.lines()
        .map(|line| {
            let entry: serde_json::Value = serde_json::from_str(line).unwrap();
            entry["key"].as_str().unwrap().to_owned()
        })
        .collect()
}

/// The offset and event id of the logged node write labelled `label`.
async fn logged_note(harness: &Harness, label: &str) -> (LogOffset, String) {
    let reader = SequentialScanReader::local(&log_dir_in(&harness.root)).unwrap();
    let records: Vec<_> = futures_util::StreamExt::collect(reader.scan(None)).await;
    records
        .into_iter()
        .map(|record| record.unwrap())
        .find_map(|record| match record.event.payload {
            LogPayload::NodeWrite(row) if row.label == label => {
                Some((record.offset, record.event.event_id))
            }
            _ => None,
        })
        .expect("the note was logged")
}

async fn logged_payloads(harness: &Harness) -> Vec<LogPayload> {
    let reader = SequentialScanReader::local(&log_dir_in(&harness.root)).unwrap();
    let records: Vec<_> = futures_util::StreamExt::collect(reader.scan(None)).await;
    records
        .into_iter()
        .map(|record| record.unwrap().event.payload)
        .collect()
}

#[tokio::test]
async fn a_run_with_extraction_writes_the_facts_and_moves_the_cursor_to_the_last_event() {
    // Arrange
    let harness = Harness::new();
    let server = seeded(&harness, PEOPLE.len()).await;
    let store = server.store().await.unwrap();
    let last_event = store.last_applied_offset().await.unwrap();
    let provider = ProviderDouble::new();
    let job = harness.job_with_hooks(
        &server,
        honest(),
        None,
        hooks_of(extraction_part(&harness, &provider, config())),
    );

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    let facts = facts_of(&store).await;
    assert_eq!(facts.len(), PEOPLE.len());
    assert!(facts.iter().all(|fact| fact.object == ORG));
    assert_eq!(cursor_of(&store).await, last_event);
    let manifest = RunManifest::load(&harness.manifest_path())
        .unwrap()
        .unwrap();
    assert!(done(&manifest, "extract"));
    assert!(!done(&manifest, "extract:partial"));
    let model = manifest.extraction_model.expect("the model was recorded");
    assert_eq!(model.model_id, MODEL_ID);
    assert_eq!(model.model_sha256, MODEL_SHA);
    assert_eq!(model.prompt_version, PROMPT_VERSION);
}

#[tokio::test]
async fn a_model_pass_without_a_model_identity_fails_the_step_partial_and_starts_nothing() {
    // Arrange
    let harness = Harness::new();
    let server = seeded(&harness, PEOPLE.len()).await;
    let provider = ProviderDouble::new();
    let mut part = extraction_part(&harness, &provider, config());
    part.identity = None;
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    let manifest = RunManifest::load(&harness.manifest_path())
        .unwrap()
        .unwrap();
    assert!(done(&manifest, "extract:partial"));
    assert_eq!(provider.opens(), 0);
}

#[tokio::test]
async fn a_second_run_resumes_from_the_cursor_and_asks_the_model_nothing() {
    // Arrange
    let harness = Harness::new();
    let server = seeded(&harness, PEOPLE.len()).await;
    let provider = ProviderDouble::new();
    let job = harness.job_with_hooks(
        &server,
        honest(),
        None,
        hooks_of(extraction_part(&harness, &provider, config())),
    );
    job.tick().await;
    let calls = provider.model.calls();

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert_eq!(provider.model.calls(), calls);
    assert_eq!(provider.opens(), 1);
}

#[tokio::test]
async fn a_rolled_back_run_leaves_no_facts_and_keeps_the_cache_and_a_client_write() {
    // Arrange
    let harness = Harness::new();
    let server = seeded(&harness, PEOPLE.len()).await;
    let before = server.store().await.unwrap();
    let (facts_before, cursor_before) = (facts_of(&before).await, cursor_of(&before).await);
    drop(before);
    let cache_before = cache_keys(&harness);
    let provider = ProviderDouble::new();
    let probes = Arc::new(ClientWritesThenRegresses {
        calls: AtomicUsize::new(0),
    });
    let job = harness.job_with_hooks(
        &server,
        probes,
        None,
        hooks_of(extraction_part(&harness, &provider, config())),
    );

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::RolledBack));
    let store = server.store().await.unwrap();
    assert_eq!(facts_of(&store).await, facts_before);
    assert_eq!(cursor_of(&store).await, cursor_before);
    let cache_after = cache_keys(&harness);
    assert!(cache_after.is_superset(&cache_before));
    assert!(cache_after.len() > cache_before.len());
    assert!(logged_payloads(&harness)
        .await
        .iter()
        .any(|payload| matches!(payload, LogPayload::RunRolledBack { .. })));
    let survivors = store
        .query(&liam_store::Query::text("written during the run"))
        .await
        .unwrap();
    assert!(
        !survivors.is_empty(),
        "the client write survived the rollback"
    );
}

#[tokio::test]
async fn the_run_after_a_rollback_rebuilds_its_facts_from_the_cache_without_the_model() {
    // Arrange
    let harness = Harness::new();
    let server = seeded(&harness, PEOPLE.len()).await;
    let provider = ProviderDouble::new();
    let part = extraction_part(&harness, &provider, config());
    let rolled_back = harness.job_with_hooks(&server, regresses(), None, hooks_of(part));
    assert_eq!(rolled_back.tick().await, Some(RunOutcome::RolledBack));
    let calls = provider.model.calls();
    let part = extraction_part(&harness, &provider, config());
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert_eq!(provider.model.calls(), calls);
    let store = server.store().await.unwrap();
    assert_eq!(facts_of(&store).await.len(), PEOPLE.len());
}

#[tokio::test]
async fn a_rolled_back_runs_facts_stay_gone_after_a_rebuild() {
    // Arrange
    let harness = Harness::new();
    let server = seeded(&harness, PEOPLE.len()).await;
    let provider = ProviderDouble::new();
    let part = extraction_part(&harness, &provider, config());
    let job = harness.job_with_hooks(&server, regresses(), None, hooks_of(part));
    assert_eq!(job.tick().await, Some(RunOutcome::RolledBack));
    let fresh = harness.root.join("rebuilt.db");
    let empty = DefaultGraph::open(fresh.to_str().unwrap(), GraphConfig::new(DIMS))
        .await
        .unwrap();

    // Act
    let (rebuilt, _report) = empty
        .rebuild_from_log(harness.log.clone(), RebuildMode::RequireEmpty)
        .await
        .unwrap();

    // Assert
    assert!(facts_of(&rebuilt).await.is_empty());
}

#[tokio::test]
async fn a_window_the_cache_already_holds_never_starts_the_model() {
    // Arrange
    let harness = Harness::new();
    let server = seeded(&harness, PEOPLE.len()).await;
    let provider = ProviderDouble::new();
    let part = extraction_part(&harness, &provider, config());
    let rolled_back = harness.job_with_hooks(&server, regresses(), None, hooks_of(part));
    rolled_back.tick().await;
    let opens = provider.opens();
    let part = extraction_part(&harness, &provider, config());
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));

    // Act
    job.tick().await;

    // Assert
    assert_eq!(provider.opens(), opens);
}

#[tokio::test]
async fn a_single_generation_permit_does_not_deadlock_the_run() {
    // Arrange
    let harness = Harness::new();
    let server = seeded(&harness, PEOPLE.len()).await;
    let provider = ProviderDouble::new();
    let part = extraction_part(&harness, &provider, config());
    let permits = part.permits.clone();
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));

    // Act
    let outcome = tokio::time::timeout(DEADLINE, job.tick()).await.unwrap();

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    let store = server.store().await.unwrap();
    assert_eq!(facts_of(&store).await.len(), PEOPLE.len());
    assert_eq!(permits.available_permits(), 1);
}

#[tokio::test]
async fn a_model_that_cannot_start_makes_the_step_partial_and_the_run_still_commits() {
    // Arrange
    let harness = Harness::new();
    let server = seeded(&harness, PEOPLE.len()).await;
    let provider = ProviderDouble::new();
    provider.fail_open();
    let job = harness.job_with_hooks(
        &server,
        honest(),
        None,
        hooks_of(extraction_part(&harness, &provider, config())),
    );

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    let manifest = RunManifest::load(&harness.manifest_path())
        .unwrap()
        .unwrap();
    assert!(done(&manifest, "extract:partial"));
    let store = server.store().await.unwrap();
    assert!(facts_of(&store).await.is_empty());
    assert_eq!(cursor_of(&store).await, None);
}

#[tokio::test]
async fn a_model_lost_mid_run_keeps_the_batches_it_finished() {
    // Arrange: one event per batch, the process gone after the second answer
    let harness = Harness::new();
    let server = seeded(&harness, 3).await;
    let provider = ProviderDouble::dying_on_call(3);
    let part = extraction_part(
        &harness,
        &provider,
        ExtractionConfig {
            batch_events: 1,
            ..config()
        },
    );
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));

    // Act
    let outcome = job.tick().await;

    // Assert: the note is batch one, ada and bob are two and three, cy is lost
    assert_eq!(outcome, Some(RunOutcome::Committed));
    let manifest = RunManifest::load(&harness.manifest_path())
        .unwrap()
        .unwrap();
    assert!(done(&manifest, "extract:partial"));
    let store = server.store().await.unwrap();
    assert_eq!(facts_of(&store).await.len(), 2);
    let (bob, _) = logged_note(&harness, &employment("bob", ORG)).await;
    assert_eq!(cursor_of(&store).await, Some(bob));
}

#[tokio::test]
async fn disabled_extraction_is_a_no_op() {
    // Arrange
    let harness = Harness::new();
    let server = seeded(&harness, PEOPLE.len()).await;
    let provider = ProviderDouble::new();
    let mut part = extraction_part(&harness, &provider, config());
    part.config.enabled = false;
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert_eq!(provider.opens(), 0);
    let store = server.store().await.unwrap();
    assert_eq!(cursor_of(&store).await, None);
    assert_eq!(rules_cursor_of(&store).await, None);
    assert!(facts_of(&store).await.is_empty());
}

#[tokio::test]
async fn a_window_is_capped_at_the_events_per_run_and_the_cursor_stops_there() {
    // Arrange: the probe note and three employment notes, two events a run
    let harness = Harness::new();
    let server = seeded(&harness, 3).await;
    let provider = ProviderDouble::new();
    let part = extraction_part(
        &harness,
        &provider,
        ExtractionConfig {
            max_events_per_run: 2,
            ..config()
        },
    );
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));

    // Act
    job.tick().await;

    // Assert: the note and ada were read first, then bob and cy
    let store = server.store().await.unwrap();
    assert_eq!(facts_of(&store).await.len(), 1);
    job.tick().await;
    assert_eq!(facts_of(&store).await.len(), 3);
}

#[tokio::test]
async fn a_daemon_that_begins_stopping_mid_pass_keeps_the_log_writable_and_the_next_run_goes_on() {
    // Arrange: the daemon starts stopping during the second answer
    let harness = Harness::new();
    let server = seeded(&harness, 3).await;
    let stop = CancellationToken::new();
    let provider = ProviderDouble::stopping_on_call(2, stop.clone());
    let slow = ExtractionConfig {
        batch_events: 1,
        ..config()
    };
    let mut part = extraction_part(&harness, &provider, slow.clone());
    part.stopping = stop;
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));

    // Act
    job.tick().await;

    // Assert: the first batch is kept, the log still takes writes, and the
    // next run finishes the window
    let manifest = RunManifest::load(&harness.manifest_path())
        .unwrap()
        .unwrap();
    assert!(done(&manifest, "extract:partial"));
    let store = server.store().await.unwrap();
    let (first, _) = logged_note(&harness, "kept").await;
    assert_eq!(cursor_of(&store).await, Some(first));
    store
        .insert(client_note("written after the stop"))
        .await
        .expect("the log is not poisoned");
    drop(store);
    let next = harness.job_with_hooks(
        &server,
        honest(),
        None,
        hooks_of(extraction_part(&harness, &provider, slow)),
    );
    assert_eq!(next.tick().await, Some(RunOutcome::Committed));
    let store = server.store().await.unwrap();
    assert_eq!(facts_of(&store).await.len(), 3);
}

#[tokio::test]
async fn a_run_deadline_far_past_any_clock_still_runs() {
    // Arrange: a hand built config, which validation would have refused
    let harness = Harness::new();
    let server = seeded(&harness, 2).await;
    let provider = ProviderDouble::new();
    let part = extraction_part(
        &harness,
        &provider,
        ExtractionConfig {
            run_deadline_secs: u64::MAX,
            ..config()
        },
    );
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    let store = server.store().await.unwrap();
    assert_eq!(facts_of(&store).await.len(), 2);
}

/// Reads the log as usual, except that on one scan it first moves a cursor
/// ahead of anything the run will finish, as another writer of the cursor would.
struct CursorBumpingReader {
    inner: SequentialScanReader,
    store: Arc<DefaultGraph>,
    cursor: ExtractorName,
    to: LogOffset,
    /// Which scan of the run bumps, counted from 0.
    on_scan: usize,
    scans: AtomicUsize,
}

/// The rules pass scans first and the model pass second.
const RULES_PASS_SCAN: usize = 0;
const MODEL_PASS_SCAN: usize = 1;

impl LogReader for CursorBumpingReader {
    fn scan(&self, from: Option<LogOffset>) -> liam_log::reader::LogStream {
        self.inner.scan(from)
    }

    fn scan_through(
        &self,
        from: Option<LogOffset>,
        through: LogOffset,
    ) -> liam_log::reader::LogStream {
        let (store, to, inner) = (self.store.clone(), self.to, self.inner.clone());
        let cursor = self.cursor;
        let bumps = self.scans.fetch_add(1, Ordering::SeqCst) == self.on_scan;
        let bump_then_read = async move {
            if bumps {
                store.advance_extraction_cursor(cursor, to).await.unwrap();
            }
            inner.scan_through(from, through)
        };
        Box::pin(futures_util::StreamExt::flatten(
            futures_util::stream::once(bump_then_read),
        ))
    }
}

#[tokio::test]
async fn a_cursor_moved_by_another_writer_makes_the_step_partial_and_the_next_run_asks_nothing() {
    // Arrange
    let harness = Harness::new();
    let (server, store) = seeded_shared(&harness, PEOPLE.len()).await;
    let ahead = LogOffset {
        segment: 0,
        index: BUMPED_INDEX,
    };
    let provider = ProviderDouble::new();
    let mut part = extraction_part(&harness, &provider, config());
    part.reader = Arc::new(CursorBumpingReader {
        inner: SequentialScanReader::local(&log_dir_in(&harness.root)).unwrap(),
        store: store.clone(),
        cursor: ExtractorName::Llm,
        to: ahead,
        on_scan: MODEL_PASS_SCAN,
        scans: AtomicUsize::new(0),
    });
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    let manifest = RunManifest::load(&harness.manifest_path())
        .unwrap()
        .unwrap();
    assert!(done(&manifest, "extract:partial"));
    assert_eq!(facts_of(&store).await.len(), PEOPLE.len());
    assert_eq!(cursor_of(&store).await, Some(ahead));
    let calls = provider.model.calls();
    let part = extraction_part(&harness, &provider, config());
    let next = harness.job_with_hooks(&server, honest(), None, hooks_of(part));
    next.tick().await;
    assert_eq!(provider.model.calls(), calls);
}

mod restore;
mod rules;
