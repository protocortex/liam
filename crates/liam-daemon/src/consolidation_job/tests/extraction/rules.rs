// SPDX-License-Identifier: Apache-2.0
//! The rules pass inside whole runs: it needs no model, and the model pass
//! failing or being absent must not stall it.

use liam_daemon::consolidation::hooks::{ExtractCtx, ExtractionHook};
use liam_store::relation;

use super::*;

const MENTIONED_IN: &str = "mentioned_in";
const LONGER_THAN_ANY_EVENT: usize = 10_000;

/// The facts the rules wrote about the entity page for Ada.
async fn mention_facts(store: &DefaultGraph) -> Vec<Fact> {
    let page = store
        .facts_valid_at("ada", None, ALL_LIVE, FACT_LIMIT)
        .await
        .unwrap();
    page.facts
        .into_iter()
        .filter(|fact| fact.predicate == MENTIONED_IN && fact.extractor_id == ExtractorName::Rules)
        .collect()
}

/// A store with an entity page for Ada that mentions the first note, and
/// `mentions` notes that name her.
async fn ada_store(harness: &Harness, mentions: usize) -> DefaultGraph {
    let store = filled(harness, 0).await;
    let ada = store
        .insert(NewNode::entity("person", "Ada"))
        .await
        .unwrap();
    for n in 1..=mentions {
        let note = client_note(&format!("Ada reviewed item {n} today"));
        store.insert(note).await.unwrap();
    }
    let kept = store.query(&liam_store::Query::text("kept")).await.unwrap();
    let kept = kept.first().expect("the kept note").id.clone();
    store
        .link(NewEdge::new(&ada, &kept, relation::MENTIONS))
        .await
        .unwrap();
    store
}

async fn with_ada(harness: &Harness, mentions: usize) -> MemoryServer {
    let store = ada_store(harness, mentions).await;
    harness.server(store, Arc::new(liam_model::MockLlm))
}

fn ctx_of<'a>(store: &'a DefaultGraph, run: &str) -> ExtractCtx<'a> {
    ExtractCtx {
        store,
        source: format!("consolidation:{run}"),
    }
}

/// Moves the Rules cursor ahead of the run during the rules pass's scan.
fn rules_cursor_bumped(harness: &Harness, store: &Arc<DefaultGraph>) -> CursorBumpingReader {
    CursorBumpingReader {
        inner: SequentialScanReader::local(&log_dir_in(&harness.root)).unwrap(),
        store: store.clone(),
        cursor: ExtractorName::Rules,
        to: LogOffset {
            segment: 0,
            index: BUMPED_INDEX,
        },
        on_scan: RULES_PASS_SCAN,
        scans: AtomicUsize::new(0),
    }
}

/// Fails the test if a pass reads the log without a bound.
struct BoundedScansOnly(SequentialScanReader);

impl LogReader for BoundedScansOnly {
    fn scan(&self, _from: Option<LogOffset>) -> liam_log::reader::LogStream {
        panic!("a pass must stop at the last event the store applied");
    }

    fn scan_through(
        &self,
        from: Option<LogOffset>,
        through: LogOffset,
    ) -> liam_log::reader::LogStream {
        self.0.scan_through(from, through)
    }
}

#[tokio::test]
async fn rules_only_extraction_writes_mention_facts_and_starts_no_model() {
    // Arrange
    let harness = Harness::new();
    let server = with_ada(&harness, 2).await;
    let store = server.store().await.unwrap();
    let last_event = store.last_applied_offset().await.unwrap();
    let provider = ProviderDouble::new();
    let part = extraction_part(
        &harness,
        &provider,
        ExtractionConfig {
            llm_enabled: false,
            ..config()
        },
    );
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert_eq!(mention_facts(&store).await.len(), 2);
    assert_eq!(rules_cursor_of(&store).await, last_event);
    assert_eq!(cursor_of(&store).await, None);
    assert_eq!(provider.opens(), 0);
}

#[tokio::test]
async fn a_model_that_never_starts_does_not_stall_the_rules_over_three_runs() {
    // Arrange: two events a run, so the later notes are only in later windows
    let harness = Harness::new();
    let server = with_ada(&harness, 3).await;
    let provider = ProviderDouble::new();
    provider.fail_open();
    let part = extraction_part(
        &harness,
        &provider,
        ExtractionConfig {
            max_events_per_run: 2,
            ..config()
        },
    );
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));
    let store = server.store().await.unwrap();

    // Act
    let mut cursors = Vec::new();
    for _ in 0..3 {
        job.tick().await;
        cursors.push(rules_cursor_of(&store).await);
    }

    // Assert
    assert!(cursors.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(mention_facts(&store).await.len(), 3);
    let facts = facts_of(&store).await;
    assert!(facts
        .iter()
        .all(|fact| fact.extractor_id == ExtractorName::Rules));
    let manifest = RunManifest::load(&harness.manifest_path())
        .unwrap()
        .unwrap();
    assert!(done(&manifest, "extract:partial"));
}

#[tokio::test]
async fn the_rules_facts_of_a_rolled_back_run_are_gone_and_the_next_run_writes_them_again() {
    // Arrange
    let harness = Harness::new();
    let server = with_ada(&harness, 2).await;
    let provider = ProviderDouble::new();
    let part = extraction_part(&harness, &provider, config());
    let regressing = harness.job_with_hooks(&server, regresses(), None, hooks_of(part));

    // Act
    let outcome = regressing.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::RolledBack));
    let store = server.store().await.unwrap();
    assert!(mention_facts(&store).await.is_empty());
    assert_eq!(rules_cursor_of(&store).await, None);
    drop(store);
    let part = extraction_part(&harness, &provider, config());
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));
    job.tick().await;
    let store = server.store().await.unwrap();
    assert_eq!(mention_facts(&store).await.len(), 2);
}

#[tokio::test]
async fn a_rules_cursor_moved_by_another_writer_makes_the_step_partial_and_keeps_the_facts() {
    // Arrange
    let harness = Harness::new();
    let store = Arc::new(ada_store(&harness, 2).await);
    let server = harness.server_arc(store.clone(), Arc::new(liam_model::MockLlm));
    let provider = ProviderDouble::new();
    let mut part = extraction_part(&harness, &provider, config());
    part.reader = Arc::new(rules_cursor_bumped(&harness, &store));
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    let manifest = RunManifest::load(&harness.manifest_path())
        .unwrap()
        .unwrap();
    assert!(done(&manifest, "extract:partial"));
    assert_eq!(mention_facts(&store).await.len(), 2);
    assert_eq!(
        rules_cursor_of(&store).await,
        Some(LogOffset {
            segment: 0,
            index: BUMPED_INDEX
        })
    );
}

#[tokio::test]
async fn a_failed_rules_pass_is_reported_as_partial() {
    // Arrange
    let harness = Harness::new();
    let store = Arc::new(ada_store(&harness, 1).await);
    let provider = ProviderDouble::new();
    let mut part = extraction_part(&harness, &provider, config());
    part.reader = Arc::new(rules_cursor_bumped(&harness, &store));

    // Act
    let report = part.extract(&ctx_of(&store, "r1")).await.unwrap();

    // Assert
    assert!(report.partial);
}

#[tokio::test(start_paused = true)]
async fn a_model_pass_that_outlasts_the_run_deadline_makes_the_step_partial() {
    // Arrange: the second answer would take an hour, the run has a minute
    let harness = Harness::new();
    let server = seeded(&harness, 3).await;
    let provider = ProviderDouble::slow_on_call(2, Duration::from_secs(3600));
    let part = extraction_part(
        &harness,
        &provider,
        ExtractionConfig {
            batch_events: 1,
            run_deadline_secs: 60,
            ..config()
        },
    );
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    let manifest = RunManifest::load(&harness.manifest_path())
        .unwrap()
        .unwrap();
    assert!(done(&manifest, "extract:partial"));
}

#[tokio::test]
async fn a_window_with_nothing_worth_reading_still_moves_both_cursors() {
    // Arrange: every event is shorter than the least text worth reading
    let harness = Harness::new();
    let server = seeded(&harness, 0).await;
    let store = server.store().await.unwrap();
    let last_event = store.last_applied_offset().await.unwrap();
    let provider = ProviderDouble::new();
    let mut part = extraction_part(&harness, &provider, config());
    part.config.min_event_chars = LONGER_THAN_ANY_EVENT;
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert!(last_event.is_some());
    assert_eq!(cursor_of(&store).await, last_event);
    assert_eq!(rules_cursor_of(&store).await, last_event);
    assert_eq!(provider.opens(), 0);
}

#[tokio::test]
async fn both_passes_read_only_up_to_the_last_event_the_store_applied() {
    // Arrange
    let harness = Harness::new();
    let store = ada_store(&harness, 1).await;
    let provider = ProviderDouble::new();
    let mut part = extraction_part(&harness, &provider, config());
    part.reader = Arc::new(BoundedScansOnly(
        SequentialScanReader::local(&log_dir_in(&harness.root)).unwrap(),
    ));

    // Act
    let report = part.extract(&ctx_of(&store, "r1")).await.unwrap();

    // Assert
    assert!(!report.partial);
    assert_eq!(mention_facts(&store).await.len(), 1);
}

#[tokio::test]
async fn a_rebuilt_store_run_again_asserts_the_same_facts_without_asking_the_model() {
    // Arrange: a committed run, then a store rebuilt from the log alone
    let harness = Harness::new();
    let server = with_ada(&harness, 1).await;
    let provider = ProviderDouble::new();
    let job = harness.job_with_hooks(
        &server,
        honest(),
        None,
        hooks_of(extraction_part(&harness, &provider, config())),
    );
    assert_eq!(job.tick().await, Some(RunOutcome::Committed));
    let calls = provider.model.calls();
    let store = server.store().await.unwrap();
    let before = triples(&store).await;
    assert!(!before.is_empty());
    let fresh = harness.root.join("rebuilt.db");
    let empty = DefaultGraph::open(fresh.to_str().unwrap(), GraphConfig::new(DIMS))
        .await
        .unwrap();
    let (rebuilt, _report) = empty
        .rebuild_from_log(harness.log.clone(), RebuildMode::RequireEmpty)
        .await
        .unwrap();
    let part = extraction_part(&harness, &provider, config());

    // Act
    let report = part.extract(&ctx_of(&rebuilt, "again")).await.unwrap();

    // Assert
    assert!(!report.partial);
    assert_eq!(triples(&rebuilt).await, before);
    assert_eq!(provider.model.calls(), calls);
}

/// Subject, predicate and object of every live fact the run could have written.
async fn triples(store: &DefaultGraph) -> Vec<(String, String, String)> {
    let mut all = facts_of(store).await;
    all.extend(mention_facts(store).await);
    let mut triples: Vec<_> = all
        .into_iter()
        .map(|fact| (fact.subject, fact.predicate, fact.object))
        .collect();
    triples.sort();
    triples.dedup();
    triples
}
