// SPDX-License-Identifier: Apache-2.0
//! Summaries in the event log: what a write appends, and what a rebuild or a
//! replay makes of it, including the writes of a run that was rolled back.

use liam_log::event::{DerivedArtifact, LogEvent, LogPayload, ARTIFACT_KIND_SUMMARY};
use liam_log::test_support::FailingLogWriter;
use liam_log::LogWriter;

use super::log_write::{events, open_with, RecordingLog};
use super::support::{count, event, fact, share, Env, StubEmbedder};
use super::*;
use crate::types::{ChildKind, EventRef, NewSummary, PutOutcome, SummaryChild};
use crate::DefaultBackend;

const DIMS: usize = 8;
const RUN: &str = "consolidation:7";
const OTHER_RUN: &str = "consolidation:8";

fn clock() -> Arc<FixedClock> {
    Arc::new(FixedClock::new(Millis(1000)))
}

fn new_summary(text: &str) -> NewSummary {
    NewSummary {
        level: 1,
        scope: String::new(),
        text: text.to_string(),
        events_fed: 1,
        valid_min: Millis(100),
        valid_max: Millis(900),
        model_id: "model-a".to_string(),
        prompt_version: "prompt-1".to_string(),
    }
}

fn children(ids: &[&str]) -> Vec<SummaryChild> {
    ids.iter()
        .map(|id| SummaryChild {
            id: (*id).to_string(),
            kind: ChildKind::Node,
        })
        .collect()
}

fn sources() -> Vec<EventRef> {
    vec![
        EventRef {
            event_id: "e1".to_string(),
            synthetic: false,
        },
        EventRef {
            event_id: "e2".to_string(),
            synthetic: true,
        },
    ]
}

fn vector() -> Vec<f32> {
    vec![0.5; DIMS]
}

/// Puts a level 1 summary of `ids` that replaces `closes`, written by `run`.
async fn put<B: Backend>(
    g: &Graph<B>,
    text: &str,
    ids: &[&str],
    closes: &[String],
    run: &str,
) -> String {
    let outcome = g
        .put_summary(
            new_summary(text),
            &children(ids),
            &sources(),
            &vector(),
            closes,
            run,
        )
        .await
        .unwrap();
    match outcome {
        PutOutcome::Inserted(id) => id,
        PutOutcome::Unchanged(id) => panic!("expected a new summary, {id} is live"),
    }
}

/// Every summary table in full, ordered, as text. Embeddings are left out
/// because a replay recomputes them.
async fn dump<B: Backend>(g: &Graph<B>) -> Vec<String> {
    let mut dump = Vec::new();
    for sql in [
        "SELECT id, level, scope, child_hash, text, derivation, events_fed, valid_min, valid_max,
                model_id, prompt_version, tx_from, tx_to FROM summaries ORDER BY id",
        "SELECT parent_id, child_id, child_kind FROM summary_children
         ORDER BY parent_id, child_id",
        "SELECT summary_id, event_id, synthetic FROM summary_sources
         ORDER BY summary_id, event_id",
    ] {
        let rows = g.backend.query(sql, &[]).await.unwrap();
        dump.push(format!("{rows:?}"));
    }
    dump
}

fn embedded_texts(embedder: &StubEmbedder) -> Vec<String> {
    embedder.calls()
}

async fn rebuilt(g: DefaultGraph, log: SharedLog) -> (DefaultGraph, crate::RebuildReport) {
    g.rebuild_from_log(log, RebuildMode::ResetChecked)
        .await
        .unwrap()
}

#[tokio::test]
async fn summaries_written_through_the_log_come_back_from_a_rebuild() {
    // Arrange: one summary replaced by another, a second closed on its own
    let env = Env::new();
    let embedder = StubEmbedder::new(DIMS);
    let (g, log) = env.open::<DefaultBackend>("live.db", clock()).await;
    let g = g.with_embedder(embedder.clone());
    let first = put(&g, "first", &["a", "b"], &[], RUN).await;
    let lone = put(&g, "lone", &["c", "d"], &[], RUN).await;
    put(&g, "second", &["a", "b", "e"], &[first], RUN).await;
    g.close_summary(&lone, Millis(2000), RUN).await.unwrap();
    let before = dump(&g).await;

    // Act
    let (rebuilt, report) = rebuilt(g, log).await;

    // Assert: the rows match, nothing superseded is live, and the live
    // summary's vector was computed again from its text
    assert_eq!(dump(&rebuilt).await, before);
    let live = rebuilt.live_summaries(1, "").await.unwrap();
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].text.as_str(), "second");
    assert_eq!(
        rebuilt.summary_embeddings("", 1).await.unwrap(),
        [(live[0].id.clone(), StubEmbedder::vector_for(DIMS, "second"))]
    );
    assert_eq!(report.replayed.reembedded.re_embedded, 1);
    assert_eq!(embedded_texts(&embedder), ["second"]);
}

#[tokio::test]
async fn a_child_set_that_returns_to_an_earlier_value_keeps_both_rows_after_a_rebuild() {
    // Arrange: A, then B in its place, then A again in B's place
    let env = Env::new();
    let (g, log) = env.open::<DefaultBackend>("live.db", clock()).await;
    let a = put(&g, "a", &["x", "y"], &[], RUN).await;
    let b = put(&g, "b", &["x", "y", "z"], std::slice::from_ref(&a), RUN).await;
    let again = put(&g, "a again", &["x", "y"], std::slice::from_ref(&b), RUN).await;
    let before = dump(&g).await;

    // Act
    let (rebuilt, _) = rebuilt(g, log).await;

    // Assert
    assert_eq!(dump(&rebuilt).await, before);
    let live = rebuilt.live_summaries(1, "").await.unwrap();
    assert_eq!(
        live.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
        [again]
    );
    assert_eq!(count(&rebuilt, "summaries").await, 3);
}

#[tokio::test]
async fn a_replay_of_events_the_store_already_holds_changes_nothing() {
    // Arrange
    let env = Env::new();
    let (g, _log) = env.open::<DefaultBackend>("live.db", clock()).await;
    let first = put(&g, "first", &["a", "b"], &[], RUN).await;
    put(&g, "second", &["a", "b", "c"], &[first], RUN).await;
    let before = dump(&g).await;
    g.backend
        .execute("DELETE FROM log_cursor", &[])
        .await
        .unwrap();

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    assert_eq!(report.already_applied, 2);
    assert_eq!(report.applied, 0);
    assert_eq!(dump(&g).await, before);
}

#[tokio::test]
async fn an_unchanged_put_appends_nothing() {
    // Arrange
    let env = Env::new();
    let (g, _log) = env.open::<DefaultBackend>("live.db", clock()).await;
    let first = put(&g, "first", &["a", "b"], &[], RUN).await;
    let logged = env.records().await.len();

    // Act
    let outcome = g
        .put_summary(
            new_summary("regenerated"),
            &children(&["a", "b"]),
            &sources(),
            &vector(),
            &[],
            RUN,
        )
        .await
        .unwrap();

    // Assert
    assert_eq!(outcome, PutOutcome::Unchanged(first));
    assert_eq!(env.records().await.len(), logged);
}

#[tokio::test]
async fn a_summary_is_logged_whole_with_the_source_of_its_run_and_no_hash_to_dedup_on() {
    // Arrange
    let env = Env::new();
    let (g, _log) = env.open::<DefaultBackend>("live.db", clock()).await;

    // Act
    let id = put(&g, "first", &["b", "a"], &[], RUN).await;

    // Assert
    let records = env.records().await;
    let [record] = records.as_slice() else {
        panic!("expected one record, got {}", records.len());
    };
    let LogPayload::DerivedArtifact(artifact) = &record.event.payload else {
        panic!("expected an artifact, got {:?}", record.event.payload);
    };
    assert_eq!(record.event.source, RUN);
    assert_eq!(record.event.content_hash, [0; 32]);
    assert_eq!(artifact.artifact_id, id);
    assert_eq!(artifact.kind, ARTIFACT_KIND_SUMMARY);
    assert_eq!(artifact.child_ids, ["b", "a"]);
    assert_eq!(artifact.at, 1000);
    assert_eq!(count(&g, "log_hash_index").await, 0);
}

#[tokio::test]
async fn the_writes_of_a_rolled_back_run_are_gone_after_a_rebuild_and_a_client_write_stays() {
    // Arrange: a run writes a summary, a client writes a node in the same
    // window, then the run is rolled back
    let env = Env::new();
    let (g, log) = env.open::<DefaultBackend>("live.db", clock()).await;
    put(&g, "doomed", &["a", "b"], &[], RUN).await;
    let client = g.insert(fact("client write")).await.unwrap();
    g.mark_run_rolled_back("7").await.unwrap();

    // Act
    let (rebuilt, _) = rebuilt(g, log).await;

    // Assert
    assert_eq!(count(&rebuilt, "summaries").await, 0);
    assert_eq!(count(&rebuilt, "summary_children").await, 0);
    assert_eq!(count(&rebuilt, "summary_sources").await, 0);
    assert!(super::support::has_node(&rebuilt, client.as_str()).await);
}

#[tokio::test]
async fn a_summary_written_again_after_a_rollback_is_not_a_duplicate_and_survives_a_rebuild() {
    // Arrange: the same child set and text, in the run that was rolled back and
    // in the next one
    let env = Env::new();
    let (g, log) = env.open::<DefaultBackend>("live.db", clock()).await;
    put(&g, "same", &["a", "b"], &[], RUN).await;
    g.mark_run_rolled_back("7").await.unwrap();
    g.backend
        .execute("DELETE FROM summaries", &[])
        .await
        .unwrap();
    let again = put(&g, "same", &["a", "b"], &[], OTHER_RUN).await;

    // Act
    let (rebuilt, _) = rebuilt(g, log).await;

    // Assert
    let live = rebuilt.live_summaries(1, "").await.unwrap();
    assert_eq!(
        live.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
        [again]
    );
    let duplicates = env
        .records()
        .await
        .into_iter()
        .filter(|record| matches!(record.event.payload, LogPayload::DuplicateOf { .. }))
        .count();
    assert_eq!(duplicates, 0);
}

fn artifact_event(id: &str, source: &str, edit: impl FnOnce(&mut DerivedArtifact)) -> LogEvent {
    let mut artifact = DerivedArtifact {
        artifact_id: id.to_string(),
        kind: ARTIFACT_KIND_SUMMARY.to_string(),
        level: 1,
        scope: String::new(),
        text: "text".to_string(),
        child_ids: vec!["a".to_string()],
        source_event_ids: vec![liam_log::event::BasisRef {
            event_id: "e1".to_string(),
            synthetic: false,
        }],
        events_fed: 1,
        derivation: "inferred".to_string(),
        model_id: "model".to_string(),
        prompt_version: "prompt".to_string(),
        closes: Vec::new(),
        valid_min: 1,
        valid_max: 2,
        at: 1000,
    };
    edit(&mut artifact);
    let mut event = event(
        &format!("event-{id}"),
        [0; 32],
        LogPayload::DerivedArtifact(artifact),
    );
    event.source = source.to_string();
    event
}

async fn fresh(env: &Env) -> DefaultGraph {
    let (g, _log) = env
        .open_over::<DefaultBackend>(env.log(super::support::ONE_SEGMENT), "fresh.db", clock())
        .await;
    g
}

#[tokio::test]
async fn a_voided_artifact_is_skipped_by_the_replay() {
    // Arrange: an artifact followed by the void its failed projection appended
    let env = Env::new();
    let written = artifact_event("voided", RUN, |_| {});
    let void = event(
        "void-1",
        [0; 32],
        LogPayload::Voided {
            target_event_id: written.event_id.clone(),
        },
    );
    env.append(&[
        written,
        artifact_event("kept", RUN, |a| a.child_ids = vec!["b".into()]),
        void,
    ]);
    let fresh = fresh(&env).await;

    // Act
    let report = fresh.catch_up().await.unwrap();

    // Assert
    assert_eq!(report.skipped_voided, 1);
    let ids: Vec<String> = fresh
        .backend
        .query("SELECT id FROM summaries ORDER BY id", &[])
        .await
        .unwrap()
        .iter()
        .map(|row| row.get_string(0).unwrap())
        .collect();
    assert_eq!(ids, ["kept"]);
}

#[tokio::test]
async fn an_artifact_the_projection_refuses_is_quarantined_and_the_replay_goes_on() {
    // Arrange: an unknown kind, an unknown derivation, then a good one
    let env = Env::new();
    env.append(&[
        artifact_event("odd-kind", RUN, |a| a.kind = "page".to_string()),
        artifact_event("odd-derivation", RUN, |a| {
            a.derivation = "guessed".to_string()
        }),
        artifact_event("good", RUN, |a| a.child_ids = vec!["b".into()]),
    ]);
    let fresh = fresh(&env).await;

    // Act
    let report = fresh.catch_up().await.unwrap();

    // Assert
    assert_eq!(report.quarantined, 2);
    assert_eq!(count(&fresh, "summaries").await, 1);
}

#[tokio::test]
async fn a_replacement_whose_predecessor_was_rolled_back_still_applies() {
    // Arrange: run 7 wrote the first, run 8 replaced it, then run 7 was rolled back
    let env = Env::new();
    env.append(&[
        artifact_event("first", RUN, |_| {}),
        artifact_event("second", OTHER_RUN, |a| {
            a.child_ids = vec!["a".into(), "b".into()];
            a.closes = vec!["first".into()];
        }),
        event(
            "marker",
            [0; 32],
            LogPayload::RunRolledBack {
                run_id: "7".to_string(),
            },
        ),
    ]);
    let fresh = fresh(&env).await;

    // Act
    fresh.catch_up().await.unwrap();

    // Assert
    let live = fresh.live_summaries(1, "").await.unwrap();
    assert_eq!(
        live.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
        ["second"]
    );
    assert_eq!(count(&fresh, "summaries").await, 1);
}

#[tokio::test]
async fn a_replay_without_an_embedder_leaves_the_summaries_counted_as_pending() {
    // Arrange
    let env = Env::new();
    env.append(&[artifact_event("one", RUN, |_| {})]);
    let fresh = fresh(&env).await;

    // Act
    let report = fresh.catch_up().await.unwrap();

    // Assert
    assert_eq!(report.reembedded.pending, 1);
    assert_eq!(count(&fresh, "summary_embeddings").await, 0);
}

#[tokio::test]
async fn a_close_with_no_replacement_is_logged_and_replayed() {
    // Arrange
    let env = Env::new();
    let (g, log) = env.open::<DefaultBackend>("live.db", clock()).await;
    let lone = put(&g, "lone", &["a", "b"], &[], RUN).await;
    g.close_summary(&lone, Millis(2000), RUN).await.unwrap();
    let before = dump(&g).await;

    // Act
    let (rebuilt, _) = rebuilt(g, log).await;

    // Assert
    assert_eq!(dump(&rebuilt).await, before);
    assert!(rebuilt.live_summaries(1, "").await.unwrap().is_empty());
    let closed = env
        .records()
        .await
        .into_iter()
        .filter(|record| {
            matches!(
                record.event.payload,
                LogPayload::DerivedArtifactClosed { .. }
            )
        })
        .count();
    assert_eq!(closed, 1);
}

#[tokio::test]
async fn a_failed_void_after_a_failed_summary_write_poisons_the_log_until_reopen() {
    // Arrange: the write appends, its projection fails, and the void append
    // (the log's second) fails too
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("graph.db");
    let path = path.to_str().unwrap();
    let (log, appended) = RecordingLog::over(FailingLogWriter::fail_on_nth(2));
    let _ = log.log_id();
    let g = open_with::<FailingBackend>(path, Millis(1000), share(log)).await;
    g.backend.set_fail_on_execute(0);

    // Act
    let first = g
        .put_summary(
            new_summary("first"),
            &children(&["a"]),
            &sources(),
            &vector(),
            &[],
            RUN,
        )
        .await;
    g.backend.set_fail_on_execute(usize::MAX);
    let second = g
        .put_summary(
            new_summary("second"),
            &children(&["b"]),
            &sources(),
            &vector(),
            &[],
            RUN,
        )
        .await;

    // Assert
    assert!(matches!(first, Err(Error::Backend(_))), "{first:?}");
    assert!(matches!(second, Err(Error::LogPoisoned)), "{second:?}");
    assert_eq!(events(&appended).len(), 1);
    assert_eq!(count(&g, "summaries").await, 0);
}
