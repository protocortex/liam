// SPDX-License-Identifier: Apache-2.0
//! Usage, verdicts and tier moves through the log: they are appended before they
//! are applied, a rebuild brings them back, and a rolled-back run's moves do not
//! come back with it.

use liam_log::event::LogPayload;
use liam_log::test_support::FailingLogWriter;

use super::log_write::{events, open_with, RecordingLog};
use super::support::{share, Env, StubEmbedder};
use super::*;
use crate::types::{LineageKey, Tier, TierMove, UsageEntry};
use crate::DefaultBackend;

const DIMS: usize = 8;
const RUN: &str = "run-1";
const RUN_SOURCE: &str = "consolidation:run-1";
const DAY: i64 = 86_400_000;

fn fact_node(label: &str) -> NewNode {
    NewNode::now("fact", label, format!("text of {label}"))
        .with_subject(label)
        .with_producer("agent-a")
}

fn move_to(lineage: &LineageKey, to: Tier, at: i64, residency: Option<i64>) -> TierMove {
    TierMove {
        lineage_key: lineage.clone(),
        to,
        run_id: RUN.to_string(),
        reason: "test".to_string(),
        changed_at: Millis(at),
        residency_until: residency.map(Millis),
    }
}

fn usage(lineage: &LineageKey, hits: u32, at: i64) -> UsageEntry {
    UsageEntry {
        lineage_key: lineage.clone(),
        producer: String::new(),
        hits,
        last_access_at: Millis(at),
    }
}

/// A store over a real log, its embedder, and the handles a test steers by.
struct Live {
    /// Keeps the log directory alive.
    _env: Env,
    graph: DefaultGraph,
    log: SharedLog,
    embedder: Arc<StubEmbedder>,
}

async fn live() -> Live {
    let env = Env::new();
    let embedder = StubEmbedder::new(DIMS);
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let (graph, log) = env.open::<DefaultBackend>("live.db", clock).await;
    let graph = graph.with_embedder(embedder.clone());
    Live {
        _env: env,
        graph,
        log,
        embedder,
    }
}

/// The content of every table that carries tier state, sorted.
async fn state<B: Backend>(g: &Graph<B>) -> Vec<String> {
    let mut rows = Vec::new();
    for table in [
        "item_tier",
        "item_usage_events",
        "usage_epochs",
        "item_feedback",
    ] {
        let found = g
            .backend
            .query(&format!("SELECT * FROM {table} ORDER BY 1, 2"), &[])
            .await
            .unwrap();
        rows.extend(found.iter().map(|row| format!("{table} {:?}", row.0)));
    }
    rows
}

async fn column<B: Backend>(g: &Graph<B>, sql: &str) -> Vec<String> {
    let rows = g.backend.query(sql, &[]).await.unwrap();
    rows.iter().map(|row| row.get_string(0).unwrap()).collect()
}

async fn vector_ids<B: Backend>(g: &Graph<B>) -> Vec<String> {
    column(g, "SELECT node_id FROM node_vectors ORDER BY node_id").await
}

async fn rebuilt(live: Live) -> DefaultGraph {
    let (graph, _) = live
        .graph
        .rebuild_from_log(live.log, RebuildMode::Replace)
        .await
        .unwrap();
    graph
}

/// Three embedded nodes: `a` demoted, `b` used, `c` demoted and promoted again.
async fn tiered(live: &Live) -> [(NodeId, LineageKey); 3] {
    let g = &live.graph;
    let mut made = Vec::new();
    for label in ["a", "b", "c"] {
        let id = g.insert(fact_node(label)).await.unwrap();
        let key = g.lineage_of(&id).await.unwrap().unwrap();
        made.push((id, key));
    }
    g.reembed_missing().await.unwrap();
    let [(_, a), (_, b), (_, c)] = &made[..] else {
        unreachable!()
    };
    g.put_usage_logged(19_000, &[]).await.unwrap();
    g.put_tier_logged(&move_to(a, Tier::Cold, 2000, None), RUN_SOURCE)
        .await
        .unwrap();
    g.put_tier_logged(&move_to(c, Tier::Cold, 2000, None), RUN_SOURCE)
        .await
        .unwrap();
    g.put_tier_logged(&move_to(c, Tier::Hot, 3000, Some(9_000)), RUN_SOURCE)
        .await
        .unwrap();
    g.settle_cold_tiers().await.unwrap();
    g.put_usage_logged(19_001, &[usage(b, 3, 4 * DAY), usage(c, 1, 5)])
        .await
        .unwrap();
    g.put_feedback_logged(b, "agent-a", true).await.unwrap();
    made.try_into().ok().unwrap()
}

#[tokio::test]
async fn state_written_through_the_log_is_rebuilt_identically_and_only_hot_lineages_keep_vectors() {
    // Arrange
    let live = live().await;
    let [(a, _), (b, _), (c, _)] = tiered(&live).await;
    let before = state(&live.graph).await;
    let signatures = column(
        &live.graph,
        "SELECT node_id FROM node_quant ORDER BY node_id",
    )
    .await;
    assert_eq!(vector_ids(&live.graph).await.len(), 2);

    // Act
    let graph = rebuilt(live).await;

    // Assert: the same rows, the residency the log carried, f32 forms for the
    // hot lineages only, and a signature for every item
    assert_eq!(state(&graph).await, before);
    let tiers = column(&graph, "SELECT tier || ':' || changed_at || ':' || ifnull(residency_until, '-') FROM item_tier ORDER BY lineage_key").await;
    assert_eq!(tiers, ["cold:2000:-", "hot:3000:9000"]);
    let mut hot = vec![b.as_str().to_string(), c.as_str().to_string()];
    hot.sort();
    assert_eq!(vector_ids(&graph).await, hot);
    assert_eq!(
        column(&graph, "SELECT node_id FROM node_quant ORDER BY node_id").await,
        signatures
    );
    let int8_gone = column(&graph, "SELECT node_id FROM node_quant WHERE i8 IS NULL").await;
    assert_eq!(int8_gone, [a.as_str()]);
    assert!(graph
        .quant_index()
        .snapshot()
        .iter()
        .any(|(id, cold, _)| id == a.as_str() && *cold));
}

#[tokio::test]
async fn each_producers_hits_come_back_under_that_producer_after_a_rebuild() {
    // Arrange
    let live = live().await;
    let key = LineageKey::from_raw("n:1:a");
    let by = |producer: &str, hits| UsageEntry {
        producer: producer.to_owned(),
        ..usage(&key, hits, 5)
    };
    live.graph
        .put_usage_logged(19_000, &[by("p1", 3), by("p2", 4), by("", 1)])
        .await
        .unwrap();
    let before = state(&live.graph).await;

    // Act
    let graph = rebuilt(live).await;

    // Assert
    assert_eq!(state(&graph).await, before);
    let producers = column(
        &graph,
        "SELECT producer FROM item_usage_events ORDER BY producer",
    )
    .await;
    assert_eq!(producers, ["", "p1", "p2"]);
}

#[tokio::test]
async fn replaying_the_log_again_changes_nothing() {
    // Arrange
    let live = live().await;
    tiered(&live).await;
    let before = state(&live.graph).await;
    live.graph
        .backend
        .execute(
            "UPDATE log_cursor SET last_segment = NULL, last_index = NULL",
            &[],
        )
        .await
        .unwrap();

    // Act
    let report = live.graph.catch_up().await.unwrap();

    // Assert
    assert!(report.applied + report.already_applied > 0, "{report:?}");
    assert_eq!(state(&live.graph).await, before);
}

#[tokio::test]
async fn a_rolled_back_runs_moves_do_not_come_back_but_a_client_write_in_the_window_does() {
    // Arrange: a run demotes `a`, the run is rolled back, and a client writes
    // while it was running
    let live = live().await;
    let g = &live.graph;
    let a = g.insert(fact_node("a")).await.unwrap();
    let a_key = g.lineage_of(&a).await.unwrap().unwrap();
    g.reembed_missing().await.unwrap();
    g.put_tier_logged(&move_to(&a_key, Tier::Cold, 2000, None), RUN_SOURCE)
        .await
        .unwrap();
    g.settle_cold_tiers().await.unwrap();
    let client = g.insert(fact_node("client")).await.unwrap();
    g.reembed_missing().await.unwrap();
    g.put_usage_logged(19_000, &[usage(&a_key, 2, 5)])
        .await
        .unwrap();
    g.mark_run_rolled_back(RUN).await.unwrap();
    assert!(vector_ids(g).await.contains(&client.as_str().to_string()));
    assert!(!vector_ids(g).await.contains(&a.as_str().to_string()));

    // Act
    let graph = rebuilt(live).await;

    // Assert: `a` is hot again with its vector, and the usage and the client's
    // node, which the run did not write, are still there
    assert_eq!(graph.tier_of(&a_key).await.unwrap(), Tier::Hot);
    assert_eq!(
        column(&graph, "SELECT lineage_key FROM item_tier")
            .await
            .len(),
        0
    );
    let vectors = vector_ids(&graph).await;
    assert!(vectors.contains(&a.as_str().to_string()));
    assert!(vectors.contains(&client.as_str().to_string()));
    assert_eq!(
        column(&graph, "SELECT lineage_key FROM item_usage_events")
            .await
            .len(),
        1
    );
}

#[tokio::test]
async fn a_write_for_a_cold_lineage_replays_to_the_same_hot_lineage() {
    // Arrange: `a` is demoted, then a new version is written
    let live = live().await;
    let g = &live.graph;
    let a = g.insert(fact_node("a")).await.unwrap();
    let key = g.lineage_of(&a).await.unwrap().unwrap();
    g.reembed_missing().await.unwrap();
    g.put_tier_logged(&move_to(&key, Tier::Cold, 2000, None), RUN_SOURCE)
        .await
        .unwrap();
    g.settle_cold_tiers().await.unwrap();
    let newer = g.supersede(&a, fact_node("a")).await.unwrap();
    assert_eq!(g.tier_of(&key).await.unwrap(), Tier::Hot);
    let before = state(g).await;

    // Act
    let graph = rebuilt(live).await;

    // Assert
    assert_eq!(graph.tier_of(&key).await.unwrap(), Tier::Hot);
    assert_eq!(state(&graph).await, before);
    assert_eq!(vector_ids(&graph).await, [newer.as_str()]);
}

#[tokio::test]
async fn a_demoted_node_is_not_embedded_again_by_the_re_embed_pass() {
    // Arrange
    let live = live().await;
    let [(a, _), ..] = tiered(&live).await;
    let calls = live.embedder.calls();

    // Act
    let report = live.graph.reembed_missing().await.unwrap();

    // Assert
    assert_eq!(
        (report.re_embedded, report.failed, report.pending),
        (0, 0, 0)
    );
    assert_eq!(live.embedder.calls(), calls);
    assert!(!vector_ids(&live.graph)
        .await
        .contains(&a.as_str().to_string()));
}

#[tokio::test]
async fn identical_batches_and_verdicts_are_each_logged_and_none_is_a_duplicate() {
    // Arrange
    let (log, appended) = RecordingLog::new();
    let g = open_with::<DefaultBackend>(":memory:", Millis(1000), share(log)).await;
    let id = g.insert(fact_node("a")).await.unwrap();
    let key = g.lineage_of(&id).await.unwrap().unwrap();

    // Act
    for _ in 0..2 {
        g.put_usage_logged(19_000, &[usage(&key, 1, 5)])
            .await
            .unwrap();
        g.put_feedback_logged(&key, "agent-a", true).await.unwrap();
    }

    // Assert
    let logged = events(&appended);
    let batches = logged
        .iter()
        .filter(|event| matches!(event.payload, LogPayload::AccessBatch { .. }))
        .count();
    let verdicts = logged
        .iter()
        .filter(|event| matches!(event.payload, LogPayload::Feedback { .. }))
        .count();
    assert_eq!((batches, verdicts), (2, 2));
    assert!(!logged
        .iter()
        .any(|event| matches!(event.payload, LogPayload::DuplicateOf { .. })));
    let sources: Vec<_> = logged.iter().skip(1).map(|e| e.source.as_str()).collect();
    assert_eq!(sources, ["usage", "feedback", "usage", "feedback"]);
}

#[tokio::test]
async fn a_move_to_the_tier_a_lineage_is_in_appends_nothing() {
    // Arrange
    let (log, appended) = RecordingLog::new();
    let g = open_with::<DefaultBackend>(":memory:", Millis(1000), share(log)).await;
    let id = g.insert(fact_node("a")).await.unwrap();
    let key = g.lineage_of(&id).await.unwrap().unwrap();
    let before = events(&appended).len();

    // Act
    let to_hot = g
        .put_tier_logged(&move_to(&key, Tier::Hot, 5, None), RUN_SOURCE)
        .await
        .unwrap();
    let to_cold = g
        .put_tier_logged(&move_to(&key, Tier::Cold, 6, None), RUN_SOURCE)
        .await
        .unwrap();
    let cold_again = g
        .put_tier_logged(&move_to(&key, Tier::Cold, 7, None), RUN_SOURCE)
        .await
        .unwrap();

    // Assert
    assert_eq!((to_hot, to_cold, cold_again), (false, true, false));
    assert_eq!(events(&appended).len(), before + 1);
}

#[tokio::test]
async fn a_write_that_cannot_be_applied_is_voided_and_a_second_store_replays_without_it() {
    // Arrange: the transaction fails on its first statement
    let env = Env::new();
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let (g, log) = env.open::<FailingBackend>("live.db", clock.clone()).await;
    let id = g.insert(fact_node("a")).await.unwrap();
    let key = g.lineage_of(&id).await.unwrap().unwrap();
    g.backend.set_fail_on_execute(0);

    // Act
    let usage_result = g.put_usage_logged(19_000, &[usage(&key, 1, 5)]).await;
    let tier_result = g
        .put_tier_logged(&move_to(&key, Tier::Cold, 5, None), RUN_SOURCE)
        .await;
    g.backend.set_fail_on_execute(usize::MAX);
    drop((g, log));

    // Assert: each was appended and then cancelled, and a store built from the
    // log has neither
    assert!(usage_result.is_err() && tier_result.is_err());
    let records = env.records().await;
    let kinds: Vec<_> = records
        .iter()
        .map(|record| match &record.event.payload {
            LogPayload::AccessBatch { .. } => "usage",
            LogPayload::TierMove { .. } => "tier",
            LogPayload::Voided { .. } => "void",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, ["other", "usage", "void", "tier", "void"]);
    let (second, _) = env.open::<DefaultBackend>("second.db", clock).await;
    second.catch_up().await.unwrap();
    assert!(state(&second).await.is_empty());
}

#[tokio::test]
async fn when_the_void_cannot_be_appended_the_next_write_is_refused() {
    // Arrange: the first write appends, its projection fails, and the append of
    // its void, the log's second, fails too
    let (log, _appended) = RecordingLog::over(FailingLogWriter::fail_on_nth(2));
    let g = open_with::<FailingBackend>(":memory:", Millis(1000), share(log)).await;
    let key = LineageKey::from_raw("n:1:x");
    g.backend.set_fail_on_execute(0);

    // Act
    let first = g.put_usage_logged(19_000, &[usage(&key, 1, 5)]).await;
    g.backend.set_fail_on_execute(usize::MAX);
    let second = g.put_feedback_logged(&key, "agent-a", true).await;

    // Assert
    assert!(matches!(first, Err(Error::Backend(_))), "{first:?}");
    assert!(matches!(second, Err(Error::LogPoisoned)), "{second:?}");
}
