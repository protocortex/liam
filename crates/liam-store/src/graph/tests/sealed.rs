// SPDX-License-Identifier: Apache-2.0
//! A log that seals its payloads: a live write leaves a sealed record and the
//! plain projection, and a replay meets records whose key was destroyed.

use liam_log::crypto::KeyRegistry;
use liam_log::envelope::{KeyHints, Sealer};
use liam_log::event::{LogEvent, LogPayload, RowEffect, UsageRow};
use liam_log::reader::ReaderError;

use super::log_write::mentions;
use super::support::{
    count, cursor_offset, edge, event, has_node, key_registry, node, quarantined, snapshot, Env,
    StubEmbedder,
};
use super::*;
use crate::DefaultBackend;

const DIMS: usize = 8;

fn clock() -> Arc<FixedClock> {
    Arc::new(FixedClock::new(Millis(1000)))
}

fn subject_node(id: &str, subject: Option<&str>) -> NodeRow {
    NodeRow {
        subject: subject.map(Into::into),
        ..node(id, &format!("content of {id}"))
    }
}

fn episode(id: &str, effects: Vec<RowEffect>) -> LogEvent {
    event(id, [0; 32], LogPayload::EpisodeBatch(effects))
}

fn plain_write(id: &str) -> LogEvent {
    super::support::node_write(id, &format!("content of {id}"))
}

fn seal(registry: &Arc<KeyRegistry>, event: &LogEvent) -> LogEvent {
    Sealer::new(Arc::clone(registry))
        .seal(event, &KeyHints::default())
        .expect("seal")
}

async fn open_over(env: &Env, registry: &Arc<KeyRegistry>, db: &str) -> (DefaultGraph, SharedLog) {
    let (graph, log) = env
        .open_over::<DefaultBackend>(env.sealing_log(registry), db, clock())
        .await;
    (graph.with_embedder(StubEmbedder::new(DIMS)), log)
}

fn key_of(event: &LogEvent, row_id: &str) -> String {
    match &event.payload {
        LogPayload::Sealed(record) => record
            .rows
            .iter()
            .find(|row| row.row_id == row_id)
            .map(|row| row.key_id.clone())
            .expect("sealed row"),
        other => panic!("expected a sealed event, got {other:?}"),
    }
}

#[tokio::test]
async fn a_sealed_write_leaves_a_sealed_record_the_plain_projection_and_the_row_keys() {
    // Arrange
    let env = Env::new();
    let (_keys, registry) = key_registry();
    let (graph, _log) = open_over(&env, &registry, "live.db").await;

    // Act
    let result = graph
        .ingest_episode(
            vec![
                NewNode::now("fact", "a", "alice secret").with_subject("alice"),
                NewNode::now("fact", "b", "free text"),
            ],
            vec![mentions(0, 1)],
        )
        .await
        .unwrap();

    // Assert
    let records = env.records().await;
    assert_eq!(records.len(), 1);
    let LogPayload::Sealed(sealed) = &records[0].event.payload else {
        panic!("the log must hold a sealed payload");
    };
    assert_eq!(sealed.rows.len(), 3);
    assert_eq!(count(&graph, "nodes").await, 2);
    assert_eq!(count(&graph, "node_keys").await, 2);
    let alice = result.node_ids[0].as_str();
    let row = graph
        .backend
        .query(
            "SELECT key_id FROM node_keys WHERE node_id = ?1",
            &[alice.into()],
        )
        .await
        .unwrap();
    assert_eq!(
        row[0].get_string(0).unwrap(),
        key_of(&records[0].event, alice)
    );
}

#[tokio::test]
async fn a_rebuild_from_a_sealed_log_reproduces_the_store_and_the_row_keys() {
    // Arrange
    let env = Env::new();
    let (_keys, registry) = key_registry();
    let (graph, log) = open_over(&env, &registry, "live.db").await;
    graph
        .ingest_episode(
            vec![
                NewNode::now("fact", "a", "alice secret").with_subject("alice"),
                NewNode::now("fact", "b", "free text"),
            ],
            vec![mentions(0, 1)],
        )
        .await
        .unwrap();
    let expected = snapshot(&graph).await;
    drop((graph, log));

    // Act
    let (fresh, _log) = open_over(&env, &registry, "fresh.db").await;
    let report = fresh.catch_up().await.unwrap();

    // Assert
    assert_eq!(report.applied, 1);
    assert_eq!(snapshot(&fresh).await, expected);
    assert_eq!(count(&fresh, "node_keys").await, 2);
}

#[tokio::test]
async fn a_fully_shredded_record_in_the_middle_only_moves_the_cursor() {
    // Arrange
    let env = Env::new();
    let (_keys, registry) = key_registry();
    let doomed = seal(
        &registry,
        &event(
            "event-doomed",
            [0; 32],
            LogPayload::NodeWrite(subject_node("doomed", Some("alice"))),
        ),
    );
    registry.destroy(&key_of(&doomed, "doomed")).unwrap();
    let offsets = env.append(&[plain_write("before"), doomed, plain_write("after")]);
    let (graph, _log) = open_over(&env, &registry, "replay.db").await;

    // Act
    let report = graph.catch_up().await.unwrap();

    // Assert
    assert_eq!(report.applied, 2);
    assert!(has_node(&graph, "before").await && has_node(&graph, "after").await);
    assert!(!has_node(&graph, "doomed").await);
    assert!(quarantined(&graph).await.is_empty());
    assert_eq!(
        cursor_offset(&graph).await,
        Some((offsets[2].segment as i64, offsets[2].index as i64))
    );
}

#[tokio::test]
async fn an_edge_to_a_shredded_node_is_skipped_and_breaks_no_foreign_key() {
    // Arrange
    let env = Env::new();
    let (_keys, registry) = key_registry();
    let batch = episode(
        "event-batch",
        vec![
            RowEffect::Node(subject_node("alice", Some("alice"))),
            RowEffect::Node(subject_node("bob", Some("bob"))),
            RowEffect::Edge(edge("a-to-b", "alice", "bob")),
            RowEffect::Edge(edge("b-to-a", "bob", "alice")),
        ],
    );
    let sealed = seal(&registry, &batch);
    registry.destroy(&key_of(&sealed, "bob")).unwrap();
    env.append(&[sealed]);
    let (graph, _log) = open_over(&env, &registry, "replay.db").await;

    // Act
    let report = graph.catch_up().await.unwrap();

    // Assert
    assert_eq!(report.applied, 1);
    assert_eq!(report.quarantined, 0);
    assert!(has_node(&graph, "alice").await && !has_node(&graph, "bob").await);
    assert_eq!(count(&graph, "edges").await, 0);
    assert_eq!(
        count(&graph, "node_keys").await,
        2,
        "the shredded node keeps its group"
    );
}

#[tokio::test]
async fn a_key_the_registry_lost_stops_the_replay_with_a_typed_error() {
    // Arrange
    let env = Env::new();
    let (_keys, registry) = key_registry();
    env.append(&[
        plain_write("before"),
        seal(
            &registry,
            &event(
                "event-sealed",
                [0; 32],
                LogPayload::NodeWrite(subject_node("sealed", Some("alice"))),
            ),
        ),
    ]);
    let (_other_keys, lost) = key_registry();
    let (graph, _log) = open_over(&env, &lost, "replay.db").await;

    // Act
    let result = graph.catch_up().await;

    // Assert
    assert!(matches!(
        result,
        Err(Error::LogRead(ReaderError::KeyMissing { .. }))
    ));
    assert!(quarantined(&graph).await.is_empty());
}

#[tokio::test]
async fn rows_under_a_key_marked_destroying_read_as_shredded_and_do_not_wedge_replay() {
    // Arrange
    let env = Env::new();
    let (_keys, registry) = key_registry();
    let sealed = seal(
        &registry,
        &event(
            "event-sealed",
            [0; 32],
            LogPayload::NodeWrite(subject_node("marked", Some("alice"))),
        ),
    );
    registry.begin_destroy(&key_of(&sealed, "marked")).unwrap();
    env.append(&[sealed, plain_write("after")]);
    let (graph, _log) = open_over(&env, &registry, "replay.db").await;

    // Act
    let report = graph.catch_up().await.unwrap();

    // Assert
    assert_eq!(report.applied, 1);
    assert!(has_node(&graph, "after").await && !has_node(&graph, "marked").await);
}

#[tokio::test]
async fn delete_records_replay_as_no_ops_and_twice_change_nothing() {
    // Arrange
    let env = Env::new();
    let (_keys, registry) = key_registry();
    let records = [
        plain_write("keep"),
        event(
            "event-destroyed",
            [0; 32],
            LogPayload::KeyDestroyed {
                key_id: "key-1".into(),
                at: 5,
            },
        ),
        event(
            "event-recorded",
            [0; 32],
            LogPayload::DeletionRecorded {
                job_id: "job-1".into(),
                event_ids: vec!["event-x".into()],
                key_ids: vec!["key-1".into()],
                at: 5,
            },
        ),
        event(
            "event-removal",
            [0; 32],
            LogPayload::LineageRemoval {
                job_id: "job-1".into(),
                store: "cache".into(),
                event_ids: vec!["event-x".into()],
                key_ids: vec!["key-1".into()],
                node_ids: Vec::new(),
            },
        ),
        event(
            "event-cancelled",
            [0; 32],
            LogPayload::DeletionCancelled {
                job_id: "job-2".into(),
            },
        ),
    ];
    env.append(&records);
    let (graph, _log) = open_over(&env, &registry, "replay.db").await;

    // Act
    let first = graph.catch_up().await.unwrap();
    let before = snapshot(&graph).await;
    let second = graph.catch_up().await.unwrap();

    // Assert
    assert_eq!(first.applied, 1);
    assert_eq!(second, CatchUpReport::default());
    assert_eq!(snapshot(&graph).await, before);
}

#[tokio::test]
async fn a_shredded_usage_batch_still_records_its_day() {
    // Arrange
    let env = Env::new();
    let (_keys, registry) = key_registry();
    let owner = registry.key_for_subject(Some("proj/a"), "alice").unwrap();
    let hints = KeyHints {
        lineages: [("e:proj/a:alice".to_owned(), owner.clone())].into(),
        ..KeyHints::default()
    };
    let usage = event(
        "event-usage",
        [0; 32],
        LogPayload::AccessBatch {
            day: 19_000,
            entries: vec![UsageRow {
                lineage_key: "e:proj/a:alice".into(),
                hits: 3,
                last_access_at: 7,
            }],
        },
    );
    let sealed = Sealer::new(Arc::clone(&registry))
        .seal(&usage, &hints)
        .unwrap();
    registry.destroy(&owner).unwrap();
    env.append(&[sealed]);
    let (graph, _log) = open_over(&env, &registry, "replay.db").await;

    // Act
    graph.catch_up().await.unwrap();

    // Assert
    assert_eq!(count(&graph, "usage_epochs").await, 1);
    assert_eq!(count(&graph, "item_usage_events").await, 0);
}

#[tokio::test]
async fn gc_leaves_the_row_keys_of_the_nodes_it_removes() {
    // Arrange
    const DAY: i64 = 86_400_000;
    let env = Env::new();
    let (_keys, registry) = key_registry();
    let clock = clock();
    let (graph, _log) = env
        .open_over::<DefaultBackend>(env.sealing_log(&registry), "gc.db", Arc::clone(&clock))
        .await;
    clock.set(Millis(100 * DAY));
    graph
        .insert(
            NewNode::now("episode", "old", "x")
                .with_subject("alice")
                .with_valid_from(Millis(10 * DAY)),
        )
        .await
        .unwrap();

    // Act
    let report = graph
        .gc(&RetentionPolicy::keep("episode", Millis::days(30)).without_reclaim())
        .await
        .unwrap();

    // Assert
    assert_eq!(report.nodes_removed, 1);
    assert_eq!(count(&graph, "nodes").await, 0);
    assert_eq!(count(&graph, "node_keys").await, 1);
}

#[tokio::test]
async fn a_write_into_a_destroying_group_is_refused_and_nothing_is_appended() {
    // Arrange
    let env = Env::new();
    let (_keys, registry) = key_registry();
    let (graph, _log) = open_over(&env, &registry, "live.db").await;
    graph
        .insert(NewNode::now("fact", "a", "first").with_subject("alice"))
        .await
        .unwrap();
    let group = registry.key_for_subject(None, "alice").unwrap();
    registry.begin_destroy(&group).unwrap();

    // Act
    let result = graph
        .insert(NewNode::now("fact", "b", "second").with_subject("alice"))
        .await;

    // Assert
    assert!(
        matches!(result, Err(Error::KeyGroupLocked(_))),
        "{result:?}"
    );
    assert_eq!(env.records().await.len(), 1);
    assert_eq!(count(&graph, "nodes").await, 1);
}
