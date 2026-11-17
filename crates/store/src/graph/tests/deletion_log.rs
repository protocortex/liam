// SPDX-License-Identifier: Apache-2.0
//! The records a delete job appends and the key groups of a subject.

use liam_log::event::LogPayload;

use super::support::{key_registry, Env, StubEmbedder};
use super::*;
use crate::graph::deletion_log::DeletionRecord;
use crate::DefaultBackend;

const DIMS: usize = 8;
const JOB: &str = "job-1";
const KEY: &str = "key-1";
const TOP_SCOPE: &str = "work";
const CHILD_SCOPE: &str = "work/liam";
const SIBLING_SCOPE: &str = "work-archive";

async fn open(env: &Env) -> DefaultGraph {
    let (_keys, registry) = key_registry();
    let (graph, _log) = env
        .open_over::<DefaultBackend>(
            env.sealing_log(&registry),
            "live.db",
            Arc::new(FixedClock::new(Millis(1000))),
        )
        .await;
    graph.with_embedder(StubEmbedder::new(DIMS))
}

fn recorded() -> DeletionRecord {
    DeletionRecord::Recorded {
        job_id: JOB.to_owned(),
        event_ids: vec!["event-1".to_owned()],
        key_ids: vec![KEY.to_owned()],
    }
}

fn destroyed() -> DeletionRecord {
    DeletionRecord::KeyDestroyed {
        key_id: KEY.to_owned(),
    }
}

fn cancelled() -> DeletionRecord {
    DeletionRecord::Cancelled {
        job_id: JOB.to_owned(),
    }
}

#[tokio::test]
async fn each_record_lands_in_the_log_as_its_payload() {
    // Arrange
    let env = Env::new();
    let graph = open(&env).await;

    // Act
    for record in [recorded(), destroyed(), cancelled()] {
        assert!(graph.record_deletion(&record).await.unwrap());
    }

    // Assert
    let payloads: Vec<LogPayload> = env
        .records()
        .await
        .into_iter()
        .map(|record| record.event.payload)
        .collect();
    assert!(matches!(
        &payloads[0],
        LogPayload::DeletionRecorded { job_id, event_ids, key_ids, .. }
            if job_id == JOB && event_ids == &["event-1"] && key_ids == &[KEY]
    ));
    assert!(matches!(&payloads[1], LogPayload::KeyDestroyed { key_id, .. } if key_id == KEY));
    assert!(matches!(&payloads[2], LogPayload::DeletionCancelled { job_id } if job_id == JOB));
}

#[tokio::test]
async fn a_record_already_in_the_log_is_not_appended_again() {
    // Arrange
    let env = Env::new();
    let graph = open(&env).await;
    for record in [recorded(), destroyed(), cancelled()] {
        graph.record_deletion(&record).await.unwrap();
    }
    let before = env.records().await.len();

    // Act
    let mut appended = Vec::new();
    for record in [recorded(), destroyed(), cancelled()] {
        appended.push(graph.record_deletion(&record).await.unwrap());
    }

    // Assert
    assert_eq!(appended, [false, false, false]);
    assert_eq!(env.records().await.len(), before);
}

#[tokio::test]
async fn a_record_of_another_job_or_key_is_appended() {
    // Arrange
    let env = Env::new();
    let graph = open(&env).await;
    graph.record_deletion(&destroyed()).await.unwrap();

    // Act
    let other = graph
        .record_deletion(&DeletionRecord::KeyDestroyed {
            key_id: "key-2".to_owned(),
        })
        .await
        .unwrap();

    // Assert
    assert!(other);
}

#[tokio::test]
async fn a_store_without_a_log_records_nothing() {
    // Arrange
    let graph = DefaultGraph::open_with_clock(
        ":memory:",
        GraphConfig::new(DIMS),
        Arc::new(FixedClock::new(Millis(1000))),
    )
    .await
    .unwrap();

    // Act
    let appended = graph.record_deletion(&recorded()).await.unwrap();

    // Assert
    assert!(!appended);
}

async fn subject_store(env: &Env) -> DefaultGraph {
    let graph = open(env).await;
    for (subject, scope) in [
        ("Alice", None),
        ("alice", Some(TOP_SCOPE)),
        ("alice", Some(CHILD_SCOPE)),
        ("alice", Some(SIBLING_SCOPE)),
        ("bob", None),
    ] {
        let mut note =
            NewNode::now("fact", format!("{subject}-{scope:?}"), "secret").with_subject(subject);
        if let Some(scope) = scope {
            note = note.with_scope(scope);
        }
        graph.insert(note).await.unwrap();
    }
    graph
}

#[tokio::test]
async fn a_subject_lists_the_one_group_of_its_exact_scope() {
    // Arrange
    let env = Env::new();
    let graph = subject_store(&env).await;

    // Act
    let unscoped = graph
        .key_groups_of_subject(" ALICE ", None, false)
        .await
        .unwrap();
    let top = graph
        .key_groups_of_subject("alice", Some(TOP_SCOPE), false)
        .await
        .unwrap();

    // Assert
    assert_eq!(unscoped.len(), 1);
    assert_eq!(top.len(), 1);
    assert_ne!(unscoped, top);
}

#[tokio::test]
async fn including_children_adds_the_groups_below_the_scope_and_not_its_siblings() {
    // Arrange
    let env = Env::new();
    let graph = subject_store(&env).await;

    // Act
    let tree = graph
        .key_groups_of_subject("alice", Some(TOP_SCOPE), true)
        .await
        .unwrap();
    let every_scope = graph
        .key_groups_of_subject("alice", None, true)
        .await
        .unwrap();

    // Assert
    assert_eq!(tree.len(), 2);
    assert_eq!(every_scope.len(), 4);
}

#[tokio::test]
async fn a_subject_with_no_node_has_no_group() {
    // Arrange
    let env = Env::new();
    let graph = subject_store(&env).await;

    // Act
    let groups = graph
        .key_groups_of_subject("nobody", None, true)
        .await
        .unwrap();

    // Assert
    assert!(groups.is_empty());
}
