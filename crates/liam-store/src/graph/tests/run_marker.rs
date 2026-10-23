// SPDX-License-Identifier: Apache-2.0
//! The `RunRolledBack` marker: once appended, a replay skips every write of the
//! run whatever its payload, and leaves other runs and client writes alone. Each
//! test uses a real WAL and database file in its own temp directory.

use super::support::{count, fact, fact_at, Env};
use super::*;
use crate::types::relation;
use crate::DefaultBackend;

const RUN: &str = "7";
const RUN_SOURCE: &str = "consolidation:7";
const LONGER_RUN_SOURCE: &str = "consolidation:70";

fn clock() -> Arc<FixedClock> {
    Arc::new(FixedClock::new(Millis(1000)))
}

async fn contents(g: &DefaultGraph) -> Vec<String> {
    g.backend
        .query("SELECT content FROM nodes ORDER BY content", &[])
        .await
        .unwrap()
        .iter()
        .map(|row| row.get_string(0).unwrap())
        .collect()
}

#[tokio::test]
async fn a_replay_skips_the_nodes_and_edges_of_a_marked_run_only() {
    // Arrange
    let env = Env::new();
    {
        let (live, _log) = env.open::<DefaultBackend>("live.db", clock()).await;
        let client = live.insert(fact("client note")).await.unwrap();
        let run_node = live
            .insert(fact("run note").with_producer(RUN_SOURCE))
            .await
            .unwrap();
        let other_run_node = live
            .insert(fact("longer run note").with_producer(LONGER_RUN_SOURCE))
            .await
            .unwrap();
        live.relate_from(&run_node, &client, "near_duplicate_of", RUN_SOURCE)
            .await
            .unwrap();
        live.relate_from(
            &other_run_node,
            &client,
            "near_duplicate_of",
            LONGER_RUN_SOURCE,
        )
        .await
        .unwrap();
        live.mark_run_rolled_back(RUN).await.unwrap();
    }
    let (replayed, _log) = env.open::<DefaultBackend>("replayed.db", clock()).await;

    // Act
    replayed.catch_up().await.unwrap();

    // Assert
    assert_eq!(
        contents(&replayed).await,
        ["client note", "longer run note"]
    );
    assert_eq!(count(&replayed, "edges").await, 1);
}

#[tokio::test]
async fn a_write_after_the_marker_still_lands() {
    // Arrange
    let env = Env::new();
    let (live, _log) = env.open::<DefaultBackend>("live.db", clock()).await;
    live.mark_run_rolled_back(RUN).await.unwrap();

    // Act
    let written = live.insert(fact("after the marker")).await;

    // Assert
    assert!(written.is_ok());
    assert_eq!(count(&live, "nodes").await, 1);
}

#[tokio::test]
async fn marking_without_an_event_log_is_refused() {
    // Arrange
    let g = DefaultGraph::open(":memory:", GraphConfig::new(8))
        .await
        .unwrap();

    // Act
    let result = g.mark_run_rolled_back(RUN).await;

    // Assert
    assert!(matches!(result, Err(Error::NoEventLog)));
}

#[tokio::test]
async fn a_sweep_stamped_with_a_run_is_cancelled_by_its_marker() {
    // Arrange
    let env = Env::new();
    {
        let (live, _log) = env.open::<DefaultBackend>("live.db", clock()).await;
        live.insert(fact_at("old note")).await.unwrap();
        let policy = RetentionPolicy::keep("fact", Millis(1)).without_reclaim();
        let report = live.gc_from(&policy, RUN_SOURCE).await.unwrap();
        assert_eq!(report.nodes_removed, 1);
        live.mark_run_rolled_back(RUN).await.unwrap();
    }
    let (replayed, _log) = env.open::<DefaultBackend>("replayed.db", clock()).await;

    // Act
    replayed.catch_up().await.unwrap();

    // Assert
    assert_eq!(contents(&replayed).await, ["old note"]);
}

#[tokio::test]
async fn a_supersession_by_a_marked_run_is_cancelled_with_its_closing_edge() {
    // Arrange
    let env = Env::new();
    {
        let (live, _log) = env.open::<DefaultBackend>("live.db", clock()).await;
        live.upsert_by(NewNode::entity("person", "Kim").with_producer("client"))
            .await
            .unwrap();
        let mut renewed = NewNode::entity("person", "Kim").with_producer(RUN_SOURCE);
        renewed.content = "written by the run".to_owned();
        live.upsert_by(renewed).await.unwrap();
        live.mark_run_rolled_back(RUN).await.unwrap();
    }
    let (replayed, _log) = env.open::<DefaultBackend>("replayed.db", clock()).await;

    // Act
    replayed.catch_up().await.unwrap();

    // Assert: the original entity is open again and nothing of the run is left.
    let live: Vec<(String, i64)> = replayed
        .backend
        .query("SELECT content, tx_to FROM nodes", &[])
        .await
        .unwrap()
        .iter()
        .map(|row| (row.get_string(0).unwrap(), row.get_i64(1).unwrap()))
        .collect();
    assert_eq!(live, [(String::new(), FOREVER.0)]);
    assert_eq!(count(&replayed, "edges").await, 0);
}

#[tokio::test]
async fn a_repair_stamped_with_a_run_is_cancelled_by_its_marker() {
    // Arrange: a client supersedes an entity that a fact mentions.
    let env = Env::new();
    let before = {
        let (live, _log) = env.open::<DefaultBackend>("live.db", clock()).await;
        let entity = live
            .upsert_by(NewNode::entity("person", "Kim").with_producer("client"))
            .await
            .unwrap();
        let note = live.insert(fact("Kim likes tea")).await.unwrap();
        live.relate(&note, &entity, relation::MENTIONS)
            .await
            .unwrap();
        let mut renewed = NewNode::entity("person", "Kim").with_producer("client");
        renewed.content = "renewed".to_owned();
        live.upsert_by(renewed).await.unwrap();
        let before = count(&live, "edges").await;
        let repaired = live
            .repair_superseded_mentions_from(RUN_SOURCE)
            .await
            .unwrap();
        assert_eq!(repaired, 1);
        assert_eq!(count(&live, "edges").await, before + 1);
        live.mark_run_rolled_back(RUN).await.unwrap();
        before
    };
    let (replayed, _log) = env.open::<DefaultBackend>("replayed.db", clock()).await;

    // Act
    replayed.catch_up().await.unwrap();

    // Assert
    assert_eq!(count(&replayed, "edges").await, before);
}
