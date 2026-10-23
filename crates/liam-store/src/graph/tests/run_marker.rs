// SPDX-License-Identifier: Apache-2.0
//! The `RunRolledBack` marker: once appended, a replay skips every write of the
//! run whatever its payload, and leaves other runs and client writes alone. Each
//! test uses a real WAL and database file in its own temp directory.

use super::support::{count, fact, Env};
use super::*;
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
