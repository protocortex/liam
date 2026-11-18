// SPDX-License-Identifier: Apache-2.0
//! `nodes_swept_by` reads the sweeps one write source logged. Each test uses a
//! real WAL and database file in its own temp directory.

use super::support::{fact_at, Env};
use super::*;
use crate::DefaultBackend;

const RUN_SOURCE: &str = "consolidation:7";
const OTHER_SOURCE: &str = "consolidation:8";

fn clock() -> Arc<FixedClock> {
    Arc::new(FixedClock::new(Millis(1000)))
}

fn sweep_facts() -> RetentionPolicy {
    RetentionPolicy::keep("fact", Millis(1)).without_reclaim()
}

#[tokio::test]
async fn the_nodes_a_source_swept_are_named_and_another_sources_are_not() {
    // Arrange
    let env = Env::new();
    let (live, _log) = env.open::<DefaultBackend>("live.db", clock()).await;
    let swept = live.insert(fact_at("old note")).await.unwrap();
    live.gc_from(&sweep_facts(), RUN_SOURCE).await.unwrap();
    live.insert(fact_at("older note")).await.unwrap();
    live.gc_from(&sweep_facts(), OTHER_SOURCE).await.unwrap();

    // Act
    let named = live.nodes_swept_by(RUN_SOURCE).await.unwrap();

    // Assert
    assert_eq!(named, HashSet::from([swept]));
}

#[tokio::test]
async fn a_source_that_swept_nothing_names_no_node() {
    let env = Env::new();
    let (live, _log) = env.open::<DefaultBackend>("live.db", clock()).await;
    live.insert(fact_at("kept note")).await.unwrap();

    let named = live.nodes_swept_by(RUN_SOURCE).await.unwrap();

    assert!(named.is_empty());
}

#[tokio::test]
async fn a_store_with_no_log_cannot_say() {
    let g = DefaultGraph::open(":memory:", GraphConfig::new(8))
        .await
        .unwrap();

    let result = g.nodes_swept_by(RUN_SOURCE).await;

    assert!(matches!(result, Err(Error::NoEventLog)));
}
