// SPDX-License-Identifier: Apache-2.0
//! A search that runs while a lineage moves between tiers.

use super::quant::signed;
use super::support::Env;
use super::tiers::{cold_at, hot_at, lineage_of_node, subject_node};
use super::*;
use crate::DefaultBackend;

const SOURCE: &str = "consolidation:run-1";
const CYCLES: usize = 15;
const SEARCHES: usize = 200;
const RESIDENT_UNTIL: i64 = 9000;

#[tokio::test]
async fn a_search_during_tier_moves_returns_each_item_at_most_once_and_never_misses_a_text_match() {
    // Arrange: the query has text, so the lexical arm finds the items whatever
    // tier they are in, and the vector and cold arms overlap on them while a
    // move is half done
    let env = Env::new();
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let (graph, _log) = env.open::<DefaultBackend>("race.db", clock).await;
    let a = graph.insert(subject_node("a", 0b01)).await.unwrap();
    let b = graph.insert(subject_node("b", 0b10)).await.unwrap();
    let key = lineage_of_node(&graph, &a).await;
    let graph = Arc::new(graph);
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let mover = {
        let (graph, barrier, key) = (Arc::clone(&graph), Arc::clone(&barrier), key.clone());
        let vectors = [(a.clone(), signed(0b01)), (b.clone(), signed(0b10))];
        tokio::spawn(async move {
            barrier.wait().await;
            for _ in 0..CYCLES {
                graph
                    .demote_lineage(&cold_at(&key, 1000), SOURCE)
                    .await
                    .unwrap();
                graph
                    .promote_lineage(&hot_at(&key, 2000, Some(RESIDENT_UNTIL)), &vectors, SOURCE)
                    .await
                    .unwrap();
            }
        })
    };
    let query = Query::text("content of a")
        .with_embedding(signed(0b01))
        .with_k(10);

    // Act
    barrier.wait().await;
    let mut seen = Vec::new();
    for _ in 0..SEARCHES {
        let hits = graph.query(&query).await.unwrap();
        seen.push(hits.into_iter().map(|hit| hit.id).collect::<Vec<_>>());
    }
    mover.await.unwrap();

    // Assert
    for ids in seen {
        let times = |wanted: &NodeId| ids.iter().filter(|id| *id == wanted).count();
        assert_eq!(times(&a), 1, "{ids:?}");
        assert!(times(&b) <= 1, "{ids:?}");
    }
}
