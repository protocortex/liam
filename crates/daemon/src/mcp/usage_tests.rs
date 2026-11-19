// SPDX-License-Identifier: Apache-2.0
//! What `recall` counts as use of a memory.

use std::sync::Arc;

use liam_model::{MockEmbedder, MockLlm};
use liam_store::{DefaultGraph, GraphConfig, NewEdge, NewNode, NodeId, Query, HANDLE_LEN};
use rmcp::handler::server::wrapper::Parameters;

use super::{MemoryServer, RecallArgs};
use crate::tiering::usage::UsageRecorder;

const DIMS: usize = 8;
const TRACKED: usize = 100;
const WIDE_POOL_MATCHES: usize = 20;
const CUT_K: usize = 3;

struct Fixture {
    server: MemoryServer,
    store: Arc<DefaultGraph>,
    usage: Arc<UsageRecorder>,
}

async fn fixture() -> Fixture {
    let store = Arc::new(
        DefaultGraph::open(":memory:", GraphConfig::new(DIMS))
            .await
            .unwrap(),
    );
    let usage = Arc::new(UsageRecorder::new(TRACKED));
    let server = MemoryServer::new(
        store.clone(),
        Arc::new(MockEmbedder::new(DIMS)),
        Arc::new(MockLlm),
        1,
        8192,
        1,
    )
    .with_usage_recorder(usage.clone());
    Fixture {
        server,
        store,
        usage,
    }
}

async fn note(store: &DefaultGraph, label: &str, content: &str) -> NodeId {
    store
        .insert(NewNode::now("fact", label, content))
        .await
        .unwrap()
}

fn recall_args(query: &str, k: Option<usize>) -> RecallArgs {
    RecallArgs {
        query: query.to_string(),
        kind: None,
        scope: None,
        k,
        as_of: None,
        mode: None,
    }
}

fn counted(fixture: &Fixture) -> Vec<String> {
    fixture
        .usage
        .hits()
        .into_iter()
        .map(|(node, _)| node.as_str().to_string())
        .collect()
}

#[tokio::test]
async fn recall_counts_the_ranked_results_it_returns_and_no_more() {
    // Arrange: ten matches, three asked for
    let fixture = fixture().await;
    for index in 0..10 {
        note(
            &fixture.store,
            &format!("note {index}"),
            "the zorbnax gadget",
        )
        .await;
    }

    // Act
    let out = fixture
        .server
        .recall(Parameters(recall_args("zorbnax gadget", Some(3))))
        .await;

    // Assert
    assert_eq!(out.matches("[fact ").count(), 3, "{out}");
    let hits = fixture.usage.hits();
    assert_eq!(hits.len(), 3);
    assert!(hits.iter().all(|(_, hits)| *hits == 1));
}

#[tokio::test]
async fn recall_counts_only_the_cut_hits_when_it_fetches_a_wider_pool() {
    // Arrange: more matches than the candidate pool holds, three asked for
    let fixture = fixture().await;
    for index in 0..WIDE_POOL_MATCHES {
        note(
            &fixture.store,
            &format!("note {index}"),
            "the zorbnax gadget",
        )
        .await;
    }

    // Act
    let out = fixture
        .server
        .recall(Parameters(recall_args("zorbnax gadget", Some(CUT_K))))
        .await;

    // Assert: the counted ids are exactly the returned ones
    let counted = counted(&fixture);
    assert_eq!(counted.len(), CUT_K);
    assert_eq!(out.matches("[fact ").count(), CUT_K, "{out}");
    for id in &counted {
        let handle = &id[..HANDLE_LEN];
        assert!(out.contains(handle), "{handle} was counted but not shown");
    }
}

#[tokio::test]
async fn recall_does_not_count_a_neighbour_pulled_in_by_an_edge() {
    // Arrange
    let fixture = fixture().await;
    let seed = note(&fixture.store, "seed", "quartzmarker").await;
    let neighbour = note(&fixture.store, "neighbour", "unrelated words").await;
    fixture
        .store
        .link(NewEdge::new(&seed, &neighbour, "relates_to"))
        .await
        .unwrap();

    // Act
    let out = fixture
        .server
        .recall(Parameters(recall_args("quartzmarker", Some(2))))
        .await;

    // Assert: the neighbour is shown and not counted
    assert!(out.contains("unrelated words"), "{out}");
    assert_eq!(counted(&fixture), [seed.as_str()]);
}

#[tokio::test]
async fn a_recall_with_no_match_counts_nothing() {
    // Arrange
    let fixture = fixture().await;

    // Act
    let out = fixture
        .server
        .recall(Parameters(recall_args("nothing stored", None)))
        .await;

    // Assert
    assert_eq!(out, "no relevant memory");
    assert!(counted(&fixture).is_empty());
}

#[tokio::test]
async fn a_query_made_straight_on_the_store_counts_nothing() {
    // Arrange
    let fixture = fixture().await;
    note(&fixture.store, "note", "the zorbnax gadget").await;

    // Act
    let hits = fixture
        .store
        .query_explained(&Query::text("zorbnax"))
        .await
        .unwrap();

    // Assert
    assert_eq!(hits.len(), 1);
    assert!(counted(&fixture).is_empty());
}
