// SPDX-License-Identifier: Apache-2.0
//! File-backed graphs, reads by id, and reads as of a past instant.

use super::*;

#[tokio::test]
async fn file_backed_graph_inserts_and_queries() {
    // Arrange
    let (_dir, g) = file_graph_at(Millis(1000)).await;

    // Act
    g.insert(NewNode::now("decision", "Use libSQL", "single file"))
        .await
        .unwrap();
    let hits = g.query(&Query::text("libSQL")).await.unwrap();

    // Assert
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].label, "Use libSQL");
}

#[tokio::test]
async fn file_backed_graph_removes_temp_dir_when_guard_drops() {
    // Arrange
    let (dir, g) = file_graph_at(Millis(1000)).await;
    let dir_path = dir.path().to_path_buf();

    // Act
    drop(g);
    drop(dir);

    // Assert
    assert!(!dir_path.exists());
}

#[tokio::test]
async fn insert_then_query() {
    let g = graph_at(Millis(1000)).await;
    g.insert(NewNode::now("decision", "Use libSQL", "single file"))
        .await
        .unwrap();
    let hits = g.query(&Query::text("libSQL")).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].label, "Use libSQL");
}

#[tokio::test]
async fn kind_filter_holds_for_graph_expanded_neighbours() {
    // Arrange: a matching `decision` linked to an `episode` that the query
    // text does not match. The episode enters the candidate set only through
    // graph expansion off the seed.
    let g = graph_at(Millis(1000)).await;
    let seed = g
        .insert(NewNode::now(
            "decision",
            "Rollout",
            "zorbnax rollout approved",
        ))
        .await
        .unwrap();
    let neighbour = g
        .insert(NewNode::now(
            "episode",
            "Chatter",
            "unrelated standup notes",
        ))
        .await
        .unwrap();
    g.link(NewEdge::new(&seed, &neighbour, "mentions"))
        .await
        .unwrap();

    // Act
    let hits = g
        .query(&Query::text("zorbnax rollout").with_kind("decision"))
        .await
        .unwrap();

    // Assert: `with_kind` is a filter on the result set, not merely on the
    // retrieval paths, so an expanded neighbour of another kind must not ride
    // along.
    let kinds: Vec<&str> = hits.iter().map(|h| h.kind.as_str()).collect();
    assert!(
        hits.iter().all(|h| h.kind == "decision"),
        "kind filter leaked expanded neighbours: {kinds:?}"
    );
}

#[tokio::test]
async fn scope_filter_holds_for_graph_expanded_neighbours() {
    // Arrange: same shape as the kind test, partitioned by scope instead.
    // This one guards behaviour that already held, so the shared filter
    // refactor cannot regress it silently.
    let g = graph_at(Millis(1000)).await;
    let seed = g
        .insert(
            NewNode::now("decision", "Rollout", "zorbnax rollout approved").with_scope("proj-a"),
        )
        .await
        .unwrap();
    let neighbour = g
        .insert(NewNode::now("decision", "Chatter", "unrelated notes").with_scope("proj-b"))
        .await
        .unwrap();
    g.link(NewEdge::new(&seed, &neighbour, "mentions"))
        .await
        .unwrap();

    // Act
    let hits = g
        .query(&Query::text("zorbnax rollout").with_scope("proj-a"))
        .await
        .unwrap();

    // Assert
    let labels: Vec<&str> = hits.iter().map(|h| h.label.as_str()).collect();
    assert!(
        hits.iter().all(|h| h.label != "Chatter"),
        "scope filter leaked expanded neighbours: {labels:?}"
    );
    assert!(
        hits.iter().any(|h| h.label == "Rollout"),
        "in-scope match missing: {labels:?}"
    );
}

#[tokio::test]
async fn as_of_recovers_superseded_history() {
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let old = g
        .insert(NewNode::now("decision", "Deno", "runtime").with_valid_from(Millis(1000)))
        .await
        .unwrap();
    clock.set(Millis(2000));
    g.supersede(
        &old,
        NewNode::now("decision", "Rust", "runtime").with_valid_from(Millis(2000)),
    )
    .await
    .unwrap();

    let now_hits = g.query(&Query::text("runtime")).await.unwrap();
    assert!(now_hits.iter().any(|h| h.label == "Rust"));
    assert!(now_hits.iter().all(|h| h.label != "Deno"));

    let past = g
        .query(&Query::text("runtime").with_as_of(Millis(1500)))
        .await
        .unwrap();
    assert!(past.iter().any(|h| h.label == "Deno"));
    assert!(past.iter().all(|h| h.label != "Rust"));
}

#[tokio::test]
async fn get_returns_live_node_at_as_of() {
    // Arrange: a node live at the query instant.
    let g = graph_at(Millis(1000)).await;
    let id = g
        .insert(NewNode::now("decision", "Use libSQL", "single file"))
        .await
        .unwrap();

    // Act
    let found = g.get(&id, Millis(1000)).await.unwrap();

    // Assert
    let candidate = found.expect("live node must be found");
    assert_eq!(candidate.id, id);
    assert_eq!(candidate.kind, "decision");
    assert_eq!(candidate.label, "Use libSQL");
    assert_eq!(candidate.retrieved.as_text(), "single file");
}

#[tokio::test]
async fn get_returns_candidate_scope_matching_stored_scope() {
    // Arrange: a live node stored under a scope.
    let g = graph_at(Millis(1000)).await;
    let id = g
        .insert(NewNode::now("decision", "Use libSQL", "single file").with_scope("proj-a"))
        .await
        .unwrap();

    // Act
    let found = g.get(&id, Millis(1000)).await.unwrap();

    // Assert
    let candidate = found.expect("live node must be found");
    assert_eq!(candidate.scope, Some("proj-a".to_string()));
}

#[tokio::test]
async fn get_returns_none_scope_for_unscoped_node() {
    // Arrange: a live node stored with no scope, the common case.
    let g = graph_at(Millis(1000)).await;
    let id = g
        .insert(NewNode::now("decision", "Use libSQL", "single file"))
        .await
        .unwrap();

    // Act
    let found = g.get(&id, Millis(1000)).await.unwrap();

    // Assert
    let candidate = found.expect("live node must be found");
    assert_eq!(candidate.scope, None);
}

#[tokio::test]
async fn get_returns_none_for_node_superseded_before_as_of() {
    // Arrange: advance the clock so supersede closes tx_to, not just
    // valid_from, since only tx_from/tx_to gate the live window here.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let old = g
        .insert(NewNode::now("decision", "Deno", "runtime").with_valid_from(Millis(1000)))
        .await
        .unwrap();
    clock.set(Millis(2000));
    g.supersede(
        &old,
        NewNode::now("decision", "Rust", "runtime").with_valid_from(Millis(2000)),
    )
    .await
    .unwrap();

    // Act
    let found = g.get(&old, Millis(2500)).await.unwrap();

    // Assert
    assert!(found.is_none());
}

#[tokio::test]
async fn get_returns_none_for_unknown_id() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let unknown = NodeId::from_raw("nonexistent");

    // Act
    let found = g.get(&unknown, Millis(1000)).await.unwrap();

    // Assert
    assert!(found.is_none());
}

#[tokio::test]
async fn as_of_recovers_superseded_history_vector() {
    // Same contract as the text-only test above, but exercised purely
    // through the vector channel: the vector path must obey the same
    // "live at T" predicate as the lexical path.
    let e = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let old = g
        .insert(
            NewNode::now("decision", "Deno", "runtime")
                .with_valid_from(Millis(1000))
                .with_embedding(e.clone()),
        )
        .await
        .unwrap();
    clock.set(Millis(2000));
    g.supersede(
        &old,
        NewNode::now("decision", "Rust", "runtime")
            .with_valid_from(Millis(2000))
            .with_embedding(e.clone()),
    )
    .await
    .unwrap();

    let now_hits = g.query(&Query::vector(e.clone())).await.unwrap();
    assert!(
        now_hits.iter().any(|h| h.label == "Rust"),
        "current: Rust live"
    );
    assert!(
        now_hits.iter().all(|h| h.label != "Deno"),
        "current: Deno superseded"
    );

    let past = g
        .query(&Query::vector(e.clone()).with_as_of(Millis(1500)))
        .await
        .unwrap();
    assert!(
        past.iter().any(|h| h.label == "Deno"),
        "as-of 1500: Deno was live"
    );
    assert!(
        past.iter().all(|h| h.label != "Rust"),
        "as-of 1500: Rust not yet recorded"
    );
}

#[tokio::test]
async fn query_returns_all_matching_candidates() {
    // Guards the batched candidate fetch: several live nodes match one
    // query and all must come back.
    let g = graph_at(Millis(1000)).await;
    for i in 0..5 {
        g.insert(NewNode::now(
            "fact",
            format!("n{i}"),
            "shared keyword topic",
        ))
        .await
        .unwrap();
    }
    let hits = g.query(&Query::text("keyword").with_k(10)).await.unwrap();
    assert_eq!(hits.len(), 5);
}
