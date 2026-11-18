// SPDX-License-Identifier: Apache-2.0
//! `ingest_episode`: the atomic write of nodes and edges, and its failures.

use super::*;

// ---- WU-2: Graph::ingest_episode core (no dedup) ----

#[tokio::test]
async fn two_new_facts_and_an_edge_between_them_commit_together() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let nodes = vec![
        NewNode::now("fact", "Alpha", "alpha content"),
        NewNode::now("fact", "Beta", "beta content"),
    ];
    let edges = vec![EpisodeEdge {
        from: EpisodeRef::New(0),
        to: EpisodeRef::New(1),
        kind: "mentions".to_string(),
        attributes: serde_json::json!({}),
    }];

    // Act
    let result = g.ingest_episode(nodes, edges).await.unwrap();

    // Assert
    assert_eq!(result.node_ids.len(), 2);
    assert_eq!(result.edge_ids.len(), 1);
    let node_rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM nodes WHERE id IN (?1, ?2)",
            &[
                result.node_ids[0].as_str().into(),
                result.node_ids[1].as_str().into(),
            ],
        )
        .await
        .unwrap();
    assert_eq!(node_rows[0].get_i64(0).unwrap(), 2);
    let edge_rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE src = ?1 AND dst = ?2 AND type = 'mentions'",
            &[
                result.node_ids[0].as_str().into(),
                result.node_ids[1].as_str().into(),
            ],
        )
        .await
        .unwrap();
    assert_eq!(edge_rows[0].get_i64(0).unwrap(), 1);
}

#[tokio::test]
async fn an_edge_can_reference_an_existing_node() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let existing = g
        .insert(NewNode::now("fact", "Existing", "x"))
        .await
        .unwrap();
    let nodes = vec![NewNode::now("fact", "New", "y")];
    let edges = vec![EpisodeEdge {
        from: EpisodeRef::New(0),
        to: EpisodeRef::Existing(existing.clone()),
        kind: "mentions".to_string(),
        attributes: serde_json::json!({}),
    }];

    // Act
    let result = g.ingest_episode(nodes, edges).await.unwrap();

    // Assert
    let rows = g
        .backend
        .query(
            "SELECT dst FROM edges WHERE src = ?1",
            &[result.node_ids[0].as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(rows[0].get_string(0).unwrap(), existing.as_str());
}

#[tokio::test]
async fn an_out_of_bounds_new_index_returns_an_error_not_a_panic() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let nodes = vec![NewNode::now("fact", "Only", "x")];
    let edges = vec![EpisodeEdge {
        from: EpisodeRef::New(0),
        to: EpisodeRef::New(5),
        kind: "mentions".to_string(),
        attributes: serde_json::json!({}),
    }];

    // Act
    let err = g.ingest_episode(nodes, edges).await.unwrap_err();

    // Assert
    assert!(matches!(err, Error::InvalidReference(_)), "{err}");
    let rows = g
        .backend
        .query("SELECT COUNT(*) FROM nodes", &[])
        .await
        .unwrap();
    assert_eq!(rows[0].get_i64(0).unwrap(), 0, "nothing must be written");
}

#[tokio::test]
async fn an_edge_to_a_stale_existing_endpoint_fails_the_whole_call() {
    // Arrange: the same setup relate_refuses_an_endpoint_that_is_no_longer_live
    // uses, so `old` is no longer live by the time ingest_episode runs.
    let g = graph_at(Millis(1000)).await;
    let old = g
        .upsert_by(NewNode::now("fact", "Rollout", "first").with_subject("rollout"))
        .await
        .unwrap();
    g.upsert_by(NewNode::now("fact", "Rollout", "second").with_subject("rollout"))
        .await
        .unwrap();
    let nodes = vec![NewNode::now("fact", "NewFact", "y")];
    let edges = vec![EpisodeEdge {
        from: EpisodeRef::New(0),
        to: EpisodeRef::Existing(old.clone()),
        kind: "mentions".to_string(),
        attributes: serde_json::json!({}),
    }];

    // Act
    let err = g.ingest_episode(nodes, edges).await.unwrap_err();

    // Assert: matches edge_refusal's exact wording, not a generic message.
    assert!(
        matches!(&err, Error::RelateRefused(m)
            if m.contains("is not live") && m.contains(old.as_str())),
        "{err}"
    );
    let node_rows = g
        .backend
        .query("SELECT COUNT(*) FROM nodes WHERE label = 'NewFact'", &[])
        .await
        .unwrap();
    assert_eq!(
        node_rows[0].get_i64(0).unwrap(),
        0,
        "the new node must not exist"
    );
    // Filtered to `mentions`, not just `dst = old`: the setup above
    // already left a `supersedes` edge pointing at `old`, which is not
    // what this assertion is about.
    let edge_rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE dst = ?1 AND type = 'mentions'",
            &[old.as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(
        edge_rows[0].get_i64(0).unwrap(),
        0,
        "the mentions edge must not exist"
    );
}

#[tokio::test]
async fn two_edges_with_the_same_triple_in_one_call_the_second_is_refused_not_duplicated() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let nodes = vec![
        NewNode::now("fact", "Alpha", "x"),
        NewNode::now("fact", "Beta", "y"),
    ];
    let edges = vec![
        EpisodeEdge {
            from: EpisodeRef::New(0),
            to: EpisodeRef::New(1),
            kind: "mentions".to_string(),
            attributes: serde_json::json!({}),
        },
        EpisodeEdge {
            from: EpisodeRef::New(0),
            to: EpisodeRef::New(1),
            kind: "mentions".to_string(),
            attributes: serde_json::json!({}),
        },
    ];

    // Act
    let err = g.ingest_episode(nodes, edges).await.unwrap_err();

    // Assert: the whole call rolls back, so not even the first edge lands.
    assert!(matches!(&err, Error::RelateRefused(_)), "{err}");
    let node_rows = g
        .backend
        .query("SELECT COUNT(*) FROM nodes", &[])
        .await
        .unwrap();
    assert_eq!(
        node_rows[0].get_i64(0).unwrap(),
        0,
        "no node from the episode must exist"
    );
    let edge_rows = g
        .backend
        .query("SELECT COUNT(*) FROM edges", &[])
        .await
        .unwrap();
    assert_eq!(
        edge_rows[0].get_i64(0).unwrap(),
        0,
        "not even the first edge must exist"
    );
}

#[tokio::test]
async fn a_mid_transaction_db_failure_leaves_zero_rows() {
    // Arrange: fail the SECOND node's insert, after the first has already
    // run for real inside the transaction.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g: Graph<FailingBackend> = Graph::open_with_clock(":memory:", GraphConfig::new(8), clock)
        .await
        .unwrap();
    g.backend.set_fail_on_execute(1);
    let nodes = vec![
        NewNode::now("fact", "One", "x"),
        NewNode::now("fact", "Two", "y"),
        NewNode::now("fact", "Three", "z"),
    ];

    // Act
    let err = g.ingest_episode(nodes, vec![]).await.unwrap_err();

    // Assert: query the real backend directly (FailingBackend's own query
    // is a pure passthrough, never intercepted), proving a real rollback
    // of the first, individually-valid statement.
    assert!(matches!(err, Error::Backend(_)), "{err}");
    let rows = g
        .backend
        .query("SELECT COUNT(*) FROM nodes", &[])
        .await
        .unwrap();
    assert_eq!(
        rows[0].get_i64(0).unwrap(),
        0,
        "no node from this call must exist"
    );
}

#[tokio::test]
async fn empty_nodes_and_edges_succeeds_trivially() {
    // Arrange
    let g = graph_at(Millis(1000)).await;

    // Act
    let result = g.ingest_episode(vec![], vec![]).await.unwrap();

    // Assert
    assert!(result.node_ids.is_empty());
    assert!(result.edge_ids.is_empty());
}

// ---- WU-3: subject-based dedup inside ingest_episode ----

#[tokio::test]
async fn a_node_with_a_subject_already_live_outside_the_call_gets_superseded() {
    // Arrange: a node with subject "alice" already live before this call.
    let g = graph_at(Millis(1000)).await;
    let old = g
        .insert(NewNode::now("fact", "v1", "alice is 30").with_subject("alice"))
        .await
        .unwrap();
    let nodes = vec![NewNode::now("fact", "v2", "alice is 31").with_subject("alice")];

    // Act
    let result = g.ingest_episode(nodes, vec![]).await.unwrap();
    let new_id = &result.node_ids[0];

    // Assert
    let old_rows = g
        .backend
        .query(
            "SELECT tx_to FROM nodes WHERE id = ?1",
            &[old.as_str().into()],
        )
        .await
        .unwrap();
    assert_ne!(
        old_rows[0].get_i64(0).unwrap(),
        FOREVER.0,
        "old must be closed"
    );
    let new_rows = g
        .backend
        .query(
            "SELECT tx_to FROM nodes WHERE id = ?1",
            &[new_id.as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(
        new_rows[0].get_i64(0).unwrap(),
        FOREVER.0,
        "new must be live"
    );
    let edge_rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE type = 'supersedes' AND src = ?1 AND dst = ?2",
            &[new_id.as_str().into(), old.as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(
        edge_rows[0].get_i64(0).unwrap(),
        1,
        "a supersedes edge must link new -> old"
    );
}

#[tokio::test]
async fn two_items_in_one_call_with_the_same_subject_supersede_in_order() {
    // Arrange: two nodes in one call, both subject "alice", no prior state.
    let g = graph_at(Millis(1000)).await;
    let nodes = vec![
        NewNode::now("fact", "v1", "alice is 30").with_subject("alice"),
        NewNode::now("fact", "v2", "alice is 31").with_subject("alice"),
    ];

    // Act
    let result = g.ingest_episode(nodes, vec![]).await.unwrap();
    let (id0, id1) = (&result.node_ids[0], &result.node_ids[1]);

    // Assert: index 0 becomes live first, then index 1 supersedes it
    // within the same transaction.
    let id0_rows = g
        .backend
        .query(
            "SELECT tx_to FROM nodes WHERE id = ?1",
            &[id0.as_str().into()],
        )
        .await
        .unwrap();
    assert_ne!(
        id0_rows[0].get_i64(0).unwrap(),
        FOREVER.0,
        "index 0 must be superseded"
    );
    let id1_rows = g
        .backend
        .query(
            "SELECT tx_to FROM nodes WHERE id = ?1",
            &[id1.as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(
        id1_rows[0].get_i64(0).unwrap(),
        FOREVER.0,
        "index 1 must be live"
    );
    let edge_rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE type = 'supersedes' AND src = ?1 AND dst = ?2",
            &[id1.as_str().into(), id0.as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(
        edge_rows[0].get_i64(0).unwrap(),
        1,
        "index 1 must supersede index 0 within the same call"
    );
}

#[tokio::test]
async fn an_edge_to_an_already_superseded_same_episode_reference_fails_the_whole_call() {
    // Arrange: node 0 and node 1 share a subject, so node 1 supersedes
    // node 0 during the node-writing phase; the edge references node 0
    // by its original index, after it has already been closed.
    let g = graph_at(Millis(1000)).await;
    let nodes = vec![
        NewNode::now("fact", "v1", "alice is 30").with_subject("alice"),
        NewNode::now("fact", "v2", "alice is 31").with_subject("alice"),
        NewNode::now("fact", "Unrelated", "z"),
    ];
    let edges = vec![EpisodeEdge {
        from: EpisodeRef::New(0),
        to: EpisodeRef::New(2),
        kind: "mentions".to_string(),
        attributes: serde_json::json!({}),
    }];

    // Act
    let err = g.ingest_episode(nodes, edges).await.unwrap_err();

    // Assert: the whole call fails, naming a specific (not generic) id,
    // matching edge_refusal's wording. Node 0's real id is generated
    // inside ingest_episode and never returned on failure since the
    // whole episode rolls back, so this checks message shape (a real
    // 26-character ULID, not a placeholder) rather than equality
    // against an independently captured id.
    let msg = match &err {
        Error::RelateRefused(m) => m.clone(),
        other => panic!("expected RelateRefused: {other:?}"),
    };
    assert!(msg.contains("is not live"), "{msg}");
    let id_token = msg.split_whitespace().nth(2).expect("id token in message");
    assert_eq!(id_token.len(), 26, "{msg}");
    assert!(id_token.chars().all(|c| c.is_ascii_alphanumeric()), "{msg}");

    // Nothing from the episode was written, including node 1 and the
    // edge, both of which would have been individually valid alone.
    let node_rows = g
        .backend
        .query("SELECT COUNT(*) FROM nodes", &[])
        .await
        .unwrap();
    assert_eq!(
        node_rows[0].get_i64(0).unwrap(),
        0,
        "nothing from the episode must exist"
    );
    let edge_rows = g
        .backend
        .query("SELECT COUNT(*) FROM edges", &[])
        .await
        .unwrap();
    assert_eq!(
        edge_rows[0].get_i64(0).unwrap(),
        0,
        "not even the supersedes edge from node 1 -> node 0 must exist"
    );
}
