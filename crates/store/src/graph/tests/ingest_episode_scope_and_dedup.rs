// SPDX-License-Identifier: Apache-2.0
//! `ingest_episode`: scope checks, subject dedup, and vectors after commit.

use super::*;

#[tokio::test]
async fn ingest_episode_rejects_a_malformed_scope_before_any_transaction_opens() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let nodes = vec![NewNode::now("fact", "v1", "alice is 30").with_scope("proj//a")];

    // Act
    let err = g.ingest_episode(nodes, vec![]).await.unwrap_err();

    // Assert
    assert!(matches!(err, Error::InvalidScope(_)), "{err}");
    let node_rows = g
        .backend
        .query("SELECT COUNT(*) FROM nodes", &[])
        .await
        .unwrap();
    assert_eq!(
        node_rows[0].get_i64(0).unwrap(),
        0,
        "nothing must be written when scope validation fails"
    );
}

#[tokio::test]
async fn ingest_episode_rejects_the_whole_call_when_a_later_node_has_a_malformed_scope() {
    // Arrange: node 0 carries a valid scope, node 1's scope is
    // malformed. The pre-flight loop mutates nodes[0].scope in place
    // (trimming it) before it reaches nodes[1] and fails, so this pins
    // that the in-memory mutation never reaches the database: no
    // transaction opened, nothing to roll back.
    let g = graph_at(Millis(1000)).await;
    let nodes = vec![
        NewNode::now("fact", "v1", "alice is 30").with_scope("proj-a"),
        NewNode::now("fact", "v2", "bob is 40").with_scope("proj//a"),
    ];

    // Act
    let err = g.ingest_episode(nodes, vec![]).await.unwrap_err();

    // Assert
    assert!(matches!(err, Error::InvalidScope(_)), "{err}");
    let node_rows = g
        .backend
        .query("SELECT COUNT(*) FROM nodes", &[])
        .await
        .unwrap();
    assert_eq!(
        node_rows[0].get_i64(0).unwrap(),
        0,
        "nothing from the episode must exist, not even node 0"
    );
}

#[tokio::test]
async fn ingest_episode_normalizes_scope_before_finding_live_competitor() {
    // Arrange: a live node under a trimmed scope, then an episode whose
    // node names the same scope with trailing whitespace, mirroring
    // `upsert_by_normalizes_scope_before_finding_live_competitor` but
    // through `ingest_episode`'s own `open_by_subject_query` path.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let old = g
        .upsert_by(
            NewNode::now("fact", "v1", "widget price is 10")
                .with_subject("s1")
                .with_scope("proj-a"),
        )
        .await
        .unwrap();

    // Act
    clock.set(Millis(2000));
    let nodes = vec![NewNode::now("fact", "v2", "widget price is 20")
        .with_subject("s1")
        .with_scope("proj-a ")];
    let result = g.ingest_episode(nodes, vec![]).await.unwrap();
    let new_id = &result.node_ids[0];

    // Assert: the old node is superseded, not duplicated
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
        "trailing-space scope must match the live node, not duplicate it"
    );
    let new_rows = g
        .backend
        .query(
            "SELECT tx_to, scope FROM nodes WHERE id = ?1",
            &[new_id.as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(
        new_rows[0].get_i64(0).unwrap(),
        FOREVER.0,
        "new must be live"
    );
    assert_eq!(
        new_rows[0].get_string(1).unwrap(),
        "proj-a",
        "the stored scope must be trimmed"
    );
}

#[tokio::test]
async fn ingest_episode_keeps_facts_live_across_producers() {
    // Arrange: a live fact from producer "agent-a" in one episode call
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let first = g
        .ingest_episode(
            vec![NewNode::now("fact", "v1", "price is 10")
                .with_subject("price")
                .with_producer("agent-a")],
            vec![],
        )
        .await
        .unwrap();
    let a_id = &first.node_ids[0];

    // Act: a second episode call from producer "agent-b" carries a fact
    // with the same subject and scope
    clock.set(Millis(2000));
    let second = g
        .ingest_episode(
            vec![NewNode::now("fact", "v2", "price is 20")
                .with_subject("price")
                .with_producer("agent-b")],
            vec![],
        )
        .await
        .unwrap();
    let b_id = &second.node_ids[0];

    // Assert: both facts stay live
    let rows = g
        .backend
        .query(
            "SELECT id, tx_to FROM nodes WHERE id IN (?1, ?2)",
            &[a_id.as_str().into(), b_id.as_str().into()],
        )
        .await
        .unwrap();
    let tx_to_of = |id: &str| -> i64 {
        rows.iter()
            .find(|r| r.get_string(0).unwrap() == id)
            .unwrap()
            .get_i64(1)
            .unwrap()
    };
    assert_eq!(
        tx_to_of(a_id.as_str()),
        FOREVER.0,
        "producer agent-a's fact stays live"
    );
    assert_eq!(
        tx_to_of(b_id.as_str()),
        FOREVER.0,
        "producer agent-b's fact stays live"
    );
}

#[tokio::test]
async fn ingest_episode_collapses_entity_pages_across_producers_to_one_live() {
    // Arrange: a live "Alice" entity page from producer "agent-a"'s episode
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    g.ingest_episode(
        vec![NewNode::entity("person", "Alice")
            .with_scope("s")
            .with_producer("agent-a")],
        vec![],
    )
    .await
    .unwrap();

    // Act: producer "agent-b"'s episode carries the same entity page in the same scope
    clock.set(Millis(2000));
    g.ingest_episode(
        vec![NewNode::entity("person", "Alice")
            .with_scope("s")
            .with_producer("agent-b")],
        vec![],
    )
    .await
    .unwrap();

    // Assert: exactly one live "Alice" page remains, the "agent-a" page superseded
    let rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM nodes WHERE subject = ?1 AND scope = ?2 AND tx_to = ?3",
            &["alice".into(), "s".into(), FOREVER.into()],
        )
        .await
        .unwrap();
    assert_eq!(
        rows[0].get_i64(0).unwrap(),
        1,
        "exactly one live Alice page must remain, ignoring producer"
    );

    // Assert: the live page belongs to "agent-b", proving the correct
    // page won rather than some other count happening to equal one
    let live_producer = g
        .backend
        .query(
            "SELECT producer FROM nodes WHERE subject = ?1 AND scope = ?2 AND tx_to = ?3",
            &["alice".into(), "s".into(), FOREVER.into()],
        )
        .await
        .unwrap();
    assert_eq!(
        live_producer[0].get_string(0).unwrap(),
        "agent-b",
        "the live Alice page must belong to agent-b"
    );

    // Assert: agent-a's superseded page is still present in history
    let total_rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM nodes WHERE subject = ?1 AND scope = ?2",
            &["alice".into(), "s".into()],
        )
        .await
        .unwrap();
    assert_eq!(
        total_rows[0].get_i64(0).unwrap(),
        2,
        "agent-a's superseded Alice page must remain in history alongside agent-b's"
    );
}

#[tokio::test]
async fn ingest_episode_picks_collision_key_per_node_not_per_call() {
    // Arrange: producer "agent-a" already has a live "price" fact and a
    // live "Alice" entity page in scope "s"
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    g.ingest_episode(
        vec![
            NewNode::now("fact", "v1", "price is 10")
                .with_subject("price")
                .with_scope("s")
                .with_producer("agent-a"),
            NewNode::entity("person", "Alice")
                .with_scope("s")
                .with_producer("agent-a"),
        ],
        vec![],
    )
    .await
    .unwrap();

    // Act: one episode call from producer "agent-b" carries both a fact
    // with subject "price" and an "Alice" entity page in the same scope
    clock.set(Millis(2000));
    g.ingest_episode(
        vec![
            NewNode::now("fact", "v2", "price is 20")
                .with_subject("price")
                .with_scope("s")
                .with_producer("agent-b"),
            NewNode::entity("person", "Alice")
                .with_scope("s")
                .with_producer("agent-b"),
        ],
        vec![],
    )
    .await
    .unwrap();

    // Assert: both "price" facts stay live
    let fact_rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM nodes WHERE subject = ?1 AND scope = ?2 AND tx_to = ?3",
            &["price".into(), "s".into(), FOREVER.into()],
        )
        .await
        .unwrap();
    assert_eq!(
        fact_rows[0].get_i64(0).unwrap(),
        2,
        "both price facts must stay live, one per producer"
    );

    // Assert: exactly one "Alice" entity page stays live
    let entity_rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM nodes WHERE subject = ?1 AND scope = ?2 AND tx_to = ?3",
            &["alice".into(), "s".into(), FOREVER.into()],
        )
        .await
        .unwrap();
    assert_eq!(
        entity_rows[0].get_i64(0).unwrap(),
        1,
        "exactly one Alice page must stay live, ignoring producer"
    );

    // Assert: the live Alice page belongs to "agent-b", proving the
    // correct page won rather than some other count happening to equal one
    let live_entity_producer = g
        .backend
        .query(
            "SELECT producer FROM nodes WHERE subject = ?1 AND scope = ?2 AND tx_to = ?3",
            &["alice".into(), "s".into(), FOREVER.into()],
        )
        .await
        .unwrap();
    assert_eq!(
        live_entity_producer[0].get_string(0).unwrap(),
        "agent-b",
        "the live Alice page must belong to agent-b"
    );
}

#[tokio::test]
async fn ingest_episode_entity_write_does_not_supersede_a_coincidentally_same_subject_fact() {
    // Arrange: producer "agent-a" writes an entity page for "Alice" in scope "s"
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    g.ingest_episode(
        vec![NewNode::entity("person", "Alice")
            .with_scope("s")
            .with_producer("agent-a")],
        vec![],
    )
    .await
    .unwrap();

    // Act: producer "agent-b" writes, in a separate episode, an unrelated
    // fact whose subject coincidentally collides with the entity's subject
    clock.set(Millis(2000));
    let fact_result = g
        .ingest_episode(
            vec![NewNode::now("fact", "some label", "some content")
                .with_subject("alice")
                .with_scope("s")
                .with_producer("agent-b")],
            vec![],
        )
        .await
        .unwrap();
    let fact_id = fact_result.node_ids[0].clone();

    // Act: producer "agent-c" re-observes the same entity in a third episode
    clock.set(Millis(3000));
    g.ingest_episode(
        vec![NewNode::entity("person", "Alice")
            .with_scope("s")
            .with_producer("agent-c")],
        vec![],
    )
    .await
    .unwrap();

    // Assert: exactly one live entity page remains for subject "alice"/scope "s"
    let entity_rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM nodes WHERE subject = ?1 AND scope = ?2 AND tx_to = ?3 AND content = ''",
            &["alice".into(), "s".into(), FOREVER.into()],
        )
        .await
        .unwrap();
    assert_eq!(
        entity_rows[0].get_i64(0).unwrap(),
        1,
        "exactly one live Alice entity page must remain"
    );

    // Assert: agent-b's fact is still live, untouched by the entity collision
    let fact_rows = g
        .backend
        .query(
            "SELECT tx_to FROM nodes WHERE id = ?1",
            &[fact_id.as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(
        fact_rows[0].get_i64(0).unwrap(),
        FOREVER.0,
        "agent-b's fact must stay live, not superseded by an unrelated entity write"
    );
}

#[tokio::test]
async fn a_subject_less_node_is_a_plain_insert() {
    // Arrange: a subject-less node alongside a subject-bearing node that
    // does supersede something, so the assertion below distinguishes
    // "no supersedes edges at all" from "the subject-less node is never
    // one of their endpoints."
    let g = graph_at(Millis(1000)).await;
    let old = g
        .insert(NewNode::now("fact", "v1", "alice is 30").with_subject("alice"))
        .await
        .unwrap();
    let nodes = vec![
        NewNode::now("fact", "NoSubject", "x"),
        NewNode::now("fact", "v2", "alice is 31").with_subject("alice"),
    ];

    // Act
    let result = g.ingest_episode(nodes, vec![]).await.unwrap();
    let no_subject_id = &result.node_ids[0];
    let with_subject_id = &result.node_ids[1];

    // Assert
    let supersede_rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE type = 'supersedes' AND src = ?1 AND dst = ?2",
            &[with_subject_id.as_str().into(), old.as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(
        supersede_rows[0].get_i64(0).unwrap(),
        1,
        "the subject-bearing node must still supersede the prior live node"
    );
    let no_subject_rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE type = 'supersedes' AND (src = ?1 OR dst = ?1)",
            &[no_subject_id.as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(
        no_subject_rows[0].get_i64(0).unwrap(),
        0,
        "the subject-less node must never be a supersedes edge endpoint"
    );
}

#[tokio::test]
async fn vectors_are_written_after_commit_for_every_embedded_node() {
    // Arrange: two nodes with distinct embeddings, dims = 8 per
    // `graph_at`'s `GraphConfig::new(8)`.
    let g = graph_at(Millis(1000)).await;
    let e1 = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let e2 = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let nodes = vec![
        NewNode::now("fact", "One", "x").with_embedding(e1.clone()),
        NewNode::now("fact", "Two", "y").with_embedding(e2.clone()),
    ];

    // Act
    let result = g.ingest_episode(nodes, vec![]).await.unwrap();

    // Assert
    let hits1 = g
        .backend
        .vector_search(&e1, 5, None, None, Millis(1000), InferredRows::Include)
        .await
        .unwrap();
    assert!(
        hits1.contains(&result.node_ids[0]),
        "the first node's embedding must be searchable after commit"
    );
    let hits2 = g
        .backend
        .vector_search(&e2, 5, None, None, Millis(1000), InferredRows::Include)
        .await
        .unwrap();
    assert!(
        hits2.contains(&result.node_ids[1]),
        "the second node's embedding must be searchable after commit"
    );
}

#[tokio::test]
async fn a_vector_upsert_failure_does_not_fail_the_call_or_orphan_later_vectors() {
    // Arrange: 3 new embedded facts; the double fails the FIRST
    // `vector_upsert` call (node 1's) and lets the rest through, dims = 8
    // per `graph_at`'s `GraphConfig::new(8)`.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g: Graph<FailingVectorBackend> =
        Graph::open_with_clock(":memory:", GraphConfig::new(8), clock)
            .await
            .unwrap();
    g.backend.set_fail_on_vector_upsert(0);
    let e1 = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let e2 = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let e3 = vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let nodes = vec![
        NewNode::now("fact", "One", "x").with_embedding(e1),
        NewNode::now("fact", "Two", "y").with_embedding(e2.clone()),
        NewNode::now("fact", "Three", "z").with_embedding(e3.clone()),
    ];

    // Act: node 1's vector_upsert fails, but the call as a whole must
    // still report success, since the transaction already committed.
    let result = g.ingest_episode(nodes, vec![]).await.unwrap();

    // Assert: all 3 node ROWS exist, proving the commit was unaffected.
    let node_rows = g
        .backend
        .query("SELECT COUNT(*) FROM nodes", &[])
        .await
        .unwrap();
    assert_eq!(
        node_rows[0].get_i64(0).unwrap(),
        3,
        "all 3 node rows must be committed regardless of the later vector failure"
    );

    // Nodes 2 and 3's vectors ARE present, proving the loop kept going
    // past node 1's failure instead of aborting.
    let hits2 = g
        .backend
        .vector_search(&e2, 5, None, None, Millis(1000), InferredRows::Include)
        .await
        .unwrap();
    assert!(
        hits2.contains(&result.node_ids[1]),
        "node 2's vector must be written even though node 1's vector_upsert failed first"
    );
    let hits3 = g
        .backend
        .vector_search(&e3, 5, None, None, Millis(1000), InferredRows::Include)
        .await
        .unwrap();
    assert!(
        hits3.contains(&result.node_ids[2]),
        "node 3's vector must be written even though node 1's vector_upsert failed first"
    );
}

#[tokio::test]
async fn a_check_dims_failure_does_not_fail_the_call() {
    // Arrange: dims = 8 per `graph_at`'s `GraphConfig::new(8)`; this
    // embedding is the wrong length, so `check_dims` rejects it before
    // `vector_upsert` is ever called.
    let g = graph_at(Millis(1000)).await;
    let wrong_length_embedding = vec![1.0f32, 0.0, 0.0, 0.0];
    let nodes = vec![NewNode::now("fact", "One", "x").with_embedding(wrong_length_embedding)];

    // Act: a dims mismatch must be handled the same way as a
    // vector_upsert failure, logged and skipped, not propagated.
    let result = g.ingest_episode(nodes, vec![]).await.unwrap();

    // Assert: the row is committed despite the dims mismatch.
    let node_rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM nodes WHERE id = ?1",
            &[result.node_ids[0].as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(
        node_rows[0].get_i64(0).unwrap(),
        1,
        "the node row must be committed despite the dims mismatch"
    );
}

#[tokio::test]
async fn vector_search_scope_filter_includes_descendant_scopes() {
    // Arrange: the same embedding at an exact match ("work"), a true
    // descendant ("work/liam"), a false string prefix with no "/"
    // boundary ("work-archive"), and a query scope that is too short to
    // be an ancestor of any of them ("wo"). Dims = 8 per `graph_at`'s
    // `GraphConfig::new(8)`.
    let g = graph_at(Millis(1000)).await;
    let e = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let work_id = g
        .insert(
            NewNode::now("fact", "Exact", "x")
                .with_scope("work")
                .with_embedding(e.clone()),
        )
        .await
        .unwrap();
    let work_liam_id = g
        .insert(
            NewNode::now("fact", "Child", "x")
                .with_scope("work/liam")
                .with_embedding(e.clone()),
        )
        .await
        .unwrap();
    g.insert(
        NewNode::now("fact", "Archived", "x")
            .with_scope("work-archive")
            .with_embedding(e.clone()),
    )
    .await
    .unwrap();
    g.insert(
        NewNode::now("fact", "TooShort", "x")
            .with_scope("wo")
            .with_embedding(e.clone()),
    )
    .await
    .unwrap();

    // Act: search with the ancestor scope.
    let hits = g
        .backend
        .vector_search(
            &e,
            10,
            None,
            Some("work"),
            Millis(1000),
            InferredRows::Include,
        )
        .await
        .unwrap();

    // Assert: exactly the exact-match and descendant-scoped nodes come
    // back, proving both inclusion and exclusion in one assertion.
    let hit_set: HashSet<NodeId> = hits.into_iter().collect();
    let expected: HashSet<NodeId> = [work_id, work_liam_id].into_iter().collect();
    assert_eq!(hit_set, expected);
}
