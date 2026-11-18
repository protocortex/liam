// SPDX-License-Identifier: Apache-2.0
//! Edge and mention fingerprints, and which entities are stale.

use super::*;

// ---- WU-7: the fingerprint seam ----

#[tokio::test]
async fn the_fingerprint_of_a_store_with_no_edges_is_zero_and_not_an_error() {
    // `MAX()` over zero rows is SQL NULL, which `Row::get_i64` rejects. A
    // plain `get_i64` here would fail on every store until its first
    // `relate`. ADR-0002 Amendment 3.
    let g = graph_at(Millis(1000)).await;

    let fp = g.edge_fingerprint().await.unwrap();

    assert_eq!(fp.edge_count, 0);
    assert_eq!(fp.max_tx_from, Millis(0));
}

#[tokio::test]
async fn the_fingerprint_ignores_supersedes_edges() {
    // Without this filter every ordinary `remember` that updates an
    // existing subject would invalidate an assignment it cannot affect,
    // turning normal write traffic into repeated Leiden runs.
    let g = graph_at(Millis(1000)).await;
    for content in ["first", "second", "third"] {
        g.upsert_by(NewNode::now("fact", "Rollout", content).with_subject("rollout"))
            .await
            .unwrap();
    }

    let fp = g.edge_fingerprint().await.unwrap();

    assert_eq!(fp.edge_count, 0, "version history is not a semantic edge");
}

#[tokio::test]
async fn the_fingerprint_falls_when_gc_sweeps_an_edge() {
    // The reason COUNT(*) is in the fingerprint at all. `gc` hard-deletes,
    // and removing a row never advances a maximum, so a timestamp alone
    // cannot see a deletion. ADR-0002 records this as the trap its first
    // draft fell into.
    // The clock has to move: `gc` sweeps `valid_from < now - max_age`, so
    // with a fixed clock the nodes are never old enough to expire.
    let t0 = Millis(1_000_000);
    let clock = Arc::new(FixedClock::new(t0));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let a = g.insert(NewNode::now("fact", "a", "x")).await.unwrap();
    let b = g.insert(NewNode::now("fact", "b", "x")).await.unwrap();
    g.relate(&a, &b, "mentions").await.unwrap();
    let before = g.edge_fingerprint().await.unwrap();
    assert_eq!(before.edge_count, 1);

    clock.set(Millis(t0.0 + 10_000));
    g.gc(&RetentionPolicy::keep("fact", Millis(1)))
        .await
        .unwrap();

    let after = g.edge_fingerprint().await.unwrap();
    assert_eq!(after.edge_count, 0, "a swept edge must lower the count");
    assert_ne!(before, after, "the fingerprint must notice a deletion");
}

#[tokio::test]
async fn the_fingerprint_ignores_an_edge_closed_in_transaction_time() {
    // Nothing in production closes an edge today, so this goes through the
    // backend directly. It pins the `tx_to` half of the predicate, which is
    // otherwise unexercised.
    let g = graph_at(Millis(1000)).await;
    let a = g.insert(NewNode::now("fact", "a", "x")).await.unwrap();
    let b = g.insert(NewNode::now("fact", "b", "x")).await.unwrap();
    g.relate(&a, &b, "mentions").await.unwrap();

    g.backend
        .execute("UPDATE edges SET tx_to = ?1", &[Millis(2000).into()])
        .await
        .unwrap();

    assert_eq!(g.edge_fingerprint().await.unwrap().edge_count, 0);
}

#[tokio::test]
async fn a_store_that_has_never_clustered_reads_as_no_prior_run() {
    // `None` must never be defaulted into a zero fingerprint. A database
    // written before `cluster_state` existed arrives with a populated
    // assignment and no state row, and on an edgeless store a defaulted
    // `(0, 0)` would MATCH the live fingerprint and serve that stale
    // assignment forever.
    let g = graph_at(Millis(1000)).await;

    assert!(g.read_cluster_state().await.unwrap().is_none());
}

// ---- entity mentions fingerprint ----

#[tokio::test]
async fn the_entity_mentions_fingerprint_counts_live_mentions_edges() {
    // Arrange: an entity with two live `mentions` edges written at different instants.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let person = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let fact_a = g.insert(NewNode::now("fact", "first", "x")).await.unwrap();
    let fact_b = g.insert(NewNode::now("fact", "second", "x")).await.unwrap();
    clock.set(Millis(2000));
    g.link(NewEdge::new(
        &person,
        &fact_a,
        crate::types::relation::MENTIONS,
    ))
    .await
    .unwrap();
    clock.set(Millis(3000));
    g.link(NewEdge::new(
        &person,
        &fact_b,
        crate::types::relation::MENTIONS,
    ))
    .await
    .unwrap();

    // Act
    let fp = g
        .entity_mentions_fingerprint("ada", None, Millis(4000))
        .await
        .unwrap();

    // Assert
    assert_eq!(fp.edge_count, 2);
    assert_eq!(fp.max_tx_from, Millis(3000));
}

#[tokio::test]
async fn the_entity_mentions_fingerprint_is_zero_for_a_subject_with_no_live_entity() {
    // Arrange
    let g = graph_at(Millis(1000)).await;

    // Act
    let fp = g
        .entity_mentions_fingerprint("nobody", None, Millis(1000))
        .await
        .unwrap();

    // Assert
    assert_eq!(fp.edge_count, 0);
    assert_eq!(fp.max_tx_from, Millis(0));
}

#[tokio::test]
async fn the_entity_mentions_fingerprint_keeps_scopes_distinct_for_the_same_subject() {
    // Arrange: two entities sharing a subject, differing only by scope, each
    // with its own mention count, so resolving by subject alone would
    // conflate them.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let global = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let scoped = g
        .insert(NewNode::entity("person", "Ada").with_scope("proj-x"))
        .await
        .unwrap();
    let fact = g.insert(NewNode::now("fact", "note", "x")).await.unwrap();
    let other = g.insert(NewNode::now("fact", "memo", "x")).await.unwrap();
    clock.set(Millis(2000));
    g.link(NewEdge::new(
        &global,
        &fact,
        crate::types::relation::MENTIONS,
    ))
    .await
    .unwrap();
    clock.set(Millis(3000));
    g.link(NewEdge::new(
        &scoped,
        &fact,
        crate::types::relation::MENTIONS,
    ))
    .await
    .unwrap();
    clock.set(Millis(4000));
    g.link(NewEdge::new(
        &scoped,
        &other,
        crate::types::relation::MENTIONS,
    ))
    .await
    .unwrap();

    // Act
    let global_fp = g
        .entity_mentions_fingerprint("ada", None, Millis(5000))
        .await
        .unwrap();
    let scoped_fp = g
        .entity_mentions_fingerprint("ada", Some("proj-x"), Millis(5000))
        .await
        .unwrap();

    // Assert
    assert_eq!(global_fp.edge_count, 1);
    assert_eq!(global_fp.max_tx_from, Millis(2000));
    assert_eq!(scoped_fp.edge_count, 2);
    assert_eq!(scoped_fp.max_tx_from, Millis(4000));
}

#[tokio::test]
async fn the_entity_mentions_fingerprint_ignores_an_edge_closed_in_transaction_time() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let person = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let fact = g.insert(NewNode::now("fact", "note", "x")).await.unwrap();
    g.link(NewEdge::new(
        &person,
        &fact,
        crate::types::relation::MENTIONS,
    ))
    .await
    .unwrap();
    g.backend
        .execute("UPDATE edges SET tx_to = ?1", &[Millis(1500).into()])
        .await
        .unwrap();

    // Act
    let fp = g
        .entity_mentions_fingerprint("ada", None, Millis(2000))
        .await
        .unwrap();

    // Assert
    assert_eq!(fp.edge_count, 0);
}

#[tokio::test]
async fn the_entity_mentions_fingerprint_ignores_an_edge_not_yet_live_at_as_of() {
    // Arrange: the mentions edge is written after the instant being fingerprinted.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let person = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let fact = g.insert(NewNode::now("fact", "note", "x")).await.unwrap();
    clock.set(Millis(3000));
    g.link(NewEdge::new(
        &person,
        &fact,
        crate::types::relation::MENTIONS,
    ))
    .await
    .unwrap();

    // Act
    let fp = g
        .entity_mentions_fingerprint("ada", None, Millis(2000))
        .await
        .unwrap();

    // Assert
    assert_eq!(
        fp.edge_count, 0,
        "an edge written after as_of must not count"
    );
}

#[tokio::test]
async fn the_entity_mentions_fingerprint_follows_a_superseded_entity_to_its_new_node() {
    // Arrange: the entity is superseded once; only the new node's mention counts.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let old = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let old_fact = g.insert(NewNode::now("fact", "first", "x")).await.unwrap();
    g.link(NewEdge::new(
        &old,
        &old_fact,
        crate::types::relation::MENTIONS,
    ))
    .await
    .unwrap();

    clock.set(Millis(2000));
    let new = g
        .supersede(&old, NewNode::entity("person", "Ada"))
        .await
        .unwrap();
    let new_fact = g.insert(NewNode::now("fact", "second", "x")).await.unwrap();
    clock.set(Millis(3000));
    g.link(NewEdge::new(
        &new,
        &new_fact,
        crate::types::relation::MENTIONS,
    ))
    .await
    .unwrap();

    // Act
    let fp = g
        .entity_mentions_fingerprint("ada", None, Millis(4000))
        .await
        .unwrap();

    // Assert
    assert_eq!(
        fp.edge_count, 1,
        "must follow the new live node, not the superseded one"
    );
    assert_eq!(fp.max_tx_from, Millis(3000));
}

// ---- stale entities ----

#[tokio::test]
async fn stale_entities_reports_the_current_live_id_when_the_fingerprint_has_moved() {
    // Arrange: mark synthesized against the original node, then supersede
    // it and add a new mention, so the live id has changed since the mark.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let old = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let fp0 = g
        .entity_mentions_fingerprint("ada", None, Millis(1000))
        .await
        .unwrap();
    g.mark_entity_synthesized("ada", None, fp0, Millis(1000))
        .await
        .unwrap();

    clock.set(Millis(2000));
    let new = g
        .supersede(&old, NewNode::entity("person", "Ada"))
        .await
        .unwrap();
    let fact = g.insert(NewNode::now("fact", "note", "x")).await.unwrap();
    clock.set(Millis(3000));
    g.link(NewEdge::new(&new, &fact, relation::MENTIONS))
        .await
        .unwrap();

    // Act
    let stale = g.stale_entities(10, Millis(4000)).await.unwrap();

    // Assert
    assert_eq!(
        stale,
        vec![(new, "ada".to_string(), None)],
        "must report the current live id, not the superseded one"
    );
}

#[tokio::test]
async fn stale_entities_omits_a_subject_whose_fingerprint_matches_its_stored_state() {
    // Arrange: mark synthesized with the fingerprint the entity already has.
    let g = graph_at(Millis(1000)).await;
    g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let fp = g
        .entity_mentions_fingerprint("ada", None, Millis(1000))
        .await
        .unwrap();
    g.mark_entity_synthesized("ada", None, fp, Millis(1000))
        .await
        .unwrap();

    // Act
    let stale = g.stale_entities(10, Millis(2000)).await.unwrap();

    // Assert
    assert!(
        stale.is_empty(),
        "an up-to-date fingerprint must not be reported as stale"
    );
}

#[tokio::test]
async fn stale_entities_includes_a_live_subject_with_no_stored_state() {
    // Arrange: a live entity that has never been synthesized.
    let g = graph_at(Millis(1000)).await;
    let entity = g.insert(NewNode::entity("person", "Ada")).await.unwrap();

    // Act
    let stale = g.stale_entities(10, Millis(2000)).await.unwrap();

    // Assert
    assert_eq!(stale, vec![(entity, "ada".to_string(), None)]);
}

#[tokio::test]
async fn stale_entities_omits_a_live_fact_with_a_subject_and_no_stored_state() {
    // Arrange: an ordinary fact, not an entity page, carrying a subject
    // (RememberArgs.subject applies to any kind) and never synthesized.
    let g = graph_at(Millis(1000)).await;
    g.insert(NewNode::now("fact", "price note", "the price is $5").with_subject("price"))
        .await
        .unwrap();

    // Act
    let stale = g.stale_entities(10, Millis(2000)).await.unwrap();

    // Assert
    assert!(
        stale.is_empty(),
        "a fact with non-empty content must never be treated as an unsynthesized entity"
    );
}

#[tokio::test]
async fn stale_entities_honors_limit_and_orders_oldest_last_synthesized_first() {
    // Arrange: a never-synthesized entity (treated as oldest), a
    // long-ago-synthesized entity that has since drifted, a
    // recently-synthesized entity that has also since drifted, and an
    // up-to-date entity that must never appear.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();

    let alice = g.insert(NewNode::entity("person", "Alice")).await.unwrap();

    let bob = g.insert(NewNode::entity("person", "Bob")).await.unwrap();
    let bob_fp = g
        .entity_mentions_fingerprint("bob", None, Millis(1000))
        .await
        .unwrap();
    g.mark_entity_synthesized("bob", None, bob_fp, Millis(1000))
        .await
        .unwrap();

    let carol = g.insert(NewNode::entity("person", "Carol")).await.unwrap();
    let carol_fp = g
        .entity_mentions_fingerprint("carol", None, Millis(5000))
        .await
        .unwrap();
    g.mark_entity_synthesized("carol", None, carol_fp, Millis(5000))
        .await
        .unwrap();

    let dave = g.insert(NewNode::entity("person", "Dave")).await.unwrap();
    let dave_fp = g
        .entity_mentions_fingerprint("dave", None, Millis(6000))
        .await
        .unwrap();
    g.mark_entity_synthesized("dave", None, dave_fp, Millis(6000))
        .await
        .unwrap();

    let fact = g.insert(NewNode::now("fact", "note", "x")).await.unwrap();
    clock.set(Millis(7000));
    for entity in [&alice, &bob, &carol] {
        g.link(NewEdge::new(entity, &fact, relation::MENTIONS))
            .await
            .unwrap();
    }

    // Act
    let stale = g.stale_entities(2, Millis(8000)).await.unwrap();

    // Assert
    assert_eq!(
        stale,
        vec![
            (alice, "alice".to_string(), None),
            (bob, "bob".to_string(), None),
        ],
        "never-synthesized sorts before long-ago, and the limit caps the rest"
    );
    assert!(
        !stale.iter().any(|(id, _, _)| id == &dave),
        "an up-to-date entity must never be reported as stale"
    );
}

#[tokio::test]
async fn stale_entities_omits_a_subject_marked_synthesized_after_its_id_changed() {
    // Arrange: mark synthesized, supersede, add a mention, then mark
    // synthesized again with the NEW node's fingerprint.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let old = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let fp0 = g
        .entity_mentions_fingerprint("ada", None, Millis(1000))
        .await
        .unwrap();
    g.mark_entity_synthesized("ada", None, fp0, Millis(1000))
        .await
        .unwrap();

    clock.set(Millis(2000));
    let new = g
        .supersede(&old, NewNode::entity("person", "Ada"))
        .await
        .unwrap();
    let fact = g.insert(NewNode::now("fact", "note", "x")).await.unwrap();
    clock.set(Millis(3000));
    g.link(NewEdge::new(&new, &fact, relation::MENTIONS))
        .await
        .unwrap();

    let fp1 = g
        .entity_mentions_fingerprint("ada", None, Millis(4000))
        .await
        .unwrap();
    g.mark_entity_synthesized("ada", None, fp1, Millis(4000))
        .await
        .unwrap();

    // Act
    let stale = g.stale_entities(10, Millis(4000)).await.unwrap();

    // Assert
    assert!(
        stale.is_empty(),
        "resynthesizing against the new id must clear the staleness, id churn notwithstanding"
    );
}
