// SPDX-License-Identifier: Apache-2.0
//! Recall across `same_as` links, and the closure those links form.

use super::*;

#[tokio::test]
async fn fetch_candidates_scope_filter_includes_descendant_scopes() {
    // Arrange: same shape as `scope_filter_holds_for_graph_expanded_neighbours`,
    // but the neighbour sits in a scope that is a descendant of the
    // query scope rather than an unrelated one, and is reachable only
    // through graph expansion off the lexically-matched seed.
    let g = graph_at(Millis(1000)).await;
    let seed = g
        .insert(NewNode::now("decision", "Rollout", "zorbnax rollout approved").with_scope("work"))
        .await
        .unwrap();
    let neighbour = g
        .insert(NewNode::now("decision", "Detail", "unrelated notes").with_scope("work/liam"))
        .await
        .unwrap();
    g.link(NewEdge::new(&seed, &neighbour, "mentions"))
        .await
        .unwrap();
    // A second expanded neighbour whose scope is a false string prefix
    // of "work" (no "/" boundary), not a true descendant.
    let false_prefix_neighbour = g
        .insert(NewNode::now("decision", "Archived", "unrelated notes").with_scope("work-archive"))
        .await
        .unwrap();
    g.link(NewEdge::new(&seed, &false_prefix_neighbour, "mentions"))
        .await
        .unwrap();

    // Act
    let hits = g
        .query(&Query::text("zorbnax rollout").with_scope("work"))
        .await
        .unwrap();

    // Assert: the descendant-scoped neighbour, hydrated only through
    // `fetch_candidates`, survives its scope filter, while the
    // false-prefix-scoped neighbour is excluded by that same filter.
    let labels: Vec<&str> = hits.iter().map(|h| h.label.as_str()).collect();
    assert!(
        labels.contains(&"Detail"),
        "expected descendant-scoped expanded neighbour in hits: {labels:?}"
    );
    assert!(
        !labels.contains(&"Archived"),
        "false-prefix-scoped expanded neighbour must not leak into hits: {labels:?}"
    );
}

#[tokio::test]
async fn recall_surfaces_same_as_linked_entity_in_different_scope() {
    // Arrange: A (scope "meeting") matches the query text directly; B
    // (scope "code") only via a same_as link to A, with no scope filter.
    let g = graph_at(Millis(1000)).await;
    let a = g
        .insert(NewNode::now("decision", "Standup", "zorbnax meeting recap").with_scope("meeting"))
        .await
        .unwrap();
    let b = g
        .insert(NewNode::now("decision", "Snippet", "unrelated code notes").with_scope("code"))
        .await
        .unwrap();
    g.relate(&a, &b, relation::SAME_AS).await.unwrap();

    // Act
    let hits = g
        .query_explained(&Query::text("zorbnax meeting"))
        .await
        .unwrap();

    // Assert
    let by_label = |label: &str| hits.iter().find(|h| h.hit.label == label);
    assert!(
        by_label("Standup").is_some(),
        "direct match A missing from hits"
    );
    let b_hit = by_label("Snippet");
    assert!(b_hit.is_some(), "same_as-linked B missing from hits");
    assert!(
        b_hit.unwrap().expanded,
        "B must be scored at the down-weighted expansion tier"
    );
}

#[tokio::test]
async fn recall_same_as_bridge_survives_explicit_scope_filter_while_unrelated_candidate_stays_excluded(
) {
    // Arrange: A (scope "meeting") matches the query text directly. B
    // (scope "code") is same_as-linked to A but has no lexical overlap
    // with the query. C (scope "unrelated") also has no lexical overlap
    // with the query; it only becomes a candidate through A's untyped
    // neighbor expansion (a "mentions" edge, not same_as). That is the
    // only way C ever reaches candidacy here, so its absence from hits
    // actually proves the scope filter dropped it, rather than C simply
    // never matching the query text in the first place.
    let g = graph_at(Millis(1000)).await;
    let a = g
        .insert(NewNode::now("decision", "Standup", "zorbnax meeting recap").with_scope("meeting"))
        .await
        .unwrap();
    let b = g
        .insert(NewNode::now("decision", "Snippet", "unrelated code notes").with_scope("code"))
        .await
        .unwrap();
    g.relate(&a, &b, relation::SAME_AS).await.unwrap();
    let c = g
        .insert(
            NewNode::now("decision", "Bystander", "wombat archive filler").with_scope("unrelated"),
        )
        .await
        .unwrap();
    g.link(NewEdge::new(&a, &c, "mentions")).await.unwrap();

    // Act: an explicit scope filter matching A's own scope, but not B's
    // or C's (neither is a hierarchy descendant of "meeting" either).
    let hits = g
        .query(&Query::text("zorbnax meeting").with_scope("meeting"))
        .await
        .unwrap();

    // Assert
    let by_label = |label: &str| hits.iter().find(|h| h.label == label);
    assert!(
        by_label("Standup").is_some(),
        "direct match A missing from hits"
    );
    assert!(
        by_label("Snippet").is_some(),
        "same_as-linked B was dropped by the explicit scope filter"
    );
    assert!(
        by_label("Bystander").is_none(),
        "unrelated expanded candidate C should stay excluded by the scope filter"
    );
}

#[tokio::test]
async fn recall_same_as_linked_entity_in_own_scope_appears_exactly_once() {
    // Arrange: A and B share scope "meeting", so B is reachable both via
    // the untyped neighbor-expansion edge AND the same_as-specific path;
    // B has no lexical overlap with the query text of its own.
    let g = graph_at(Millis(1000)).await;
    let a = g
        .insert(NewNode::now("decision", "Standup", "zorbnax meeting recap").with_scope("meeting"))
        .await
        .unwrap();
    let b_id = g
        .insert(NewNode::now("decision", "Recap", "unrelated notes").with_scope("meeting"))
        .await
        .unwrap();
    g.relate(&a, &b_id, relation::SAME_AS).await.unwrap();

    // Act: scope filter includes both A's and B's scope.
    let hits = g
        .query_explained(&Query::text("zorbnax meeting").with_scope("meeting"))
        .await
        .unwrap();

    // Assert
    let b_hits: Vec<_> = hits.iter().filter(|h| h.hit.label == "Recap").collect();
    assert_eq!(b_hits.len(), 1, "B must appear exactly once: {b_hits:?}");
    assert!(
        b_hits[0].expanded,
        "B must be scored at the down-weighted expansion tier"
    );
}

#[tokio::test]
async fn recall_same_as_linked_entity_with_direct_match_keeps_its_own_score() {
    // Arrange: A and B share scope "meeting" and are same_as-linked, but
    // unlike the own-scope test above, B's own content also matches the
    // query text directly, so B is already a genuine lexical hit before
    // the same_as path ever runs. This proves the same_as merge's
    // `or_insert` never downgrades a row that already scored on its own.
    let g = graph_at(Millis(1000)).await;
    let a = g
        .insert(NewNode::now("decision", "Standup", "zorbnax meeting recap").with_scope("meeting"))
        .await
        .unwrap();
    let b_id = g
        .insert(NewNode::now("decision", "Recap", "zorbnax standalone notes").with_scope("meeting"))
        .await
        .unwrap();
    g.relate(&a, &b_id, relation::SAME_AS).await.unwrap();

    // Act: scope filter includes both A's and B's scope.
    let hits = g
        .query_explained(&Query::text("zorbnax meeting").with_scope("meeting"))
        .await
        .unwrap();

    // Assert
    let b_hits: Vec<_> = hits.iter().filter(|h| h.hit.label == "Recap").collect();
    assert_eq!(b_hits.len(), 1, "B must appear exactly once: {b_hits:?}");
    assert!(
        !b_hits[0].expanded,
        "B's own direct match must not be downgraded by the same_as expansion path"
    );
}

#[tokio::test]
async fn query_explained_surfaces_same_as_linked_entity() {
    // Arrange: same cross-scope same_as link as the baseline recall
    // scenario, but read through query_explained, the entry point that
    // returns each hit with its explanation.
    let g = graph_at(Millis(1000)).await;
    let a = g
        .insert(NewNode::now("decision", "Retro", "glimmerfox retro notes").with_scope("meeting"))
        .await
        .unwrap();
    let b = g
        .insert(NewNode::now("decision", "Patch", "unrelated code change").with_scope("code"))
        .await
        .unwrap();
    g.relate(&a, &b, relation::SAME_AS).await.unwrap();

    // Act
    let hits = g
        .query_explained(&Query::text("glimmerfox retro"))
        .await
        .unwrap();

    // Assert: each half checked independently.
    let labels: Vec<&str> = hits.iter().map(|h| h.hit.label.as_str()).collect();
    assert!(
        labels.contains(&"Retro"),
        "direct match A missing from the explained hits: {labels:?}"
    );
    assert!(
        labels.contains(&"Patch"),
        "same_as-linked B missing from the explained hits: {labels:?}"
    );
}

#[tokio::test]
async fn same_as_edge_to_superseded_node_is_dropped_gracefully() {
    // Arrange: A same_as B, then B is superseded; the same_as edge stays
    // live in `edges`, but B itself is no longer a live node.
    let g = graph_at(Millis(1000)).await;
    let a = g
        .insert(NewNode::now(
            "decision",
            "Kickoff",
            "wibblenar kickoff notes",
        ))
        .await
        .unwrap();
    let b = g
        .insert(NewNode::now("decision", "Old", "wibblenar stale copy"))
        .await
        .unwrap();
    g.relate(&a, &b, relation::SAME_AS).await.unwrap();
    g.supersede(&b, NewNode::now("decision", "New", "wibblenar replacement"))
        .await
        .unwrap();

    // Premise: the same_as edge to the now-dead node is still live and
    // traversed, so the assertions below prove it gets filtered as dead
    // data, not that it was simply never reached.
    assert!(
        g.same_as_closure(
            std::slice::from_ref(&a),
            Millis(1000),
            InferredRows::Include
        )
        .await
        .unwrap()
        .contains(&b),
        "premise: the same_as edge to the superseded node must still be live"
    );

    // Act: the closure traverses the dead edge; this must not error.
    let hits = g.query(&Query::text("wibblenar kickoff")).await.unwrap();

    // Assert
    let labels: Vec<&str> = hits.iter().map(|h| h.label.as_str()).collect();
    assert!(
        labels.contains(&"Kickoff"),
        "direct match A missing from hits: {labels:?}"
    );
    assert!(
        !labels.contains(&"Old"),
        "same_as edge to a superseded node must not surface the dead id: {labels:?}"
    );
}

#[tokio::test]
async fn recall_surfaces_full_same_as_chain_not_just_one_hop() {
    // Arrange: A same_as B same_as C, a 3-node chain. C is two same_as
    // hops from A, unreachable via a single neighbors() call off A, so
    // this is what actually proves the closure wiring runs end to end.
    let g = graph_at(Millis(1000)).await;
    let a = g
        .insert(NewNode::now("decision", "Alpha", "quixolt planning notes").with_scope("meeting"))
        .await
        .unwrap();
    let b = g
        .insert(NewNode::now("decision", "Beta", "unrelated middle link").with_scope("code"))
        .await
        .unwrap();
    let c = g
        .insert(NewNode::now("decision", "Gamma", "unrelated far link").with_scope("archive"))
        .await
        .unwrap();
    g.relate(&a, &b, relation::SAME_AS).await.unwrap();
    g.relate(&b, &c, relation::SAME_AS).await.unwrap();

    // Act
    let hits = g.query(&Query::text("quixolt planning")).await.unwrap();

    // Assert: each chain member checked independently.
    let labels: Vec<&str> = hits.iter().map(|h| h.label.as_str()).collect();
    assert!(
        labels.contains(&"Alpha"),
        "direct match A missing: {labels:?}"
    );
    assert!(
        labels.contains(&"Beta"),
        "one-hop chain member B missing: {labels:?}"
    );
    assert!(
        labels.contains(&"Gamma"),
        "two-hop chain member C missing, the case that proves full-chain wiring: {labels:?}"
    );
}

#[tokio::test]
async fn query_core_same_as_wiring_respects_as_of() {
    // Arrange: A and B exist from the start; the same_as edge between
    // them lands at a LATER instant than the as_of the query asks for.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let a = g
        .insert(NewNode::now("decision", "Draft", "flurbis draft notes").with_scope("meeting"))
        .await
        .unwrap();
    let b = g
        .insert(NewNode::now("decision", "Copy", "unrelated code copy").with_scope("code"))
        .await
        .unwrap();
    clock.set(Millis(2000));
    g.relate(&a, &b, relation::SAME_AS).await.unwrap();

    // Act: as_of before the same_as edge existed.
    let past_hits = g
        .query(&Query::text("flurbis draft").with_as_of(Millis(1500)))
        .await
        .unwrap();

    // Assert
    let past_labels: Vec<&str> = past_hits.iter().map(|h| h.label.as_str()).collect();
    assert!(
        past_labels.contains(&"Draft"),
        "direct match A missing: {past_labels:?}"
    );
    assert!(
        !past_labels.contains(&"Copy"),
        "same_as-linked B must not appear before the edge existed: {past_labels:?}"
    );

    // Act: as_of after the same_as edge existed, same fixture. This
    // contrast rules out B being absent above for some unrelated reason.
    let present_hits = g
        .query(&Query::text("flurbis draft").with_as_of(Millis(2500)))
        .await
        .unwrap();

    // Assert
    let present_labels: Vec<&str> = present_hits.iter().map(|h| h.label.as_str()).collect();
    assert!(
        present_labels.contains(&"Copy"),
        "same_as-linked B must appear once the edge exists: {present_labels:?}"
    );
}

/// Substring unique to `fetch_candidates`'s `SELECT` shape among every
/// other query this module issues, for the `QUERY_MATCH_COUNT` watch.
const FETCH_CANDIDATES_SELECT: &str =
    "SELECT id, kind, label, content, scope, attributes, confidence, valid_from";

#[tokio::test]
async fn fetch_candidates_no_same_as_edges_incurs_no_extra_round_trip() {
    let _guard = TEST_LOCK.lock().await;
    // Arrange: an entity with no same_as edge anywhere in the store.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g: Graph<Interposing> =
        Graph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
    g.insert(NewNode::now("decision", "Solo", "yonderlisp solo notes"))
        .await
        .unwrap();

    *QUERY_MATCH_COUNT.lock().unwrap() = (FETCH_CANDIDATES_SELECT.to_string(), 0);

    // Act
    g.query(&Query::text("yonderlisp solo")).await.unwrap();

    // Assert: only the first, unconditional fetch_candidates call fires.
    // Read the count into a local before asserting so the MutexGuard
    // drops before a failing assertion could panic while still holding
    // it, which would poison the mutex for every later test.
    let round_trips = QUERY_MATCH_COUNT.lock().unwrap().1;
    assert_eq!(
        round_trips, 1,
        "a query touching no same_as edge must not issue a second fetch_candidates call"
    );
}

#[tokio::test]
async fn fetch_candidates_same_as_bridge_incurs_exactly_one_extra_round_trip() {
    let _guard = TEST_LOCK.lock().await;
    // Arrange: A same_as B, in different scopes, so the scope-exemption
    // path is expected to fire its one extra fetch_candidates call.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g: Graph<Interposing> =
        Graph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
    let a = g
        .insert(
            NewNode::now("decision", "Bridgehead", "vandorix bridge notes").with_scope("meeting"),
        )
        .await
        .unwrap();
    let b = g
        .insert(NewNode::now("decision", "Bridged", "unrelated far scope").with_scope("code"))
        .await
        .unwrap();
    g.relate(&a, &b, relation::SAME_AS).await.unwrap();

    *QUERY_MATCH_COUNT.lock().unwrap() = (FETCH_CANDIDATES_SELECT.to_string(), 0);

    // Act: an explicit scope filter on A's own scope, so the first call
    // does not already hydrate B and the bridging call is forced to run.
    g.query(&Query::text("vandorix bridge").with_scope("meeting"))
        .await
        .unwrap();

    // Assert: the base call plus exactly one bridging call. Read the
    // count into a local first so the MutexGuard drops before a failing
    // assertion could panic while still holding it, which would poison
    // the mutex for every later test.
    let round_trips = QUERY_MATCH_COUNT.lock().unwrap().1;
    assert_eq!(
        round_trips, 2,
        "a same_as bridge must incur exactly one extra fetch_candidates call"
    );
}

#[tokio::test]
async fn fetch_candidates_multi_seed_same_as_bridge_still_incurs_exactly_one_extra_round_trip() {
    let _guard = TEST_LOCK.lock().await;
    // Arrange: two direct-match seeds A1, A2 in scope "meeting", each
    // same_as-linked to a different out-of-scope node. Locks in that the
    // multi-source same_as_closure traversal shares one bridging call
    // across both seeds rather than reverting to a per-seed loop.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g: Graph<Interposing> =
        Graph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
    let a1 = g
        .insert(
            NewNode::now("decision", "BridgeheadOne", "vandorix bridge one notes")
                .with_scope("meeting"),
        )
        .await
        .unwrap();
    let a2 = g
        .insert(
            NewNode::now("decision", "BridgeheadTwo", "vandorix bridge two notes")
                .with_scope("meeting"),
        )
        .await
        .unwrap();
    let b1 = g
        .insert(
            NewNode::now("decision", "BridgedOne", "unrelated far scope one").with_scope("code"),
        )
        .await
        .unwrap();
    let b2 = g
        .insert(
            NewNode::now("decision", "BridgedTwo", "unrelated far scope two").with_scope("archive"),
        )
        .await
        .unwrap();
    g.relate(&a1, &b1, relation::SAME_AS).await.unwrap();
    g.relate(&a2, &b2, relation::SAME_AS).await.unwrap();

    *QUERY_MATCH_COUNT.lock().unwrap() = (FETCH_CANDIDATES_SELECT.to_string(), 0);

    // Act: an explicit scope filter on the seeds' own scope, so the
    // first call does not already hydrate B1 or B2 and the bridging
    // call is forced to run.
    g.query(&Query::text("vandorix bridge").with_scope("meeting"))
        .await
        .unwrap();

    // Assert: still the base call plus exactly one bridging call, not
    // one bridging call per seed. Read the count into a local first so
    // the MutexGuard drops before a failing assertion could panic while
    // still holding it, which would poison the mutex for every later
    // test.
    let round_trips = QUERY_MATCH_COUNT.lock().unwrap().1;
    assert_eq!(
        round_trips, 2,
        "two same_as-linked seeds must still incur exactly one bridging fetch_candidates call"
    );
}

#[tokio::test]
async fn same_as_closure_with_no_edges_returns_only_the_seed() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let seed = g.insert(NewNode::entity("person", "Ada")).await.unwrap();

    // Act
    let closure = g
        .same_as_closure(
            std::slice::from_ref(&seed),
            Millis(1000),
            InferredRows::Include,
        )
        .await
        .unwrap();

    // Assert
    assert_eq!(closure, HashSet::from([seed]));
}

#[tokio::test]
async fn same_as_closure_traverses_both_directions_of_one_edge() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let a = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let b = g.insert(NewNode::entity("person", "Ada B")).await.unwrap();
    g.relate(&a, &b, relation::SAME_AS).await.unwrap();

    // Act
    let from_a = g
        .same_as_closure(
            std::slice::from_ref(&a),
            Millis(1000),
            InferredRows::Include,
        )
        .await
        .unwrap();
    let from_b = g
        .same_as_closure(
            std::slice::from_ref(&b),
            Millis(1000),
            InferredRows::Include,
        )
        .await
        .unwrap();

    // Assert: the edge is stored src=A,dst=B, but traversal from either
    // endpoint must find the other.
    assert_eq!(from_a, HashSet::from([a.clone(), b.clone()]));
    assert_eq!(from_b, HashSet::from([a, b]));
}

#[tokio::test]
async fn same_as_closure_traverses_full_chain() {
    // Arrange: A same_as B same_as C same_as D.
    let g = graph_at(Millis(1000)).await;
    let a = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let b = g.insert(NewNode::entity("person", "Ada B")).await.unwrap();
    let c = g.insert(NewNode::entity("person", "Ada C")).await.unwrap();
    let d = g.insert(NewNode::entity("person", "Ada D")).await.unwrap();
    g.relate(&a, &b, relation::SAME_AS).await.unwrap();
    g.relate(&b, &c, relation::SAME_AS).await.unwrap();
    g.relate(&c, &d, relation::SAME_AS).await.unwrap();

    // Act
    let closure = g
        .same_as_closure(
            std::slice::from_ref(&a),
            Millis(1000),
            InferredRows::Include,
        )
        .await
        .unwrap();

    // Assert
    assert_eq!(closure, HashSet::from([a, b, c, d]));
}

#[test]
fn same_as_closure_terminates_on_cycle() {
    // Arrange/Act: A same_as B, B same_as A, run on a real OS thread. A
    // cycle-safety bug here would be a tight loop of synchronous DB calls
    // that never yields to the executor, so a same-task
    // tokio::time::timeout would never even get to check its deadline;
    // only a deadline enforced from a separate thread can catch it.
    let result = run_with_real_timeout(std::time::Duration::from_secs(5), || async {
        let g = graph_at(Millis(1000)).await;
        let a = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
        let b = g.insert(NewNode::entity("person", "Ada B")).await.unwrap();
        g.relate(&a, &b, relation::SAME_AS).await.unwrap();
        g.relate(&b, &a, relation::SAME_AS).await.unwrap();
        let closure = g
            .same_as_closure(
                std::slice::from_ref(&a),
                Millis(1000),
                InferredRows::Include,
            )
            .await
            .unwrap();
        (closure, a, b)
    });

    // Assert
    let (closure, a, b) = result.expect(
        "same_as_closure must terminate within the timeout; a RecvTimeoutError here means the cycle-safety guard regressed and the closure hung",
    );
    assert_eq!(closure, HashSet::from([a, b]));
}

#[test]
fn same_as_closure_terminates_on_three_node_cycle() {
    // Arrange/Act: A same_as B, B same_as C, C same_as A. A two-node
    // cycle is already a cycle under bidirectional traversal, so it
    // cannot distinguish a proper visited-set guard from a hypothetical
    // parent-skip guard, which would still loop forever past two hops;
    // this longer cycle can.
    let result = run_with_real_timeout(std::time::Duration::from_secs(5), || async {
        let g = graph_at(Millis(1000)).await;
        let a = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
        let b = g.insert(NewNode::entity("person", "Ada B")).await.unwrap();
        let c = g.insert(NewNode::entity("person", "Ada C")).await.unwrap();
        g.relate(&a, &b, relation::SAME_AS).await.unwrap();
        g.relate(&b, &c, relation::SAME_AS).await.unwrap();
        g.relate(&c, &a, relation::SAME_AS).await.unwrap();
        let closure = g
            .same_as_closure(
                std::slice::from_ref(&a),
                Millis(1000),
                InferredRows::Include,
            )
            .await
            .unwrap();
        (closure, a, b, c)
    });

    // Assert
    let (closure, a, b, c) = result.expect(
        "same_as_closure must terminate within the timeout; a RecvTimeoutError here means the cycle-safety guard regressed and the closure hung",
    );
    assert_eq!(closure, HashSet::from([a, b, c]));
}

#[tokio::test]
async fn same_as_closure_respects_as_of() {
    // Arrange: the same_as edge is created at t=2000.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let seed = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let linked = g.insert(NewNode::entity("person", "Ada B")).await.unwrap();
    clock.set(Millis(2000));
    g.relate(&seed, &linked, relation::SAME_AS).await.unwrap();

    // Act: as_of a time before the edge existed.
    let closure = g
        .same_as_closure(
            std::slice::from_ref(&seed),
            Millis(1500),
            InferredRows::Include,
        )
        .await
        .unwrap();

    // Assert
    assert_eq!(closure, HashSet::from([seed]));
}
