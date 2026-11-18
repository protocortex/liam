// SPDX-License-Identifier: Apache-2.0
//! Entity mentions, their recency order, and the scoring of candidates.

use super::*;

// ---- Graph::live_entity_subjects ----

#[tokio::test]
async fn live_entity_subjects_lists_only_nodes_that_mention_something_sorted_and_capped() {
    // Arrange: two entity pages that mention a fact, an entity with no
    // mention, and a fact node with a subject of its own
    let g = graph_at(Millis(1000)).await;
    let fact = g
        .insert(NewNode::now("fact", "note", "a note").with_subject("note-subject"))
        .await
        .unwrap();
    for name in ["Zed", "Ada"] {
        let entity = g.insert(NewNode::entity("person", name)).await.unwrap();
        g.link(NewEdge::new(&entity, &fact, relation::MENTIONS))
            .await
            .unwrap();
    }
    g.insert(NewNode::entity("person", "Lonely")).await.unwrap();

    // Act
    let all = g.live_entity_subjects(10).await.unwrap();
    let capped = g.live_entity_subjects(1).await.unwrap();

    // Assert
    assert_eq!(all, ["ada", "zed"]);
    assert_eq!(capped, ["ada"]);
}

// ---- Graph::mentions ----

#[tokio::test]
async fn mentions_returns_all_live_mentions_most_recent_first() {
    // Arrange: three mentions-linked facts, each written at a later tick.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let entity = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let first = g
        .insert(NewNode::now("fact", "first", "note one"))
        .await
        .unwrap();
    g.link(NewEdge::new(&entity, &first, relation::MENTIONS))
        .await
        .unwrap();
    clock.set(Millis(2000));
    let second = g
        .insert(NewNode::now("fact", "second", "note two"))
        .await
        .unwrap();
    g.link(NewEdge::new(&entity, &second, relation::MENTIONS))
        .await
        .unwrap();
    clock.set(Millis(3000));
    let third = g
        .insert(NewNode::now("fact", "third", "note three"))
        .await
        .unwrap();
    g.link(NewEdge::new(&entity, &third, relation::MENTIONS))
        .await
        .unwrap();

    // Act
    let found = g.mentions(&entity, Millis(3000), 10).await.unwrap();

    // Assert: most-recently-mentioned first.
    let ids: Vec<NodeId> = found.into_iter().map(|c| c.id).collect();
    assert_eq!(ids, vec![third, second, first]);
}

#[tokio::test]
async fn mentions_excludes_edges_of_a_different_type() {
    // Arrange: a mentions edge and an unrelated edge to the same entity.
    let g = graph_at(Millis(1000)).await;
    let entity = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let mentioned = g
        .insert(NewNode::now("fact", "mentioned", "note"))
        .await
        .unwrap();
    let other = g
        .insert(NewNode::now("fact", "other", "unrelated"))
        .await
        .unwrap();
    g.link(NewEdge::new(&entity, &mentioned, relation::MENTIONS))
        .await
        .unwrap();
    g.link(NewEdge::new(&entity, &other, "references"))
        .await
        .unwrap();

    // Act
    let found = g.mentions(&entity, Millis(1000), 10).await.unwrap();

    // Assert: only the mentions-typed edge's target comes back.
    let ids: Vec<NodeId> = found.into_iter().map(|c| c.id).collect();
    assert_eq!(ids, vec![mentioned]);
}

#[tokio::test]
async fn mentions_respects_limit_keeping_the_most_recent() {
    // Arrange: more live mentions than the limit, each at a later tick.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let entity = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let mut ids = Vec::new();
    for i in 0..5 {
        clock.set(Millis(1000 + i * 1000));
        let fact = g
            .insert(NewNode::now("fact", "note", "note"))
            .await
            .unwrap();
        g.link(NewEdge::new(&entity, &fact, relation::MENTIONS))
            .await
            .unwrap();
        ids.push(fact);
    }

    // Act
    let found = g.mentions(&entity, Millis(5000), 2).await.unwrap();

    // Assert: exactly `limit` results, the two most recent.
    let found_ids: Vec<NodeId> = found.into_iter().map(|c| c.id).collect();
    assert_eq!(found_ids, vec![ids[4].clone(), ids[3].clone()]);
}

#[tokio::test]
async fn mentions_excludes_an_edge_expired_before_as_of() {
    // Arrange: a mentions edge closed (superseded) before the as-of instant.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let entity = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let fact = g
        .insert(NewNode::now("fact", "note", "note"))
        .await
        .unwrap();
    g.link(NewEdge::new(&entity, &fact, relation::MENTIONS))
        .await
        .unwrap();
    g.backend
        .execute("UPDATE edges SET tx_to = ?1", &[Millis(2000).into()])
        .await
        .unwrap();

    // Act
    let found = g.mentions(&entity, Millis(2500), 10).await.unwrap();

    // Assert
    assert!(found.is_empty());
}

#[test]
fn reorder_by_recency_restores_recency_order_from_scrambled_input() {
    // Arrange: candidates hydrated scrambled, true order given separately.
    let a = NodeId::from_raw("a");
    let b = NodeId::from_raw("b");
    let c = NodeId::from_raw("c");
    let candidate = |id: &NodeId| Candidate::new(id.clone(), "fact", "", "");
    let scrambled = vec![candidate(&c), candidate(&a), candidate(&b)];
    let ordered_ids = vec![a.clone(), b.clone(), c.clone()];

    // Act
    let result = reorder_by_recency(scrambled, &ordered_ids);

    // Assert
    let ids: Vec<NodeId> = result.into_iter().map(|cand| cand.id).collect();
    assert_eq!(ids, vec![a, b, c]);
}

fn scoring_candidate(id: &str, confidence: f64, valid_from: Millis) -> Candidate {
    let mut c = Candidate::new(NodeId::from_raw(id), "fact", "", "");
    c.confidence = confidence;
    c.valid_from = valid_from;
    c
}

#[test]
fn score_and_rank_applies_weight_one_when_not_expanded() {
    // Arrange: a hydrated row whose candidate entry marks it non-expanded.
    let now = Millis(1000);
    let row = scoring_candidate("a", 0.5, now);
    let mut candidates = HashMap::new();
    candidates.insert("a".to_string(), (2.0, false));

    // Act
    let out = score_and_rank(
        vec![row],
        &candidates,
        &Ranks::default(),
        now,
        None,
        3.0,
        0.1,
        10,
    );

    // Assert: score uses weight 1.0, not the expansion weight.
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].hit.score, 2.0 * 0.5 * 1.0 * 1.0);
    assert!(!out[0].expanded);
}

#[test]
fn score_and_rank_applies_expansion_weight_when_expanded() {
    // Arrange: a hydrated row whose candidate entry marks it expansion-only.
    let now = Millis(1000);
    let row = scoring_candidate("a", 0.5, now);
    let mut candidates = HashMap::new();
    candidates.insert("a".to_string(), (2.0, true));

    // Act
    let out = score_and_rank(
        vec![row],
        &candidates,
        &Ranks::default(),
        now,
        None,
        3.0,
        0.1,
        10,
    );

    // Assert: score uses the expansion weight, not 1.0.
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].hit.score, 2.0 * 0.5 * 1.0 * 3.0);
    assert!(out[0].expanded);
}

#[test]
fn score_and_rank_falls_back_to_floor_and_expanded_when_missing() {
    // Arrange: a hydrated row with no entry in the candidate map at all.
    let now = Millis(1000);
    let row = scoring_candidate("a", 1.0, now);
    let candidates = HashMap::new();
    let floor = 0.25;

    // Act
    let out = score_and_rank(
        vec![row],
        &candidates,
        &Ranks::default(),
        now,
        None,
        4.0,
        floor,
        10,
    );

    // Assert: falls back to the floor score and counts as expanded.
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].rrf, floor);
    assert!(out[0].expanded);
    assert_eq!(out[0].hit.score, floor * 1.0 * 1.0 * 4.0);
}

#[test]
fn score_and_rank_reports_rank_map_presence_per_row() {
    // Arrange: one row present in both rank maps, one absent from both.
    let now = Millis(1000);
    let present = scoring_candidate("present", 1.0, now);
    let absent = scoring_candidate("absent", 1.0, now);
    let mut candidates = HashMap::new();
    candidates.insert("present".to_string(), (1.0, false));
    candidates.insert("absent".to_string(), (1.0, false));
    let mut lex_rank = HashMap::new();
    lex_rank.insert("present".to_string(), 2usize);
    let mut vec_rank = HashMap::new();
    vec_rank.insert("present".to_string(), 5usize);

    // Act
    let out = score_and_rank(
        vec![present, absent],
        &candidates,
        &Ranks {
            lexical: lex_rank,
            vector: vec_rank,
            cold: HashMap::new(),
        },
        now,
        None,
        1.0,
        0.1,
        10,
    );

    // Assert
    let by_id: HashMap<&str, &ExplainedHit> = out.iter().map(|h| (h.hit.id.as_str(), h)).collect();
    assert_eq!(by_id["present"].lexical_rank, Some(2));
    assert_eq!(by_id["present"].vector_rank, Some(5));
    assert_eq!(by_id["absent"].lexical_rank, None);
    assert_eq!(by_id["absent"].vector_rank, None);
}

#[test]
fn score_and_rank_sorts_descending_and_truncates_to_k() {
    // Arrange: three rows with distinct scores, k smaller than the count.
    let now = Millis(1000);
    let rows = vec![
        scoring_candidate("low", 0.1, now),
        scoring_candidate("high", 0.9, now),
        scoring_candidate("mid", 0.5, now),
    ];
    let mut candidates = HashMap::new();
    candidates.insert("low".to_string(), (1.0, false));
    candidates.insert("high".to_string(), (1.0, false));
    candidates.insert("mid".to_string(), (1.0, false));

    // Act
    let out = score_and_rank(rows, &candidates, &Ranks::default(), now, None, 1.0, 0.1, 2);

    // Assert
    let ids: Vec<&str> = out.iter().map(|h| h.hit.id.as_str()).collect();
    assert_eq!(ids, vec!["high", "mid"]);
}

#[test]
fn score_and_rank_decay_is_one_when_half_life_is_none() {
    // Arrange: a row valid long before `now`; with no half-life the real
    // `decay_factor` helper should still report no decay.
    let now = Millis(10_000);
    let row = scoring_candidate("a", 1.0, Millis(0));
    let mut candidates = HashMap::new();
    candidates.insert("a".to_string(), (1.0, false));

    // Act
    let out = score_and_rank(
        vec![row],
        &candidates,
        &Ranks::default(),
        now,
        None,
        1.0,
        0.1,
        10,
    );

    // Assert
    assert_eq!(out[0].decay, 1.0);
}

#[tokio::test]
async fn mentions_matches_recency_order_despite_interleaved_insert_order() {
    // Arrange: facts inserted in ascending order, then mentioned in a
    // scrambled order, decoupling physical id order from recency order.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let entity = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let mut facts = Vec::new();
    for _ in 0..5 {
        facts.push(
            g.insert(NewNode::now("fact", "note", "note"))
                .await
                .unwrap(),
        );
    }
    let link_order = [2, 0, 4, 1, 3];
    let mut expected_recency_order = Vec::new();
    for (step, &idx) in link_order.iter().enumerate() {
        clock.set(Millis(5000 - step as i64 * 1000));
        g.link(NewEdge::new(&entity, &facts[idx], relation::MENTIONS))
            .await
            .unwrap();
        expected_recency_order.push(facts[idx].clone());
    }

    // Act
    let found = g.mentions(&entity, Millis(5000), 10).await.unwrap();

    // Assert
    let ids: Vec<NodeId> = found.into_iter().map(|c| c.id).collect();
    assert_eq!(ids, expected_recency_order);
}

#[tokio::test]
async fn mentions_does_not_undercount_when_the_most_recent_edge_targets_a_superseded_node() {
    // Arrange: two live mentions, then a third, most-recent-by-tx_from
    // mentions edge to a node that is later superseded. The edge itself
    // stays live forever (`supersede` never touches edges), so it would
    // previously have consumed the `LIMIT` slot ahead of a still-live
    // mention.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let entity = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let first = g
        .insert(NewNode::now("fact", "first", "note one"))
        .await
        .unwrap();
    g.link(NewEdge::new(&entity, &first, relation::MENTIONS))
        .await
        .unwrap();
    clock.set(Millis(2000));
    let second = g
        .insert(NewNode::now("fact", "second", "note two"))
        .await
        .unwrap();
    g.link(NewEdge::new(&entity, &second, relation::MENTIONS))
        .await
        .unwrap();
    clock.set(Millis(3000));
    let stale = g
        .insert(NewNode::now("fact", "stale", "note three"))
        .await
        .unwrap();
    g.link(NewEdge::new(&entity, &stale, relation::MENTIONS))
        .await
        .unwrap();
    clock.set(Millis(4000));
    g.supersede(&stale, NewNode::now("fact", "stale", "replacement"))
        .await
        .unwrap();

    // Act
    let found = g.mentions(&entity, Millis(4000), 2).await.unwrap();

    // Assert: both still-live mentions come back, not just one.
    let ids: Vec<NodeId> = found.into_iter().map(|c| c.id).collect();
    assert_eq!(ids, vec![second, first]);
}
