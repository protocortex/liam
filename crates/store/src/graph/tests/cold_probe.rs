// SPDX-License-Identifier: Apache-2.0
//! The cold arm of a query: a demoted item has no vector, and a query whose
//! embedding is near its signature still finds it, behind a cutoff and a weight
//! that keep it from outranking a hot hit.

use super::quant::{node, open_at, signed, DIMS};
use super::tiers::{cold_at, drop_embeddings, lineage_of_node};
use super::*;
use crate::graph::tiers::ColdQuery;

/// Demotes `id` the way a tier step leaves it: its lineage cold, its int8 form
/// and its vector gone, its row, text index and signature kept.
async fn demote<B: Backend>(g: &Graph<B>, id: &NodeId) {
    let lineage = lineage_of_node(g, id).await;
    g.set_tier(&cold_at(&lineage, 1), None).await.unwrap();
    drop_embeddings(g, id).await;
}

/// The ids a query returned through the cold arm, sorted.
async fn cold_found<B: Backend>(g: &Graph<B>, query: &Query) -> Vec<String> {
    let mut found: Vec<String> = g
        .query_explained(query)
        .await
        .unwrap()
        .into_iter()
        .filter(|hit| hit.cold_rank.is_some())
        .map(|hit| hit.hit.id.as_str().to_string())
        .collect();
    found.sort();
    found
}

fn sorted(ids: &[&NodeId]) -> Vec<String> {
    let mut ids: Vec<String> = ids.iter().map(|id| id.as_str().to_string()).collect();
    ids.sort();
    ids
}

fn arm(pool: usize) -> ColdQuery<'static> {
    ColdQuery {
        pool,
        kind: None,
        scope: None,
        as_of: Millis(1000),
        include_inferred: false,
    }
}

#[tokio::test]
async fn a_cold_item_is_found_by_a_query_that_shares_no_words_and_ranks_below_a_hot_hit() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let hot = g.insert(node("hot", 0b1111_0000)).await.unwrap();
    let cold = g.insert(node("cold", 0b0000_1111)).await.unwrap();
    demote(&g, &cold).await;
    let query = Query::vector(signed(0b0000_1111)).with_k(5);

    // Act
    let hits = g.query_explained(&query).await.unwrap();

    // Assert
    let by_id = |id: &NodeId| hits.iter().find(|hit| hit.hit.id == *id).unwrap();
    let (cold_hit, hot_hit) = (by_id(&cold), by_id(&hot));
    assert_eq!(cold_hit.cold_rank, Some(0));
    assert_eq!((cold_hit.lexical_rank, cold_hit.vector_rank), (None, None));
    assert_eq!(hot_hit.vector_rank, Some(0));
    assert_eq!(hot_hit.cold_rank, None);
    // Both lead their own list, and the weight keeps the cold one behind.
    assert!(hot_hit.hit.score > cold_hit.hit.score);
    assert!((cold_hit.rrf * 2.0 - hot_hit.rrf).abs() < 1e-12);
}

#[tokio::test]
async fn a_cold_item_beyond_the_cutoff_is_not_a_candidate_and_the_hot_hit_keeps_its_place() {
    // Arrange: the decoy's signature is 4 of 8 bits from the query's, past the
    // cutoff of 2 that a ratio of 0.35 allows
    let g = open_at(Millis(1000)).await;
    let hot = g.insert(node("hot", 0b0000_0001)).await.unwrap();
    let decoy = g.insert(node("decoy", 0b0000_1111)).await.unwrap();
    demote(&g, &decoy).await;

    // Act
    let hits = g
        .query_explained(&Query::vector(signed(0)).with_k(5))
        .await
        .unwrap();

    // Assert
    assert!(!hits.iter().any(|hit| hit.hit.id == decoy));
    assert_eq!(hits[0].hit.id, hot);
    assert_eq!(hits[0].vector_rank, Some(0));
}

#[tokio::test]
async fn a_text_query_still_finds_a_cold_item_lexically_and_the_arm_is_skipped() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let cold = g
        .insert(
            NewNode::now("fact", "orchard", "apples grow in the orchard").with_embedding(signed(3)),
        )
        .await
        .unwrap();
    demote(&g, &cold).await;

    // Act
    let hits = g
        .query_explained(&Query::text("orchard apples"))
        .await
        .unwrap();

    // Assert
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].hit.id, cold);
    assert_eq!(hits[0].lexical_rank, Some(0));
    assert_eq!(hits[0].cold_rank, None);
}

#[tokio::test]
async fn the_tier_row_decides_what_is_cold_not_the_index_flag() {
    // Arrange: a hot item the index wrongly flags cold, and a cold item whose
    // vector a promotion has already written
    let g = open_at(Millis(1000)).await;
    let hot = g.insert(node("hot", 0b11)).await.unwrap();
    g.quant_index().set_cold([hot.as_str()], true);
    let promoting = g.insert(node("promoting", 0b10)).await.unwrap();
    let lineage = lineage_of_node(&g, &promoting).await;
    g.set_tier(&cold_at(&lineage, 1), None).await.unwrap();

    // Act
    let found = cold_found(&g, &Query::vector(signed(0b10)).with_k(5)).await;

    // Assert
    assert_eq!(found, sorted(&[&promoting]));
}

#[tokio::test]
async fn a_query_with_the_wrong_width_fails_as_it_did_before_the_arm() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let cold = g.insert(node("cold", 1)).await.unwrap();
    demote(&g, &cold).await;

    // Act
    let result = g.query_explained(&Query::vector(vec![1.0, 2.0])).await;

    // Assert
    assert!(matches!(result, Err(Error::Dimension { .. })), "{result:?}");
}

#[tokio::test]
async fn the_arm_respects_kind_scope_time_supersession_and_the_strict_filter() {
    // Arrange: every item has the query's own signature, so only the filters
    // decide
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock.clone())
        .await
        .unwrap();
    let signature = 0b0101_0101;
    let make = |label: &str, kind: &str, scope: &str| {
        NewNode::now(kind, label, format!("text of {label}"))
            .with_scope(scope)
            .with_embedding(signed(signature))
    };
    let in_scope = g.insert(make("in", "fact", "work/a")).await.unwrap();
    let elsewhere = g.insert(make("elsewhere", "fact", "home")).await.unwrap();
    let other_kind = g.insert(make("kind", "note", "work/a")).await.unwrap();
    let replaced = g.insert(make("replaced", "fact", "work/a")).await.unwrap();
    let replacement = g
        .supersede(&replaced, make("replacement", "fact", "work/a"))
        .await
        .unwrap();
    let page = g
        .upsert_by(
            NewNode::entity("person", "Ada")
                .with_scope("work/a")
                .with_embedding(signed(signature)),
        )
        .await
        .unwrap();
    clock.set(Millis(2000));
    let later = g.insert(make("later", "fact", "work/a")).await.unwrap();
    // The replacement shares the replaced item's lineage, so it goes cold with it.
    for id in [
        &in_scope,
        &elsewhere,
        &other_kind,
        &replaced,
        &replacement,
        &page,
        &later,
    ] {
        demote(&g, id).await;
    }
    let vector = || Query::vector(signed(signature)).with_k(20);
    let at = Millis(1500);

    // Act
    let scoped = cold_found(
        &g,
        &vector()
            .with_scope("work/a")
            .with_kind("fact")
            .with_as_of(at),
    )
    .await;
    let unscoped = cold_found(&g, &vector().with_kind("fact").with_as_of(at)).await;
    let after = cold_found(&g, &vector().with_scope("work/a").with_kind("fact")).await;
    let strict = cold_found(&g, &vector().with_scope("work/a").with_as_of(at)).await;
    let inferred = cold_found(
        &g,
        &vector()
            .with_scope("work/a")
            .with_as_of(at)
            .with_inferred(true),
    )
    .await;

    // Assert
    // The replaced item is absent everywhere: it is closed at the query time.
    assert_eq!(scoped, sorted(&[&in_scope, &replacement]));
    assert_eq!(unscoped, sorted(&[&in_scope, &elsewhere, &replacement]));
    assert_eq!(after, sorted(&[&in_scope, &replacement, &later]));
    assert_eq!(strict, sorted(&[&in_scope, &other_kind, &replacement]));
    assert_eq!(
        inferred,
        sorted(&[&in_scope, &other_kind, &replacement, &page])
    );
}

#[tokio::test]
async fn the_arm_widens_its_ask_until_enough_candidates_pass_the_filters() {
    // Arrange: six nearer decoys out of scope, three passing items farther
    // away, and an index asked for only the pool at first
    let config = GraphConfig::new(DIMS).with_cold_arm(0.5, 0.35, 1, 24);
    let g =
        DefaultGraph::open_with_clock(":memory:", config, Arc::new(FixedClock::new(Millis(1000))))
            .await
            .unwrap();
    let make = |label: &str, scope: &str, pattern: u8| {
        NewNode::now("fact", label, format!("text of {label}"))
            .with_scope(scope)
            .with_embedding(signed(pattern))
    };
    let mut passing = Vec::new();
    for index in 0..6 {
        let decoy = g
            .insert(make(&format!("decoy {index}"), "home", 0))
            .await
            .unwrap();
        demote(&g, &decoy).await;
    }
    for index in 0..3 {
        let item = g
            .insert(make(&format!("item {index}"), "work", 1))
            .await
            .unwrap();
        demote(&g, &item).await;
        passing.push(item);
    }
    let query = ColdQuery {
        scope: Some("work"),
        ..arm(3)
    };

    // Act
    let found = g.cold_probe(&signed(0), query).await.unwrap();

    // Assert
    let mut found: Vec<String> = found.iter().map(|id| id.as_str().to_string()).collect();
    found.sort();
    assert_eq!(found, sorted(&passing.iter().collect::<Vec<_>>()));
}

#[tokio::test]
async fn the_arm_returns_at_most_its_pool_nearest_first_and_ties_by_id() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let mut ids = Vec::new();
    for (label, pattern) in [
        ("far", 0b11),
        ("near", 0b1),
        ("tie a", 0b10),
        ("tie b", 0b100),
    ] {
        let id = g.insert(node(label, pattern)).await.unwrap();
        demote(&g, &id).await;
        ids.push(id);
    }
    let [far, near, tie_a, tie_b] = &ids[..] else {
        unreachable!()
    };

    // Act
    let all = g.cold_probe(&signed(0), arm(3)).await.unwrap();

    // Assert: distances 1, 1, 1 and 2; the pool of 3 keeps the three at 1, by id
    let mut ties = [near.clone(), tie_a.clone(), tie_b.clone()];
    ties.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    assert_eq!(all, ties);
    assert!(!all.contains(far));
}

// ---- recall against the exact f32 ranking ----

/// A hand-rolled generator, so the fixture needs no dependency and is the same
/// on every run.
struct Xorshift(u64);

impl Xorshift {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `[low, high)`.
    fn between(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[tokio::test]
async fn the_hamming_top_ten_holds_most_of_the_exact_top_ten_of_vectors_with_a_non_zero_mean() {
    // The fixture is a cluster of 10 vectors around a centre, whose cosine to
    // the query is about 0.95, among 190 unrelated ones. Every component is
    // uniform in [-0.4, 1.0), a mean of 0.3, so the signs carry less than they
    // would for centred data. A sign differs from the centre's when the noise
    // (at most 0.1) outweighs the component, which happens for the roughly one
    // component in seven within 0.1 of zero and half the time then: a cluster
    // member sits about 5 of 64 bits from the query. An unrelated vector
    // differs where independent signs differ, about 0.4 of the bits, 26 of 64.
    // That gap is more than five standard deviations of either distance, so a
    // top ten that keeps 8 of the 10 is a bound the signatures must clear with
    // room to spare, and it was set before the fixture was run.
    const DIMS_64: usize = 64;
    const CLUSTER: usize = 10;
    const OTHERS: usize = 190;
    const MIN_SHARE: usize = 8;
    let mut random = Xorshift(0x9E37_79B9_7F4A_7C15);
    let vector = |centre: Option<&[f32]>, random: &mut Xorshift| -> Vec<f32> {
        (0..DIMS_64)
            .map(|i| match centre {
                Some(centre) => centre[i] + random.between(-0.1, 0.1),
                None => random.between(-0.4, 1.0),
            })
            .collect()
    };
    let centre = vector(None, &mut random);
    let query = vector(Some(&centre), &mut random);
    let mut stored: Vec<Vec<f32>> = (0..CLUSTER)
        .map(|_| vector(Some(&centre), &mut random))
        .collect();
    stored.extend((0..OTHERS).map(|_| vector(None, &mut random)));
    let config = GraphConfig::new(DIMS_64).with_cold_arm(0.5, 1.0, 4, 30);
    let g =
        DefaultGraph::open_with_clock(":memory:", config, Arc::new(FixedClock::new(Millis(1000))))
            .await
            .unwrap();
    let mut by_id = Vec::new();
    for (index, embedding) in stored.iter().enumerate() {
        let id = g
            .insert(
                NewNode::now("fact", format!("n{index}"), format!("t{index}"))
                    .with_embedding(embedding.clone()),
            )
            .await
            .unwrap();
        demote(&g, &id).await;
        by_id.push((id, cosine(&query, embedding)));
    }
    by_id.sort_by(|a, b| b.1.total_cmp(&a.1));
    let exact: Vec<&NodeId> = by_id.iter().take(10).map(|(id, _)| id).collect();

    // Act
    let found = g.cold_probe(&query, arm(10)).await.unwrap();

    // Assert
    let shared = found.iter().filter(|id| exact.contains(id)).count();
    assert_eq!(found.len(), 10);
    assert!(shared >= MIN_SHARE, "kept {shared} of the exact top 10");
}

// ---- fusion ----

fn ids(names: &[&str]) -> Vec<NodeId> {
    names.iter().map(|name| NodeId::from_raw(*name)).collect()
}

#[test]
fn fusion_without_a_cold_list_scores_as_two_lists_always_did() {
    // Arrange
    let (lexical, vector) = (ids(&["a", "b"]), ids(&["b", "c"]));
    let rrf_k = 60.0;

    // Act
    let fused = rrf_fuse(&lexical, &vector, &[], rrf_k, 0.5);

    // Assert
    assert_eq!(fused["a"], 1.0 / 61.0);
    assert_eq!(fused["b"], 1.0 / 62.0 + 1.0 / 61.0);
    assert_eq!(fused["c"], 1.0 / 62.0);
    assert_eq!(fused.len(), 3);
}

#[test]
fn fusion_scales_the_cold_list_by_its_weight() {
    // Arrange
    let (vector, cold) = (ids(&["hot"]), ids(&["cold", "hot"]));

    // Act
    let fused = rrf_fuse(&[], &vector, &cold, 60.0, 0.5);

    // Assert
    assert_eq!(fused["hot"], 1.0 / 61.0 + 0.5 / 62.0);
    assert_eq!(fused["cold"], 0.5 / 61.0);
    assert!(fused["hot"] > fused["cold"]);
}
