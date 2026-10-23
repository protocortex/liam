// SPDX-License-Identifier: Apache-2.0
//! Equal scores come back in insertion order, whatever the per-instance hash
//! seed or the random suffix of the minted ids.

use super::*;

const DIMS: usize = 8;
const FRESH_STORES: usize = 20;
const NEIGHBOURS: usize = 12;
const TRUNCATED_K: usize = 5;
const HUB_TERM: &str = "hubterm";
const LEXICAL_TIES: usize = 30;
const LEXICAL_TERM: &str = "alphaterm";

async fn fresh_graph() -> DefaultGraph {
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    DefaultGraph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock)
        .await
        .unwrap()
}

fn labels(hits: &[Hit]) -> Vec<String> {
    hits.iter().map(|h| h.label.clone()).collect()
}

#[tokio::test]
async fn tied_expanded_neighbours_keep_insertion_order_and_the_same_k_members() {
    // Arrange: one seed whose twelve neighbours all share the expansion floor
    // score, so only the tie-break decides which survive truncation.
    let expected: Vec<String> = std::iter::once("hub".to_string())
        .chain((0..TRUNCATED_K - 1).map(|i| format!("n{i:02}")))
        .collect();

    for _ in 0..FRESH_STORES {
        let g = fresh_graph().await;
        let hub = g
            .insert(NewNode::now("fact", "hub", HUB_TERM))
            .await
            .unwrap();
        for i in 0..NEIGHBOURS {
            let n = g
                .insert(NewNode::now(
                    "fact",
                    format!("n{i:02}"),
                    format!("other{i}"),
                ))
                .await
                .unwrap();
            g.link(NewEdge::new(&hub, &n, relation::MENTIONS))
                .await
                .unwrap();
        }

        // Act
        let hits = g
            .query(&Query::text(HUB_TERM).with_k(TRUNCATED_K))
            .await
            .unwrap();

        // Assert
        assert_eq!(labels(&hits), expected);
    }
}

#[tokio::test]
async fn tied_seeds_keep_insertion_order_through_seed_and_final_truncation() {
    // Arrange: a lexical-only hit and a vector-only hit share rank 0 in their
    // lists, so their fused scores tie; each has one neighbour.
    for _ in 0..FRESH_STORES {
        let g = fresh_graph().await;
        let lexical = g
            .insert(NewNode::now("fact", "lexical", LEXICAL_TERM))
            .await
            .unwrap();
        let vector = g
            .insert(NewNode::now("fact", "vector", "x").with_embedding(vec![1.0; DIMS]))
            .await
            .unwrap();
        for (seed, label) in [(&lexical, "lexical-n"), (&vector, "vector-n")] {
            let n = g.insert(NewNode::now("fact", label, "y")).await.unwrap();
            g.link(NewEdge::new(seed, &n, relation::MENTIONS))
                .await
                .unwrap();
        }

        // Act
        let hits = g
            .query(
                &Query::text(LEXICAL_TERM)
                    .with_embedding(vec![1.0; DIMS])
                    .with_k(1),
            )
            .await
            .unwrap();

        // Assert
        assert_eq!(labels(&hits), ["lexical"]);
    }
}

/// Candidates whose ids sort opposite to their rowids, so a tie-break keyed on
/// the id instead of the rowid yields the reverse order.
fn row(id: &str, rowid: i64) -> Candidate {
    let mut c = Candidate::new(NodeId::from_raw(id), "fact", id, "");
    c.rowid = rowid;
    c
}

fn ids(hits: &[ExplainedHit]) -> Vec<&str> {
    hits.iter().map(|h| h.hit.id.as_str()).collect()
}

fn rank(rows: Vec<Candidate>, k: usize) -> Vec<ExplainedHit> {
    let candidates: HashMap<String, (f64, bool)> = rows
        .iter()
        .map(|c| (c.id.as_str().to_string(), (1.0, false)))
        .collect();
    score_and_rank(
        rows,
        &candidates,
        &HashMap::new(),
        &HashMap::new(),
        Millis(0),
        None,
        0.5,
        0.1,
        k,
    )
}

#[test]
fn score_and_rank_orders_tied_rows_by_ascending_rowid_whatever_the_input_order() {
    // Arrange: equal scores, fed in reverse rowid order, ids opposite to rowid.
    let rows = vec![row("a", 3), row("m", 2), row("z", 1)];

    // Act
    let out = rank(rows, 3);

    // Assert
    assert_eq!(ids(&out), ["z", "m", "a"]);
}

#[test]
fn score_and_rank_keeps_the_lowest_rowids_when_ties_are_truncated() {
    // Arrange
    let rows = vec![row("a", 4), row("b", 3), row("c", 2), row("d", 1)];

    // Act
    let out = rank(rows, 2);

    // Assert
    assert_eq!(ids(&out), ["d", "c"]);
}

#[test]
fn score_and_rank_puts_a_higher_score_before_a_lower_rowid() {
    // Arrange
    let rows = vec![row("low", 1), row("high", 2)];
    let candidates = HashMap::from([
        ("low".to_string(), (1.0, false)),
        ("high".to_string(), (2.0, false)),
    ]);

    // Act
    let out = score_and_rank(
        rows,
        &candidates,
        &HashMap::new(),
        &HashMap::new(),
        Millis(0),
        None,
        0.5,
        0.1,
        2,
    );

    // Assert
    assert_eq!(ids(&out), ["high", "low"]);
}

#[test]
fn score_and_rank_orders_a_nan_score_the_same_way_every_time() {
    // Arrange: a NaN score among finite ones must not make the sort depend on
    // the input order.
    let build = |order: [(&str, i64); 3]| {
        let rows: Vec<Candidate> = order.iter().map(|(id, r)| row(id, *r)).collect();
        let candidates = HashMap::from([
            ("a".to_string(), (f64::NAN, false)),
            ("b".to_string(), (1.0, false)),
            ("c".to_string(), (1.0, false)),
        ]);
        let out = score_and_rank(
            rows,
            &candidates,
            &HashMap::new(),
            &HashMap::new(),
            Millis(0),
            None,
            0.5,
            0.1,
            3,
        );
        out.iter()
            .map(|h| h.hit.id.as_str().to_string())
            .collect::<Vec<_>>()
    };

    // Act
    let forward = build([("a", 1), ("b", 2), ("c", 3)]);
    let reversed = build([("c", 3), ("b", 2), ("a", 1)]);

    // Assert
    assert_eq!(forward, reversed);
}

fn tied_scores(ids: &[&str]) -> HashMap<String, f64> {
    ids.iter().map(|id| (id.to_string(), 0.5)).collect()
}

#[test]
fn rank_seeds_keeps_the_lowest_rowids_among_tied_scores() {
    // Arrange: four tied seeds, ids opposite to rowid.
    let rrf = tied_scores(&["a", "b", "c", "d"]);
    let rowids = HashMap::from([
        ("a".to_string(), 4),
        ("b".to_string(), 3),
        ("c".to_string(), 2),
        ("d".to_string(), 1),
    ]);

    // Act
    let seeds = rank_seeds(&rrf, &rowids, 2);

    // Assert
    assert_eq!(seeds, ["d", "c"]);
}

#[test]
fn rank_seeds_prefers_a_higher_score_over_a_lower_rowid() {
    // Arrange
    let rrf = HashMap::from([("old".to_string(), 0.1), ("new".to_string(), 0.9)]);
    let rowids = HashMap::from([("old".to_string(), 1), ("new".to_string(), 2)]);

    // Act
    let seeds = rank_seeds(&rrf, &rowids, 1);

    // Assert
    assert_eq!(seeds, ["new"]);
}

#[test]
fn rank_seeds_sorts_an_id_without_a_rowid_last_among_ties() {
    // Arrange
    let rrf = tied_scores(&["gone", "kept"]);
    let rowids = HashMap::from([("kept".to_string(), 9)]);

    // Act
    let seeds = rank_seeds(&rrf, &rowids, 1);

    // Assert
    assert_eq!(seeds, ["kept"]);
}

#[test]
fn with_rrf_k_clamps_a_negative_value_and_ignores_a_non_finite_one() {
    // Arrange
    let base = GraphConfig::new(DIMS);

    // Act
    let negative = base.with_rrf_k(-5.0).rrf_k;
    let nan = base.with_rrf_k(f64::NAN).rrf_k;
    let infinite = base.with_rrf_k(f64::INFINITY).rrf_k;
    let valid = base.with_rrf_k(10.0).rrf_k;

    // Assert
    assert_eq!(negative, 0.0);
    assert_eq!(nan, base.rrf_k);
    assert_eq!(infinite, base.rrf_k);
    assert_eq!(valid, 10.0);
}

#[tokio::test]
async fn tied_lexical_hits_beyond_the_pool_return_the_first_inserted() {
    // Arrange: many identical rows, so bm25 ties and the pool of three (k = 1)
    // leaves most out; only the SQL tie-break picks which three get in.
    for _ in 0..FRESH_STORES {
        let g = fresh_graph().await;
        for i in 0..LEXICAL_TIES {
            g.insert(NewNode::now("fact", format!("row{i:02}"), LEXICAL_TERM))
                .await
                .unwrap();
        }

        // Act
        let hits = g.query(&Query::text(LEXICAL_TERM).with_k(1)).await.unwrap();

        // Assert
        assert_eq!(labels(&hits), ["row00"]);
    }
}

#[tokio::test]
async fn tied_vector_hits_beyond_the_pool_return_the_first_inserted() {
    // Arrange: ten rows with the same embedding, so distances tie. The query
    // plan scans `nodes` by `tx_from`, so the first row is given the latest
    // `tx_from`: left to the scan order alone it would come last, and only the
    // SQL tie-break on `nodes.rowid` puts it first.
    for _ in 0..FRESH_STORES {
        let g = fresh_graph().await;
        let mut first = None;
        for i in 0..10 {
            let id = g
                .insert(
                    NewNode::now("fact", format!("row{i:02}"), "x").with_embedding(vec![1.0; DIMS]),
                )
                .await
                .unwrap();
            first.get_or_insert(id);
        }
        g.backend
            .execute(
                "UPDATE nodes SET tx_from = CASE WHEN id = ?1 THEN 900 ELSE 100 END",
                &[first.unwrap().as_str().into()],
            )
            .await
            .unwrap();

        // Act
        let hits = g
            .query(&Query::vector(vec![1.0; DIMS]).with_k(1))
            .await
            .unwrap();

        // Assert
        assert_eq!(labels(&hits), ["row00"]);
    }
}

#[tokio::test]
async fn a_tied_lexical_hit_with_a_higher_rowid_loses_to_a_vector_hit_at_k_one() {
    // Arrange: the vector hit is inserted first (lower rowid) while the
    // lexical hit comes first in the fused list, so list position and rowid
    // disagree about which tied seed wins.
    for _ in 0..FRESH_STORES {
        let g = fresh_graph().await;
        let vector = g
            .insert(NewNode::now("fact", "vector", "x").with_embedding(vec![1.0; DIMS]))
            .await
            .unwrap();
        let lexical = g
            .insert(NewNode::now("fact", "lexical", LEXICAL_TERM))
            .await
            .unwrap();
        for (seed, label) in [(&vector, "vector-n"), (&lexical, "lexical-n")] {
            let n = g.insert(NewNode::now("fact", label, "y")).await.unwrap();
            g.link(NewEdge::new(seed, &n, relation::MENTIONS))
                .await
                .unwrap();
        }

        // Act: k = 2 keeps one seed; its neighbour shows which was picked.
        let hits = g
            .query(
                &Query::text(LEXICAL_TERM)
                    .with_embedding(vec![1.0; DIMS])
                    .with_k(2),
            )
            .await
            .unwrap();

        // Assert
        assert_eq!(labels(&hits), ["vector", "lexical"]);
    }
}

#[tokio::test]
async fn mentions_with_equal_tx_from_follow_edge_id_descending() {
    // Arrange: six mention edges at one instant; limit 1 keeps a pool of three.
    for _ in 0..FRESH_STORES {
        let g = fresh_graph().await;
        let entity = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
        for i in 0..6 {
            let n = g
                .insert(NewNode::now("fact", format!("m{i}"), "x"))
                .await
                .unwrap();
            g.link(NewEdge::new(&entity, &n, relation::MENTIONS))
                .await
                .unwrap();
        }
        let top = g
            .backend
            .query("SELECT dst FROM edges ORDER BY id DESC LIMIT 1", &[])
            .await
            .unwrap()[0]
            .get_string(0)
            .unwrap();

        // Act
        let found = g.mentions(&entity, Millis(1000), 1).await.unwrap();

        // Assert
        assert_eq!(found[0].id.as_str(), top);
    }
}

#[tokio::test]
async fn stale_entities_never_synthesized_come_back_in_insertion_order() {
    // Arrange: six entities that all tie at last_synthesized_at = 0.
    const NAMES: [&str; 6] = ["Ada", "Bob", "Cyd", "Dee", "Eve", "Fay"];
    for _ in 0..FRESH_STORES {
        let g = fresh_graph().await;
        let mut inserted = Vec::new();
        for name in NAMES {
            inserted.push(g.insert(NewNode::entity("person", name)).await.unwrap());
        }

        // Act
        let stale = g.stale_entities(10, Millis(2000)).await.unwrap();

        // Assert: oldest insert first, whatever the random ids sort like
        let got: Vec<&NodeId> = stale.iter().map(|(id, _, _)| id).collect();
        assert_eq!(got, inserted.iter().collect::<Vec<_>>());
    }
}
