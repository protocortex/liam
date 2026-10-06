// SPDX-License-Identifier: Apache-2.0
//! A write finds the row it replaces by its open transaction, never by the
//! caller's clock, so a clock that steps backwards cannot leave two open rows
//! for one subject or a version that starts before the one it closes.

use liam_log::event::{LogPayload, RowEffect};

use super::log_write::{dump_rows, events, mentions, share, Appended, RecordingLog};
use super::*;
use crate::graph::projection::{apply_steps, steps_for};

const PRICE: &str = "price";

/// A graph over a clock the test moves, with or without a log.
struct Rig {
    g: DefaultGraph,
    clock: Arc<FixedClock>,
    appended: Option<Appended>,
}

async fn rig(logged: bool, t: Millis) -> Rig {
    let clock = Arc::new(FixedClock::new(t));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    if !logged {
        return Rig {
            g,
            clock,
            appended: None,
        };
    }
    let (log, appended) = RecordingLog::new();
    Rig {
        g: g.with_log(share(log)).await.unwrap(),
        clock,
        appended: Some(appended),
    }
}

fn fact(content: &str) -> NewNode {
    NewNode::now("fact", "label", content).with_producer("agent-a")
}

fn priced(content: &str) -> NewNode {
    fact(content).with_subject(PRICE)
}

/// `(content, tx_from, tx_to)` of every version of `subject`, oldest write first.
async fn versions(g: &DefaultGraph, subject: &str) -> Vec<(String, i64, i64)> {
    g.backend
        .query(
            "SELECT content, tx_from, tx_to FROM nodes WHERE subject = ?1 ORDER BY rowid",
            &[subject.into()],
        )
        .await
        .unwrap()
        .iter()
        .map(|row| {
            (
                row.get_string(0).unwrap(),
                row.get_i64(1).unwrap(),
                row.get_i64(2).unwrap(),
            )
        })
        .collect()
}

/// `(src content, dst content, tx_from)` of every `supersedes` edge.
async fn supersedes(g: &DefaultGraph) -> Vec<(String, String, i64)> {
    g.backend
        .query(
            "SELECT s.content, d.content, e.tx_from FROM edges e
             JOIN nodes s ON s.id = e.src JOIN nodes d ON d.id = e.dst
             WHERE e.type = ?1 ORDER BY e.rowid",
            &[relation::SUPERSEDES.into()],
        )
        .await
        .unwrap()
        .iter()
        .map(|row| {
            (
                row.get_string(0).unwrap(),
                row.get_string(1).unwrap(),
                row.get_i64(2).unwrap(),
            )
        })
        .collect()
}

fn version(content: &str, tx_from: i64, tx_to: i64) -> (String, i64, i64) {
    (content.to_string(), tx_from, tx_to)
}

fn link(new: &str, old: &str, tx_from: i64) -> (String, String, i64) {
    (new.to_string(), old.to_string(), tx_from)
}

#[tokio::test]
async fn upsert_by_after_the_clock_steps_back_supersedes_instead_of_adding_a_second_live_row() {
    for logged in [false, true] {
        // Arrange: the only live row started at 2000, the clock now reads 1000.
        let r = rig(logged, Millis(2000)).await;
        r.g.upsert_by(priced("v1")).await.unwrap();
        r.clock.set(Millis(1000));

        // Act
        r.g.upsert_by(priced("v2")).await.unwrap();

        // Assert: one open row, and the version starts where the old one ended.
        let expected = vec![version("v1", 2000, 2000), version("v2", 2000, FOREVER.0)];
        assert_eq!(versions(&r.g, PRICE).await, expected, "logged: {logged}");
        assert_eq!(
            supersedes(&r.g).await,
            vec![link("v2", "v1", 2000)],
            "logged: {logged}"
        );
    }
}

#[tokio::test]
async fn upsert_by_chain_keeps_non_decreasing_times_over_repeated_backward_steps() {
    // Arrange
    let r = rig(true, Millis(3000)).await;
    r.g.upsert_by(priced("v1")).await.unwrap();

    // Act: the clock keeps reading earlier than the row it replaces.
    for (content, at) in [("v2", 2000), ("v3", 1000)] {
        r.clock.set(Millis(at));
        r.g.upsert_by(priced(content)).await.unwrap();
    }

    // Assert
    let expected = vec![
        version("v1", 3000, 3000),
        version("v2", 3000, 3000),
        version("v3", 3000, FOREVER.0),
    ];
    assert_eq!(versions(&r.g, PRICE).await, expected);
}

#[tokio::test]
async fn supersede_after_the_clock_steps_back_closes_the_old_row_at_a_non_decreasing_time() {
    for logged in [false, true] {
        // Arrange
        let r = rig(logged, Millis(2000)).await;
        let old = r.g.insert(priced("v1")).await.unwrap();
        r.clock.set(Millis(1000));

        // Act
        r.g.supersede(&old, priced("v2")).await.unwrap();

        // Assert
        let expected = vec![version("v1", 2000, 2000), version("v2", 2000, FOREVER.0)];
        assert_eq!(versions(&r.g, PRICE).await, expected, "logged: {logged}");
        assert_eq!(
            supersedes(&r.g).await,
            vec![link("v2", "v1", 2000)],
            "logged: {logged}"
        );
    }
}

#[tokio::test]
async fn supersede_of_a_closed_row_is_refused_whatever_the_clock_reads() {
    for logged in [false, true] {
        // Arrange: v1 was closed at 3000, and the clock then steps back to 2500,
        // inside v1's old transaction interval.
        let r = rig(logged, Millis(2000)).await;
        let old = r.g.insert(priced("v1")).await.unwrap();
        r.clock.set(Millis(3000));
        r.g.supersede(&old, priced("v2")).await.unwrap();
        r.clock.set(Millis(2500));

        // Act
        let refused = r.g.supersede(&old, priced("v3")).await;

        // Assert
        assert!(
            matches!(refused, Err(Error::NodeNotFound(_))),
            "logged: {logged}: {refused:?}"
        );
        assert_eq!(versions(&r.g, PRICE).await.len(), 2, "logged: {logged}");
    }
}

#[tokio::test]
async fn episode_node_after_the_clock_steps_back_supersedes_the_stored_competitor() {
    for logged in [false, true] {
        // Arrange
        let r = rig(logged, Millis(2000)).await;
        r.g.insert(priced("v1")).await.unwrap();
        r.clock.set(Millis(1000));

        // Act: the second node replaces the first, which replaced the stored row.
        let nodes = vec![priced("e0"), priced("e1")];
        r.g.ingest_episode(nodes, Vec::new()).await.unwrap();

        // Assert
        let expected = vec![
            version("v1", 2000, 2000),
            version("e0", 2000, 2000),
            version("e1", 2000, FOREVER.0),
        ];
        assert_eq!(versions(&r.g, PRICE).await, expected, "logged: {logged}");
        let chain = vec![link("e0", "v1", 2000), link("e1", "e0", 2000)];
        assert_eq!(supersedes(&r.g).await, chain, "logged: {logged}");
    }
}

#[tokio::test]
async fn relate_after_the_clock_steps_back_still_finds_both_endpoints_open() {
    // Endpoints are open rows (`tx_to = FOREVER`), never a function of the
    // clock, so a row that started after `now` is still related.
    for logged in [false, true] {
        // Arrange
        let r = rig(logged, Millis(2000)).await;
        let a = r.g.insert(fact("a")).await.unwrap();
        let b = r.g.insert(fact("b")).await.unwrap();
        r.clock.set(Millis(1000));

        // Act
        let related = r.g.relate(&a, &b, relation::MENTIONS).await;

        // Assert
        assert!(related.is_ok(), "logged: {logged}: {related:?}");
        assert_eq!(count_edges(&r.g).await, 1, "logged: {logged}");
    }
}

#[tokio::test]
async fn relate_refuses_an_endpoint_closed_after_the_clock_steps_back() {
    for logged in [false, true] {
        // Arrange: `a` was superseded at 3000, so it is closed even though the
        // clock now reads 2500, inside its old interval.
        let r = rig(logged, Millis(2000)).await;
        let a = r.g.insert(priced("a")).await.unwrap();
        let b = r.g.insert(fact("b")).await.unwrap();
        r.clock.set(Millis(3000));
        r.g.supersede(&a, priced("a2")).await.unwrap();
        r.clock.set(Millis(2500));

        // Act
        let refused = r.g.relate(&a, &b, relation::MENTIONS).await;

        // Assert
        assert!(
            matches!(refused, Err(Error::RelateRefused(_))),
            "logged: {logged}: {refused:?}"
        );
    }
}

async fn count_edges(g: &DefaultGraph) -> i64 {
    super::log_write::count(g, "edges").await
}

#[tokio::test]
async fn reads_as_of_before_a_row_started_do_not_see_it() {
    // Arrange: v1 started at 2000; v2 replaced it under a clock reading 1000.
    let r = rig(true, Millis(2000)).await;
    let v1 = r.g.upsert_by(priced("v1")).await.unwrap();
    r.clock.set(Millis(1000));
    let v2 = r.g.upsert_by(priced("v2")).await.unwrap();

    // Act
    let seen = |id: NodeId, at: i64| {
        let g = &r.g;
        async move { g.get(&id, Millis(at)).await.unwrap().is_some() }
    };

    // Assert: reads keep the `live_at(as_of)` window of the stored times.
    assert!(!seen(v1.clone(), 1000).await, "v1 before it started");
    assert!(!seen(v2.clone(), 1000).await, "v2 before it started");
    assert!(!seen(v2.clone(), 1999).await, "v2 before it started");
    assert!(!seen(v1, 2000).await, "v1 is closed the instant it started");
    assert!(seen(v2, 2000).await, "v2 at its start");
}

/// `(ingested_at, node tx_from, edge tx_from)` of each `supersedes` edge a
/// batch event logged, with the node row the edge starts.
fn logged_supersedes(appended: &Appended) -> Vec<(i64, i64, i64)> {
    let mut found = Vec::new();
    for event in events(appended) {
        let LogPayload::EpisodeBatch(effects) = event.payload else {
            continue;
        };
        for effect in &effects {
            let RowEffect::Edge(edge) = effect else {
                continue;
            };
            let new_row = effects.iter().find_map(|effect| match effect {
                RowEffect::Node(row) if row.id == edge.src => Some(row.tx_from),
                _ => None,
            });
            if edge.edge_type == relation::SUPERSEDES {
                let node_from = new_row.expect("the new row rides in the same batch");
                found.push((event.ingested_at, node_from, edge.tx_from));
            }
        }
    }
    found
}

#[tokio::test]
async fn logged_batches_carry_the_clamped_times_and_replay_to_the_same_rows() {
    // Arrange: each supersede shape runs under a clock reading earlier than the
    // row it replaces.
    let r = rig(true, Millis(3000)).await;
    let appended = r.appended.clone().unwrap();
    r.g.upsert_by(priced("a1")).await.unwrap();
    r.clock.set(Millis(2000));
    r.g.upsert_by(priced("a2")).await.unwrap();
    let b = r.g.insert(fact("b1").with_subject("other")).await.unwrap();
    r.clock.set(Millis(1000));
    r.g.supersede(&b, fact("b2").with_subject("other"))
        .await
        .unwrap();
    let nodes = vec![priced("a3"), priced("a4"), fact("c")];
    let edges = vec![mentions(1, 2)];
    r.g.ingest_episode(nodes, edges).await.unwrap();

    // Act: apply each payload's own statements to an empty store.
    let replica = graph_at(Millis(1000)).await;
    for event in events(&appended) {
        let mut tx = replica.backend.begin().await.unwrap();
        apply_steps(&mut *tx, &steps_for(&event.payload))
            .await
            .unwrap();
        tx.commit().await.unwrap();
    }

    // Assert: the node and its edge start at the one clamped time, which is the
    // time the old row closes at, while the event keeps the real ingest time.
    let expected = vec![
        (2000, 3000, 3000),
        (1000, 2000, 2000),
        (1000, 3000, 3000),
        (1000, 3000, 3000),
    ];
    assert_eq!(logged_supersedes(&appended), expected);
    assert_eq!(dump_rows(&replica).await, dump_rows(&r.g).await);
    let open_prices = versions(&r.g, PRICE).await;
    assert_eq!(open_prices.iter().filter(|v| v.2 == FOREVER.0).count(), 1);
}
