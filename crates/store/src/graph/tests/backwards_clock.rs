// SPDX-License-Identifier: Apache-2.0
//! A write finds the row it replaces by its open transaction, never by the
//! caller's clock, so a clock that steps backwards cannot leave two open rows
//! for one subject or a version that starts before the one it closes.

use liam_log::event::{LogEvent, LogPayload, RowEffect};

use super::log_write::{dump_rows, events, mentions, Appended, RecordingLog};
use super::support::share;
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

/// `(content, valid_from, tx_from, tx_to)` of every version of `subject`, oldest
/// write first.
async fn versions(g: &DefaultGraph, subject: &str) -> Vec<(String, i64, i64, i64)> {
    g.backend
        .query(
            "SELECT content, valid_from, tx_from, tx_to FROM nodes WHERE subject = ?1
             ORDER BY rowid",
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
                row.get_i64(3).unwrap(),
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

fn version(content: &str, valid_from: i64, tx_from: i64, tx_to: i64) -> (String, i64, i64, i64) {
    (content.to_string(), valid_from, tx_from, tx_to)
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
        // Valid time is the world's, so the clamp leaves v2 at the clock's 1000.
        let expected = vec![
            version("v1", 2000, 2000, 2000),
            version("v2", 1000, 2000, FOREVER.0),
        ];
        assert_eq!(versions(&r.g, PRICE).await, expected, "logged: {logged}");
        assert_eq!(
            supersedes(&r.g).await,
            vec![link("v2", "v1", 2000)],
            "logged: {logged}"
        );
        if let Some(appended) = &r.appended {
            let last = events(appended).pop().unwrap();
            assert_eq!((last.observed_at, last.ingested_at), (1000, 1000));
        }
    }
}

#[tokio::test]
async fn upsert_by_chain_keeps_non_decreasing_times_over_repeated_backward_steps() {
    for logged in [false, true] {
        // Arrange
        let r = rig(logged, Millis(3000)).await;
        r.g.upsert_by(priced("v1")).await.unwrap();

        // Act: the clock keeps reading earlier than the row it replaces.
        for (content, at) in [("v2", 2000), ("v3", 1000)] {
            r.clock.set(Millis(at));
            r.g.upsert_by(priced(content)).await.unwrap();
        }

        // Assert
        let expected = vec![
            version("v1", 3000, 3000, 3000),
            version("v2", 2000, 3000, 3000),
            version("v3", 1000, 3000, FOREVER.0),
        ];
        assert_eq!(versions(&r.g, PRICE).await, expected, "logged: {logged}");
        let chain = vec![link("v2", "v1", 3000), link("v3", "v2", 3000)];
        assert_eq!(supersedes(&r.g).await, chain, "logged: {logged}");
    }
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
        let expected = vec![
            version("v1", 2000, 2000, 2000),
            version("v2", 1000, 2000, FOREVER.0),
        ];
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
        let before = logged_events(&r);

        // Act
        let refused = r.g.supersede(&old, priced("v3")).await;

        // Assert: nothing is logged, not even a write that is voided after.
        assert!(
            matches!(refused, Err(Error::NodeNotFound(_))),
            "logged: {logged}: {refused:?}"
        );
        let expected = vec![
            version("v1", 2000, 2000, 3000),
            version("v2", 3000, 3000, FOREVER.0),
        ];
        assert_eq!(versions(&r.g, PRICE).await, expected, "logged: {logged}");
        assert_log_untouched(&r, &before).await;
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
            version("v1", 2000, 2000, 2000),
            version("e0", 1000, 2000, 2000),
            version("e1", 1000, 2000, FOREVER.0),
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

        // Assert: only a `supersedes` edge is clamped. A relation records when it
        // was asserted, which is what the clock read.
        assert!(related.is_ok(), "logged: {logged}: {related:?}");
        let mentions = edge_starts(&r.g, relation::MENTIONS).await;
        assert_eq!(mentions, vec![1000], "logged: {logged}");
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
        let before = logged_events(&r);

        // Act
        let refused = r.g.relate(&a, &b, relation::MENTIONS).await;

        // Assert: nothing is logged, not even a write that is voided after.
        assert!(
            matches!(refused, Err(Error::RelateRefused(_))),
            "logged: {logged}: {refused:?}"
        );
        assert!(
            edge_starts(&r.g, relation::MENTIONS).await.is_empty(),
            "logged: {logged}"
        );
        assert_log_untouched(&r, &before).await;
    }
}

/// `tx_from` of every edge of `edge_type`, oldest write first.
async fn edge_starts(g: &DefaultGraph, edge_type: &str) -> Vec<i64> {
    g.backend
        .query(
            "SELECT tx_from FROM edges WHERE type = ?1 ORDER BY rowid",
            &[edge_type.into()],
        )
        .await
        .unwrap()
        .iter()
        .map(|row| row.get_i64(0).unwrap())
        .collect()
}

fn logged_events(r: &Rig) -> Vec<LogEvent> {
    r.appended.as_ref().map(events).unwrap_or_default()
}

/// A refused write leaves the log as it was and does not stop the next write
/// from being logged.
async fn assert_log_untouched(r: &Rig, before: &[LogEvent]) {
    assert_eq!(logged_events(r), before);
    r.g.insert(fact("later")).await.unwrap();
    let grew = logged_events(r).len() - before.len();
    assert_eq!(grew, usize::from(r.appended.is_some()));
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

    // Assert: reads keep the `live_at(as_of)` window of the stored times. v2's
    // valid time began at 1000, so it is its transaction time that hides it.
    assert!(
        !seen(v2.clone(), 1000).await,
        "v2 before its transaction began"
    );
    assert!(
        !seen(v2.clone(), 1999).await,
        "v2 before its transaction began"
    );
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
        let steps = steps_for(&event.payload);
        let vector_delete = replica.backend.vector_delete_sql();
        apply_steps(&mut *tx, &steps, vector_delete).await.unwrap();
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
    let chain = vec![
        version("a1", 3000, 3000, 3000),
        version("a2", 2000, 3000, 3000),
        version("a3", 1000, 3000, 3000),
        version("a4", 1000, 3000, FOREVER.0),
    ];
    assert_eq!(versions(&r.g, PRICE).await, chain);
}

#[tokio::test]
async fn upsert_by_replaces_an_open_row_whose_valid_time_has_not_begun() {
    for logged in [false, true] {
        // Arrange: v1 is open but valid only from 5000, and the clock reads 1000.
        let r = rig(logged, Millis(1000)).await;
        r.g.insert(priced("v1").with_valid_from(Millis(5000)))
            .await
            .unwrap();

        // Act
        r.g.upsert_by(priced("v2")).await.unwrap();

        // Assert: a write replaces the open row, whatever its valid time says.
        let expected = vec![
            version("v1", 5000, 1000, 1000),
            version("v2", 1000, 1000, FOREVER.0),
        ];
        assert_eq!(versions(&r.g, PRICE).await, expected, "logged: {logged}");
    }
}

#[tokio::test]
async fn supersede_replaces_an_open_row_whose_valid_time_has_not_begun() {
    for logged in [false, true] {
        // Arrange
        let r = rig(logged, Millis(1000)).await;
        let old =
            r.g.insert(priced("v1").with_valid_from(Millis(5000)))
                .await
                .unwrap();

        // Act
        r.g.supersede(&old, priced("v2")).await.unwrap();

        // Assert
        let expected = vec![
            version("v1", 5000, 1000, 1000),
            version("v2", 1000, 1000, FOREVER.0),
        ];
        assert_eq!(versions(&r.g, PRICE).await, expected, "logged: {logged}");
    }
}

async fn open_ids(g: &DefaultGraph, subject: &str) -> Vec<String> {
    g.backend
        .query(
            "SELECT id FROM nodes WHERE subject = ?1 AND tx_to = ?2",
            &[subject.into(), FOREVER.into()],
        )
        .await
        .unwrap()
        .iter()
        .map(|row| row.get_string(0).unwrap())
        .collect()
}

#[tokio::test]
async fn upsert_by_closes_the_larger_id_when_two_open_rows_share_a_start() {
    // Ids within one millisecond are random, so repeat until either order of
    // the two rows would have shown a tie-break by anything but id.
    for logged in [false, true] {
        for _ in 0..8 {
            // Arrange: two open rows started at the same clock reading.
            let r = rig(logged, Millis(2000)).await;
            let first = r.g.insert(priced("a")).await.unwrap();
            let second = r.g.insert(priced("b")).await.unwrap();
            let (smaller, larger) = if first.as_str() < second.as_str() {
                (first, second)
            } else {
                (second, first)
            };
            r.clock.set(Millis(1000));

            // Act
            let written = r.g.upsert_by(priced("c")).await.unwrap();

            // Assert: the larger id is the one replaced, the other stays open.
            let mut open = open_ids(&r.g, PRICE).await;
            open.sort();
            let mut expected = vec![smaller.as_str().to_string(), written.as_str().to_string()];
            expected.sort();
            assert_eq!(open, expected, "logged: {logged}, replaced {larger:?}");
        }
    }
}

#[tokio::test]
async fn mentions_hide_a_relation_asserted_after_the_read_time() {
    for logged in [false, true] {
        // Arrange: both nodes exist from 1000, the mention is asserted at 2000.
        let r = rig(logged, Millis(1000)).await;
        let entity = r.g.insert(NewNode::entity("person", "Ada")).await.unwrap();
        let fact_id = r.g.insert(fact("x")).await.unwrap();
        r.clock.set(Millis(2000));
        r.g.relate(&entity, &fact_id, relation::MENTIONS)
            .await
            .unwrap();

        // Act
        let before = r.g.mentions(&entity, Millis(1500), 10).await.unwrap();
        let at = r.g.mentions(&entity, Millis(2000), 10).await.unwrap();

        // Assert
        assert!(before.is_empty(), "logged: {logged}");
        let ids: Vec<_> = at.iter().map(|c| c.id.clone()).collect();
        assert_eq!(ids, vec![fact_id], "logged: {logged}");
    }
}

#[tokio::test]
async fn repair_watermark_stays_behind_the_clock_after_a_clamped_supersede() {
    for logged in [false, true] {
        // Arrange: a supersede at clock 1000 over a row started at 3000 leaves a
        // `supersedes` edge at 3000, later than the clock.
        let r = rig(logged, Millis(3000)).await;
        r.g.upsert_by(priced("v1")).await.unwrap();
        r.clock.set(Millis(1000));
        let entity = r.g.insert(NewNode::entity("person", "Ada")).await.unwrap();
        let fact_id = r.g.insert(fact("x")).await.unwrap();
        r.g.relate(&entity, &fact_id, relation::MENTIONS)
            .await
            .unwrap();
        r.g.upsert_by(priced("v2")).await.unwrap();
        assert_eq!(edge_starts(&r.g, relation::SUPERSEDES).await, vec![3000]);

        // Act: repair now, then supersede the entity page at 1500 and repair again.
        r.g.repair_superseded_mentions().await.unwrap();
        let after_first = r.g.repair_watermark().await.unwrap();
        r.clock.set(Millis(1500));
        let successor =
            r.g.supersede(&entity, NewNode::entity("person", "Ada"))
                .await
                .unwrap();
        let moved = r.g.repair_superseded_mentions().await.unwrap();
        let after_second = r.g.repair_watermark().await.unwrap();

        // Assert: the later supersession is not skipped behind a future watermark.
        assert!(
            after_first <= Millis(1000),
            "logged: {logged}: {after_first:?}"
        );
        assert!(
            after_second <= Millis(1500),
            "logged: {logged}: {after_second:?}"
        );
        assert_eq!(moved, 1, "logged: {logged}");
        let mentioned = r.g.mentions(&successor, Millis(1500), 10).await.unwrap();
        let ids: Vec<_> = mentioned.iter().map(|c| c.id.clone()).collect();
        assert_eq!(ids, vec![fact_id], "logged: {logged}");
    }
}
