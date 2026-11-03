// SPDX-License-Identifier: Apache-2.0
//! The rows that carry tiering: the lineage backfill, tiers, usage and feedback,
//! the reconcile of vectors with the tier rows, and the snapshot the planner
//! reads.

use std::collections::BTreeMap;

use super::log_write::{race, within_deadline};
use super::quant::{column, indexed_ids, lineages, node, open_at, quant_ids, DIMS};
use super::support::count;
use super::*;
use crate::types::{
    ChildKind, EventRef, FeedbackEvent, LineageKey, NewSummary, ReconcileReport, RetentionPolicy,
    SummaryChild, Tier, TierMove, UsageEntry,
};

const DAY: i64 = 86_400_000;
const WRITER: &str = "consolidation:test";

fn key(text: &str) -> LineageKey {
    LineageKey::from_raw(text)
}

pub(super) fn cold_at(lineage: &LineageKey, at: i64) -> TierMove {
    TierMove {
        lineage_key: lineage.clone(),
        to: Tier::Cold,
        run_id: "run-1".to_string(),
        reason: "unused".to_string(),
        changed_at: Millis(at),
        residency_until: None,
    }
}

pub(super) fn hot_at(lineage: &LineageKey, at: i64, residency_until: Option<i64>) -> TierMove {
    TierMove {
        to: Tier::Hot,
        reason: "used".to_string(),
        residency_until: residency_until.map(Millis),
        ..cold_at(lineage, at)
    }
}

/// A node with a subject, so the lineage of every one written with the same
/// producer and scope is the same.
pub(super) fn subject_node(label: &str, pattern: u8) -> NewNode {
    node(label, pattern)
        .with_subject("ada")
        .with_producer("agent-a")
}

pub(super) async fn lineage_of_node<B: Backend>(g: &Graph<B>, id: &NodeId) -> LineageKey {
    g.lineage_of(id).await.unwrap().expect("a lineage")
}

/// Marks a node as a cold one holds it: its int8 form is gone and its vector
/// with it.
pub(super) async fn drop_embeddings<B: Backend>(g: &Graph<B>, id: &NodeId) {
    g.backend
        .execute(
            "UPDATE node_quant SET i8 = NULL WHERE node_id = ?1",
            &[id.as_str().into()],
        )
        .await
        .unwrap();
    g.backend.vector_delete(id.as_str()).await.unwrap();
}

async fn vector_ids<B: Backend>(g: &Graph<B>) -> Vec<String> {
    column(g, "SELECT node_id FROM node_vectors ORDER BY node_id").await
}

// ---- lineage backfill and lookup ----

#[tokio::test]
async fn the_backfill_gives_an_upgraded_store_its_lineages_once_and_chains_share_a_key() {
    // Arrange: a store from before lineage, with a chain longer than one chunk
    let g = open_at(Millis(1000)).await;
    let page = g.upsert_by(NewNode::entity("person", "Ada")).await.unwrap();
    let fact = g.insert(subject_node("fact", 1)).await.unwrap();
    let mut chain = vec![g.insert(NewNode::now("fact", "v0", "x")).await.unwrap()];
    for version in 1..=250 {
        let next = g
            .supersede(
                chain.last().unwrap(),
                NewNode::now("fact", format!("v{version}"), "x"),
            )
            .await
            .unwrap();
        chain.push(next);
    }
    let before = lineages(&g).await;
    for sql in [
        "DELETE FROM node_lineage",
        "DELETE FROM lineage_backfill_state",
    ] {
        g.backend.execute(sql, &[]).await.unwrap();
    }

    // Act
    let first = g.backfill_lineage().await.unwrap();
    let again = g.backfill_lineage().await.unwrap();

    // Assert
    assert_eq!((first, again), (before.len(), 0));
    assert_eq!(lineages(&g).await, before);
    let chain_keys: Vec<_> = chain.iter().map(|id| before[id.as_str()].clone()).collect();
    assert!(chain_keys.iter().all(|k| *k == chain_keys[0]));
    assert_ne!(before[page.as_str()], before[fact.as_str()]);
}

#[tokio::test]
async fn the_backfill_gives_a_chain_whose_root_is_gone_a_key_from_what_is_left() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let root = g.insert(NewNode::now("fact", "root", "x")).await.unwrap();
    let child = g
        .supersede(&root, NewNode::now("fact", "child", "y"))
        .await
        .unwrap();
    let grandchild = g
        .supersede(&child, NewNode::now("fact", "grandchild", "z"))
        .await
        .unwrap();
    g.backend
        .execute("DELETE FROM nodes WHERE id = ?1", &[root.as_str().into()])
        .await
        .unwrap();
    for sql in [
        "DELETE FROM node_lineage",
        "DELETE FROM lineage_backfill_state",
    ] {
        g.backend.execute(sql, &[]).await.unwrap();
    }

    // Act
    g.backfill_lineage().await.unwrap();

    // Assert
    let keys = lineages(&g).await;
    assert_eq!(keys[child.as_str()], keys[grandchild.as_str()]);
    assert!(keys[child.as_str()].starts_with("n:"));
}

#[tokio::test]
async fn opening_a_store_backfills_lineage_before_it_takes_a_write() {
    // Arrange
    let env = super::support::Env::new();
    let path = env.db("upgraded.db");
    let open = || {
        DefaultGraph::open_with_clock(
            &path,
            GraphConfig::new(DIMS),
            Arc::new(FixedClock::new(Millis(1000))),
        )
    };
    let g = open().await.unwrap();
    let old = g.insert(NewNode::now("fact", "old", "x")).await.unwrap();
    for sql in [
        "DELETE FROM node_lineage",
        "DELETE FROM lineage_backfill_state",
    ] {
        g.backend.execute(sql, &[]).await.unwrap();
    }
    drop(g);

    // Act
    let g = open().await.unwrap();
    let new = g
        .supersede(&old, NewNode::now("fact", "new", "y"))
        .await
        .unwrap();

    // Assert
    let keys = lineages(&g).await;
    assert_eq!(keys[old.as_str()], keys[new.as_str()]);
}

#[tokio::test]
async fn a_node_with_no_stored_key_is_given_the_key_it_would_start_with() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let id = g.insert(subject_node("n", 1)).await.unwrap();
    let stored = lineage_of_node(&g, &id).await;
    g.backend
        .execute("DELETE FROM node_lineage", &[])
        .await
        .unwrap();

    // Act
    let computed = g.lineage_of(&id).await.unwrap();
    let missing = g.lineage_of(&NodeId::new()).await.unwrap();

    // Assert
    assert_eq!(computed, Some(stored));
    assert_eq!(missing, None);
}

// ---- tiers ----

#[tokio::test]
async fn a_lineage_with_no_row_is_hot_and_the_last_move_wins_with_the_index_following() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let id = g.insert(subject_node("n", 0b101)).await.unwrap();
    let lineage = lineage_of_node(&g, &id).await;
    let cold_flag = |g: &DefaultGraph| g.quant_index().snapshot()[0].1;

    // Act and assert
    assert_eq!(g.tier_of(&lineage).await.unwrap(), Tier::Hot);
    g.set_tier(&cold_at(&lineage, 5), Some("event-1"))
        .await
        .unwrap();
    assert_eq!(g.tier_of(&lineage).await.unwrap(), Tier::Cold);
    assert!(cold_flag(&g));
    g.set_tier(&hot_at(&lineage, 6, Some(9)), Some("event-2"))
        .await
        .unwrap();
    assert_eq!(g.tier_of(&lineage).await.unwrap(), Tier::Hot);
    assert!(!cold_flag(&g));
    let row = g
        .backend
        .query(
            "SELECT changed_at, residency_until, last_event_id FROM item_tier",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(row[0].get_i64(0).unwrap(), 6);
    assert_eq!(row[0].get_i64(1).unwrap(), 9);
    assert_eq!(row[0].get_string(2).unwrap(), "event-2");
}

#[tokio::test]
async fn a_reload_reads_the_cold_flag_from_the_tier_row() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let cold = g.insert(subject_node("cold", 1)).await.unwrap();
    let hot = g.insert(node("hot", 2)).await.unwrap();
    let lineage = lineage_of_node(&g, &cold).await;
    g.set_tier(&cold_at(&lineage, 5), None).await.unwrap();

    // Act
    g.reload_quant_index().await.unwrap();

    // Assert
    let flags: BTreeMap<_, _> = g
        .quant_index()
        .snapshot()
        .into_iter()
        .map(|(id, cold, _)| (id, cold))
        .collect();
    assert!(flags[cold.as_str()]);
    assert!(!flags[hot.as_str()]);
}

#[test]
fn a_tier_name_that_is_neither_hot_nor_cold_is_an_error_not_a_default() {
    // Arrange
    let names = ["hot", "cold", "warm", ""];

    // Act
    let parsed: Vec<_> = names.iter().map(|name| Tier::parse(name)).collect();

    // Assert
    assert!(matches!(parsed[0], Ok(Tier::Hot)));
    assert!(matches!(parsed[1], Ok(Tier::Cold)));
    assert!(matches!(&parsed[2], Err(Error::UnknownTier(text)) if text == "warm"));
    assert!(matches!(parsed[3], Err(Error::UnknownTier(_))));
}

#[tokio::test]
async fn a_write_for_a_cold_lineage_makes_it_hot_when_no_other_node_is_still_cold() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let old = g.insert(subject_node("old", 1)).await.unwrap();
    let lineage = lineage_of_node(&g, &old).await;
    drop_embeddings(&g, &old).await;
    g.set_tier(&cold_at(&lineage, 5), Some("event-1"))
        .await
        .unwrap();

    // Act
    let new = g.supersede(&old, subject_node("new", 2)).await.unwrap();

    // Assert: hot in the same write, with the new node's entry hot
    assert_eq!(g.tier_of(&lineage).await.unwrap(), Tier::Hot);
    assert_eq!(
        g.quant_index().snapshot(),
        [(new.as_str().to_string(), false, vec![2])]
    );
    let row = g
        .backend
        .query("SELECT changed_at, residency_until FROM item_tier", &[])
        .await
        .unwrap();
    assert_eq!(row[0].get_i64(0).unwrap(), 1000);
    assert!(matches!(row[0].0[1], Value::Null));
}

#[tokio::test]
async fn a_write_leaves_a_lineage_cold_while_a_sibling_is_still_cold() {
    // Arrange: two live nodes of one lineage, both without embeddings
    let g = open_at(Millis(1000)).await;
    let first = g.insert(subject_node("first", 1)).await.unwrap();
    let second = g.insert(subject_node("second", 2)).await.unwrap();
    let lineage = lineage_of_node(&g, &first).await;
    assert_eq!(lineage_of_node(&g, &second).await, lineage);
    for id in [&first, &second] {
        drop_embeddings(&g, id).await;
    }
    g.set_tier(&cold_at(&lineage, 5), None).await.unwrap();

    // Act
    let third = g.supersede(&first, subject_node("third", 4)).await.unwrap();

    // Assert: the sibling still holds nothing, so the lineage stays cold
    assert_eq!(g.tier_of(&lineage).await.unwrap(), Tier::Cold);
    let flags: BTreeMap<_, _> = g
        .quant_index()
        .snapshot()
        .into_iter()
        .map(|(id, cold, _)| (id, cold))
        .collect();
    assert!(flags[second.as_str()]);
    assert!(flags[third.as_str()]);
}

#[tokio::test]
async fn reconcile_deletes_the_vectors_of_a_cold_lineage_once_and_skips_what_it_must() {
    // Arrange: a cold lineage with a vector, a hot one without, a node written
    // after the lineage went cold, and a vector with no signature
    let g = open_at(Millis(1000)).await;
    let cold = g.insert(subject_node("cold", 1)).await.unwrap();
    let cold_lineage = lineage_of_node(&g, &cold).await;
    let hot = g
        .insert(NewNode::now("fact", "hot", "hot text").with_subject("other"))
        .await
        .unwrap();
    let late = g
        .insert(
            subject_node("late", 2)
                .with_subject("later")
                .with_producer("agent-b"),
        )
        .await
        .unwrap();
    let unsigned = g
        .insert(
            subject_node("unsigned", 3)
                .with_subject("bare")
                .with_producer("agent-c"),
        )
        .await
        .unwrap();
    g.backend
        .execute(
            "DELETE FROM node_quant WHERE node_id = ?1",
            &[unsigned.as_str().into()],
        )
        .await
        .unwrap();
    g.set_tier(&cold_at(&cold_lineage, 1000), None)
        .await
        .unwrap();
    g.set_tier(&cold_at(&lineage_of_node(&g, &unsigned).await, 1000), None)
        .await
        .unwrap();
    g.set_tier(&cold_at(&lineage_of_node(&g, &late).await, 999), None)
        .await
        .unwrap();

    // Act
    let first = g.reconcile_tier_vectors().await.unwrap();
    let second = g.reconcile_tier_vectors().await.unwrap();

    // Assert
    assert_eq!(
        first,
        ReconcileReport {
            deleted: 1,
            skipped_late: 1,
            skipped_unsigned: 1
        }
    );
    assert_eq!(
        second,
        ReconcileReport {
            deleted: 0,
            skipped_late: 1,
            skipped_unsigned: 1
        }
    );
    let mut kept = vec![late.as_str().to_string(), unsigned.as_str().to_string()];
    kept.sort();
    assert_eq!(vector_ids(&g).await, kept);
    assert!(!vector_ids(&g).await.contains(&hot.as_str().to_string()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tier_moves_and_reloads_racing_readers_leave_the_index_equal_to_the_tables() {
    const ROUNDS: usize = 40;
    // Arrange
    let g = Arc::new(open_at(Millis(1000)).await);
    let id = g.insert(subject_node("n", 0b11)).await.unwrap();
    let lineage = lineage_of_node(&g, &id).await;
    let query = quantize(&signed_zero()).unwrap().bin;

    // Act
    let seen = within_deadline(race(3, |worker| {
        let (g, lineage, query) = (Arc::clone(&g), lineage.clone(), query.clone());
        async move {
            let mut cold_seen = 0;
            for round in 0..ROUNDS {
                match worker {
                    0 => {
                        let to_cold = round % 2 == 0;
                        let change = if to_cold {
                            cold_at(&lineage, round as i64)
                        } else {
                            hot_at(&lineage, round as i64, None)
                        };
                        g.set_tier(&change, None).await.unwrap();
                    }
                    1 => g.reload_quant_index().await.unwrap(),
                    _ => {
                        let found = g.quant_index().hamming_top(&query, 4, false).unwrap();
                        assert!(found.len() <= 1, "{found:?}");
                        cold_seen += g.quant_index().cold_count();
                    }
                }
            }
            cold_seen
        }
    }))
    .await;

    // Assert: every observation was a whole state, and the end state is the table's
    assert!(seen.iter().all(|cold| *cold <= ROUNDS));
    let tier = g.tier_of(&lineage).await.unwrap();
    assert_eq!(
        g.quant_index().snapshot()[0].1,
        tier == Tier::Cold,
        "{tier:?}"
    );
    assert_eq!(indexed_ids(&g), [id.as_str().to_string()]);
}

fn signed_zero() -> Vec<f32> {
    vec![-1.0; DIMS]
}

// ---- usage ----

fn usage(lineage: &LineageKey, hits: u32, at: i64) -> UsageEntry {
    UsageEntry {
        lineage_key: lineage.clone(),
        hits,
        last_access_at: Millis(at),
    }
}

#[tokio::test]
async fn a_usage_batch_applied_twice_leaves_one_set_of_rows_and_an_empty_one_leaves_an_epoch() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let a = key("n:1:a");

    // Act
    g.apply_usage("event-1", 10, &[usage(&a, 3, 5)])
        .await
        .unwrap();
    g.apply_usage("event-1", 10, &[usage(&a, 3, 5)])
        .await
        .unwrap();
    g.apply_usage("event-2", 11, &[]).await.unwrap();

    // Assert
    assert_eq!(count(&g, "item_usage_events").await, 1);
    assert_eq!(count(&g, "usage_epochs").await, 2);
}

#[tokio::test]
async fn counting_has_started_once_any_batch_even_an_empty_one_is_recorded() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let before = g.usage_started().await.unwrap();

    // Act
    g.apply_usage("event-1", 10, &[]).await.unwrap();

    // Assert
    assert!(!before);
    assert!(g.usage_started().await.unwrap());
}

#[tokio::test]
async fn of_a_producers_verdicts_on_a_lineage_in_a_day_the_first_counts() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let a = key("n:1:a");
    let verdict = |event: &str, producer: &str, at: i64, correct: bool| FeedbackEvent {
        event_id: event.to_string(),
        lineage_key: a.clone(),
        producer: producer.to_string(),
        at: Millis(at),
        correct,
    };

    // Act
    let results = [
        g.apply_feedback(&verdict("e1", "p1", 5, true))
            .await
            .unwrap(),
        g.apply_feedback(&verdict("e1", "p1", 5, true))
            .await
            .unwrap(),
        g.apply_feedback(&verdict("e2", "p1", 6, false))
            .await
            .unwrap(),
        g.apply_feedback(&verdict("e3", "p2", 6, false))
            .await
            .unwrap(),
        g.apply_feedback(&verdict("e4", "p1", DAY + 1, false))
            .await
            .unwrap(),
    ];

    // Assert: a repeat of one event, and the second of the day, do not count
    assert_eq!(results, [true, false, false, true, true]);
    assert_eq!(count(&g, "item_feedback").await, 4);
    let counted = column(
        &g,
        "SELECT event_id FROM item_feedback WHERE counted = 1 ORDER BY event_id",
    )
    .await;
    assert_eq!(counted, ["e1", "e3", "e4"]);
}

#[tokio::test]
async fn the_snapshot_reads_each_lineage_with_its_usage_feedback_and_holdings() {
    // Arrange
    let clock = Arc::new(FixedClock::new(Millis(100 * DAY)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock.clone())
        .await
        .unwrap();
    let held = g.insert(subject_node("held", 1)).await.unwrap();
    let bare = g
        .insert(
            NewNode::now("fact", "bare", "no vector")
                .with_subject("bare")
                .with_producer("agent-b"),
        )
        .await
        .unwrap();
    let held_key = lineage_of_node(&g, &held).await;
    let bare_key = lineage_of_node(&g, &bare).await;
    let window = 30;
    let today = 100;
    g.apply_usage("old", today - window - 1, &[usage(&held_key, 7, 70 * DAY)])
        .await
        .unwrap();
    g.apply_usage("edge", today - window, &[usage(&held_key, 2, 71 * DAY)])
        .await
        .unwrap();
    g.apply_usage("recent", today - 1, &[usage(&held_key, 4, 99 * DAY)])
        .await
        .unwrap();
    for (event, producer, at, correct) in [
        ("f1", "p1", 98 * DAY, true),
        ("f2", "p1", 98 * DAY + 5, false),
        ("f3", "p2", 99 * DAY, false),
    ] {
        g.apply_feedback(&FeedbackEvent {
            event_id: event.to_string(),
            lineage_key: held_key.clone(),
            producer: producer.to_string(),
            at: Millis(at),
            correct,
        })
        .await
        .unwrap();
    }
    g.set_tier(&cold_at(&bare_key, 50 * DAY), None)
        .await
        .unwrap();

    // Act
    let snapshot = g.usage_snapshot(Millis(100 * DAY), window).await.unwrap();

    // Assert
    assert_eq!(snapshot.now, Millis(100 * DAY));
    assert_eq!(
        snapshot.observation_since,
        Some(Millis::start_of_day(today - window - 1))
    );
    let by_key: BTreeMap<_, _> = snapshot
        .items
        .iter()
        .map(|item| (item.lineage_key.clone(), item))
        .collect();
    let held_view = by_key[&held_key];
    assert_eq!(held_view.hits_by_day, [(today - window, 2), (today - 1, 4)]);
    assert_eq!(held_view.last_access_at, Some(Millis(99 * DAY)));
    let verdicts: Vec<_> = held_view
        .feedback
        .iter()
        .map(|f| (f.producer.as_str(), f.correct))
        .collect();
    assert_eq!(verdicts, [("p2", false), ("p1", true)]);
    assert_eq!(held_view.tier, Tier::Hot);
    assert!(held_view.has_vector && held_view.has_signature);
    assert_eq!(held_view.producer, "agent-a");
    assert_eq!(held_view.newest_tx_from, Millis(100 * DAY));
    let bare_view = by_key[&bare_key];
    assert_eq!(bare_view.tier, Tier::Cold);
    assert_eq!(bare_view.changed_at, Some(Millis(50 * DAY)));
    assert!(bare_view.hits_by_day.is_empty() && bare_view.last_access_at.is_none());
    assert!(!bare_view.has_vector && bare_view.has_signature);
    assert_eq!(snapshot.items.len(), 2);
}

#[tokio::test]
async fn a_node_with_a_vector_and_no_signature_is_reported_unsigned() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let id = g.insert(subject_node("n", 1)).await.unwrap();
    g.backend
        .execute("DELETE FROM node_quant", &[])
        .await
        .unwrap();

    // Act
    let snapshot = g.usage_snapshot(Millis(2000), 30).await.unwrap();

    // Assert
    let view = &snapshot.items[0];
    assert!(view.has_vector);
    assert!(!view.has_signature);
    let nodes = g.lineage_nodes(&view.lineage_key).await.unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].node_id, id);
    assert!(nodes[0].has_vector && !nodes[0].has_signature);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_snapshot_racing_usage_batches_never_holds_part_of_one() {
    const BATCHES: usize = 30;
    const ENTRIES: usize = 4;
    // Arrange: one lineage per entry, so a batch touches ENTRIES rows
    let g = Arc::new(open_at(Millis(1000)).await);
    let mut keys = Vec::new();
    for entry in 0..ENTRIES {
        let id = g
            .insert(
                NewNode::now("fact", format!("n{entry}"), "x").with_subject(format!("s{entry}")),
            )
            .await
            .unwrap();
        keys.push(lineage_of_node(&g, &id).await);
    }

    // Act
    let outcomes = within_deadline(race(2, |worker| {
        let (g, keys) = (Arc::clone(&g), keys.clone());
        async move {
            let mut totals = Vec::new();
            for batch in 0..BATCHES {
                if worker == 0 {
                    let entries: Vec<_> = keys.iter().map(|k| usage(k, 1, 5)).collect();
                    g.apply_usage(&format!("event-{batch}"), 7, &entries)
                        .await
                        .unwrap();
                } else {
                    let snapshot = g.usage_snapshot(Millis(8 * DAY), 30).await.unwrap();
                    let hits: Vec<i64> = snapshot
                        .items
                        .iter()
                        .map(|item| item.hits_by_day.iter().map(|(_, h)| h).sum())
                        .collect();
                    totals.push(hits);
                }
            }
            totals
        }
    }))
    .await;

    // Assert: every lineage shows the same number of batches in any one snapshot
    for hits in outcomes.iter().flatten() {
        assert!(hits.iter().all(|h| *h == hits[0]), "torn snapshot {hits:?}");
    }
}

#[tokio::test]
async fn pruning_usage_keeps_the_window_and_the_earliest_epoch() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let a = key("n:1:a");
    for (event, day) in [("e1", 1), ("e2", 5), ("e3", 9)] {
        g.apply_usage(event, day, &[usage(&a, 1, day * DAY)])
            .await
            .unwrap();
    }
    let since = g
        .usage_snapshot(Millis(10 * DAY), 30)
        .await
        .unwrap()
        .observation_since;

    // Act
    let removed = g.prune_usage(6).await.unwrap();

    // Assert: two usage rows and one epoch went, the earliest epoch stayed
    assert_eq!(removed, 3);
    assert_eq!(
        column(&g, "SELECT event_id FROM usage_epochs ORDER BY event_id").await,
        ["e1", "e3"]
    );
    assert_eq!(count(&g, "item_usage_events").await, 1);
    let after = g
        .usage_snapshot(Millis(10 * DAY), 30)
        .await
        .unwrap()
        .observation_since;
    assert_eq!(after, since);
    assert_eq!(since, Some(Millis::start_of_day(1)));
}

#[tokio::test]
async fn pruning_tier_state_drops_a_lineage_no_node_has_and_leaves_the_others() {
    // Arrange
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock.clone())
        .await
        .unwrap();
    let doomed = g
        .insert(NewNode::now("scratch", "d", "x").with_subject("d"))
        .await
        .unwrap();
    let kept = g.insert(subject_node("k", 1)).await.unwrap();
    let (doomed_key, kept_key) = (
        lineage_of_node(&g, &doomed).await,
        lineage_of_node(&g, &kept).await,
    );
    for lineage in [&doomed_key, &kept_key] {
        g.set_tier(&cold_at(lineage, 1), None).await.unwrap();
        g.apply_usage(
            &format!("u-{}", lineage.as_str()),
            1,
            &[usage(lineage, 1, 1)],
        )
        .await
        .unwrap();
        g.apply_feedback(&FeedbackEvent {
            event_id: format!("f-{}", lineage.as_str()),
            lineage_key: lineage.clone(),
            producer: "p".to_string(),
            at: Millis(1),
            correct: true,
        })
        .await
        .unwrap();
    }
    clock.set(Millis(1000 + 10_000));

    // Act: the sweep deletes the node and prunes what was keyed by it
    g.gc(&RetentionPolicy::keep("scratch", Millis(1000)))
        .await
        .unwrap();

    // Assert
    for table in ["item_tier", "item_usage_events", "item_feedback"] {
        assert_eq!(
            column(&g, &format!("SELECT lineage_key FROM {table}")).await,
            [kept_key.as_str()],
            "{table}"
        );
    }
}

#[tokio::test]
async fn entity_pages_and_summary_children_are_exempt_from_demotion() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let page = g.upsert_by(NewNode::entity("person", "Ada")).await.unwrap();
    let child = g.insert(subject_node("child", 1)).await.unwrap();
    let plain = g
        .insert(subject_node("plain", 2).with_subject("plain"))
        .await
        .unwrap();
    g.put_summary(
        NewSummary {
            level: 1,
            scope: String::new(),
            text: "a summary".to_string(),
            events_fed: 1,
            valid_min: Millis(1),
            valid_max: Millis(2),
            model_id: "m".to_string(),
            prompt_version: "p".to_string(),
        },
        &[SummaryChild {
            id: child.as_str().to_string(),
            kind: ChildKind::Node,
        }],
        &[EventRef {
            event_id: "e1".to_string(),
            synthetic: false,
        }],
        &[0.5; DIMS],
        &[],
        WRITER,
    )
    .await
    .unwrap();

    // Act
    let exempt = g.tier_exempt_lineages().await.unwrap();

    // Assert
    let expected: std::collections::HashSet<_> = [
        lineage_of_node(&g, &page).await,
        lineage_of_node(&g, &child).await,
    ]
    .into();
    assert_eq!(exempt, expected);
    assert!(!exempt.contains(&lineage_of_node(&g, &plain).await));
}

#[tokio::test]
async fn a_closed_summary_no_longer_exempts_its_children() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let child = g.insert(subject_node("child", 1)).await.unwrap();
    g.backend
        .execute(
            "INSERT INTO summaries (id, level, child_hash, text, events_fed, valid_min, valid_max,
                model_id, prompt_version, tx_from, tx_to)
             VALUES ('s1', 1, 'h', 't', 1, 1, 1, 'm', 'p', 1, 2)",
            &[],
        )
        .await
        .unwrap();
    g.backend
        .execute(
            "INSERT INTO summary_children (parent_id, child_id, child_kind) VALUES ('s1', ?1, 'node')",
            &[child.as_str().into()],
        )
        .await
        .unwrap();

    // Act
    let exempt = g.tier_exempt_lineages().await.unwrap();

    // Assert
    assert!(exempt.is_empty(), "{exempt:?}");
    assert_eq!(quant_ids(&g).await.len(), 1);
}
