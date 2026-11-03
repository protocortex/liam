// SPDX-License-Identifier: Apache-2.0
//! Demoting and promoting a lineage: the order of the transaction and the
//! vector calls, and what each fault in between leaves for reconcile.

use liam_log::event::LogPayload;

use super::quant::{column, signed};
use super::support::{count, Env};
use super::tiers::{cold_at, hot_at, lineage_of_node, subject_node};
use super::*;
use crate::types::{ApplyOutcome, LineageKey, Tier, TierMove};

const SOURCE: &str = "consolidation:run-1";
const WENT_COLD_AT: i64 = 1000;
const RESIDENT_UNTIL: i64 = 9000;

struct Rig {
    env: Env,
    graph: Graph<FailingBackend>,
}

async fn rig() -> Rig {
    let env = Env::new();
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let (graph, _log) = env.open::<FailingBackend>("apply.db", clock).await;
    Rig { env, graph }
}

/// Two live nodes of one lineage, each with a vector, and an edge between them.
async fn lineage_of_two(rig: &Rig) -> (LineageKey, NodeId, NodeId) {
    let g = &rig.graph;
    let a = g.insert(subject_node("a", 0b01)).await.unwrap();
    let b = g.insert(subject_node("b", 0b10)).await.unwrap();
    g.link(NewEdge::new(&a, &b, "relates_to")).await.unwrap();
    let key = lineage_of_node(g, &a).await;
    assert_eq!(lineage_of_node(g, &b).await, key);
    (key, a, b)
}

fn demotion(key: &LineageKey) -> TierMove {
    cold_at(key, WENT_COLD_AT)
}

fn promotion(key: &LineageKey) -> TierMove {
    hot_at(key, WENT_COLD_AT + 1000, Some(RESIDENT_UNTIL))
}

async fn vector_ids<B: Backend>(g: &Graph<B>) -> Vec<String> {
    column(g, "SELECT node_id FROM node_vectors ORDER BY node_id").await
}

fn sorted(ids: &[&NodeId]) -> Vec<String> {
    let mut ids: Vec<String> = ids.iter().map(|id| id.as_str().to_string()).collect();
    ids.sort();
    ids
}

async fn logged(rig: &Rig) -> Vec<LogPayload> {
    rig.env
        .records()
        .await
        .into_iter()
        .map(|record| record.event.payload)
        .collect()
}

async fn tier_moves(rig: &Rig) -> usize {
    logged(rig)
        .await
        .iter()
        .filter(|payload| matches!(payload, LogPayload::TierMove { .. }))
        .count()
}

async fn voids(rig: &Rig) -> usize {
    logged(rig)
        .await
        .iter()
        .filter(|payload| matches!(payload, LogPayload::Voided { .. }))
        .count()
}

#[tokio::test]
async fn a_demotion_makes_the_lineage_cold_and_keeps_everything_but_the_vectors() {
    // Arrange
    let rig = rig().await;
    let g = &rig.graph;
    let (key, a, b) = lineage_of_two(&rig).await;
    let events_before = count(g, "node_events").await;

    // Act
    let outcome = g.demote_lineage(&demotion(&key), SOURCE).await.unwrap();

    // Assert
    assert_eq!(outcome, ApplyOutcome::Applied);
    assert_eq!(g.tier_of(&key).await.unwrap(), Tier::Cold);
    assert_eq!(
        column(
            g,
            "SELECT node_id FROM node_quant WHERE i8 IS NULL ORDER BY node_id"
        )
        .await,
        sorted(&[&a, &b])
    );
    assert_eq!(
        column(g, "SELECT node_id FROM node_quant WHERE bin IS NOT NULL")
            .await
            .len(),
        2
    );
    assert!(vector_ids(g).await.is_empty());
    assert_eq!(count(g, "nodes").await, 2);
    assert_eq!(count(g, "edges").await, 1);
    assert_eq!(count(g, "node_events").await, events_before);
    assert_eq!(g.quant_index().cold_count(), 2);
    assert_eq!(tier_moves(&rig).await, 1);
    let found = g.query(&Query::text("content")).await.unwrap();
    assert_eq!(found.len(), 2, "still found by its text");
}

#[tokio::test]
async fn a_demotion_whose_commit_fails_changes_nothing_and_voids_its_record() {
    // Arrange
    let rig = rig().await;
    let g = &rig.graph;
    let (key, a, b) = lineage_of_two(&rig).await;
    g.backend.set_fail_commit(true);

    // Act
    let result = g.demote_lineage(&demotion(&key), SOURCE).await;
    g.backend.set_fail_commit(false);

    // Assert
    assert!(result.is_err());
    assert_eq!(g.tier_of(&key).await.unwrap(), Tier::Hot);
    assert_eq!(vector_ids(g).await, sorted(&[&a, &b]));
    assert!(column(g, "SELECT node_id FROM node_quant WHERE i8 IS NULL")
        .await
        .is_empty());
    assert_eq!(voids(&rig).await, 1);
}

#[tokio::test]
async fn a_vector_delete_that_fails_after_the_commit_leaves_a_partial_demotion_reconcile_finishes()
{
    // Arrange
    let rig = rig().await;
    let g = &rig.graph;
    let (key, _, _) = lineage_of_two(&rig).await;
    g.backend.set_fail_on_vector_delete(2);

    // Act
    let outcome = g.demote_lineage(&demotion(&key), SOURCE).await.unwrap();
    let first = g.reconcile_tier_vectors().await.unwrap();
    let second = g.reconcile_tier_vectors().await.unwrap();

    // Assert
    assert_eq!(outcome, ApplyOutcome::Partial { remaining: 1 });
    assert_eq!(first.deleted, 1);
    assert_eq!(second.deleted, 0);
    assert!(vector_ids(g).await.is_empty());
    assert_eq!(g.tier_of(&key).await.unwrap(), Tier::Cold);
}

#[tokio::test]
async fn a_node_written_after_the_lineage_went_cold_keeps_its_vector() {
    // Arrange
    let rig = rig().await;
    let g = &rig.graph;
    let (key, _, _) = lineage_of_two(&rig).await;
    let earlier = cold_at(&key, WENT_COLD_AT - 1);

    // Act: both nodes were written at 1000, after the lineage went cold at 999
    let outcome = g.demote_lineage(&earlier, SOURCE).await.unwrap();

    // Assert
    assert_eq!(outcome, ApplyOutcome::Applied);
    assert_eq!(vector_ids(g).await.len(), 2);
}

#[tokio::test]
async fn a_promotion_brings_back_the_vectors_the_int8_forms_and_a_hot_row_with_residency() {
    // Arrange
    let rig = rig().await;
    let g = &rig.graph;
    let (key, a, b) = lineage_of_two(&rig).await;
    g.demote_lineage(&demotion(&key), SOURCE).await.unwrap();
    let signatures = column(g, "SELECT hex(bin) FROM node_quant ORDER BY node_id").await;
    let vectors = [(a.clone(), signed(0b01)), (b.clone(), signed(0b10))];

    // Act
    let outcome = g
        .promote_lineage(&promotion(&key), &vectors, SOURCE)
        .await
        .unwrap();

    // Assert
    assert_eq!(outcome, ApplyOutcome::Applied);
    assert_eq!(g.tier_of(&key).await.unwrap(), Tier::Hot);
    assert_eq!(vector_ids(g).await, sorted(&[&a, &b]));
    assert!(column(g, "SELECT node_id FROM node_quant WHERE i8 IS NULL")
        .await
        .is_empty());
    assert_eq!(
        column(g, "SELECT hex(bin) FROM node_quant ORDER BY node_id").await,
        signatures
    );
    assert_eq!(
        column(g, "SELECT CAST(residency_until AS TEXT) FROM item_tier").await,
        [RESIDENT_UNTIL.to_string()]
    );
    assert_eq!(g.quant_index().cold_count(), 0);
    assert_eq!(tier_moves(&rig).await, 2);
}

#[tokio::test]
async fn a_promotion_whose_commit_fails_leaves_a_cold_row_with_vectors_reconcile_removes() {
    // Arrange
    let rig = rig().await;
    let g = &rig.graph;
    let (key, a, b) = lineage_of_two(&rig).await;
    g.demote_lineage(&demotion(&key), SOURCE).await.unwrap();
    let vectors = [(a, signed(0b01)), (b, signed(0b10))];
    g.backend.set_fail_commit(true);

    // Act
    let result = g.promote_lineage(&promotion(&key), &vectors, SOURCE).await;
    g.backend.set_fail_commit(false);
    let stored = vector_ids(g).await.len();
    let reconciled = g.reconcile_tier_vectors().await.unwrap();

    // Assert
    assert!(result.is_err());
    assert_eq!(g.tier_of(&key).await.unwrap(), Tier::Cold);
    assert_eq!(stored, 2);
    assert_eq!(reconciled.deleted, 2);
    assert!(vector_ids(g).await.is_empty());
}

#[tokio::test]
async fn a_promotion_retried_after_a_crash_takes_the_forms_from_the_vectors_already_stored() {
    // Arrange: the vectors were written and the commit was lost
    let rig = rig().await;
    let g = &rig.graph;
    let (key, a, b) = lineage_of_two(&rig).await;
    g.demote_lineage(&demotion(&key), SOURCE).await.unwrap();
    g.backend.set_fail_commit(true);
    let vectors = [(a.clone(), signed(0b01)), (b.clone(), signed(0b10))];
    g.promote_lineage(&promotion(&key), &vectors, SOURCE)
        .await
        .unwrap_err();
    g.backend.set_fail_commit(false);

    // Act: no vectors passed, the stored ones are enough
    let outcome = g
        .promote_lineage(&promotion(&key), &[], SOURCE)
        .await
        .unwrap();

    // Assert
    assert_eq!(outcome, ApplyOutcome::Applied);
    assert_eq!(vector_ids(g).await, sorted(&[&a, &b]));
    assert!(column(g, "SELECT node_id FROM node_quant WHERE i8 IS NULL")
        .await
        .is_empty());
}

#[tokio::test]
async fn a_vector_with_no_signature_is_never_dropped() {
    // Arrange
    let rig = rig().await;
    let g = &rig.graph;
    let (key, a, _) = lineage_of_two(&rig).await;
    g.backend
        .execute(
            "DELETE FROM node_quant WHERE node_id = ?1",
            &[a.as_str().into()],
        )
        .await
        .unwrap();

    // Act
    let outcome = g.demote_lineage(&demotion(&key), SOURCE).await.unwrap();

    // Assert
    assert_eq!(outcome, ApplyOutcome::NoSignature);
    assert_eq!(g.tier_of(&key).await.unwrap(), Tier::Hot);
    assert_eq!(vector_ids(g).await.len(), 2);
    assert_eq!(tier_moves(&rig).await, 0);
}

#[tokio::test]
async fn a_move_the_lineage_no_longer_needs_is_stale_and_writes_nothing() {
    // Arrange
    let rig = rig().await;
    let g = &rig.graph;
    let (key, _, _) = lineage_of_two(&rig).await;
    let unknown = LineageKey::from_raw("s:!:nobody:!");

    // Act
    let promote_hot = g
        .promote_lineage(&promotion(&key), &[], SOURCE)
        .await
        .unwrap();
    let demote_unknown = g.demote_lineage(&demotion(&unknown), SOURCE).await.unwrap();
    g.demote_lineage(&demotion(&key), SOURCE).await.unwrap();
    let demote_again = g.demote_lineage(&demotion(&key), SOURCE).await.unwrap();

    // Assert
    assert_eq!(promote_hot, ApplyOutcome::Stale);
    assert_eq!(demote_unknown, ApplyOutcome::Stale);
    assert_eq!(demote_again, ApplyOutcome::Stale);
    assert_eq!(tier_moves(&rig).await, 1);
}

#[tokio::test]
async fn a_move_in_the_wrong_direction_is_refused() {
    // Arrange
    let rig = rig().await;
    let g = &rig.graph;
    let (key, _, _) = lineage_of_two(&rig).await;

    // Act
    let demote = g.demote_lineage(&promotion(&key), SOURCE).await;
    let promote = g.promote_lineage(&demotion(&key), &[], SOURCE).await;

    // Assert
    assert!(matches!(
        demote,
        Err(Error::UnexpectedTierMove { expected: "cold" })
    ));
    assert!(matches!(
        promote,
        Err(Error::UnexpectedTierMove { expected: "hot" })
    ));
}

#[tokio::test]
async fn a_hot_node_that_lost_its_int8_form_gets_it_back_when_its_vector_is_stored_again() {
    // Arrange: as after a promotion replayed without the vectors it came with
    let rig = rig().await;
    let g = &rig.graph;
    let (_, a, _) = lineage_of_two(&rig).await;
    g.backend
        .execute(
            "UPDATE node_quant SET i8 = NULL WHERE node_id = ?1",
            &[a.as_str().into()],
        )
        .await
        .unwrap();

    // Act
    g.ensure_quant(&a, &signed(0b01)).await.unwrap();

    // Assert
    assert!(column(g, "SELECT node_id FROM node_quant WHERE i8 IS NULL")
        .await
        .is_empty());
}

#[tokio::test]
async fn a_cold_node_stays_without_its_int8_form_when_a_vector_is_stored_again() {
    // Arrange
    let rig = rig().await;
    let g = &rig.graph;
    let (key, a, b) = lineage_of_two(&rig).await;
    g.demote_lineage(&demotion(&key), SOURCE).await.unwrap();

    // Act
    g.ensure_quant(&a, &signed(0b01)).await.unwrap();

    // Assert
    assert_eq!(
        column(
            g,
            "SELECT node_id FROM node_quant WHERE i8 IS NULL ORDER BY node_id"
        )
        .await,
        sorted(&[&a, &b])
    );
}
