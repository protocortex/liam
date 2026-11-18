// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use liam_model::{Embedder, TextKind};
use liam_store::{DefaultGraph, FixedClock, LineageKey, Millis, NewNode, NodeId, OriginVerdict};
use tokio::sync::Semaphore;

use super::*;
use crate::tiering::planner::MoveReason;
use crate::tiering::test_support::{
    logged_store, signed, CountingEmbedder, FailOnNthEmbedder, FixedRam, ScriptedEmbedder,
    StubVerifier,
};

const DIMS: usize = 8;
const START: Millis = Millis(10 * 86_400_000);
const RUN: &str = "run-1";
const SOURCE: &str = "consolidation:run-1";
const RESIDENCY_DAYS: u32 = 7;

struct Fixture {
    store: DefaultGraph,
    verifier: StubVerifier,
    permits: Semaphore,
    ram: FixedRam,
    _dir: tempfile::TempDir,
}

async fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let store = logged_store(dir.path(), Arc::new(FixedClock::new(START)), DIMS).await;
    Fixture {
        store,
        verifier: StubVerifier::default(),
        permits: Semaphore::new(1),
        ram: FixedRam::plenty(),
        _dir: dir,
    }
}

fn text_of(label: &str) -> String {
    format!("text of {label}")
}

/// A node with a subject, so each label is a lineage of its own.
async fn add(store: &DefaultGraph, label: &str, pattern: u8) -> (NodeId, LineageKey) {
    let id = store
        .insert(
            NewNode::now("fact", label, text_of(label))
                .with_subject(label)
                .with_producer("agent-a")
                .with_embedding(signed(pattern, DIMS)),
        )
        .await
        .unwrap();
    let lineage = store.lineage_of(&id).await.unwrap().unwrap();
    (id, lineage)
}

/// Demotes `lineage` straight through the store, as an earlier run did.
async fn demoted(store: &DefaultGraph, lineage: &LineageKey) {
    let change = TierMove {
        lineage_key: lineage.clone(),
        to: Tier::Cold,
        run_id: "earlier".to_string(),
        reason: "test".to_string(),
        changed_at: START,
        residency_until: None,
    };
    let outcome = store.demote_lineage(&change, "consolidation:earlier").await;
    assert_eq!(outcome.unwrap(), ApplyOutcome::Applied);
}

fn planned(lineage: &LineageKey, to: Tier) -> PlannedMove {
    PlannedMove {
        lineage_key: lineage.clone(),
        to,
        utility: 0.1,
        reason: match to {
            Tier::Cold => MoveReason::LowUtility,
            Tier::Hot => MoveReason::HighUtility,
        },
    }
}

fn plan(promotions: Vec<PlannedMove>, demotions: Vec<PlannedMove>) -> TierPlan {
    TierPlan {
        promotions,
        demotions,
        skips: Default::default(),
        plan_hash: "hash".to_string(),
    }
}

fn deps<'a>(fixture: &'a Fixture, embedder: &'a dyn Embedder, deadline: Instant) -> ApplyDeps<'a> {
    ApplyDeps {
        store: &fixture.store,
        verifier: &fixture.verifier,
        embedder,
        permits: &fixture.permits,
        free_ram: &fixture.ram,
        source: SOURCE,
        run_id: RUN,
        now: START,
        min_residency_days: RESIDENCY_DAYS,
        deadline,
    }
}

fn far() -> Instant {
    Instant::now() + Duration::from_secs(60)
}

async fn holds_vector(store: &DefaultGraph, lineage: &LineageKey) -> bool {
    let texts = store.lineage_texts(lineage).await.unwrap();
    texts.iter().all(|text| text.has_vector)
}

#[tokio::test]
async fn demotions_then_promotions_are_applied_logged_and_reported() {
    // Arrange
    let fixture = fixture().await;
    let (_, going_cold) = add(&fixture.store, "a", 0b01).await;
    let (_, coming_back) = add(&fixture.store, "b", 0b10).await;
    demoted(&fixture.store, &coming_back).await;
    let embedder = CountingEmbedder::new(ScriptedEmbedder::new(
        DIMS,
        &[(&text_of("b"), signed(0b10, DIMS))],
    ));
    let moves = plan(
        vec![planned(&coming_back, Tier::Hot)],
        vec![planned(&going_cold, Tier::Cold)],
    );

    // Act
    let report = apply_moves(&moves, &deps(&fixture, &embedder, far())).await;

    // Assert
    assert_eq!(
        report,
        TierReport {
            planned: 2,
            demoted: 1,
            promoted: 1,
            reembedded: 1,
            plan_hash: "hash".to_string(),
            moved_lineages: vec![going_cold.clone(), coming_back.clone()],
            ..TierReport::default()
        }
    );
    assert!(!report.partial());
    assert_eq!(
        fixture.store.tier_of(&going_cold).await.unwrap(),
        Tier::Cold
    );
    assert_eq!(
        fixture.store.tier_of(&coming_back).await.unwrap(),
        Tier::Hot
    );
    assert!(holds_vector(&fixture.store, &coming_back).await);
    assert!(!holds_vector(&fixture.store, &going_cold).await);
    assert_eq!(embedder.kinds(), [TextKind::Document]);
}

#[tokio::test]
async fn a_promotion_is_resident_for_the_configured_days() {
    // Arrange
    let fixture = fixture().await;
    let (_, lineage) = add(&fixture.store, "a", 0b01).await;
    demoted(&fixture.store, &lineage).await;
    let embedder = ScriptedEmbedder::new(DIMS, &[(&text_of("a"), signed(0b01, DIMS))]);

    // Act
    apply_moves(
        &plan(vec![planned(&lineage, Tier::Hot)], vec![]),
        &deps(&fixture, &embedder, far()),
    )
    .await;

    // Assert
    let snapshot = fixture.store.usage_snapshot(START, 30).await.unwrap();
    let item = &snapshot.items[0];
    assert_eq!(
        item.residency_until,
        Some(Millis(START.0 + Millis::days(i64::from(RESIDENCY_DAYS)).0))
    );
}

#[tokio::test]
async fn one_verification_covers_every_demotion() {
    // Arrange
    let fixture = fixture().await;
    let mut demotions = Vec::new();
    for (label, pattern) in [("a", 1), ("b", 2), ("c", 4)] {
        let (_, lineage) = add(&fixture.store, label, pattern).await;
        demotions.push(planned(&lineage, Tier::Cold));
    }
    let embedder = ScriptedEmbedder::new(DIMS, &[]);

    // Act
    let report = apply_moves(&plan(vec![], demotions), &deps(&fixture, &embedder, far())).await;

    // Assert
    assert_eq!(report.demoted, 3);
    assert_eq!(fixture.verifier.calls(), 1);
    assert_eq!(fixture.verifier.claims_seen(), 3);
}

#[tokio::test]
async fn a_demotion_the_log_does_not_vouch_for_is_skipped_and_the_others_go_on() {
    // Arrange
    let fixture = fixture().await;
    let (unproven, unproven_lineage) = add(&fixture.store, "a", 0b01).await;
    let (_, proven_lineage) = add(&fixture.store, "b", 0b10).await;
    fixture
        .verifier
        .judging(&unproven, OriginVerdict::OriginMissing);
    let embedder = ScriptedEmbedder::new(DIMS, &[]);
    let moves = plan(
        vec![],
        vec![
            planned(&unproven_lineage, Tier::Cold),
            planned(&proven_lineage, Tier::Cold),
        ],
    );

    // Act
    let report = apply_moves(&moves, &deps(&fixture, &embedder, far())).await;

    // Assert
    assert_eq!(report.skipped_unverified, 1);
    assert_eq!(report.demoted, 1);
    assert_eq!(
        fixture.store.tier_of(&unproven_lineage).await.unwrap(),
        Tier::Hot
    );
    assert!(holds_vector(&fixture.store, &unproven_lineage).await);
}

#[tokio::test]
async fn the_real_log_vouches_for_a_node_it_wrote_and_demotes_it() {
    // Arrange: the store itself is the verifier, reading its own log
    let fixture = fixture().await;
    let (_, lineage) = add(&fixture.store, "a", 0b01).await;
    let embedder = ScriptedEmbedder::new(DIMS, &[]);
    let mut apply = deps(&fixture, &embedder, far());
    apply.verifier = &fixture.store;

    // Act
    let report = apply_moves(&plan(vec![], vec![planned(&lineage, Tier::Cold)]), &apply).await;

    // Assert
    assert_eq!(report.demoted, 1);
    assert_eq!(report.skipped_unverified, 0);
}

#[tokio::test]
async fn a_promotion_whose_embedding_fails_changes_nothing_and_the_others_apply() {
    // Arrange
    let fixture = fixture().await;
    let (_, first) = add(&fixture.store, "a", 0b01).await;
    let (_, second) = add(&fixture.store, "b", 0b10).await;
    demoted(&fixture.store, &first).await;
    demoted(&fixture.store, &second).await;
    let embedder = FailOnNthEmbedder::new(
        ScriptedEmbedder::new(
            DIMS,
            &[
                (&text_of("a"), signed(0b01, DIMS)),
                (&text_of("b"), signed(0b10, DIMS)),
            ],
        ),
        1,
    );
    let moves = plan(
        vec![planned(&first, Tier::Hot), planned(&second, Tier::Hot)],
        vec![],
    );

    // Act
    let report = apply_moves(&moves, &deps(&fixture, &embedder, far())).await;

    // Assert
    assert_eq!(report.failed, 1);
    assert_eq!(report.promoted, 1);
    assert!(report.partial());
    assert_eq!(fixture.store.tier_of(&first).await.unwrap(), Tier::Cold);
    assert!(!holds_vector(&fixture.store, &first).await);
    assert_eq!(fixture.store.tier_of(&second).await.unwrap(), Tier::Hot);
}

#[tokio::test]
async fn a_deadline_that_passes_during_a_promotion_leaves_it_and_the_rest_for_the_next_run() {
    // Arrange: each embedding takes longer than the run has left
    let fixture = fixture().await;
    let (_, first) = add(&fixture.store, "a", 0b01).await;
    let (_, second) = add(&fixture.store, "b", 0b10).await;
    demoted(&fixture.store, &first).await;
    demoted(&fixture.store, &second).await;
    let embedder = FailOnNthEmbedder::new(
        ScriptedEmbedder::new(
            DIMS,
            &[
                (&text_of("a"), signed(0b01, DIMS)),
                (&text_of("b"), signed(0b10, DIMS)),
            ],
        ),
        usize::MAX,
    )
    .slow(Duration::from_millis(400));
    let moves = plan(
        vec![planned(&first, Tier::Hot), planned(&second, Tier::Hot)],
        vec![],
    );
    let deadline = Instant::now() + Duration::from_millis(100);

    // Act
    let report = apply_moves(&moves, &deps(&fixture, &embedder, deadline)).await;

    // Assert
    assert!(report.deadline_hit);
    assert_eq!(report.promoted, 0);
    assert!(report.partial());
    assert_eq!(fixture.store.tier_of(&first).await.unwrap(), Tier::Cold);
    assert_eq!(fixture.store.tier_of(&second).await.unwrap(), Tier::Cold);
}

#[tokio::test]
async fn a_run_already_past_its_deadline_moves_nothing() {
    // Arrange
    let fixture = fixture().await;
    let (_, going_cold) = add(&fixture.store, "a", 0b01).await;
    let embedder = ScriptedEmbedder::new(DIMS, &[]);
    let deadline = Instant::now();

    // Act
    let report = apply_moves(
        &plan(vec![], vec![planned(&going_cold, Tier::Cold)]),
        &deps(&fixture, &embedder, deadline),
    )
    .await;

    // Assert
    assert!(report.deadline_hit);
    assert_eq!(report.demoted, 0);
    assert_eq!(fixture.store.tier_of(&going_cold).await.unwrap(), Tier::Hot);
}

#[tokio::test]
async fn low_memory_stops_the_promotions_before_the_first_embedding() {
    // Arrange
    let fixture = fixture().await;
    let (_, lineage) = add(&fixture.store, "a", 0b01).await;
    demoted(&fixture.store, &lineage).await;
    fixture
        .ram
        .0
        .store(EMBED_FREE_RAM_FLOOR_MB - 1, Ordering::SeqCst);
    let embedder = ScriptedEmbedder::new(DIMS, &[(&text_of("a"), signed(0b01, DIMS))]);

    // Act
    let report = apply_moves(
        &plan(vec![planned(&lineage, Tier::Hot)], vec![]),
        &deps(&fixture, &embedder, far()),
    )
    .await;

    // Assert
    assert_eq!(report.failed, 1);
    assert_eq!(embedder.calls(), 0);
    assert_eq!(fixture.store.tier_of(&lineage).await.unwrap(), Tier::Cold);
}

#[tokio::test]
async fn embedding_waits_for_a_generation_permit_and_gives_up_at_the_deadline() {
    // Arrange: no permit is ever free
    let mut fixture = fixture().await;
    fixture.permits = Semaphore::new(0);
    let (_, lineage) = add(&fixture.store, "a", 0b01).await;
    demoted(&fixture.store, &lineage).await;
    let embedder = ScriptedEmbedder::new(DIMS, &[(&text_of("a"), signed(0b01, DIMS))]);
    let deadline = Instant::now() + Duration::from_millis(100);

    // Act
    let report = apply_moves(
        &plan(vec![planned(&lineage, Tier::Hot)], vec![]),
        &deps(&fixture, &embedder, deadline),
    )
    .await;

    // Assert
    assert!(report.deadline_hit);
    assert_eq!(embedder.calls(), 0);
}

#[tokio::test]
async fn a_move_for_a_lineage_that_has_since_changed_is_counted_as_stale() {
    // Arrange: the plan wants the lineage cold, and it already is
    let fixture = fixture().await;
    let (_, lineage) = add(&fixture.store, "a", 0b01).await;
    demoted(&fixture.store, &lineage).await;
    let embedder = ScriptedEmbedder::new(DIMS, &[]);

    // Act
    let report = apply_moves(
        &plan(vec![], vec![planned(&lineage, Tier::Cold)]),
        &deps(&fixture, &embedder, far()),
    )
    .await;

    // Assert
    assert_eq!(report.skipped_stale, 1);
    assert_eq!(report.demoted, 0);
}

#[tokio::test]
async fn an_empty_plan_calls_nothing() {
    // Arrange
    let fixture = fixture().await;
    let embedder = ScriptedEmbedder::new(DIMS, &[]);

    // Act
    let report = apply_moves(&plan(vec![], vec![]), &deps(&fixture, &embedder, far())).await;

    // Assert
    assert_eq!(
        report,
        TierReport {
            plan_hash: "hash".to_string(),
            ..TierReport::default()
        }
    );
    assert_eq!(fixture.verifier.calls(), 0);
    assert_eq!(embedder.calls(), 0);
}
