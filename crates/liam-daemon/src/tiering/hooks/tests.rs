// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use liam_daemon::consolidation::tier_history::Cooldown;
use liam_model::Embedder;
use liam_store::{FixedClock, Millis, NewNode, Tier, TierMove, UsageEntry};

use super::*;
use crate::tiering::test_support::{
    logged_store, signed, FailOnNthEmbedder, FixedRam, ScriptedEmbedder,
};

const DIMS: usize = 8;
const DAY_MS: i64 = 86_400_000;
const START: Millis = Millis(10 * DAY_MS);
const WINDOW_DAYS: u32 = 30;
const HITS_CAP: u32 = 2;
const SOURCE: &str = "consolidation:run-1";
const RUN: &str = "run-1";

struct Fixture {
    store: DefaultGraph,
    clock: Arc<FixedClock>,
    _dir: tempfile::TempDir,
}

/// A store whose counting began at `START`, read 45 days later: long enough for
/// every item to be judged.
async fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(FixedClock::new(START));
    let store = logged_store(dir.path(), clock.clone(), DIMS).await;
    Fixture {
        store,
        clock,
        _dir: dir,
    }
}

fn config() -> TieringConfig {
    TieringConfig {
        enabled: true,
        w1: 1.0,
        w2: 0.0,
        w3: 0.0,
        w4: 0.0,
        hits_cap: HITS_CAP,
        window_days: WINDOW_DAYS,
        max_demotions_per_run: 10,
        max_promotions_per_run: 10,
        ..TieringConfig::default()
    }
}

fn text_of(label: &str) -> String {
    format!("text of {label}")
}

async fn add(fixture: &Fixture, label: &str, pattern: u8) -> (NodeId, LineageKey) {
    let id = fixture
        .store
        .insert(
            NewNode::now("fact", label, text_of(label))
                .with_subject(label)
                .with_producer("agent-a")
                .with_embedding(signed(pattern, DIMS)),
        )
        .await
        .unwrap();
    let lineage = fixture.store.lineage_of(&id).await.unwrap().unwrap();
    (id, lineage)
}

async fn demote(fixture: &Fixture, lineage: &LineageKey) {
    let change = TierMove {
        lineage_key: lineage.clone(),
        to: Tier::Cold,
        run_id: "earlier".to_string(),
        reason: "test".to_string(),
        changed_at: START,
        residency_until: None,
    };
    fixture
        .store
        .demote_lineage(&change, "consolidation:earlier")
        .await
        .unwrap();
}

/// Starts counting at `START`, then moves the clock 45 days on, where the items
/// in `used` have been hit enough to count fully.
async fn observed(fixture: &Fixture, used: &[&LineageKey]) {
    fixture
        .store
        .put_usage_logged(START.day_number(), &[])
        .await
        .unwrap();
    let later = Millis(START.0 + 45 * DAY_MS);
    fixture.clock.set(later);
    let entries: Vec<UsageEntry> = used
        .iter()
        .map(|lineage| UsageEntry {
            lineage_key: (*lineage).clone(),
            hits: HITS_CAP,
            last_access_at: later,
        })
        .collect();
    fixture
        .store
        .put_usage_logged(later.day_number(), &entries)
        .await
        .unwrap();
}

fn hooks(fixture: &Fixture, config: TieringConfig, embedder: Arc<dyn Embedder>) -> TierHooks {
    TierHooks {
        config,
        extraction: ExtractionConfig::default(),
        embedder,
        permits: Arc::new(Semaphore::new(1)),
        free_ram: Arc::new(FixedRam::plenty()),
        clock: fixture.clock.clone(),
    }
}

fn ctx<'a>(fixture: &'a Fixture, probed: &[&NodeId], history: TierHistory) -> TierCtx<'a> {
    TierCtx {
        store: &fixture.store,
        source: SOURCE.to_string(),
        run_id: RUN.to_string(),
        probed_nodes: probed.iter().map(|id| (*id).clone()).collect(),
        history,
    }
}

fn scripted(labels: &[&str]) -> Arc<ScriptedEmbedder> {
    Arc::new(scripted_owned(labels))
}

fn scripted_owned(labels: &[&str]) -> ScriptedEmbedder {
    let vectors: Vec<(String, Vec<f32>)> = labels
        .iter()
        .map(|label| (text_of(label), signed(0b0101_0101, DIMS)))
        .collect();
    let named: Vec<(&str, Vec<f32>)> = vectors
        .iter()
        .map(|(text, vector)| (text.as_str(), vector.clone()))
        .collect();
    ScriptedEmbedder::new(DIMS, &named)
}

async fn tier_of(fixture: &Fixture, lineage: &LineageKey) -> Tier {
    fixture.store.tier_of(lineage).await.unwrap()
}

async fn holds_vector(fixture: &Fixture, lineage: &LineageKey) -> bool {
    let texts = fixture.store.lineage_texts(lineage).await.unwrap();
    texts.iter().all(|text| text.has_vector)
}

#[tokio::test]
async fn an_unused_item_is_demoted_and_a_used_one_stays_hot() {
    // Arrange
    let fixture = fixture().await;
    let (_, unused) = add(&fixture, "unused", 0b0000_0011).await;
    let (_, used) = add(&fixture, "used", 0b0000_1100).await;
    observed(&fixture, &[&used]).await;
    let hook = hooks(&fixture, config(), scripted(&[]));

    // Act
    let report = hook
        .tier(&ctx(&fixture, &[], TierHistory::default()))
        .await
        .unwrap();

    // Assert
    assert_eq!(report.moved_lineages, vec![unused.as_str().to_string()]);
    assert!(!report.partial);
    assert!(!report.plan_hash.is_empty());
    assert_eq!(tier_of(&fixture, &unused).await, Tier::Cold);
    assert!(!holds_vector(&fixture, &unused).await);
    assert_eq!(tier_of(&fixture, &used).await, Tier::Hot);
    assert!(holds_vector(&fixture, &used).await);
}

#[tokio::test]
async fn a_cold_item_that_has_been_used_is_promoted_with_a_fresh_vector() {
    // Arrange
    let fixture = fixture().await;
    let (_, lineage) = add(&fixture, "wanted", 0b0000_0011).await;
    demote(&fixture, &lineage).await;
    observed(&fixture, &[&lineage]).await;
    let embedder = scripted(&["wanted"]);
    let hook = hooks(&fixture, config(), embedder.clone());

    // Act
    let report = hook
        .tier(&ctx(&fixture, &[], TierHistory::default()))
        .await
        .unwrap();

    // Assert
    assert_eq!(report.moved_lineages, vec![lineage.as_str().to_string()]);
    assert_eq!(tier_of(&fixture, &lineage).await, Tier::Hot);
    assert!(holds_vector(&fixture, &lineage).await);
    assert_eq!(embedder.calls(), 1);
}

#[tokio::test]
async fn a_second_step_over_the_same_store_moves_nothing_and_embeds_nothing() {
    // Arrange
    let fixture = fixture().await;
    let (_, unused) = add(&fixture, "unused", 0b0000_0011).await;
    let (_, wanted) = add(&fixture, "wanted", 0b0000_1100).await;
    demote(&fixture, &wanted).await;
    observed(&fixture, &[&wanted]).await;
    let embedder = scripted(&["wanted"]);
    let hook = hooks(&fixture, config(), embedder.clone());
    hook.tier(&ctx(&fixture, &[], TierHistory::default()))
        .await
        .unwrap();
    let calls_after_first = embedder.calls();

    // Act
    let again = hook
        .tier(&ctx(&fixture, &[], TierHistory::default()))
        .await
        .unwrap();

    // Assert
    assert!(again.moved_lineages.is_empty());
    assert_eq!(embedder.calls(), calls_after_first);
    assert_eq!(tier_of(&fixture, &unused).await, Tier::Cold);
    assert_eq!(tier_of(&fixture, &wanted).await, Tier::Hot);
}

#[tokio::test]
async fn an_item_a_probe_expects_to_find_is_not_demoted() {
    // Arrange
    let fixture = fixture().await;
    let (probed, pinned) = add(&fixture, "probed", 0b0000_0011).await;
    let (_, other) = add(&fixture, "other", 0b0000_1100).await;
    observed(&fixture, &[]).await;
    let hook = hooks(&fixture, config(), scripted(&[]));

    // Act
    let report = hook
        .tier(&ctx(&fixture, &[&probed], TierHistory::default()))
        .await
        .unwrap();

    // Assert
    assert_eq!(report.moved_lineages, vec![other.as_str().to_string()]);
    assert_eq!(tier_of(&fixture, &pinned).await, Tier::Hot);
}

#[tokio::test]
async fn a_lineage_cooling_down_after_a_rollback_is_left_alone() {
    // Arrange
    let fixture = fixture().await;
    let (_, cooling) = add(&fixture, "cooling", 0b0000_0011).await;
    let (_, other) = add(&fixture, "other", 0b0000_1100).await;
    observed(&fixture, &[]).await;
    let history = TierHistory {
        consecutive_rollbacks: 0,
        cooldown: vec![Cooldown {
            lineage_key: cooling.as_str().to_string(),
            runs_left: 2,
        }],
    };
    let hook = hooks(&fixture, config(), scripted(&[]));

    // Act
    let report = hook.tier(&ctx(&fixture, &[], history)).await.unwrap();

    // Assert
    assert_eq!(report.moved_lineages, vec![other.as_str().to_string()]);
    assert_eq!(tier_of(&fixture, &cooling).await, Tier::Hot);
}

#[tokio::test]
async fn each_rollback_in_a_row_halves_the_demotions_of_the_next_run() {
    // Arrange
    let fixture = fixture().await;
    for (label, pattern) in [("a", 0b001), ("b", 0b010), ("c", 0b100), ("d", 0b111)] {
        add(&fixture, label, pattern).await;
    }
    observed(&fixture, &[]).await;
    let hook = hooks(
        &fixture,
        TieringConfig {
            max_demotions_per_run: 4,
            ..config()
        },
        scripted(&[]),
    );
    let history = TierHistory {
        consecutive_rollbacks: 2,
        cooldown: Vec::new(),
    };

    // Act
    let report = hook.tier(&ctx(&fixture, &[], history)).await.unwrap();

    // Assert
    assert_eq!(report.moved_lineages.len(), 1);
}

#[tokio::test]
async fn the_demotion_cap_bounds_one_run() {
    // Arrange
    let fixture = fixture().await;
    for (label, pattern) in [("a", 0b001), ("b", 0b010), ("c", 0b100)] {
        add(&fixture, label, pattern).await;
    }
    observed(&fixture, &[]).await;
    let hook = hooks(
        &fixture,
        TieringConfig {
            max_demotions_per_run: 2,
            ..config()
        },
        scripted(&[]),
    );

    // Act
    let report = hook
        .tier(&ctx(&fixture, &[], TierHistory::default()))
        .await
        .unwrap();

    // Assert
    assert_eq!(report.moved_lineages.len(), 2);
}

#[tokio::test]
async fn a_promotion_that_cannot_be_embedded_makes_the_step_partial() {
    // Arrange
    let fixture = fixture().await;
    let (_, lineage) = add(&fixture, "wanted", 0b0000_0011).await;
    demote(&fixture, &lineage).await;
    observed(&fixture, &[&lineage]).await;
    let failing = Arc::new(FailOnNthEmbedder::new(scripted_owned(&["wanted"]), 1));
    let hook = hooks(&fixture, config(), failing);

    // Act
    let report = hook
        .tier(&ctx(&fixture, &[], TierHistory::default()))
        .await
        .unwrap();

    // Assert
    assert!(report.partial);
    assert!(report.moved_lineages.is_empty());
    assert_eq!(tier_of(&fixture, &lineage).await, Tier::Cold);
}

#[tokio::test]
async fn a_run_deadline_that_has_passed_leaves_the_moves_for_a_later_run() {
    // Arrange
    let fixture = fixture().await;
    let (_, unused) = add(&fixture, "unused", 0b0000_0011).await;
    observed(&fixture, &[]).await;
    let hook = hooks(
        &fixture,
        TieringConfig {
            run_deadline_secs: 0,
            ..config()
        },
        scripted(&[]),
    );

    // Act
    let report = hook
        .tier(&ctx(&fixture, &[], TierHistory::default()))
        .await
        .unwrap();

    // Assert
    assert!(report.partial);
    assert!(report.moved_lineages.is_empty());
    assert_eq!(tier_of(&fixture, &unused).await, Tier::Hot);
}
