// SPDX-License-Identifier: Apache-2.0
//! The tier step inside whole runs: unused items are demoted and kept demoted
//! by a committed run, a rolled back run puts them back and leaves a cooldown,
//! and a daemon with tiering off runs no tier step at all.

use liam_daemon::config::{ExtractionConfig, TieringConfig};
use liam_daemon::consolidation::tier_history::{history_path_for, load as load_history};
use liam_store::{Clock, LineageKey, NewNode, Tier, UsageEntry};
use tokio::sync::Semaphore;

use super::*;
use crate::extraction::sidecar::SystemRam;
use crate::tiering::hooks::TierHooks;
use crate::tiering::test_support::{signed, ScriptedEmbedder};

const NODES: usize = 4;
const OBSERVED_DAYS: i64 = 45;

fn config_on() -> TieringConfig {
    TieringConfig {
        enabled: true,
        w1: 1.0,
        w2: 0.0,
        w3: 0.0,
        w4: 0.0,
        ..TieringConfig::default()
    }
}

/// Four unused facts whose counting began 45 days before the clock now reads.
async fn aged(harness: &Harness) -> (MemoryServer, Arc<FixedClock>, Vec<LineageKey>) {
    let (store, clock) = harness.store_at(Millis(10 * DAY)).await;
    let mut lineages = Vec::new();
    for n in 0..NODES {
        let id = store
            .insert(
                NewNode::now("fact", format!("Note {n}"), format!("note number {n}"))
                    .with_subject(format!("subject {n}"))
                    .with_producer("agent-a")
                    .with_embedding(signed(1 << n, DIMS)),
            )
            .await
            .unwrap();
        lineages.push(store.lineage_of(&id).await.unwrap().unwrap());
    }
    store
        .put_usage_logged(clock.now().day_number(), &[] as &[UsageEntry])
        .await
        .unwrap();
    clock.set(Millis(clock.now().0 + OBSERVED_DAYS * DAY));
    (
        harness.server(store, Arc::new(liam_model::MockLlm)),
        clock,
        lineages,
    )
}

fn tier_hooks(clock: &Arc<FixedClock>) -> ConsolidationHookSet {
    ConsolidationHookSet {
        tiering: Some(Arc::new(TierHooks {
            config: config_on(),
            extraction: ExtractionConfig::default(),
            embedder: Arc::new(ScriptedEmbedder::new(DIMS, &[])),
            permits: Arc::new(Semaphore::new(1)),
            free_ram: Arc::new(SystemRam),
            clock: clock.clone(),
        })),
        ..Default::default()
    }
}

async fn cold_count(server: &MemoryServer, lineages: &[LineageKey]) -> usize {
    let store = server.store().await.unwrap();
    let mut cold = 0;
    for lineage in lineages {
        cold += usize::from(store.tier_of(lineage).await.unwrap() == Tier::Cold);
    }
    cold
}

fn one_probe(harness: &mut Harness) {
    harness.config.consolidation.probe_count = 1;
}

fn manifest_of(harness: &Harness) -> RunManifest {
    RunManifest::load(&harness.manifest_path())
        .unwrap()
        .unwrap()
}

#[tokio::test]
async fn a_committed_run_demotes_the_unused_items_but_not_the_probed_one() {
    // Arrange
    let mut harness = Harness::new();
    one_probe(&mut harness);
    let (server, clock, lineages) = aged(&harness).await;
    let job = harness.job_with_hooks(&server, honest(), None, tier_hooks(&clock));

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert_eq!(cold_count(&server, &lineages).await, NODES - 1);
    let manifest = manifest_of(&harness);
    assert_eq!(manifest.tier_moved_lineages.len(), NODES - 1);
    assert!(manifest.tier_plan_hash.is_some());
    assert!(manifest
        .completed_steps
        .iter()
        .all(|step| step != "tier:partial"));
}

#[tokio::test]
async fn a_rolled_back_run_puts_the_items_back_and_the_next_run_leaves_them_alone() {
    // Arrange
    let mut harness = Harness::new();
    one_probe(&mut harness);
    let (server, clock, lineages) = aged(&harness).await;
    let rolled_back = harness.job_with_hooks(&server, regresses(), None, tier_hooks(&clock));

    // Act
    let first = rolled_back.tick().await;
    let moved = manifest_of(&harness).tier_moved_lineages;

    // Assert
    assert_eq!(first, Some(RunOutcome::RolledBack));
    assert_eq!(moved.len(), NODES - 1);
    assert_eq!(cold_count(&server, &lineages).await, 0);
    let history = load_history(&history_path_for(&database_in(&harness.root))).unwrap();
    assert_eq!(history.consecutive_rollbacks, 1);
    assert_eq!(history.cooldown.len(), NODES - 1);

    // Act
    let retry = harness.job_with_hooks(&server, honest(), None, tier_hooks(&clock));
    let second = retry.tick().await;

    // Assert
    assert_eq!(second, Some(RunOutcome::Committed));
    let still_cooling = manifest_of(&harness)
        .tier_moved_lineages
        .iter()
        .any(|lineage| moved.contains(lineage));
    assert!(!still_cooling);
    assert!(cold_count(&server, &lineages).await <= 1);
}

#[tokio::test]
async fn with_tiering_off_the_tier_step_is_skipped() {
    // Arrange
    let mut harness = Harness::new();
    one_probe(&mut harness);
    let (server, _clock, lineages) = aged(&harness).await;
    let job = ConsolidationJob::new(JobSetup {
        config: &harness.config,
        database_path: &database_in(&harness.root),
        log_dir: &log_dir_in(&harness.root),
        log: harness.log.clone(),
        embedder: embedder(),
        server: &server,
        shutdown: &harness.shutdown,
    })
    .unwrap();

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert!(manifest_of(&harness)
        .completed_steps
        .iter()
        .any(|step| step == "tier:skipped"));
    assert_eq!(cold_count(&server, &lineages).await, 0);
}
