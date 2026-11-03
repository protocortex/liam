// SPDX-License-Identifier: Apache-2.0
//! The tier step inside whole runs: unused items are demoted and kept demoted
//! by a committed run, a rolled back run puts them back and leaves a cooldown,
//! and a daemon with tiering off runs no tier step at all.

use liam_daemon::config::{ExtractionConfig, TieringConfig};
use liam_daemon::consolidation::hooks::{TierCtx, TierHook, TierReport};
use liam_daemon::consolidation::tier_history::{history_path_for, load as load_history};
use liam_store::{Clock, LineageKey, NewNode, NodeId, Tier, TierMove, UsageEntry};
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
async fn aged(harness: &Harness) -> Aged {
    let (store, clock) = harness.store_at(Millis(10 * DAY)).await;
    let mut lineages = Vec::new();
    let mut ids = Vec::new();
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
        ids.push(id);
    }
    store
        .put_usage_logged(clock.now().day_number(), &[] as &[UsageEntry])
        .await
        .unwrap();
    clock.set(Millis(clock.now().0 + OBSERVED_DAYS * DAY));
    Aged {
        server: harness.server(store, Arc::new(liam_model::MockLlm)),
        clock,
        lineages,
        ids,
    }
}

struct Aged {
    server: MemoryServer,
    clock: Arc<FixedClock>,
    lineages: Vec<LineageKey>,
    ids: Vec<NodeId>,
}

fn tier_hooks(clock: &Arc<FixedClock>) -> ConsolidationHookSet {
    ConsolidationHookSet {
        tiering: Some(Arc::new(TierHooks {
            config: config_on(),
            extraction: ExtractionConfig::default(),
            unknown_producer: "unknown".to_owned(),
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

/// One probe, and a dedup threshold no pair of mock vectors reaches, so the
/// near-duplicate merge links no two notes and a demoted one has no neighbour
/// to be found through.
fn one_probe(harness: &mut Harness) {
    harness.config.consolidation.probe_count = 1;
    harness.config.consolidation.dedup_similarity_threshold = 1.0;
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
    let Aged {
        server,
        clock,
        lineages,
        ..
    } = aged(&harness).await;
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
    let Aged {
        server,
        clock,
        lineages,
        ..
    } = aged(&harness).await;
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
    let Aged {
        server, lineages, ..
    } = aged(&harness).await;
    let job = ConsolidationJob::new(JobSetup {
        keys: None,
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

// ---- probes after tier moves ----

/// The dimensions of a node's vector whose signs the paraphrase flips: half of
/// them, so the query's signature is well past the cold arm's cutoff.
const FLIPPED_DIMS: usize = DIMS / 2;
const NO_WORDS: &str = "qqq www";

/// Scores each probe with a query that shares no word with its node and whose
/// vector is the node's with half its signs flipped. Only the f32 arm can
/// answer it: the cold arm's signature cutoff rejects it.
struct ParaphraseProbes {
    queries: Vec<(NodeId, Vec<f32>)>,
}

impl ParaphraseProbes {
    fn over(ids: &[NodeId]) -> Arc<Self> {
        let queries = ids
            .iter()
            .enumerate()
            .map(|(n, id)| {
                let mut vector = signed(1 << n, DIMS);
                vector[..FLIPPED_DIMS]
                    .iter_mut()
                    .for_each(|component| *component = -*component);
                (id.clone(), vector)
            })
            .collect();
        Arc::new(Self { queries })
    }

    fn paraphrased(&self, probes: &[Probe]) -> Vec<Probe> {
        probes
            .iter()
            .map(|probe| {
                let embedding = self
                    .queries
                    .iter()
                    .find(|(id, _)| probe.relevant.contains(id))
                    .map(|(_, vector)| vector.clone());
                Probe {
                    query: NO_WORDS.to_string(),
                    embedding,
                    ..probe.clone()
                }
            })
            .collect()
    }
}

#[async_trait]
impl ProbeRunner for ParaphraseProbes {
    async fn run(&self, store: &DefaultGraph, probes: &[Probe]) -> Result<ProbeReport, ProbeError> {
        run_probes(store, &self.paraphrased(probes)).await
    }
}

/// Demotes the lineages of the probed nodes whatever the planner would say.
struct DemotesProbed {
    clock: Arc<FixedClock>,
}

#[async_trait]
impl TierHook for DemotesProbed {
    async fn tier(&self, ctx: &TierCtx<'_>) -> anyhow::Result<TierReport> {
        let mut moved = Vec::new();
        for node in &ctx.probed_nodes {
            let lineage_key = ctx.store.lineage_of(node).await?.unwrap();
            let change = TierMove {
                lineage_key: lineage_key.clone(),
                to: Tier::Cold,
                run_id: ctx.run_id.clone(),
                reason: "forced".to_string(),
                changed_at: self.clock.now(),
                residency_until: None,
            };
            ctx.store.demote_lineage(&change, &ctx.source).await?;
            moved.push(lineage_key.as_str().to_string());
        }
        Ok(TierReport {
            partial: false,
            plan_hash: "forced".to_string(),
            moved_lineages: moved,
        })
    }
}

#[tokio::test]
async fn demoting_an_item_only_the_f32_arm_reaches_costs_its_probe_more_than_the_margin() {
    for index in 0..NODES {
        // Arrange
        let harness = Harness::new();
        let aged = aged(&harness).await;
        let probes = ParaphraseProbes::over(&aged.ids);
        let target = Probe {
            name: "target".to_string(),
            query: "unused".to_string(),
            relevant: vec![aged.ids[index].clone()],
            embedding: None,
        };
        let store = aged.server.store().await.unwrap();
        let before = probes
            .run(&store, std::slice::from_ref(&target))
            .await
            .unwrap();
        let change = TierMove {
            lineage_key: aged.lineages[index].clone(),
            to: Tier::Cold,
            run_id: "earlier".to_string(),
            reason: "test".to_string(),
            changed_at: aged.clock.now(),
            residency_until: None,
        };

        // Act
        store
            .demote_lineage(&change, "consolidation:earlier")
            .await
            .unwrap();
        let after = probes.run(&store, &[target]).await.unwrap();

        // Assert
        let drop = before.scores["target"].recall - after.scores["target"].recall;
        assert!(
            drop >= harness.config.consolidation.probe_recall_margin,
            "node {index}"
        );
    }
}

#[tokio::test]
async fn the_planner_keeps_the_probed_item_hot_so_the_run_passes_its_probes() {
    // Arrange
    let mut harness = Harness::new();
    one_probe(&mut harness);
    let aged = aged(&harness).await;
    let probes = ParaphraseProbes::over(&aged.ids);
    let job = harness.job_with_hooks(&aged.server, probes, None, tier_hooks(&aged.clock));

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert_eq!(cold_count(&aged.server, &aged.lineages).await, NODES - 1);
}

#[tokio::test]
async fn a_forced_demotion_of_a_probed_item_rolls_the_run_back_and_the_next_run_leaves_it_alone() {
    // Arrange
    let mut harness = Harness::new();
    one_probe(&mut harness);
    let aged = aged(&harness).await;
    let probes = ParaphraseProbes::over(&aged.ids);
    let hooks = ConsolidationHookSet {
        tiering: Some(Arc::new(DemotesProbed {
            clock: aged.clock.clone(),
        })),
        ..Default::default()
    };
    let forced = harness.job_with_hooks(&aged.server, probes.clone(), None, hooks);

    // Act
    let first = forced.tick().await;
    let forced_lineages = manifest_of(&harness).tier_moved_lineages;

    // Assert
    assert_eq!(first, Some(RunOutcome::RolledBack));
    assert_eq!(forced_lineages.len(), 1);
    assert_eq!(cold_count(&aged.server, &aged.lineages).await, 0);

    // Act
    let retry = harness.job_with_hooks(&aged.server, probes, None, tier_hooks(&aged.clock));
    let second = retry.tick().await;

    // Assert
    assert_eq!(second, Some(RunOutcome::Committed));
    let repeated = manifest_of(&harness)
        .tier_moved_lineages
        .iter()
        .any(|lineage| forced_lineages.contains(lineage));
    assert!(
        !repeated,
        "the plan that failed the probes was not repeated"
    );
}
