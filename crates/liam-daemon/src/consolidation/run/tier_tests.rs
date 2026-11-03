// SPDX-License-Identifier: Apache-2.0
//! What the tier step hands the run and what the run keeps of it: the moved
//! lineages in the manifest, and in the history beside the database the streak
//! of rolled back runs and the lineages they moved.

use std::sync::Mutex;

use super::tests::*;
use super::*;
use crate::consolidation::hooks::{TierCtx, TierHook, TierReport};
use crate::consolidation::tier_history::{self, history_path_for, TierHistory};

const MOVED: &str = "lineage-a";
const PLAN_HASH: &str = "plan-hash";

/// Moves the lineages it is given and notes what each run handed it.
struct ScriptedTier {
    moves: Vec<String>,
    seen: Mutex<Vec<Seen>>,
}

struct Seen {
    history: TierHistory,
    probed_nodes: usize,
    run_id: String,
}

impl ScriptedTier {
    fn moving(lineages: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            moves: lineages.iter().map(|name| (*name).to_string()).collect(),
            seen: Mutex::new(Vec::new()),
        })
    }

    fn histories_seen(&self) -> Vec<TierHistory> {
        let seen = self.seen.lock().unwrap();
        seen.iter().map(|run| run.history.clone()).collect()
    }
}

#[async_trait]
impl TierHook for ScriptedTier {
    async fn tier(&self, ctx: &TierCtx<'_>) -> anyhow::Result<TierReport> {
        self.seen.lock().unwrap().push(Seen {
            history: ctx.history.clone(),
            probed_nodes: ctx.probed_nodes.len(),
            run_id: ctx.run_id.clone(),
        });
        Ok(TierReport {
            partial: false,
            plan_hash: PLAN_HASH.to_string(),
            moved_lineages: self.moves.clone(),
        })
    }
}

fn hooks_with(tier: &Arc<ScriptedTier>) -> ConsolidationHookSet {
    ConsolidationHookSet {
        tiering: Some(tier.clone()),
        ..Default::default()
    }
}

fn history_of(env: &Env) -> TierHistory {
    tier_history::load(&history_path_for(&env.database_path())).unwrap()
}

#[tokio::test]
async fn a_committed_run_records_the_plan_and_the_moved_lineages_in_its_manifest() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let tier = ScriptedTier::moving(&[MOVED]);
    let run = env.consolidator_with(maintenance(), honest(), hooks_with(&tier));

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::Committed));
    assert_eq!(manifest.tier_plan_hash.as_deref(), Some(PLAN_HASH));
    assert_eq!(manifest.tier_moved_lineages, vec![MOVED.to_string()]);
}

#[tokio::test]
async fn the_step_is_handed_the_run_and_the_nodes_the_probes_expect() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let tier = ScriptedTier::moving(&[]);
    let run = env.consolidator_with(maintenance(), honest(), hooks_with(&tier));

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    let seen = tier.seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].run_id, manifest.run_id);
    assert!(seen[0].probed_nodes > 0);
}

#[tokio::test]
async fn a_rolled_back_run_that_moved_lineages_leaves_a_cooldown_and_a_streak() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let tier = ScriptedTier::moving(&[MOVED]);
    let run = env.consolidator_with(maintenance(), env.regressing(), hooks_with(&tier));

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::RolledBack));
    let history = history_of(&env);
    assert_eq!(history.consecutive_rollbacks, 1);
    assert_eq!(history.cooldown.len(), 1);
    assert_eq!(history.cooldown[0].lineage_key, MOVED);
    assert_eq!(history.cooldown[0].runs_left, COOLDOWN_RUNS);
}

#[tokio::test]
async fn the_next_run_is_handed_the_history_a_rollback_left() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let tier = ScriptedTier::moving(&[MOVED]);
    let rolled_back = env.consolidator_with(maintenance(), env.regressing(), hooks_with(&tier));
    rolled_back.consolidate(&shared).await.unwrap();
    let next = env.consolidator_with(maintenance(), honest(), hooks_with(&tier));

    // Act
    next.consolidate(&shared).await.unwrap();

    // Assert
    let histories = tier.histories_seen();
    assert_eq!(histories[0], TierHistory::default());
    assert_eq!(histories[1].consecutive_rollbacks, 1);
    assert_eq!(histories[1].cooldown[0].lineage_key, MOVED);
}

#[tokio::test]
async fn a_rolled_back_run_that_moved_nothing_changes_no_history() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let tier = ScriptedTier::moving(&[]);
    let run = env.consolidator_with(maintenance(), env.regressing(), hooks_with(&tier));

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::RolledBack));
    assert_eq!(history_of(&env), TierHistory::default());
}

#[tokio::test]
async fn a_rollback_without_a_tier_part_changes_no_history() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let run = env.consolidator(maintenance(), env.regressing());

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::RolledBack));
    assert!(!history_path_for(&env.database_path()).exists());
}

#[tokio::test]
async fn committed_runs_count_the_cooldown_down_and_a_moving_one_ends_the_streak() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let rolled_back = ScriptedTier::moving(&[MOVED]);
    env.consolidator_with(maintenance(), env.regressing(), hooks_with(&rolled_back))
        .consolidate(&shared)
        .await
        .unwrap();
    let quiet = ScriptedTier::moving(&[]);

    // Act
    env.consolidator_with(maintenance(), honest(), hooks_with(&quiet))
        .consolidate(&shared)
        .await
        .unwrap();
    let after_quiet = history_of(&env);
    let moving = ScriptedTier::moving(&["lineage-b"]);
    env.consolidator_with(maintenance(), honest(), hooks_with(&moving))
        .consolidate(&shared)
        .await
        .unwrap();

    // Assert
    assert_eq!(after_quiet.consecutive_rollbacks, 1);
    assert_eq!(after_quiet.cooldown[0].runs_left, COOLDOWN_RUNS - 1);
    let after_moving = history_of(&env);
    assert_eq!(after_moving.consecutive_rollbacks, 0);
    assert_eq!(after_moving.cooldown[0].runs_left, COOLDOWN_RUNS - 2);
}
