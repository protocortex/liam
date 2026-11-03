// SPDX-License-Identifier: Apache-2.0
//! The tier step of a consolidation run: bring the signatures up to date, read
//! the usage once, plan the moves, apply them and prune what no longer counts.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use liam_daemon::config::{ExtractionConfig, TieringConfig};
use liam_daemon::consolidation::hooks::{TierCtx, TierHook, TierReport};
use liam_daemon::consolidation::tier_history::TierHistory;
use liam_model::Embedder;
use liam_store::{Clock, DefaultGraph, LineageKey, NodeId};
use tokio::sync::Semaphore;
use tokio::time::Instant;
use tracing::info;

use super::apply::{apply_moves, ApplyDeps, TierReport as AppliedMoves};
use super::caps::apply_caps;
use super::planner::{plan, ExemptSet, PinnedSet, RollbackState};
use super::utility::TierParams;
use crate::extraction::sidecar::FreeRam;

pub struct TierHooks {
    pub config: TieringConfig,
    pub extraction: ExtractionConfig,
    /// The fallback producer id, whose trust `unknown_trust_ceiling` bounds.
    pub unknown_producer: String,
    pub embedder: Arc<dyn Embedder>,
    /// The generation permits, shared with `ask`, because the embedder shares
    /// the model host with it.
    pub permits: Arc<Semaphore>,
    pub free_ram: Arc<dyn FreeRam + Send + Sync>,
    pub clock: Arc<dyn Clock>,
}

#[async_trait]
impl TierHook for TierHooks {
    async fn tier(&self, ctx: &TierCtx<'_>) -> anyhow::Result<TierReport> {
        let store = ctx.store;
        info!(run_id = %ctx.run_id, "tier step: preparing the store");
        prepare(store).await?;

        info!(run_id = %ctx.run_id, "tier step: planning");
        let now = self.clock.now();
        let mut snapshot = store
            .usage_snapshot(now, i64::from(self.config.window_days))
            .await
            .context("could not read the usage of the items")?;
        let exempt: ExemptSet = store.tier_exempt_lineages().await?;
        let pinned = pinned_lineages(store, &ctx.probed_nodes).await?;
        let params = TierParams::new(&self.config, &self.extraction)
            .with_unknown_producer(&self.unknown_producer);
        apply_caps(&mut snapshot, &params);
        let tier_plan = plan(
            &snapshot,
            &params,
            &exempt,
            &pinned,
            &rollback_state(&ctx.history),
        );

        info!(run_id = %ctx.run_id, plan_hash = %tier_plan.plan_hash, "tier step: applying");
        let applied = apply_moves(
            &tier_plan,
            &ApplyDeps {
                store,
                verifier: store,
                embedder: self.embedder.as_ref(),
                permits: &self.permits,
                free_ram: self.free_ram.as_ref(),
                source: &ctx.source,
                run_id: &ctx.run_id,
                now,
                min_residency_days: self.config.min_residency_days,
                deadline: Instant::now() + Duration::from_secs(self.config.run_deadline_secs),
            },
        )
        .await;

        info!(run_id = %ctx.run_id, "tier step: pruning");
        prune(store, now, self.config.window_days).await?;
        Ok(report_of(applied))
    }
}

/// Gives vectors from before tiering their signatures, and makes the in-memory
/// index and the vectors agree with the tier rows, which a crash may have left
/// apart.
async fn prepare(store: &DefaultGraph) -> anyhow::Result<()> {
    store
        .backfill_quant()
        .await
        .context("could not give the stored vectors their signatures")?;
    store.reload_quant_index().await?;
    store
        .reconcile_tier_vectors()
        .await
        .context("could not make the vectors agree with the tier rows")?;
    Ok(())
}

async fn pinned_lineages(store: &DefaultGraph, probed: &[NodeId]) -> anyhow::Result<PinnedSet> {
    let mut pinned = PinnedSet::new();
    for node in probed {
        pinned.extend(store.lineage_of(node).await?);
    }
    Ok(pinned)
}

fn rollback_state(history: &TierHistory) -> RollbackState {
    RollbackState {
        consecutive: history.consecutive_rollbacks,
        cooldown: history
            .cooldown
            .iter()
            .map(|entry| (LineageKey::from_raw(&entry.lineage_key), entry.runs_left))
            .collect(),
    }
}

/// Drops the usage that fell out of the window and the rows of lineages that
/// no longer exist.
async fn prune(
    store: &DefaultGraph,
    now: liam_store::Millis,
    window_days: u32,
) -> anyhow::Result<()> {
    store
        .prune_usage(now.day_number() - i64::from(window_days))
        .await?;
    store.prune_tier_state().await?;
    Ok(())
}

fn report_of(applied: AppliedMoves) -> TierReport {
    TierReport {
        partial: applied.partial(),
        moved_lineages: applied
            .moved_lineages
            .iter()
            .map(|lineage| lineage.as_str().to_string())
            .collect(),
        plan_hash: applied.plan_hash,
    }
}

#[cfg(test)]
mod tests;
