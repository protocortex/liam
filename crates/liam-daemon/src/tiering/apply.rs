// SPDX-License-Identifier: Apache-2.0
//! Applying a plan: demotions the log vouches for, then promotions, each one a
//! store call that logs its move before it changes anything.

use std::collections::HashMap;

use liam_model::{Embedder, TextKind};
use liam_store::{
    ApplyOutcome, DefaultGraph, EventRole, LineageKey, Millis, NodeId, OriginClaim, OriginVerdict,
    Tier, TierMove,
};
use tokio::sync::Semaphore;
use tokio::time::Instant;
use tracing::{debug, error, info, warn};

use super::planner::{PlannedMove, TierPlan};
use super::verify::TierVerifier;
use crate::extraction::sidecar::FreeRam;

/// Free memory below which no more vectors are computed, so a promotion pass
/// cannot push the host into swap.
const EMBED_FREE_RAM_FLOOR_MB: u64 = 512;

/// What one application of a plan did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TierReport {
    pub planned: usize,
    pub demoted: usize,
    pub promoted: usize,
    /// Demotions the log did not vouch for.
    pub skipped_unverified: usize,
    /// Moves whose lineage had already changed or gone.
    pub skipped_stale: usize,
    /// Demotions refused because a vector had no signature.
    pub skipped_no_signature: usize,
    /// Moves that errored, and demotions that left a vector behind.
    pub failed: usize,
    pub deadline_hit: bool,
    pub plan_hash: String,
    /// The lineages whose tier changed, in the order they moved.
    pub moved_lineages: Vec<LineageKey>,
}

impl TierReport {
    /// Whether work was left for a later run.
    pub fn partial(&self) -> bool {
        self.failed > 0 || self.deadline_hit
    }
}

/// What applying a plan needs from the outside.
pub struct ApplyDeps<'a> {
    pub store: &'a DefaultGraph,
    pub verifier: &'a dyn TierVerifier,
    pub embedder: &'a dyn Embedder,
    /// The generation permits, shared with `ask`, because the embedder shares
    /// the model host with it.
    pub permits: &'a Semaphore,
    pub free_ram: &'a (dyn FreeRam + Sync),
    /// The write source the moves are logged under, naming the run.
    pub source: &'a str,
    pub run_id: &'a str,
    /// When the moves are made; a node written after it keeps its vector.
    pub now: Millis,
    pub min_residency_days: u32,
    pub deadline: Instant,
}

/// Applies `plan`: demotions first, so memory is freed before it is used.
pub async fn apply_moves(plan: &TierPlan, deps: &ApplyDeps<'_>) -> TierReport {
    let mut report = TierReport {
        planned: plan.promotions.len() + plan.demotions.len(),
        plan_hash: plan.plan_hash.clone(),
        ..TierReport::default()
    };
    demote(&plan.demotions, deps, &mut report).await;
    promote(&plan.promotions, deps, &mut report).await;
    info!(
        planned = report.planned,
        demoted = report.demoted,
        promoted = report.promoted,
        skipped_unverified = report.skipped_unverified,
        skipped_stale = report.skipped_stale,
        skipped_no_signature = report.skipped_no_signature,
        failed = report.failed,
        deadline_hit = report.deadline_hit,
        "tier moves applied"
    );
    report
}

fn expired(deps: &ApplyDeps<'_>) -> bool {
    Instant::now() >= deps.deadline
}

fn tier_move(planned: &PlannedMove, to: Tier, deps: &ApplyDeps<'_>) -> TierMove {
    let residency_until = (to == Tier::Hot)
        .then(|| Millis(deps.now.0 + Millis::days(i64::from(deps.min_residency_days)).0));
    TierMove {
        lineage_key: planned.lineage_key.clone(),
        to,
        run_id: deps.run_id.to_string(),
        reason: planned.reason.as_str().to_string(),
        changed_at: deps.now,
        residency_until,
    }
}

async fn demote(planned: &[PlannedMove], deps: &ApplyDeps<'_>, report: &mut TierReport) {
    if planned.is_empty() {
        return;
    }
    let claims = match claims_for(planned, deps).await {
        Ok(claims) => claims,
        Err(error) => {
            error!(%error, "the nodes to demote could not be read, no demotion is made");
            report.failed += planned.len();
            return;
        }
    };
    let all: Vec<OriginClaim> = claims.values().flatten().cloned().collect();
    let verdicts = match deps.verifier.verify(&all).await {
        Ok(verdicts) => verdicts,
        Err(error) => {
            error!(%error, "the log could not vouch for the demotions, none is made");
            report.failed += planned.len();
            return;
        }
    };
    for item in planned {
        if expired(deps) {
            report.deadline_hit = true;
            return;
        }
        let vouched = claims[&item.lineage_key]
            .iter()
            .all(|claim| verdicts.get(&claim.node_id) == Some(&OriginVerdict::Verified));
        if !vouched {
            debug!("a demotion was left out: the log does not vouch for every node");
            report.skipped_unverified += 1;
            continue;
        }
        let change = tier_move(item, Tier::Cold, deps);
        match deps.store.demote_lineage(&change, deps.source).await {
            Ok(ApplyOutcome::Applied) => moved(report, item, Tier::Cold),
            Ok(ApplyOutcome::Partial { remaining }) => {
                warn!(remaining, "a demotion left vectors behind for reconcile");
                moved(report, item, Tier::Cold);
                report.failed += 1;
            }
            Ok(ApplyOutcome::NoSignature) => report.skipped_no_signature += 1,
            Ok(ApplyOutcome::Stale) => report.skipped_stale += 1,
            Err(error) => {
                error!(%error, "a demotion failed");
                report.failed += 1;
            }
        }
    }
}

/// What each lineage's vector-holding nodes claim of the log.
async fn claims_for(
    planned: &[PlannedMove],
    deps: &ApplyDeps<'_>,
) -> liam_store::Result<HashMap<LineageKey, Vec<OriginClaim>>> {
    let mut claims = HashMap::with_capacity(planned.len());
    for item in planned {
        let mut lineage_claims = Vec::new();
        for text in deps.store.lineage_texts(&item.lineage_key).await? {
            if !text.has_vector {
                continue;
            }
            let origins = deps
                .store
                .node_event_ids(&text.node_id, EventRole::Origin)
                .await?;
            lineage_claims.push(OriginClaim {
                node_id: text.node_id,
                origin_event_ids: origins.into_iter().map(|event| event.event_id).collect(),
                label: text.label,
                content: text.content,
            });
        }
        claims.insert(item.lineage_key.clone(), lineage_claims);
    }
    Ok(claims)
}

fn moved(report: &mut TierReport, item: &PlannedMove, to: Tier) {
    match to {
        Tier::Cold => report.demoted += 1,
        Tier::Hot => report.promoted += 1,
    }
    report.moved_lineages.push(item.lineage_key.clone());
}

async fn promote(planned: &[PlannedMove], deps: &ApplyDeps<'_>, report: &mut TierReport) {
    for item in planned {
        if expired(deps) {
            report.deadline_hit = true;
            return;
        }
        let vectors = match embed_missing(item, deps).await {
            Ok(vectors) => vectors,
            Err(Embedding::DeadlineHit) => {
                report.deadline_hit = true;
                return;
            }
            Err(Embedding::MemoryLow(free_mb)) => {
                warn!(free_mb, "memory is low, no more promotions are made");
                report.failed += 1;
                return;
            }
            Err(Embedding::Failed(reason)) => {
                error!(%reason, "a promotion could not be embedded and is left for the next run");
                report.failed += 1;
                continue;
            }
        };
        let change = tier_move(item, Tier::Hot, deps);
        match deps
            .store
            .promote_lineage(&change, &vectors, deps.source)
            .await
        {
            Ok(ApplyOutcome::Applied) => moved(report, item, Tier::Hot),
            Ok(ApplyOutcome::Stale) => report.skipped_stale += 1,
            Ok(other) => {
                error!(?other, "a promotion ended as only a demotion can");
                report.failed += 1;
            }
            Err(error) => {
                error!(%error, "a promotion failed");
                report.failed += 1;
            }
        }
    }
}

enum Embedding {
    DeadlineHit,
    MemoryLow(u64),
    Failed(String),
}

/// The vectors of the nodes of a lineage that have none, embedded one at a
/// time behind the generation permits.
async fn embed_missing(
    item: &PlannedMove,
    deps: &ApplyDeps<'_>,
) -> Result<Vec<(NodeId, Vec<f32>)>, Embedding> {
    let texts = deps
        .store
        .lineage_texts(&item.lineage_key)
        .await
        .map_err(|error| Embedding::Failed(error.to_string()))?;
    let mut vectors = Vec::new();
    for text in texts
        .into_iter()
        .filter(|text| !text.has_vector && !text.content.trim().is_empty())
    {
        let free_mb = deps.free_ram.free_mb();
        if free_mb < EMBED_FREE_RAM_FLOOR_MB {
            return Err(Embedding::MemoryLow(free_mb));
        }
        let permit = tokio::time::timeout_at(deps.deadline, deps.permits.acquire())
            .await
            .map_err(|_| Embedding::DeadlineHit)?
            .map_err(|error| Embedding::Failed(error.to_string()))?;
        let embedded = tokio::time::timeout_at(
            deps.deadline,
            deps.embedder.embed(&text.content, TextKind::Document),
        )
        .await;
        drop(permit);
        match embedded {
            Err(_) => return Err(Embedding::DeadlineHit),
            Ok(Err(error)) => return Err(Embedding::Failed(error.to_string())),
            Ok(Ok(vector)) => vectors.push((text.node_id, vector)),
        }
    }
    Ok(vectors)
}

#[cfg(test)]
mod tests;
