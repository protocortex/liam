// SPDX-License-Identifier: Apache-2.0
//! The retention sweep. Per rule, the doomed nodes are chosen first and then
//! removed in chunks, each chunk in one transaction through the same removal
//! the log's tombstones replay. With a log attached each chunk is tombstoned
//! before it is deleted, and the log lock is held from the first selection to
//! the last deletion so no write can land between the two.

use liam_log::event::{LogPayload, TombstoneTable, TombstoneTarget};

use super::logged_plan::tombstone_event;
use super::logged_write::{abandon, commit_or_abandon, HeldLog};
use super::projection::{apply_steps, steps_for};
use super::Graph;
use crate::backend::{Backend, BackendTx};
use crate::error::Result;
use crate::ids::Millis;
use crate::types::{GcReport, RetentionPolicy, RetentionRule};
use crate::value::Value;

/// The one definition of which nodes a retention rule dooms.
const DOOMED: &str = "SELECT id FROM nodes WHERE kind = ?1 AND valid_from < ?2";

impl<B: Backend> Graph<B> {
    pub async fn gc(&self, policy: &RetentionPolicy) -> Result<GcReport> {
        let mut held = match &self.log {
            Some(log) => Some(HeldLog::acquire(log).await?),
            None => None,
        };
        let now = self.clock.now();
        let mut report = GcReport::default();
        for rule in &policy.rules {
            let doomed = self.doomed_nodes(rule, now).await?;
            tracing::debug!(kind = %rule.kind, doomed = doomed.len(), "gc selected nodes");
            for chunk in doomed.chunks(self.gc_chunk) {
                report.edges_removed += self.sweep_chunk(held.as_mut(), chunk, now).await?;
                report.nodes_removed += chunk.len() as u64;
            }
        }
        // Rows orphaned by some path other than this sweep.
        report.edges_removed += self
            .backend
            .execute(
                "DELETE FROM edges
                 WHERE src NOT IN (SELECT id FROM nodes)
                    OR dst NOT IN (SELECT id FROM nodes)",
                &[],
            )
            .await?;
        self.backend.vector_sweep_orphans().await?;
        if policy.reclaim {
            self.backend
                .execute("PRAGMA incremental_vacuum", &[])
                .await?;
        }
        tracing::info!(?report, "gc swept");
        Ok(report)
    }

    async fn doomed_nodes(&self, rule: &RetentionRule, now: Millis) -> Result<Vec<String>> {
        let cutoff = now.0 - rule.max_age.0;
        let rows = self
            .backend
            .query(DOOMED, &[rule.kind.as_str().into(), cutoff.into()])
            .await?;
        rows.iter().map(|row| row.get_string(0)).collect()
    }

    /// Removes `ids` and the edges that touch them, and returns how many edges
    /// that was. The removal is the tombstone's own projection, so a replay of
    /// the logged chunk deletes exactly what this does.
    async fn sweep_chunk(
        &self,
        held: Option<&mut HeldLog>,
        ids: &[String],
        now: Millis,
    ) -> Result<u64> {
        let targets = ids
            .iter()
            .map(|id| TombstoneTarget {
                table: TombstoneTable::Nodes,
                id: id.clone(),
            })
            .collect();
        let mut tx = self.backend.begin().await?;
        let edges = match edges_touching(&mut *tx, ids).await {
            Ok(edges) => edges,
            Err(error) => {
                abandon(tx).await;
                return Err(error);
            }
        };
        match held {
            Some(held) => {
                let event = tombstone_event(targets, now);
                let steps = steps_for(&event.payload);
                self.append_and_commit(tx, held, event, &steps).await?;
            }
            None => {
                let steps = steps_for(&LogPayload::Tombstone(targets));
                let applied = apply_steps(&mut *tx, &steps, self.backend.vector_delete_sql()).await;
                commit_or_abandon(tx, applied).await?;
            }
        }
        tracing::debug!(nodes = ids.len(), edges, "gc chunk removed");
        Ok(edges)
    }
}

/// How many edges have an end in `ids`, each counted once.
async fn edges_touching(tx: &mut dyn BackendTx, ids: &[String]) -> Result<u64> {
    let marks = (1..=ids.len())
        .map(|n| format!("?{n}"))
        .collect::<Vec<_>>()
        .join(", ");
    let params: Vec<Value> = ids.iter().map(|id| id.as_str().into()).collect();
    let sql = format!("SELECT COUNT(*) FROM edges WHERE src IN ({marks}) OR dst IN ({marks})");
    let rows = tx.query(&sql, &params).await?;
    let count = rows.first().map_or(Ok(0), |row| row.get_i64(0))?;
    Ok(count.unsigned_abs())
}
