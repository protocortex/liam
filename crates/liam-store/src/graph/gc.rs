// SPDX-License-Identifier: Apache-2.0
//! The retention sweep. Per rule, the doomed nodes are chosen first and then
//! removed in chunks, each chunk in one transaction through the same removal
//! the log's tombstones replay. With a log attached each chunk is tombstoned
//! before it is deleted, and the log lock is held from the first selection to
//! the last chunk, orphan edges included, so no write can land between the two.
//! A very large sweep therefore blocks every logged writer for its duration:
//! chunking bounds each event and transaction, not the wait.

use std::collections::HashSet;

use liam_log::event::{TombstoneTable, TombstoneTarget};

use super::logged_plan::{edges_of_node, tombstone_event};
use super::logged_write::{abandon, commit_or_abandon, HeldLog};
use super::projection::{apply_steps, steps_for};
use super::Graph;
use crate::backend::{Backend, BackendTx};
use crate::error::Result;
use crate::ids::Millis;
use crate::types::{GcReport, RetentionPolicy, RetentionRule};

/// Targets one tombstone event carries at most, which also bounds the length of
/// the transaction that applies it.
const GC_CHUNK_TARGETS: usize = 500;

/// The one definition of which nodes a retention rule dooms.
const DOOMED: &str = "SELECT id FROM nodes WHERE kind = ?1 AND valid_from < ?2";

/// Edges with an end that is no longer a node, left by some path other than
/// this sweep.
const ORPHAN_EDGES: &str = "src NOT IN (SELECT id FROM nodes) OR dst NOT IN (SELECT id FROM nodes)";

/// The ids of the edges that go with one node, as `Table::removal_sql` removes them.
const NODE_EDGE_IDS: &str = concat!("SELECT id FROM edges WHERE ", edges_of_node!());

impl<B: Backend> Graph<B> {
    pub async fn gc(&self, policy: &RetentionPolicy) -> Result<GcReport> {
        self.sweep(policy, GC_CHUNK_TARGETS).await
    }

    pub(super) async fn sweep(&self, policy: &RetentionPolicy, chunk: usize) -> Result<GcReport> {
        let mut held = match &self.log {
            Some(log) => Some(HeldLog::acquire(log).await?),
            None => None,
        };
        let now = self.clock.now();
        let mut report = GcReport::default();
        for rule in &policy.rules {
            let doomed = self.doomed_nodes(rule, now).await?;
            tracing::debug!(kind = %rule.kind, doomed = doomed.len(), "gc selected nodes");
            for ids in doomed.chunks(chunk) {
                let nodes = TombstoneTable::Nodes;
                report.edges_removed += self.sweep_chunk(held.as_mut(), nodes, ids, now).await?;
                report.nodes_removed += ids.len() as u64;
            }
        }
        report.edges_removed += self.sweep_orphan_edges(held.as_mut(), chunk, now).await?;
        // The rest is not a log write, so writers need not wait for it.
        drop(held);
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

    /// Removes the orphaned edges and returns how many there were. A rebuild
    /// replays an edge's write, so with a log the removal is tombstoned too.
    async fn sweep_orphan_edges(
        &self,
        held: Option<&mut HeldLog>,
        chunk: usize,
        now: Millis,
    ) -> Result<u64> {
        let Some(held) = held else {
            let delete = format!("DELETE FROM edges WHERE {ORPHAN_EDGES}");
            return self.backend.execute(&delete, &[]).await;
        };
        let select = format!("SELECT id FROM edges WHERE {ORPHAN_EDGES}");
        let rows = self.backend.query(&select, &[]).await?;
        let orphans = rows
            .iter()
            .map(|row| row.get_string(0))
            .collect::<Result<Vec<_>>>()?;
        let mut removed = 0;
        for ids in orphans.chunks(chunk) {
            let edges = TombstoneTable::Edges;
            removed += self.sweep_chunk(Some(&mut *held), edges, ids, now).await?;
        }
        Ok(removed)
    }

    /// Removes the `table` rows `ids` and returns how many edges that deleted.
    /// The removal is the tombstone's own projection, so a replay of the logged
    /// chunk deletes exactly what this does.
    async fn sweep_chunk(
        &self,
        held: Option<&mut HeldLog>,
        table: TombstoneTable,
        ids: &[String],
        now: Millis,
    ) -> Result<u64> {
        let targets = ids
            .iter()
            .map(|id| TombstoneTarget {
                table,
                id: id.clone(),
            })
            .collect();
        let event = tombstone_event(targets, now);
        let steps = steps_for(&event.payload);
        let mut tx = self.backend.begin().await?;
        let edges = match table {
            TombstoneTable::Nodes => edges_of_nodes(&mut *tx, ids).await,
            TombstoneTable::Edges => Ok(ids.len() as u64),
            TombstoneTable::NodeCommunity => Ok(0),
        };
        let edges = match edges {
            Ok(edges) => edges,
            Err(error) => {
                abandon(tx).await;
                return Err(error);
            }
        };
        match held {
            Some(held) => self.append_and_commit(tx, held, event, &steps).await?,
            None => {
                let applied = apply_steps(&mut *tx, &steps, self.backend.vector_delete_sql()).await;
                commit_or_abandon(tx, applied).await?;
            }
        }
        tracing::debug!(rows = ids.len(), edges, "gc chunk removed");
        Ok(edges)
    }
}

/// How many edges have an end in `ids`, each counted once even when both ends
/// are in `ids`.
async fn edges_of_nodes(tx: &mut dyn BackendTx, ids: &[String]) -> Result<u64> {
    let mut edges = HashSet::new();
    for id in ids {
        for row in tx.query(NODE_EDGE_IDS, &[id.as_str().into()]).await? {
            edges.insert(row.get_string(0)?);
        }
    }
    Ok(edges.len() as u64)
}
