// SPDX-License-Identifier: Apache-2.0
//! Rebuilding the projection from the event log alone, and checking the result
//! row by row against what the store held and what the log says should be live.

use std::collections::HashMap;

use futures_util::StreamExt;
use liam_log::event::{EdgeRow, LogPayload, NodeRow, RowEffect, TombstoneTable};
use liam_log::hash::{edge_row_hash, node_row_hash};

use super::logged_write::{commit_or_abandon, HeldLog, SharedLog};
use super::replay::{scan_ahead, CatchUpReport};
use super::{opt_string, row_f64, Graph};
use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::ids::FOREVER;
use crate::types::relation;
use crate::value::Row;

/// What a rebuild does with rows the projection already holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildMode {
    /// Refuse a projection that is not empty.
    RequireEmpty,
    /// Clear the projection first, and check the rebuilt rows against the ones
    /// it held.
    Reset,
    /// Clear the projection first and trust the log over it: rows the store
    /// held that the log does not record are dropped without a mismatch.
    Replace,
}

/// What one successful rebuild did. A mismatch is an error, never a report.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RebuildReport {
    /// The replay of the whole log, including the re-embed pass.
    pub replayed: CatchUpReport,
    /// Live nodes after the rebuild.
    pub live_nodes: usize,
    /// Live edges after the rebuild.
    pub live_edges: usize,
    /// Always true in a report: the multiset of live row hashes was verified.
    pub hash_multiset_ok: bool,
}

/// Empties what the log derives, referencing rows first because the vector
/// table has no cascade. `cluster_state` goes with the assignments, otherwise
/// its fingerprint would vouch for communities that are no longer there.
const RESET_SQL: [&str; 7] = [
    "DELETE FROM node_vectors",
    "DELETE FROM node_community",
    "DELETE FROM cluster_state",
    "DELETE FROM edges",
    "DELETE FROM nodes",
    "DELETE FROM log_hash_index",
    "DELETE FROM log_quarantine",
];

/// Puts the cursor at the start of the log it is given, creating it when the
/// store has none. The store takes on that log's id, which is what lets a store
/// that belonged to another log, or ran ahead of this one, be rebuilt from it.
const RESTART_CURSOR_SQL: &str = "INSERT INTO log_cursor (id, log_id, last_segment, last_index)
     VALUES (1, ?1, NULL, NULL)
     ON CONFLICT(id) DO UPDATE SET
       log_id = excluded.log_id, last_segment = NULL, last_index = NULL";

const LIVE_NODES_SQL: &str = "SELECT id, kind, label, content, producer, attributes, scope, \
     subject, confidence, valid_from, valid_until, tx_from, tx_to FROM nodes WHERE tx_to = ?1";
const LIVE_EDGES_SQL: &str =
    "SELECT id, src, dst, type, attributes, tx_from, tx_to FROM edges WHERE tx_to = ?1";

impl<B: Backend> Graph<B> {
    /// Attaches `log` without checking that the store belongs to it or has not
    /// run past it, so a store those checks refuse can still be rebuilt. The
    /// rebuild adopts the log. Anything else the store does before then is
    /// unchecked.
    pub fn with_log_for_rebuild(mut self, log: SharedLog) -> Self {
        self.log = Some(log);
        self
    }

    /// Empties every table the log derives and moves the log cursor back to the
    /// start of the attached log, then empties the dedup filter.
    pub async fn reset_projection(&self) -> Result<()> {
        let mut held = HeldLog::acquire(self.attached_log()?).await?;
        self.reset(&mut held).await
    }

    /// The sorted multiset of canonical row hashes of every live node and edge.
    pub async fn projection_hash_multiset(&self) -> Result<Vec<[u8; 32]>> {
        let mut hashes = Vec::new();
        for row in self
            .backend
            .query(LIVE_NODES_SQL, &[FOREVER.into()])
            .await?
        {
            hashes.push(node_row_hash(&stored_node(&row)?));
        }
        for row in self
            .backend
            .query(LIVE_EDGES_SQL, &[FOREVER.into()])
            .await?
        {
            hashes.push(edge_row_hash(&stored_edge(&row)?));
        }
        hashes.sort_unstable();
        Ok(hashes)
    }

    /// How many live rows the store holds that the log does not account for,
    /// such as rows written before the log existed. A reset drops them.
    pub async fn rows_unknown_to_log(&self) -> Result<usize> {
        let held = HeldLog::acquire(self.attached_log()?).await?;
        let known = log_expected_multiset(&held).await?;
        let stored = self.projection_hash_multiset().await?;
        Ok(multiset_difference(&stored, &known).0)
    }

    /// Rebuilds the projection from the start of the log and verifies it. The
    /// log lock is held from the reset to the end of the replay, so no write
    /// lands on a half rebuilt store. A failure after the reset leaves the
    /// store as the replay left it, and running the rebuild again starts over.
    pub async fn rebuild_from_log(&self, mode: RebuildMode) -> Result<RebuildReport> {
        let mut held = HeldLog::acquire(self.attached_log()?).await?;
        let before = match mode {
            RebuildMode::RequireEmpty if self.holds_rows().await? => {
                return Err(Error::ProjectionNotEmpty)
            }
            RebuildMode::RequireEmpty => None,
            RebuildMode::Reset => Some(self.projection_hash_multiset().await?),
            RebuildMode::Replace => None,
        };
        self.reset(&mut held).await?;
        let replayed = self.replay(&mut held).await?;
        tracing::info!(?replayed, "rebuild: log replayed");

        let rebuilt = self.projection_hash_multiset().await?;
        verify(&log_expected_multiset(&held).await?, &rebuilt)?;
        if let Some(before) = before {
            verify(&before, &rebuilt)?;
        }
        tracing::info!(rows = rebuilt.len(), "rebuild: rebuilt rows verified");
        drop(held);

        Ok(RebuildReport {
            replayed: self.reembedded(replayed).await,
            live_nodes: self.live_count("nodes").await?,
            live_edges: self.live_count("edges").await?,
            hash_multiset_ok: true,
        })
    }

    fn attached_log(&self) -> Result<&SharedLog> {
        self.log.as_ref().ok_or(Error::NoEventLog)
    }

    async fn holds_rows(&self) -> Result<bool> {
        let rows = self
            .backend
            .query(
                "SELECT EXISTS(SELECT 1 FROM nodes) OR EXISTS(SELECT 1 FROM edges)",
                &[],
            )
            .await?;
        Ok(rows.first().map(|row| row.get_i64(0)).transpose()? == Some(1))
    }

    async fn live_count(&self, table: &str) -> Result<usize> {
        let rows = self
            .backend
            .query(
                &format!("SELECT COUNT(*) FROM {table} WHERE tx_to = ?1"),
                &[FOREVER.into()],
            )
            .await?;
        let count = rows.first().map(|row| row.get_i64(0)).transpose()?;
        Ok(usize::try_from(count.unwrap_or(0)).unwrap_or(0))
    }

    /// One transaction, so a failure leaves the old projection whole. The dedup
    /// filter is emptied only once the index it mirrors is gone.
    async fn reset(&self, held: &mut HeldLog) -> Result<()> {
        let mut tx = self.backend.begin().await?;
        let cleared = async {
            for sql in RESET_SQL {
                tx.execute(sql, &[]).await?;
            }
            tx.execute(RESTART_CURSOR_SQL, &[held.log_id.to_string().into()])
                .await?;
            Ok(())
        }
        .await;
        commit_or_abandon(tx, cleared).await?;
        held.forget_hashes();
        tracing::info!("rebuild: projection reset, cursor back at the start of the log");
        Ok(())
    }
}

/// The node a stored row holds. The table does not record whether `valid_from`
/// was supplied, so it is hashed as supplied, which is how a log row is hashed
/// when the two are compared.
fn stored_node(row: &Row) -> Result<NodeRow> {
    Ok(NodeRow {
        id: row.get_string(0)?,
        kind: row.get_string(1)?,
        label: row.get_string(2)?,
        content: row.get_string(3)?,
        producer: row.get_string(4)?,
        attributes: row.get_string(5)?,
        scope: opt_string(row, 6)?,
        subject: opt_string(row, 7)?,
        confidence: row_f64(row, 8),
        valid_from: row.get_i64(9)?,
        valid_from_supplied: true,
        valid_until: row.get_i64(10)?,
        tx_from: row.get_i64(11)?,
        tx_to: row.get_i64(12)?,
    })
}

fn stored_edge(row: &Row) -> Result<EdgeRow> {
    Ok(EdgeRow {
        id: row.get_string(0)?,
        src: row.get_string(1)?,
        dst: row.get_string(2)?,
        edge_type: row.get_string(3)?,
        attributes: row.get_string(4)?,
        tx_from: row.get_i64(5)?,
        tx_to: row.get_i64(6)?,
    })
}

/// Fails when `found` is not the multiset `expected`, naming how far apart they
/// are even when the counts agree.
fn verify(expected: &[[u8; 32]], found: &[[u8; 32]]) -> Result<()> {
    let (missing, unexpected) = multiset_difference(expected, found);
    if missing + unexpected == 0 {
        return Ok(());
    }
    Err(Error::RebuildMismatch {
        expected: expected.len(),
        found: found.len(),
        missing,
        unexpected,
    })
}

/// How many entries of the sorted multiset `held` are absent from `wanted`, and
/// how many of `wanted` are absent from `held`. A hash held twice counts twice.
fn multiset_difference(held: &[[u8; 32]], wanted: &[[u8; 32]]) -> (usize, usize) {
    let (mut held_only, mut wanted_only) = (0, 0);
    let (mut held, mut wanted) = (held.iter().peekable(), wanted.iter().peekable());
    while let (Some(h), Some(w)) = (held.peek(), wanted.peek()) {
        match h.cmp(w) {
            std::cmp::Ordering::Less => {
                held_only += 1;
                held.next();
            }
            std::cmp::Ordering::Greater => {
                wanted_only += 1;
                wanted.next();
            }
            std::cmp::Ordering::Equal => {
                held.next();
                wanted.next();
            }
        }
    }
    (held_only + held.count(), wanted_only + wanted.count())
}

/// The sorted multiset of live rows the log alone describes: every event that
/// was not voided, applied in order. It is read from the log's own rows instead
/// of the projection, so it can catch a replay that dropped or invented one.
async fn log_expected_multiset(held: &HeldLog) -> Result<Vec<[u8; 32]>> {
    let Some(head) = held.head()? else {
        return Ok(Vec::new());
    };
    let reader = held.reader()?;
    let voided = scan_ahead(reader.scan_through(None, head)).await?.voided;
    let mut rows = LoggedRows::default();
    let mut records = reader.scan_through(None, head);
    while let Some(record) = records.next().await {
        let event = record?.event;
        if !voided.contains(&event.event_id) {
            rows.apply(event.payload);
        }
    }
    Ok(rows.live_hashes())
}

/// The rows the log has produced so far, by id.
#[derive(Default)]
struct LoggedRows {
    nodes: HashMap<String, NodeRow>,
    edges: HashMap<String, EdgeRow>,
}

impl LoggedRows {
    fn apply(&mut self, payload: LogPayload) {
        match payload {
            LogPayload::NodeWrite(row) => self.add_node(row),
            LogPayload::EdgeWrite(row) => self.add_edge(row),
            LogPayload::EpisodeBatch(effects) => {
                for effect in effects {
                    match effect {
                        RowEffect::Node(row) => self.add_node(row),
                        RowEffect::Edge(row) => self.add_edge(row),
                    }
                }
            }
            LogPayload::Tombstone(targets) => {
                for target in targets {
                    self.remove(target.table, &target.id);
                }
            }
            LogPayload::DuplicateOf { .. } | LogPayload::Voided { .. } => {}
        }
    }

    fn add_node(&mut self, row: NodeRow) {
        self.nodes.insert(row.id.clone(), row);
    }

    fn add_edge(&mut self, row: EdgeRow) {
        if row.edge_type == relation::SUPERSEDES {
            if let Some(closed) = self.nodes.get_mut(&row.dst) {
                closed.tx_to = row.tx_from;
            }
        }
        self.edges.insert(row.id.clone(), row);
    }

    fn remove(&mut self, table: TombstoneTable, id: &str) {
        match table {
            TombstoneTable::Nodes => {
                self.nodes.remove(id);
                self.edges
                    .retain(|_, edge| edge.src != id && edge.dst != id);
            }
            TombstoneTable::Edges => {
                self.edges.remove(id);
            }
            TombstoneTable::NodeCommunity => {}
        }
    }

    fn live_hashes(self) -> Vec<[u8; 32]> {
        let nodes = self
            .nodes
            .into_values()
            .filter(|row| row.tx_to == FOREVER.0);
        let edges = self
            .edges
            .into_values()
            .filter(|row| row.tx_to == FOREVER.0);
        let mut hashes: Vec<_> = nodes
            .map(|row| {
                node_row_hash(&NodeRow {
                    valid_from_supplied: true,
                    ..row
                })
            })
            .chain(edges.map(|row| edge_row_hash(&row)))
            .collect();
        hashes.sort_unstable();
        hashes
    }
}
