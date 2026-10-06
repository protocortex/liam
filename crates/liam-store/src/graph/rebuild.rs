// SPDX-License-Identifier: Apache-2.0
//! Rebuilding the projection from the event log alone, and checking the result
//! row by row against what the log says should be live.

use std::collections::HashMap;

use futures_util::StreamExt;
use liam_log::event::{EdgeRow, LogPayload, NodeRow, RowEffect, TombstoneTable};
use liam_log::hash::{edge_row_hash, node_row_hash};
use uuid::Uuid;

use super::log_cursor;
use super::logged_write::{commit_or_abandon, HeldLog, SharedLog};
use super::replay::{scan_ahead, CatchUpReport};
use super::{opt_string, row_f64, Graph};
use crate::backend::Backend;
use crate::error::{Error, MismatchSource, Result};
use crate::ids::FOREVER;
use crate::schema::{Clear, DERIVED_TABLES};
use crate::types::relation;
use crate::value::Row;

/// What a rebuild does with rows the projection already holds. A mode that
/// clears the projection refuses first, under the log lock, whatever it would
/// destroy that the log cannot give back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildMode {
    /// Refuses a projection that holds any row.
    RequireEmpty,
    /// Clears the projection only when its live rows are exactly what the log
    /// says should be live, and otherwise refuses with it untouched. A store
    /// that is merely behind the log differs, so it is refused too.
    ResetChecked,
    /// Clears the projection and trusts the log over it: rows the log does not
    /// record are dropped. It still refuses to wipe rows for an empty log, or to
    /// take on a different log than the one the store was built from.
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
}

/// The sorted hashes of the live rows, with how many are nodes and edges.
struct LiveRows {
    hashes: Vec<[u8; 32]>,
    nodes: usize,
    edges: usize,
}

const LIVE_NODES_SQL: &str = "SELECT id, kind, label, content, producer, attributes, scope, \
     subject, confidence, valid_from, valid_until, tx_from, tx_to FROM nodes WHERE tx_to = ?1";
const LIVE_EDGES_SQL: &str =
    "SELECT id, src, dst, type, attributes, tx_from, tx_to FROM edges WHERE tx_to = ?1";

impl<B: Backend> Graph<B> {
    /// The sorted multiset of canonical row hashes of every live node and edge.
    pub async fn projection_hash_multiset(&self) -> Result<Vec<[u8; 32]>> {
        Ok(self.live_rows().await?.hashes)
    }

    /// How many live rows the store holds that the log does not account for,
    /// such as rows written before the log existed. `Replace` drops them.
    pub async fn rows_unknown_to_log(&self) -> Result<usize> {
        let log = self.log.as_ref().ok_or(Error::NoEventLog)?;
        let held = HeldLog::acquire(log).await?;
        self.unknown_rows(&held).await
    }

    /// Rebuilds the projection from the start of `log`, verifies it against the
    /// log, and returns the store attached to `log`. A store the mode refuses
    /// is untouched, and one whose replay fails is dropped with the error, so a
    /// store that was not checked never escapes.
    ///
    /// The caller must hold the exclusive store lock and run no `gc` or other
    /// writer for the whole call, as the CLI does: the log lock only orders
    /// logged writes. An interrupted rebuild leaves the store on its log with
    /// the cursor on the last record applied, so `catch_up` resumes it, and
    /// `Replace` starts over. `ResetChecked` refuses such a partial store and
    /// leaves it as it is.
    pub async fn rebuild_from_log(
        mut self,
        log: SharedLog,
        mode: RebuildMode,
    ) -> Result<(Self, RebuildReport)> {
        let report = self.rebuild(&log, mode).await?;
        self.log = Some(log);
        Ok((self, report))
    }

    async fn rebuild(&self, log: &SharedLog, mode: RebuildMode) -> Result<RebuildReport> {
        let mut held = HeldLog::acquire(log).await?;
        let previous_log = log_cursor::read(&self.backend)
            .await?
            .map(|cursor| cursor.log_id)
            .filter(|id| *id != held.log_id);
        self.check_before_reset(&held, mode, previous_log).await?;
        tracing::info!(?mode, "rebuild: checks passed, resetting the projection");
        self.reset(&mut held, previous_log.is_some()).await?;
        let replayed = self.replay(&mut held).await?;
        tracing::info!(?replayed, "rebuild: log replayed");

        let rebuilt = self.live_rows().await?;
        let expected = log_expected_multiset(&held).await?;
        verify(MismatchSource::Log, &expected, &rebuilt.hashes)?;
        tracing::info!(
            rows = rebuilt.hashes.len(),
            "rebuild: rebuilt rows verified"
        );
        drop(held);

        Ok(RebuildReport {
            replayed: self.reembedded(replayed).await,
            live_nodes: rebuilt.nodes,
            live_edges: rebuilt.edges,
        })
    }

    /// Refuses, before anything is deleted, a rebuild that would destroy rows
    /// the log cannot give back. `previous_log` is the log the store belonged
    /// to when that is another one.
    async fn check_before_reset(
        &self,
        held: &HeldLog,
        mode: RebuildMode,
        previous_log: Option<Uuid>,
    ) -> Result<()> {
        match mode {
            RebuildMode::RequireEmpty if self.holds_rows().await? => Err(Error::ProjectionNotEmpty),
            RebuildMode::RequireEmpty => Ok(()),
            RebuildMode::ResetChecked => {
                let expected = log_expected_multiset(held).await?;
                let stored = self.live_rows().await?;
                verify(MismatchSource::PriorProjection, &expected, &stored.hashes)
            }
            RebuildMode::Replace if !self.holds_rows().await? => Ok(()),
            RebuildMode::Replace => {
                if let Some(store) = previous_log {
                    return Err(Error::ForeignLogWouldWipe {
                        store,
                        log: held.log_id,
                        rows_unknown_to_log: self.unknown_rows(held).await?,
                    });
                }
                if held.head()?.is_none() {
                    return Err(Error::EmptyLogWouldWipe {
                        rows_unknown_to_log: self.unknown_rows(held).await?,
                    });
                }
                Ok(())
            }
        }
    }

    async fn unknown_rows(&self, held: &HeldLog) -> Result<usize> {
        let known = log_expected_multiset(held).await?;
        let stored = self.live_rows().await?;
        Ok(multiset_difference(&stored.hashes, &known).0)
    }

    /// Any row at all, superseded ones included.
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

    async fn live_rows(&self) -> Result<LiveRows> {
        let live = [FOREVER.into()];
        let mut hashes = Vec::new();
        for row in self.backend.query(LIVE_NODES_SQL, &live).await? {
            hashes.push(node_row_hash(&stored_node(&row)?));
        }
        let nodes = hashes.len();
        for row in self.backend.query(LIVE_EDGES_SQL, &live).await? {
            hashes.push(edge_row_hash(&stored_edge(&row)?));
        }
        let edges = hashes.len() - nodes;
        hashes.sort_unstable();
        Ok(LiveRows {
            hashes,
            nodes,
            edges,
        })
    }

    /// One transaction, so a failure leaves the old projection whole. The dedup
    /// filter is emptied only once the index it mirrors is gone. `adopting`
    /// says the store takes on another log, which also drops what the old one
    /// left behind.
    pub(super) async fn reset(&self, held: &mut HeldLog, adopting: bool) -> Result<()> {
        // The vector table has no cascade, so it goes first.
        let statements = self
            .backend
            .vector_clear_sql()
            .map(str::to_owned)
            .into_iter()
            .chain(
                DERIVED_TABLES
                    .iter()
                    .filter(|(_, clear)| adopting || *clear == Clear::Always)
                    .map(|(table, _)| format!("DELETE FROM {table}")),
            );
        let mut tx = self.backend.begin().await?;
        let cleared = async {
            for sql in statements {
                tx.execute(&sql, &[]).await?;
            }
            log_cursor::restart_in_tx(&mut *tx, held.log_id).await
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
fn verify(against: MismatchSource, expected: &[[u8; 32]], found: &[[u8; 32]]) -> Result<()> {
    let (missing, unexpected) = multiset_difference(expected, found);
    if missing + unexpected == 0 {
        return Ok(());
    }
    Err(Error::RebuildMismatch {
        against,
        expected: expected.len(),
        found: found.len(),
        missing,
        unexpected,
    })
}

/// How many entries of the sorted multiset `left` are absent from `right`, and
/// how many of `right` are absent from `left`. A hash held twice counts twice.
fn multiset_difference(left: &[[u8; 32]], right: &[[u8; 32]]) -> (usize, usize) {
    let (mut only_left, mut only_right) = (0, 0);
    let (mut left, mut right) = (left.iter().peekable(), right.iter().peekable());
    while let (Some(l), Some(r)) = (left.peek(), right.peek()) {
        match l.cmp(r) {
            std::cmp::Ordering::Less => {
                only_left += 1;
                left.next();
            }
            std::cmp::Ordering::Greater => {
                only_right += 1;
                right.next();
            }
            std::cmp::Ordering::Equal => {
                left.next();
                right.next();
            }
        }
    }
    (only_left + left.count(), only_right + right.count())
}

/// The sorted multiset of live rows the log alone describes: every event that
/// was not voided, applied in order. It is read from the log's own rows instead
/// of the projection, so it can catch a replay that dropped or invented one.
async fn log_expected_multiset(held: &HeldLog) -> Result<Vec<[u8; 32]>> {
    let Some(head) = held.head()? else {
        return Ok(Vec::new());
    };
    let reader = held.reader()?;
    let voided = scan_ahead(reader.scan_through(None, head))
        .await?
        .into_voided();
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

/// The rows the log has produced so far, kept as little as the verdict needs:
/// the hash of each row that is live, by id.
#[derive(Default)]
struct LoggedRows {
    /// `None` once the node is closed.
    nodes: HashMap<String, Option<[u8; 32]>>,
    edges: HashMap<String, Option<[u8; 32]>>,
    /// The ids of the edges touching each node, so a node's tombstone removes
    /// them without a scan. An edge removed through its other end or on its own
    /// stays listed, and removing it again changes nothing.
    edges_of: HashMap<String, Vec<String>>,
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

    fn add_node(&mut self, mut row: NodeRow) {
        row.valid_from_supplied = true;
        let live = (row.tx_to == FOREVER.0).then(|| node_row_hash(&row));
        self.nodes.insert(row.id, live);
    }

    fn add_edge(&mut self, row: EdgeRow) {
        if row.edge_type == relation::SUPERSEDES {
            if let Some(closed) = self.nodes.get_mut(&row.dst) {
                *closed = None;
            }
        }
        for end in [&row.src, &row.dst] {
            self.edges_of
                .entry(end.clone())
                .or_default()
                .push(row.id.clone());
        }
        let live = (row.tx_to == FOREVER.0).then(|| edge_row_hash(&row));
        self.edges.insert(row.id, live);
    }

    fn remove(&mut self, table: TombstoneTable, id: &str) {
        match table {
            TombstoneTable::Nodes => {
                self.nodes.remove(id);
                for edge in self.edges_of.remove(id).unwrap_or_default() {
                    self.edges.remove(&edge);
                }
            }
            TombstoneTable::Edges => {
                self.edges.remove(id);
            }
            TombstoneTable::NodeCommunity => {}
        }
    }

    fn live_hashes(self) -> Vec<[u8; 32]> {
        let mut hashes: Vec<_> = self
            .nodes
            .into_values()
            .chain(self.edges.into_values())
            .flatten()
            .collect();
        hashes.sort_unstable();
        hashes
    }
}
