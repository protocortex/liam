// SPDX-License-Identifier: Apache-2.0
//! Copying the rows a store held before it had a log into that log, once, so a
//! rebuild from the log alone can recreate them.
//!
//! The rows are logged in the order a replay needs them: nodes, then plain
//! edges, then `supersedes` edges, because a replayed `supersedes` edge closes
//! its target and a plain edge needs both of its ends open. A superseded node is
//! logged open and left to its `supersedes` edge to close, as it was written.

use std::collections::HashSet;
use std::sync::Arc;

use futures_util::StreamExt;
use liam_log::event::{LogEvent, LogPayload, RowEffect};
use liam_log::hash::{edge_row_hash, node_row_hash};
use liam_log::reader::LogReader;
use liam_log::LogOffset;

use super::logged_plan::{event, STORE_TRUST};
use super::logged_write::{commit_or_abandon, HeldLog};
use super::projection::Step;
use super::rebuild::{stored_edge, stored_node, EDGE_COLUMNS, NODE_COLUMNS};
use super::replay::scan_ahead;
use super::{opt_string, opt_text, Graph};
use crate::backend::{Backend, BackendTx};
use crate::error::{Error, Result};
use crate::ids::{Millis, FOREVER};
use crate::types::relation;
use crate::value::{Row, Value};

/// Rows backfilled between two writes of the resume point.
const BACKFILL_BATCH: usize = 500;

const BACKFILL_SOURCE: &str = "backfill";

const PROGRESS_SQL: &str = "INSERT INTO backfill_state (id, phase, last_id, completed_at)
     VALUES (1, ?1, ?2, ?3)
     ON CONFLICT(id) DO UPDATE SET
       phase = excluded.phase, last_id = excluded.last_id,
       completed_at = excluded.completed_at";

/// What one backfill run appended to the log.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BackfillReport {
    /// Node events appended, superseded nodes included.
    pub nodes: usize,
    /// Edge events appended.
    pub edges: usize,
    /// Whether the run continued an interrupted one instead of starting over.
    pub resumed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Nodes,
    Edges,
    Supersedes,
}

impl Phase {
    const IN_ORDER: [Phase; 3] = [Phase::Nodes, Phase::Edges, Phase::Supersedes];

    fn name(self) -> &'static str {
        match self {
            Phase::Nodes => "nodes",
            Phase::Edges => "edges",
            Phase::Supersedes => "supersedes",
        }
    }

    fn parse(name: &str) -> Result<Self> {
        Self::IN_ORDER
            .into_iter()
            .find(|phase| phase.name() == name)
            .ok_or_else(|| {
                Error::CorruptLogState(format!("backfill_state holds the phase {name:?}"))
            })
    }

    /// The next `limit` rows of this phase after `after`, by id.
    fn page_query(self, after: &str, limit: usize) -> (String, Vec<Value>) {
        let (after, limit) = (after.into(), Value::Int(limit as i64));
        match self {
            Phase::Nodes => (
                format!("SELECT {NODE_COLUMNS} FROM nodes WHERE id > ?1 ORDER BY id LIMIT ?2"),
                vec![after, limit],
            ),
            Phase::Edges | Phase::Supersedes => {
                let op = if self == Phase::Edges { "!=" } else { "=" };
                let sql = format!(
                    "SELECT {EDGE_COLUMNS} FROM edges
                     WHERE id > ?1 AND tx_to = ?3 AND type {op} ?4 ORDER BY id LIMIT ?2"
                );
                let params = vec![after, limit, FOREVER.into(), relation::SUPERSEDES.into()];
                (sql, params)
            }
        }
    }
}

/// How far the backfill has got: `last_id` is the last row of `phase` accounted
/// for, and `completed_at` is set once every row is in the log.
#[derive(Debug)]
pub(super) struct Progress {
    phase: Phase,
    last_id: Option<String>,
    completed_at: Option<i64>,
}

impl Progress {
    fn start() -> Self {
        Self {
            phase: Phase::Nodes,
            last_id: None,
            completed_at: None,
        }
    }

    fn at(phase: Phase, last_id: &str) -> Self {
        Self {
            phase,
            last_id: Some(last_id.to_owned()),
            completed_at: None,
        }
    }

    fn complete(now: Millis) -> Self {
        Self {
            phase: Phase::Supersedes,
            last_id: None,
            completed_at: Some(now.0),
        }
    }

    fn from_row(row: &Row) -> Result<Self> {
        Ok(Self {
            phase: Phase::parse(&row.get_string(0)?)?,
            last_id: opt_string(row, 1)?,
            completed_at: match row.0[2] {
                Value::Null => None,
                _ => Some(row.get_i64(2)?),
            },
        })
    }

    pub(super) async fn save(&self, tx: &mut dyn BackendTx) -> Result<()> {
        let params = [
            self.phase.name().into(),
            opt_text(self.last_id.clone()),
            self.completed_at.map_or(Value::Null, Value::Int),
        ];
        tx.execute(PROGRESS_SQL, &params).await?;
        Ok(())
    }
}

/// A row waiting to be logged, with the id the walk resumes after.
struct Pending {
    id: String,
    event: LogEvent,
}

impl<B: Backend> Graph<B> {
    /// Logs every row the log does not hold yet, then marks the backfill complete.
    /// The log lock is held throughout, so no write lands between the rows.
    pub async fn backfill_log_from_projection(&self) -> Result<BackfillReport> {
        self.backfill_in_batches(BACKFILL_BATCH).await
    }

    /// Whether the backfill has not completed for this store.
    pub async fn needs_backfill(&self) -> Result<bool> {
        self.log.as_ref().ok_or(Error::NoEventLog)?;
        let saved = self.saved_progress().await?;
        Ok(!saved.is_some_and(|progress| progress.completed_at.is_some()))
    }

    pub(super) async fn backfill_in_batches(&self, batch: usize) -> Result<BackfillReport> {
        let log = self.log.as_ref().ok_or(Error::NoEventLog)?;
        let mut held = HeldLog::acquire(log).await?;
        let saved = self.saved_progress().await?;
        if saved.as_ref().is_some_and(|p| p.completed_at.is_some()) {
            return Ok(BackfillReport::default());
        }
        let mut report = BackfillReport {
            resumed: saved.is_some(),
            ..BackfillReport::default()
        };
        let start = saved.unwrap_or_else(Progress::start);
        // Settles any row of an interrupted run that the log holds past the cursor.
        self.replay(&mut held).await?;
        let logged = logged_row_ids(held.reader()?, held.head()?).await?;
        let now = self.clock.now();
        tracing::info!(
            logged = logged.len(),
            resumed = report.resumed,
            "backfill: starting"
        );

        let mut unrecorded = 0;
        for phase in Phase::IN_ORDER
            .into_iter()
            .skip_while(|p| *p != start.phase)
        {
            let mut after = match phase == start.phase {
                true => start.last_id.clone().unwrap_or_default(),
                false => String::new(),
            };
            loop {
                let page = self.pending_page(phase, &after, batch, now).await?;
                let Some(last) = page.last() else { break };
                after = last.id.clone();
                for Pending { id, event } in page {
                    if logged.contains(&id) {
                        continue;
                    }
                    unrecorded += 1;
                    let steps = if unrecorded == batch {
                        unrecorded = 0;
                        vec![Step::BackfillProgress(Progress::at(phase, &id))]
                    } else {
                        Vec::new()
                    };
                    let tx = self.backend.begin().await?;
                    self.append_and_commit(tx, &mut held, event, &steps).await?;
                    match phase {
                        Phase::Nodes => report.nodes += 1,
                        _ => report.edges += 1,
                    }
                }
            }
            tracing::info!(phase = phase.name(), "backfill: phase logged");
        }
        self.record_completion(now).await?;
        tracing::info!(?report, "backfill: complete");
        Ok(report)
    }

    async fn saved_progress(&self) -> Result<Option<Progress>> {
        let rows = self
            .backend
            .query(
                "SELECT phase, last_id, completed_at FROM backfill_state WHERE id = 1",
                &[],
            )
            .await?;
        rows.first().map(Progress::from_row).transpose()
    }

    async fn record_completion(&self, now: Millis) -> Result<()> {
        let mut tx = self.backend.begin().await?;
        let saved = Progress::complete(now).save(&mut *tx).await;
        commit_or_abandon(tx, saved).await
    }

    async fn pending_page(
        &self,
        phase: Phase,
        after: &str,
        limit: usize,
        now: Millis,
    ) -> Result<Vec<Pending>> {
        let (sql, params) = phase.page_query(after, limit);
        let rows = self.backend.query(&sql, &params).await?;
        let mut page = Vec::with_capacity(rows.len());
        for row in &rows {
            page.push(match phase {
                Phase::Nodes => self.node_pending(row, now).await?,
                Phase::Edges | Phase::Supersedes => edge_pending(row, now)?,
            });
        }
        Ok(page)
    }

    /// A node closed by a `supersedes` edge is logged open, and replaying that
    /// edge closes it at the same time. One with no such edge would stay open
    /// after a rebuild, so it is refused.
    async fn node_pending(&self, row: &Row, now: Millis) -> Result<Pending> {
        let mut node = stored_node(row)?;
        if node.tx_to != FOREVER.0 {
            let closers = self
                .backend
                .query(
                    "SELECT 1 FROM edges
                     WHERE dst = ?1 AND type = ?2 AND tx_from = ?3 AND tx_to = ?4",
                    &[
                        node.id.as_str().into(),
                        relation::SUPERSEDES.into(),
                        node.tx_to.into(),
                        FOREVER.into(),
                    ],
                )
                .await?;
            if closers.is_empty() {
                return Err(Error::BackfillUnreplayable(format!(
                    "node {} is closed but no live supersedes edge closes it",
                    node.id
                )));
            }
            node.tx_to = FOREVER.0;
        }
        let stamp = (BACKFILL_SOURCE, node.confidence);
        Ok(Pending {
            id: node.id.clone(),
            event: event(
                node_row_hash(&node),
                stamp,
                node.valid_from,
                now,
                LogPayload::NodeWrite(node),
            ),
        })
    }
}

fn edge_pending(row: &Row, now: Millis) -> Result<Pending> {
    let edge = stored_edge(row)?;
    let stamp = (BACKFILL_SOURCE, STORE_TRUST);
    Ok(Pending {
        id: edge.id.clone(),
        event: event(
            edge_row_hash(&edge),
            stamp,
            edge.tx_from,
            now,
            LogPayload::EdgeWrite(edge),
        ),
    })
}

/// The ids of every row the log holds and no later record cancels, whoever
/// wrote it: scanned rather than read from the hash index, which names only the
/// last of several rows with the same content.
async fn logged_row_ids(
    reader: Arc<dyn LogReader>,
    head: Option<LogOffset>,
) -> Result<HashSet<String>> {
    let Some(head) = head else {
        return Ok(HashSet::new());
    };
    let voided = scan_ahead(reader.scan_through(None, head))
        .await?
        .into_voided();
    let mut ids = HashSet::new();
    let mut records = reader.scan_through(None, head);
    while let Some(record) = records.next().await {
        let event = record?.event;
        if voided.contains(&event.event_id) {
            continue;
        }
        ids.extend(written_ids(event.payload));
    }
    Ok(ids)
}

fn written_ids(payload: LogPayload) -> Vec<String> {
    match payload {
        LogPayload::NodeWrite(row) => vec![row.id],
        LogPayload::EdgeWrite(row) => vec![row.id],
        LogPayload::EpisodeBatch(effects) => effects
            .into_iter()
            .map(|effect| match effect {
                RowEffect::Node(row) => row.id,
                RowEffect::Edge(row) => row.id,
            })
            .collect(),
        _ => Vec::new(),
    }
}
