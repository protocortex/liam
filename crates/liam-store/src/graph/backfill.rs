// SPDX-License-Identifier: Apache-2.0
//! Copying the rows a store held before it had a log into that log, once, so a
//! rebuild from the log alone can recreate them.
//!
//! The rows are logged in the order a replay needs them: nodes, then plain
//! edges, then `supersedes` edges, because a replayed `supersedes` edge closes
//! its target and a plain edge needs both of its ends open. A superseded node is
//! logged open and left to its `supersedes` edge to close, as it was written.
//!
//! A closed node that no live `supersedes` edge closes, such as one whose
//! superseder a retention sweep removed, is logged as it is stored. A live edge
//! that a replay would refuse because of it cannot be logged, so it is skipped
//! and counted: a rebuild then lacks that edge, and its verification names it
//! as unexpected for as long as the store still holds it.

use std::collections::HashSet;
use std::sync::Arc;

use liam_log::event::{LogEvent, LogPayload, RowEffect};
use liam_log::hash::{edge_row_hash, node_row_hash};
use liam_log::reader::LogReader;
use liam_log::LogOffset;

use super::logged_plan::{event, STORE_TRUST};
use super::logged_write::{abandon, commit_or_abandon, HeldLog};
use super::rebuild::{stored_edge, stored_node, EDGE_COLUMNS, NODE_COLUMNS};
use super::replay::{for_each_unvoided, row_effects};
use super::Graph;
use crate::backend::{Backend, BackendTx};
use crate::error::{Error, Result};
use crate::ids::{Millis, FOREVER};
use crate::types::relation;
use crate::value::{Row, Value};

/// Rows one query reads, and rows logged between two saves of the resume point.
const BACKFILL_BATCH: usize = 500;

const BACKFILL_SOURCE: &str = "backfill";

/// What one backfill run did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BackfillReport {
    /// Node events appended, superseded nodes included.
    pub nodes_logged: usize,
    /// Edge events appended.
    pub edges_logged: usize,
    /// Live edges left out because a replay would refuse them.
    pub edges_skipped: usize,
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
                Error::CorruptLogState(format!("log_backfill_state holds the phase {name:?}"))
            })
    }

    /// The next `limit` rows of this phase after `after`, by id, each with one
    /// trailing flag column that `trailing_flag` reads: for a node, whether a
    /// live `supersedes` edge closes it; for an edge, whether a replay would
    /// refuse it.
    fn page_query(self, after: &str, limit: usize) -> (String, Vec<Value>) {
        let live = FOREVER.0;
        let supersedes = relation::SUPERSEDES;
        let (columns, table, flag, condition) = match self {
            Phase::Nodes => (
                NODE_COLUMNS,
                "nodes",
                format!("tx_to != {live} AND {}", live_closer("nodes")),
                String::new(),
            ),
            Phase::Edges => {
                // A guarded edge is refused unless both ends are open once the
                // nodes are logged.
                let open_end = |end: &str| {
                    format!(
                        "EXISTS (SELECT 1 FROM nodes n WHERE n.id = edges.{end}
                         AND (n.tx_to = {live} OR {}))",
                        live_closer("n")
                    )
                };
                (
                    EDGE_COLUMNS,
                    "edges",
                    format!("NOT ({} AND {})", open_end("src"), open_end("dst")),
                    format!("AND tx_to = {live} AND type != '{supersedes}'"),
                )
            }
            Phase::Supersedes => (
                EDGE_COLUMNS,
                "edges",
                // The close it replays fails unless its target is open or this
                // edge is what closed it.
                format!(
                    "NOT EXISTS (SELECT 1 FROM nodes d WHERE d.id = edges.dst
                     AND d.tx_to IN ({live}, edges.tx_from))"
                ),
                format!("AND tx_to = {live} AND type = '{supersedes}'"),
            ),
        };
        let sql = format!(
            "SELECT {columns}, {flag} FROM {table}
             WHERE id > ?1 {condition} ORDER BY id LIMIT ?2"
        );
        (sql, vec![after.into(), Value::Int(limit as i64)])
    }
}

/// Whether a live `supersedes` edge closed node `node` at the instant the node
/// was closed, `node` being the name or alias of a table of nodes in scope.
fn live_closer(node: &str) -> String {
    format!(
        "EXISTS (SELECT 1 FROM edges closer WHERE closer.dst = {node}.id
         AND closer.type = '{}' AND closer.tx_from = {node}.tx_to AND closer.tx_to = {})",
        relation::SUPERSEDES,
        FOREVER.0
    )
}

fn trailing_flag(row: &Row) -> Result<bool> {
    match row.0.last() {
        Some(Value::Int(flag)) => Ok(*flag != 0),
        _ => Err(Error::Backend("the flag column is not an integer".into())),
    }
}

const READ_SQL: &str = "SELECT phase, last_id, completed_at FROM log_backfill_state WHERE id = 1";

const SAVE_SQL: &str = "INSERT INTO log_backfill_state (id, phase, last_id, completed_at)
     VALUES (1, ?1, ?2, ?3)
     ON CONFLICT(id) DO UPDATE SET
       phase = excluded.phase, last_id = excluded.last_id,
       completed_at = excluded.completed_at";

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

    /// `None` for a store whose backfill never started.
    pub(super) async fn load<B: Backend>(backend: &B) -> Result<Option<Self>> {
        let rows = backend.query(READ_SQL, &[]).await?;
        rows.first().map(Self::from_row).transpose()
    }

    fn from_row(row: &Row) -> Result<Self> {
        Ok(Self {
            phase: Phase::parse(&row.get_string(0)?)?,
            last_id: super::opt_string(row, 1)?,
            completed_at: match row.0[2] {
                Value::Null => None,
                _ => Some(row.get_i64(2)?),
            },
        })
    }

    pub(super) fn is_complete(&self) -> bool {
        self.completed_at.is_some()
    }

    async fn save(&self, tx: &mut dyn BackendTx) -> Result<()> {
        let params = [
            self.phase.name().into(),
            super::opt_text(self.last_id.clone()),
            self.completed_at.map_or(Value::Null, Value::Int),
        ];
        tx.execute(SAVE_SQL, &params).await?;
        Ok(())
    }
}

/// A row read from the store, with the id the walk resumes after. The event is
/// `None` for a row a replay would refuse.
struct Pending {
    id: String,
    event: Option<LogEvent>,
}

/// What a run carries from one phase to the next.
struct Run {
    batch: usize,
    now: Millis,
    /// The rows the log already holds, whoever wrote them.
    logged: HashSet<String>,
    report: BackfillReport,
    /// Rows logged since the resume point was last saved.
    unrecorded: usize,
}

impl<B: Backend> Graph<B> {
    /// Logs every row the log does not hold yet, then marks the backfill complete.
    /// The log lock is held throughout, so no write lands between the rows, and
    /// writes are refused until it completes.
    pub async fn backfill_log_from_projection(&self) -> Result<BackfillReport> {
        self.backfill_in_batches(BACKFILL_BATCH).await
    }

    /// Whether the backfill has not completed for this store.
    pub async fn needs_backfill(&self) -> Result<bool> {
        self.log.as_ref().ok_or(Error::NoEventLog)?;
        let saved = Progress::load(&self.backend).await?;
        Ok(!saved.is_some_and(|progress| progress.is_complete()))
    }

    pub(super) async fn backfill_in_batches(&self, batch: usize) -> Result<BackfillReport> {
        let log = self.log.as_ref().ok_or(Error::NoEventLog)?;
        let mut held = HeldLog::acquire(log).await?;
        let saved = Progress::load(&self.backend).await?;
        if saved.as_ref().is_some_and(Progress::is_complete) {
            return Ok(BackfillReport::default());
        }
        let resumed = saved.is_some();
        let start = saved.unwrap_or_else(Progress::start);
        // Settles any row of an interrupted run that the log holds past the cursor.
        self.replay(&mut held).await?;
        let mut run = Run {
            batch,
            now: self.clock.now(),
            logged: logged_row_ids(held.reader()?, held.head()?).await?,
            report: BackfillReport {
                resumed,
                ..BackfillReport::default()
            },
            unrecorded: 0,
        };
        tracing::info!(logged = run.logged.len(), resumed, "backfill: starting");

        for phase in Phase::IN_ORDER
            .into_iter()
            .skip_while(|p| *p != start.phase)
        {
            let after = if phase == start.phase {
                start.last_id.clone().unwrap_or_default()
            } else {
                String::new()
            };
            self.log_phase(&mut held, &mut run, phase, after).await?;
            tracing::info!(phase = phase.name(), "backfill: phase logged");
        }
        self.record_completion(&mut held, run.now).await?;
        if run.report.edges_skipped > 0 {
            tracing::warn!(
                edges_skipped = run.report.edges_skipped,
                "backfill: edges left out, an end is a closed node no live supersedes edge closes"
            );
        }
        tracing::info!(report = ?run.report, "backfill: complete");
        Ok(run.report)
    }

    /// Logs the rows of `phase` after `after` that the log does not hold.
    async fn log_phase(
        &self,
        held: &mut HeldLog,
        run: &mut Run,
        phase: Phase,
        mut after: String,
    ) -> Result<()> {
        loop {
            let page = self.pending_page(phase, &after, run).await?;
            let Some(last) = page.last() else {
                return Ok(());
            };
            after = last.id.clone();
            for Pending { id, event } in page {
                if run.logged.contains(&id) {
                    continue;
                }
                let Some(event) = event else {
                    tracing::debug!(edge = %id, "backfill: edge skipped, a replay would refuse it");
                    run.report.edges_skipped += 1;
                    continue;
                };
                run.unrecorded += 1;
                let due = (run.unrecorded == run.batch).then(|| Progress::at(phase, &id));
                if due.is_some() {
                    run.unrecorded = 0;
                }
                // The resume point commits with the row that reaches it.
                let tx = self.begin_saving(due.as_ref()).await?;
                self.append_and_commit(tx, held, event, &[]).await?;
                match phase {
                    Phase::Nodes => run.report.nodes_logged += 1,
                    Phase::Edges | Phase::Supersedes => run.report.edges_logged += 1,
                }
            }
        }
    }

    async fn begin_saving(&self, progress: Option<&Progress>) -> Result<Box<dyn BackendTx + '_>> {
        let mut tx = self.backend.begin().await?;
        if let Some(progress) = progress {
            if let Err(error) = progress.save(&mut *tx).await {
                abandon(tx).await;
                return Err(error);
            }
        }
        Ok(tx)
    }

    async fn record_completion(&self, held: &mut HeldLog, now: Millis) -> Result<()> {
        let mut tx = self.backend.begin().await?;
        let saved = Progress::complete(now).save(&mut *tx).await;
        commit_or_abandon(tx, saved).await?;
        held.set_backfill_pending(false);
        Ok(())
    }

    async fn pending_page(&self, phase: Phase, after: &str, run: &Run) -> Result<Vec<Pending>> {
        let (sql, params) = phase.page_query(after, run.batch);
        let rows = self.backend.query(&sql, &params).await?;
        rows.iter()
            .map(|row| match phase {
                Phase::Nodes => node_pending(row, run.now),
                Phase::Edges | Phase::Supersedes => edge_pending(row, run.now),
            })
            .collect()
    }
}

/// A node a live `supersedes` edge closes is logged open, and replaying that
/// edge closes it at the same time. Any other node is logged as stored.
fn node_pending(row: &Row, now: Millis) -> Result<Pending> {
    let mut node = stored_node(row)?;
    if trailing_flag(row)? {
        node.tx_to = FOREVER.0;
    }
    let stamp = (BACKFILL_SOURCE, node.confidence);
    Ok(Pending {
        id: node.id.clone(),
        event: Some(event(
            node_row_hash(&node),
            stamp,
            node.valid_from,
            now,
            LogPayload::NodeWrite(node),
        )),
    })
}

fn edge_pending(row: &Row, now: Millis) -> Result<Pending> {
    let edge = stored_edge(row)?;
    let id = edge.id.clone();
    if trailing_flag(row)? {
        return Ok(Pending { id, event: None });
    }
    let stamp = (BACKFILL_SOURCE, STORE_TRUST);
    Ok(Pending {
        id,
        event: Some(event(
            edge_row_hash(&edge),
            stamp,
            edge.tx_from,
            now,
            LogPayload::EdgeWrite(edge),
        )),
    })
}

/// The ids of every row the log holds and no later record cancels, whoever
/// wrote it: scanned rather than read from the hash index, which names only the
/// last of several rows with the same content.
async fn logged_row_ids(
    reader: Arc<dyn LogReader>,
    head: Option<LogOffset>,
) -> Result<HashSet<String>> {
    let mut ids = HashSet::new();
    for_each_unvoided(reader, head, |payload| {
        ids.extend(row_effects(payload).into_iter().map(|effect| match effect {
            RowEffect::Node(row) => row.id,
            RowEffect::Edge(row) => row.id,
        }));
    })
    .await?;
    Ok(ids)
}
