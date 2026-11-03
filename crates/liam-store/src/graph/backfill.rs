// SPDX-License-Identifier: Apache-2.0
//! Copying the rows a store held before it had a log into that log, once, so a
//! rebuild from the log alone can recreate them.
//!
//! The rows are logged in the order a replay needs them: nodes, then plain
//! edges, then `supersedes` edges, because a replayed `supersedes` edge closes
//! its target and a plain edge needs both of its ends open. A superseded node is
//! logged open and left to its `supersedes` edge to close, as it was written.
//!
//! Nodes are walked in rowid order, which is the order they were inserted in, so
//! a store rebuilt from the log gets the rowids it had and ranking ties between
//! rows that predate the log fall as they did. Edges have no ranking, so they
//! walk by id.
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
use super::projection::{origin_steps, steps_for};
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

pub(super) const BACKFILL_SOURCE: &str = "backfill";

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

    /// The column this phase walks in order and resumes by.
    fn order_column(self) -> &'static str {
        match self {
            Phase::Nodes => "rowid",
            Phase::Edges | Phase::Supersedes => "id",
        }
    }

    /// Where a walk of this phase begins: before its first row.
    fn origin(self) -> Cursor {
        match self {
            Phase::Nodes => Cursor::Rowid(0),
            Phase::Edges | Phase::Supersedes => Cursor::Id(String::new()),
        }
    }

    /// The next `limit` rows of this phase after `after`, each with a flag
    /// column that `trailing_flag` reads (for a node, whether a live
    /// `supersedes` edge closes it; for an edge, whether a replay would refuse
    /// it) and then the order column that `Cursor::of_row` reads.
    fn page_query(self, after: &Cursor, limit: usize) -> (String, Vec<Value>) {
        let (columns, table, flag, condition) = self.parts();
        let order = self.order_column();
        let sql = format!(
            "SELECT {columns}, {flag}, {order} FROM {table}
             WHERE {order} > ?1 {condition} ORDER BY {order} LIMIT ?2"
        );
        (sql, vec![after.param(), Value::Int(limit as i64)])
    }

    /// How many rows `page_query` walks through for this phase after `after`.
    fn count_query(self, after: &Cursor) -> (String, Vec<Value>) {
        let (_, table, _, condition) = self.parts();
        let order = self.order_column();
        let sql = format!("SELECT COUNT(*) FROM {table} WHERE {order} > ?1 {condition}");
        (sql, vec![after.param()])
    }

    /// The columns, table, trailing flag expression and extra condition that
    /// select this phase's rows.
    fn parts(self) -> (&'static str, &'static str, String, String) {
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
        (columns, table, flag, condition)
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

/// The flag column of a page row, which sits just before the order column.
fn trailing_flag(row: &Row) -> Result<bool> {
    match row.0.iter().rev().nth(1) {
        Some(Value::Int(flag)) => Ok(*flag != 0),
        _ => Err(Error::Backend("the flag column is not an integer".into())),
    }
}

/// A position in a phase's walk: the rowid of a node, or the id of an edge.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Cursor {
    Rowid(i64),
    Id(String),
}

impl Cursor {
    fn param(&self) -> Value {
        match self {
            Cursor::Rowid(rowid) => Value::Int(*rowid),
            Cursor::Id(id) => id.as_str().into(),
        }
    }

    /// The order column, which `page_query` selects last.
    fn of_row(row: &Row) -> Result<Self> {
        match row.0.last() {
            Some(Value::Int(rowid)) => Ok(Cursor::Rowid(*rowid)),
            Some(Value::Text(id)) => Ok(Cursor::Id(id.clone())),
            _ => Err(Error::Backend(
                "the order column is not a rowid or an id".into(),
            )),
        }
    }
}

const READ_SQL: &str =
    "SELECT phase, last_id, completed_at, last_rowid FROM log_backfill_state WHERE id = 1";

const SAVE_SQL: &str =
    "INSERT INTO log_backfill_state (id, phase, last_id, completed_at, last_rowid)
     VALUES (1, ?1, ?2, ?3, ?4)
     ON CONFLICT(id) DO UPDATE SET
       phase = excluded.phase, last_id = excluded.last_id,
       completed_at = excluded.completed_at, last_rowid = excluded.last_rowid";

/// How far the backfill has got: `last` is the last row of `phase` accounted
/// for, and `completed_at` is set once every row is in the log.
#[derive(Debug)]
pub(super) struct Progress {
    phase: Phase,
    last: Option<Cursor>,
    completed_at: Option<i64>,
}

impl Progress {
    fn start() -> Self {
        Self {
            phase: Phase::Nodes,
            last: None,
            completed_at: None,
        }
    }

    fn at(phase: Phase, last: &Cursor) -> Self {
        Self {
            phase,
            last: Some(last.clone()),
            completed_at: None,
        }
    }

    fn complete(now: Millis) -> Self {
        Self {
            phase: Phase::Supersedes,
            last: None,
            completed_at: Some(now.0),
        }
    }

    /// `None` for a store whose backfill never started.
    pub(super) async fn load<B: Backend>(backend: &B) -> Result<Option<Self>> {
        let rows = backend.query(READ_SQL, &[]).await?;
        rows.first().map(Self::from_row).transpose()
    }

    fn from_row(row: &Row) -> Result<Self> {
        let phase = Phase::parse(&row.get_string(0)?)?;
        let last = match phase {
            Phase::Nodes => match row.0[3] {
                Value::Null => None,
                _ => Some(Cursor::Rowid(row.get_i64(3)?)),
            },
            Phase::Edges | Phase::Supersedes => super::opt_string(row, 1)?.map(Cursor::Id),
        };
        Ok(Self {
            phase,
            last,
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
        let (last_id, last_rowid) = match &self.last {
            Some(Cursor::Id(id)) => (super::opt_text(Some(id.clone())), Value::Null),
            Some(Cursor::Rowid(rowid)) => (Value::Null, Value::Int(*rowid)),
            None => (Value::Null, Value::Null),
        };
        let params = [
            self.phase.name().into(),
            last_id,
            self.completed_at.map_or(Value::Null, Value::Int),
            last_rowid,
        ];
        tx.execute(SAVE_SQL, &params).await?;
        Ok(())
    }
}

/// A row read from the store, with the position the walk resumes after. The event is
/// `None` for a row a replay would refuse.
struct Pending {
    id: String,
    cursor: Cursor,
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
    /// Rows this run walks through, counted before the first is logged.
    total: usize,
    /// Rows walked through so far, whether logged, already in the log or skipped.
    examined: usize,
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
            total: 0,
            examined: 0,
        };
        let phases = || {
            Phase::IN_ORDER
                .into_iter()
                .skip_while(|p| *p != start.phase)
                .map(|phase| {
                    let after = match phase == start.phase {
                        true => start.last.clone().unwrap_or_else(|| phase.origin()),
                        false => phase.origin(),
                    };
                    (phase, after)
                })
        };
        for (phase, after) in phases() {
            run.total += self.count_phase(phase, &after).await?;
        }
        // An empty store has nothing to wait for, so a fresh install stays quiet.
        if run.total > 0 {
            tracing::info!(
                rows = run.total,
                already_logged = run.logged.len(),
                resumed,
                "backfill: logging the rows that predate the event log, writes wait until it is done"
            );
        }

        for (phase, after) in phases() {
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
        mut after: Cursor,
    ) -> Result<()> {
        loop {
            let page = self.pending_page(phase, &after, run).await?;
            let Some(last) = page.last() else {
                return Ok(());
            };
            after = last.cursor.clone();
            for Pending { id, cursor, event } in page {
                run.examined += 1;
                if run.logged.contains(&id) {
                    continue;
                }
                let Some(event) = event else {
                    tracing::debug!(edge = %id, "backfill: edge skipped, a replay would refuse it");
                    run.report.edges_skipped += 1;
                    continue;
                };
                run.unrecorded += 1;
                let due = (run.unrecorded == run.batch).then(|| Progress::at(phase, &cursor));
                if due.is_some() {
                    run.unrecorded = 0;
                }
                // The resume point commits with the row that reaches it.
                let tx = self.begin_saving(due.as_ref()).await?;
                // The row is already stored, so only its origin is written.
                let origin = origin_steps(&event, &steps_for(&event.payload));
                self.append_and_commit(tx, held, event, &origin).await?;
                match phase {
                    Phase::Nodes => run.report.nodes_logged += 1,
                    Phase::Edges | Phase::Supersedes => run.report.edges_logged += 1,
                }
                if due.is_some() {
                    tracing::info!(
                        progress = format!("{}/{}", run.examined, run.total),
                        phase = phase.name(),
                        "backfill: resume point saved"
                    );
                }
            }
        }
    }

    async fn count_phase(&self, phase: Phase, after: &Cursor) -> Result<usize> {
        let (sql, params) = phase.count_query(after);
        let rows = self.backend.query(&sql, &params).await?;
        let count = rows
            .first()
            .ok_or_else(|| Error::Backend("a count query returned no row".into()))?
            .get_i64(0)?;
        Ok(count as usize)
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

    async fn pending_page(&self, phase: Phase, after: &Cursor, run: &Run) -> Result<Vec<Pending>> {
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
        cursor: Cursor::of_row(row)?,
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
        return Ok(Pending {
            id,
            cursor: Cursor::of_row(row)?,
            event: None,
        });
    }
    let stamp = (BACKFILL_SOURCE, STORE_TRUST);
    Ok(Pending {
        id,
        cursor: Cursor::of_row(row)?,
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
