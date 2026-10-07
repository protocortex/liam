// SPDX-License-Identifier: Apache-2.0
//! Replay of the event log into the projection: the records past the log
//! cursor are applied once each, in log order.

use std::collections::HashSet;
use std::sync::Arc;

use futures_util::StreamExt;
use liam_log::event::{EdgeRow, LogPayload, RowEffect, TombstoneTable, TombstoneTarget};
use liam_log::hash::content_hashes;
use liam_log::reader::{LogReader, LogRecord, LogStream};
use liam_log::LogOffset;
use uuid::Uuid;

use super::log_cursor;
use super::logged_plan::Table;
use super::logged_write::{commit_or_abandon, follow_up, project_logged, HeldLog};
use super::projection::{steps_for, Step};
use super::reembed::ReembedReport;
use super::Graph;
use crate::backend::{Backend, BackendTx};
use crate::error::{Error, Result};
use crate::value::Value;

const QUARANTINE_SQL: &str = "INSERT INTO log_quarantine (event_id, segment, seg_index, reason, at)
     VALUES (?1, ?2, ?3, ?4, ?5) ON CONFLICT(event_id) DO NOTHING";

/// What one `catch_up` did. The four record counts describe log records only:
/// those that carry no rows to project (`Voided`, `DuplicateOf`) only move the
/// cursor and are in no count, including the `Voided` records a quarantine
/// appends, and a cancelled event is counted once, as voided.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CatchUpReport {
    /// Events whose rows were projected, tombstones included.
    pub applied: usize,
    /// Events whose rows the store already held.
    pub already_applied: usize,
    /// Events a later `Voided` record cancelled.
    pub skipped_voided: usize,
    /// Events the projection refused, recorded in `log_quarantine`.
    pub quarantined: usize,
    /// The re-embed pass that follows the replay, which covers every live
    /// node without a vector, not only the replayed ones.
    pub reembedded: ReembedReport,
}

/// What became of one record.
enum Outcome {
    Applied,
    AlreadyApplied,
    Voided,
    Quarantined,
    /// A record that carries no rows.
    Passed,
}

impl CatchUpReport {
    fn count(&mut self, outcome: Outcome) {
        match outcome {
            Outcome::Applied => self.applied += 1,
            Outcome::AlreadyApplied => self.already_applied += 1,
            Outcome::Voided => self.skipped_voided += 1,
            Outcome::Quarantined => self.quarantined += 1,
            Outcome::Passed => {}
        }
    }
}

impl<B: Backend> Graph<B> {
    /// Applies the log records the store has not accounted for, up to the
    /// writer's last acknowledged record. A refused event is quarantined and
    /// replay goes on past it; a backend error stops replay with the cursor on
    /// the last record that was accounted for, so a retry resumes from there.
    /// Holds the log lock while it replays, as a write does, and refuses a
    /// poisoned log, whose tail is unknown until the store is reopened. It then
    /// re-embeds the live nodes with no vector, replayed ones included.
    pub async fn catch_up(&self) -> Result<CatchUpReport> {
        let Some(log) = &self.log else {
            return Ok(CatchUpReport::default());
        };
        let report = {
            let mut held = HeldLog::acquire(log).await?;
            self.replay(&mut held).await?
        };
        Ok(self.reembedded(report).await)
    }

    /// Adds the re-embed pass to a replay's report. The replay has already
    /// committed, so a failed listing must not hide its report.
    pub(super) async fn reembedded(&self, mut report: CatchUpReport) -> CatchUpReport {
        match self.reembed_missing().await {
            Ok(reembedded) => report.reembedded = reembedded,
            Err(error) => {
                tracing::error!(%error, "re-embed pass skipped, the nodes could not be listed")
            }
        }
        report
    }

    /// The replay itself, for a caller that already holds the log lock.
    pub(super) async fn replay(&self, held: &mut HeldLog) -> Result<CatchUpReport> {
        let cursor = log_cursor::read(&self.backend)
            .await?
            .and_then(|cursor| cursor.last);
        let Some(head) = held.head()?.filter(|head| cursor.is_none_or(|c| c < *head)) else {
            return Ok(CatchUpReport::default());
        };
        let reader = held.reader()?;
        // A `Voided` record follows its target, so the targets are all known
        // before the first record is applied.
        let ahead = scan_ahead(reader.scan_through(cursor, head)).await?;
        tracing::debug!(
            voided = ahead.voided.len(),
            "replay: void and tombstone records collected"
        );

        let mut report = CatchUpReport::default();
        // Both scans stop at `head`, so the voids a quarantine appends below it
        // are not read back as new work.
        let mut records = reader.scan_through(cursor, head);
        let mut last_void = None;
        while let Some(record) = records.next().await {
            let record = record?;
            let event_id = &record.event.event_id;
            let outcome = if ahead.voided.contains(event_id) {
                log_cursor::advance(&self.backend, held.log_id, record.offset).await?;
                Outcome::Voided
            } else {
                self.replay_event(held, &record, &ahead.removed, &mut last_void)
                    .await?
            };
            tracing::debug!(event = %event_id, offset = %record.offset, "replay: record accounted for");
            report.count(outcome);
        }
        // Every record the scan covered is accounted for, so the cursor can pass
        // the voids appended behind them, which carry no rows.
        if let Some(void) = last_void {
            log_cursor::advance(&self.backend, held.log_id, void).await?;
        }
        Ok(report)
    }

    async fn replay_event(
        &self,
        held: &mut HeldLog,
        record: &LogRecord,
        removed: &Removed,
        last_void: &mut Option<LogOffset>,
    ) -> Result<Outcome> {
        let LogRecord { offset, event, .. } = record;
        let steps = steps_for(&event.payload);
        if steps.is_empty() {
            log_cursor::advance(&self.backend, held.log_id, *offset).await?;
            return Ok(Outcome::Passed);
        }
        let carried = content_hashes(event);
        let mut tx = self.backend.begin().await?;
        let vector_delete = self.backend.vector_delete_sql();
        let projected = replay_projection(
            &mut *tx,
            &steps,
            &carried,
            record,
            held.log_id,
            removed,
            vector_delete,
        )
        .await;
        match commit_or_abandon(tx, projected).await {
            Ok(outcome) => {
                held.remember(&carried);
                Ok(outcome)
            }
            Err(error) if is_refusal(&error) => {
                tracing::warn!(event = %event.event_id, %error, "replay: event refused, quarantining");
                let void = self.quarantine(held, record, &error.to_string()).await?;
                *last_void = Some(void);
                Ok(Outcome::Quarantined)
            }
            Err(error) => Err(error),
        }
    }

    /// Appends a `Voided` for the refused event, so a rebuild from the log
    /// skips it as this store does, then records the refusal and moves the
    /// cursor onto the event in one transaction. Returns the void's offset.
    async fn quarantine(
        &self,
        held: &mut HeldLog,
        record: &LogRecord,
        reason: &str,
    ) -> Result<LogOffset> {
        let LogRecord { offset, event, .. } = record;
        let target_event_id = event.event_id.clone();
        let void = follow_up(event, LogPayload::Voided { target_event_id });
        let void_offset = match held.append(void).await {
            Ok(void_offset) => void_offset,
            Err(error) => {
                held.set_poisoned(true);
                tracing::error!(%error, "could not void a refused event, the log is poisoned until reopen");
                return Err(error);
            }
        };
        held.reconcile(&[]);

        let mut tx = self.backend.begin().await?;
        let (segment, index) = log_cursor::offset_to_values(*offset)?;
        let recorded = async {
            let params = [
                event.event_id.as_str().into(),
                segment,
                index,
                reason.into(),
                self.clock.now().0.into(),
            ];
            tx.execute(QUARANTINE_SQL, &params).await?;
            log_cursor::advance_in_tx(&mut *tx, held.log_id, *offset).await
        }
        .await;
        commit_or_abandon(tx, recorded).await?;
        Ok(void_offset)
    }
}

/// Projects an event whose rows the store does not hold yet, or passes one it
/// already holds, and moves the cursor onto it, all inside `tx`.
async fn replay_projection(
    tx: &mut dyn BackendTx,
    steps: &[Step],
    carried: &[([u8; 32], String)],
    record: &LogRecord,
    log_id: Uuid,
    removed: &Removed,
    vector_delete: Option<&str>,
) -> Result<Outcome> {
    let LogRecord { offset, event, .. } = record;
    let rows = written_rows(steps);
    if rows.is_empty() {
        // Only removals: there is no row to look for, and running them again
        // is harmless.
        let event_id = &event.event_id;
        project_logged(tx, steps, carried, event_id, log_id, *offset, vector_delete).await?;
        return Ok(Outcome::Applied);
    }
    // A row a later tombstone removes may be gone from a store that is ahead of
    // its cursor, so it says nothing about whether the rest of the event
    // landed. Only when every row is removed later do they decide.
    let (kept, doomed): (Vec<_>, Vec<_>) = rows.into_iter().partition(|row| !removed.takes(row));
    let deciding = if kept.is_empty() { doomed } else { kept };
    let held = rows_held(tx, &deciding).await?;
    if held == deciding.len() {
        // The store may hold the rows of an event it never indexed, as a
        // backfill that stopped after its append leaves them.
        let event_id = &event.event_id;
        project_logged(tx, &[], carried, event_id, log_id, *offset, vector_delete).await?;
        return Ok(Outcome::AlreadyApplied);
    }
    if held > 0 {
        return Err(Error::PartlyApplied(event.event_id.clone()));
    }
    let event_id = &event.event_id;
    project_logged(tx, steps, carried, event_id, log_id, *offset, vector_delete).await?;
    Ok(Outcome::Applied)
}

/// What a replay range holds besides the events to apply.
pub(super) struct Ahead {
    /// The ids of the events a `Voided` record cancels.
    voided: HashSet<String>,
    /// The rows named by the tombstones that ran, so a voided one is left out.
    removed: Removed,
}

impl Ahead {
    pub(super) fn into_voided(self) -> HashSet<String> {
        self.voided
    }
}

/// Rows named by a tombstone. Ids are never reused, so once the tombstone has
/// run a row it names is gone for good.
#[derive(Default)]
struct Removed {
    nodes: HashSet<String>,
    edges: HashSet<String>,
}

impl Removed {
    fn add(&mut self, target: &TombstoneTarget) {
        match target.table {
            TombstoneTable::Nodes => self.nodes.insert(target.id.clone()),
            TombstoneTable::Edges => self.edges.insert(target.id.clone()),
            // No event writes a community row, so none is looked for.
            TombstoneTable::NodeCommunity => false,
        };
    }

    /// Whether the tombstones remove the row: a node it names, an edge it
    /// names, or an edge that ends at a node it names.
    fn takes(&self, row: &WrittenRow<'_>) -> bool {
        match row {
            WrittenRow::Node(id) => self.nodes.contains(*id),
            WrittenRow::Edge(edge) => {
                self.edges.contains(&edge.id)
                    || self.nodes.contains(&edge.src)
                    || self.nodes.contains(&edge.dst)
            }
        }
    }
}

pub(super) async fn scan_ahead(mut records: LogStream) -> Result<Ahead> {
    let mut voided = HashSet::new();
    let mut tombstones = Vec::new();
    while let Some(record) = records.next().await {
        let event = record?.event;
        match event.payload {
            LogPayload::Voided { target_event_id } => {
                voided.insert(target_event_id);
            }
            LogPayload::Tombstone(targets) => tombstones.push((event.event_id, targets)),
            _ => {}
        }
    }
    let mut removed = Removed::default();
    tombstones
        .iter()
        .filter(|(event_id, _)| !voided.contains(event_id))
        .flat_map(|(_, targets)| targets)
        .for_each(|target| removed.add(target));
    Ok(Ahead { voided, removed })
}

/// Hands `visit` the payload of every record up to `head` that no `Voided`
/// record cancels, in log order. Takes the reader and head, not the held log,
/// so a future that calls it stays `Send`.
pub(super) async fn for_each_unvoided(
    reader: Arc<dyn LogReader>,
    head: Option<LogOffset>,
    mut visit: impl FnMut(LogPayload),
) -> Result<()> {
    let Some(head) = head else {
        return Ok(());
    };
    let voided = scan_ahead(reader.scan_through(None, head))
        .await?
        .into_voided();
    let mut records = reader.scan_through(None, head);
    while let Some(record) = records.next().await {
        let event = record?.event;
        if !voided.contains(&event.event_id) {
            visit(event.payload);
        }
    }
    Ok(())
}

/// The rows a payload writes, in order.
pub(super) fn row_effects(payload: LogPayload) -> Vec<RowEffect> {
    match payload {
        LogPayload::NodeWrite(row) => vec![RowEffect::Node(row)],
        LogPayload::EdgeWrite(row) => vec![RowEffect::Edge(row)],
        LogPayload::EpisodeBatch(effects) => effects,
        LogPayload::Tombstone(_) | LogPayload::DuplicateOf { .. } | LogPayload::Voided { .. } => {
            Vec::new()
        }
    }
}

/// A row the steps insert.
enum WrittenRow<'a> {
    Node(&'a str),
    Edge(&'a EdgeRow),
}

impl WrittenRow<'_> {
    fn table_and_id(&self) -> (Table, &str) {
        match self {
            WrittenRow::Node(id) => (Table::Nodes, id),
            WrittenRow::Edge(edge) => (Table::Edges, &edge.id),
        }
    }
}

/// Every row the steps insert, read from the same steps the projection runs so
/// the rows checked are the rows written.
fn written_rows(steps: &[Step]) -> Vec<WrittenRow<'_>> {
    steps
        .iter()
        .filter_map(|step| match step {
            Step::Node(row) => Some(WrittenRow::Node(&row.id)),
            Step::Edge(row) | Step::GuardedEdge(row) => Some(WrittenRow::Edge(row)),
            Step::Close { .. } | Step::Remove(_) => None,
        })
        .collect()
}

/// How many of `rows` the store holds, live or superseded.
async fn rows_held(tx: &mut dyn BackendTx, rows: &[WrittenRow<'_>]) -> Result<usize> {
    let mut held = 0;
    for row in rows {
        let (table, id) = row.table_and_id();
        if !tx
            .query(table.exists_query(), &[Value::from(id)])
            .await?
            .is_empty()
        {
            held += 1;
        }
    }
    Ok(held)
}

/// A refusal is the projection's own verdict on the event, so replaying it
/// again refuses again; anything else may be a passing backend fault.
fn is_refusal(error: &Error) -> bool {
    matches!(
        error,
        Error::RelateRefused(_)
            | Error::NodeNotFound(_)
            | Error::Constraint(_)
            | Error::PartlyApplied(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_projections_own_verdicts_are_refusals() {
        // Arrange
        let refusals = [
            Error::RelateRefused("target node n is not live".into()),
            Error::NodeNotFound("n".into()),
            Error::Constraint("FOREIGN KEY constraint failed".into()),
            Error::PartlyApplied("event-1".into()),
        ];
        let transient = [
            Error::Backend("database is locked".into()),
            Error::ConcurrentWrite,
            Error::CorruptLogState("log_cursor holds the offset -1".into()),
            Error::LogPoisoned,
        ];

        // Act
        let verdicts = |errors: &[Error]| errors.iter().map(is_refusal).collect::<Vec<_>>();

        // Assert
        assert_eq!(verdicts(&refusals), [true; 4]);
        assert_eq!(verdicts(&transient), [false; 4]);
    }
}
