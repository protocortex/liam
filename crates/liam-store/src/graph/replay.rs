// SPDX-License-Identifier: Apache-2.0
//! Replay of the event log into the projection: the records past the log
//! cursor are applied once each, in log order.

use std::collections::HashSet;

use futures_util::StreamExt;
use liam_log::event::{LogEvent, LogPayload};
use liam_log::hash::content_hashes;
use liam_log::reader::{LogRecord, LogStream};
use liam_log::LogOffset;
use uuid::Uuid;

use super::log_cursor;
use super::logged_plan::Table;
use super::logged_write::{commit_or_abandon, follow_up, project_logged, HeldLog, SharedLog};
use super::projection::{steps_for, Step};
use super::Graph;
use crate::backend::{Backend, BackendTx};
use crate::error::{Error, Result};
use crate::value::Value;

const QUARANTINE_SQL: &str = "INSERT INTO log_quarantine (event_id, segment, seg_index, reason, at)
     VALUES (?1, ?2, ?3, ?4, ?5) ON CONFLICT(event_id) DO NOTHING";

/// What one `catch_up` did with the records past the cursor. Records that carry
/// no rows to project (`Voided`, `DuplicateOf`, `Tombstone`) only move the
/// cursor and are in no count, including the `Voided` records a quarantine
/// appends; a cancelled event is counted once, as voided.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CatchUpReport {
    /// Events whose rows were projected.
    pub applied: usize,
    /// Events whose rows the store already held.
    pub already_applied: usize,
    /// Events a later `Voided` record cancelled.
    pub skipped_voided: usize,
    /// Events the projection refused, recorded in `log_quarantine`.
    pub quarantined: usize,
    /// Nodes that had no vector and now have one.
    pub re_embedded: usize,
    /// Nodes that still have no vector because embedding them failed.
    pub embed_failed: usize,
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
        let mut report = self.replay(log).await?;
        let embedded = self.reembed_missing().await?;
        report.re_embedded = embedded.re_embedded;
        report.embed_failed = embedded.failed;
        Ok(report)
    }

    async fn replay(&self, log: &SharedLog) -> Result<CatchUpReport> {
        let mut held = HeldLog::acquire(log).await?;
        let cursor = log_cursor::read(&self.backend)
            .await?
            .and_then(|cursor| cursor.last);
        let Some(head) = held.head()?.filter(|head| cursor.is_none_or(|c| c < *head)) else {
            return Ok(CatchUpReport::default());
        };
        let reader = held.reader()?;
        // A `Voided` record follows its target, so the targets are all known
        // before the first record is applied.
        let voided = voided_targets(reader.scan_through(cursor, head)).await?;
        tracing::debug!(voided = voided.len(), "replay: void records collected");

        let mut report = CatchUpReport::default();
        // Both scans stop at `head`, so the voids a quarantine appends below it
        // are not read back as new work.
        let mut records = reader.scan_through(cursor, head);
        let mut last_void = None;
        while let Some(record) = records.next().await {
            let record = record?;
            let event_id = &record.event.event_id;
            let outcome = if voided.contains(event_id) {
                log_cursor::advance(&self.backend, held.log_id, record.offset).await?;
                Outcome::Voided
            } else {
                self.replay_event(&mut held, &record, &mut last_void)
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
        let projected =
            replay_projection(&mut *tx, &steps, &carried, event, held.log_id, *offset).await;
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
    event: &LogEvent,
    log_id: Uuid,
    offset: LogOffset,
) -> Result<Outcome> {
    let rows = written_rows(steps);
    match rows_held(tx, &rows).await? {
        held if held == rows.len() => {
            log_cursor::advance_in_tx(tx, log_id, offset).await?;
            Ok(Outcome::AlreadyApplied)
        }
        0 => {
            project_logged(tx, steps, carried, &event.event_id, log_id, offset).await?;
            Ok(Outcome::Applied)
        }
        _ => Err(Error::PartlyApplied(event.event_id.clone())),
    }
}

/// The ids a `Voided` record in `records` cancels.
async fn voided_targets(mut records: LogStream) -> Result<HashSet<String>> {
    let mut targets = HashSet::new();
    while let Some(record) = records.next().await {
        if let LogPayload::Voided { target_event_id } = record?.event.payload {
            targets.insert(target_event_id);
        }
    }
    Ok(targets)
}

/// The table and id of every row the steps insert, read from the same steps the
/// projection runs so the rows checked are the rows written.
fn written_rows(steps: &[Step]) -> Vec<(Table, &str)> {
    steps
        .iter()
        .filter_map(|step| match step {
            Step::Node(row) => Some((Table::Nodes, row.id.as_str())),
            Step::Edge(row) | Step::GuardedEdge(row) => Some((Table::Edges, row.id.as_str())),
            Step::Close { .. } => None,
        })
        .collect()
}

/// How many of `rows` the store holds, live or superseded.
async fn rows_held(tx: &mut dyn BackendTx, rows: &[(Table, &str)]) -> Result<usize> {
    let mut held = 0;
    for (table, id) in rows {
        if !tx
            .query(table.exists_query(), &[Value::from(*id)])
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
