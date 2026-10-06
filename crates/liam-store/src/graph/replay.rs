// SPDX-License-Identifier: Apache-2.0
//! Replay of the event log into the projection: the records past the log
//! cursor are applied once each, in log order.

use std::collections::HashSet;

use futures_util::StreamExt;
use liam_log::event::{LogEvent, LogPayload, RowEffect};
use liam_log::hash::content_hashes;
use liam_log::reader::{LogRecord, LogStream};
use liam_log::LogOffset;
use uuid::Uuid;

use super::log_cursor;
use super::logged_write::{abandon, commit_or_abandon, index_rows, EventLog};
use super::projection::{apply_steps, steps_for};
use super::Graph;
use crate::backend::{Backend, BackendTx};
use crate::error::{Error, Result};
use crate::value::Value;

const QUARANTINE_SQL: &str = "INSERT INTO log_quarantine (event_id, segment, seg_index, reason, at)
     VALUES (?1, ?2, ?3, ?4, ?5) ON CONFLICT(event_id) DO NOTHING";

/// What one `catch_up` did with the records past the cursor. Records that carry
/// no rows to project (`Voided`, `DuplicateOf`, `Tombstone`) only move the
/// cursor and are in no count; a cancelled event is counted once, as voided.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CatchUp {
    /// Events whose rows were projected.
    pub applied: usize,
    /// Events whose rows the store already held.
    pub skipped_applied: usize,
    /// Events a later `Voided` record cancelled.
    pub skipped_voided: usize,
    /// Events the projection refused, recorded in `log_quarantine`.
    pub quarantined: usize,
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

impl CatchUp {
    fn count(&mut self, outcome: Outcome) {
        match outcome {
            Outcome::Applied => self.applied += 1,
            Outcome::AlreadyApplied => self.skipped_applied += 1,
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
    /// Holds the log lock throughout, as a write does.
    pub async fn catch_up(&self) -> Result<CatchUp> {
        let Some(log) = &self.log else {
            return Ok(CatchUp::default());
        };
        let mut held = log.lock().await;
        let Some(reader) = held.reader() else {
            return Err(Error::LogReaderMissing);
        };
        let log_id = held.log_id();
        let cursor = log_cursor::read(&self.backend)
            .await?
            .and_then(|cursor| cursor.last);
        let Some(head) = held.head().filter(|head| cursor.is_none_or(|c| c < *head)) else {
            return Ok(CatchUp::default());
        };
        // A `Voided` record follows its target, so the targets are all known
        // before the first record is applied.
        let voided = voided_targets(reader.scan_through(cursor, head)).await?;
        tracing::debug!(voided = voided.len(), "replay: void records collected");

        let mut report = CatchUp::default();
        let mut records = reader.scan_through(cursor, head);
        while let Some(record) = records.next().await {
            let LogRecord { offset, event, .. } = record?;
            let outcome = if voided.contains(&event.event_id) {
                log_cursor::advance(&self.backend, log_id, offset).await?;
                Outcome::Voided
            } else {
                self.replay_event(&mut held, log_id, offset, &event).await?
            };
            tracing::debug!(event = %event.event_id, %offset, "replay: record accounted for");
            report.count(outcome);
        }
        Ok(report)
    }

    async fn replay_event(
        &self,
        held: &mut EventLog,
        log_id: Uuid,
        offset: LogOffset,
        event: &LogEvent,
    ) -> Result<Outcome> {
        let rows = row_refs(&event.payload);
        if rows.is_empty() {
            log_cursor::advance(&self.backend, log_id, offset).await?;
            return Ok(Outcome::Passed);
        }
        let mut tx = self.backend.begin().await?;
        match project(&mut *tx, log_id, offset, event, &rows).await {
            Ok(outcome) => {
                tx.commit().await?;
                if matches!(outcome, Outcome::Applied) {
                    let hashes: Vec<_> = content_hashes(event).iter().map(|(h, _)| *h).collect();
                    held.remember(&hashes);
                }
                Ok(outcome)
            }
            Err(error) if is_refusal(&error) => {
                abandon(tx).await;
                tracing::warn!(event = %event.event_id, %error, "replay: event refused, quarantining");
                self.quarantine(log_id, offset, &event.event_id, &error.to_string())
                    .await?;
                Ok(Outcome::Quarantined)
            }
            Err(error) => {
                abandon(tx).await;
                Err(error)
            }
        }
    }

    /// Records the refusal and moves the cursor past the event in one
    /// transaction.
    async fn quarantine(
        &self,
        log_id: Uuid,
        offset: LogOffset,
        event_id: &str,
        reason: &str,
    ) -> Result<()> {
        let mut tx = self.backend.begin().await?;
        let (segment, index) = log_cursor::offset_to_values(offset)?;
        let recorded = async {
            let params = [
                event_id.into(),
                segment,
                index,
                reason.into(),
                self.clock.now().0.into(),
            ];
            tx.execute(QUARANTINE_SQL, &params).await?;
            log_cursor::advance_in_tx(&mut *tx, log_id, offset).await
        }
        .await;
        commit_or_abandon(tx, recorded).await
    }
}

/// Projects an event whose rows the store does not hold yet, or passes one it
/// already holds, and moves the cursor onto it, all inside `tx`.
async fn project(
    tx: &mut dyn BackendTx,
    log_id: Uuid,
    offset: LogOffset,
    event: &LogEvent,
    rows: &[(&'static str, &str)],
) -> Result<Outcome> {
    let outcome = match rows_held(tx, rows).await? {
        held if held == rows.len() => Outcome::AlreadyApplied,
        0 => {
            apply_steps(tx, &steps_for(&event.payload)).await?;
            index_rows(tx, &content_hashes(event), &event.event_id).await?;
            Outcome::Applied
        }
        _ => {
            return Err(Error::CorruptLogState(format!(
                "event {} is only partly in the store",
                event.event_id
            )))
        }
    };
    log_cursor::advance_in_tx(tx, log_id, offset).await?;
    Ok(outcome)
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

/// The table and id of every row an event writes.
fn row_refs(payload: &LogPayload) -> Vec<(&'static str, &str)> {
    match payload {
        LogPayload::NodeWrite(row) => vec![("nodes", row.id.as_str())],
        LogPayload::EdgeWrite(row) => vec![("edges", row.id.as_str())],
        LogPayload::EpisodeBatch(effects) => effects
            .iter()
            .map(|effect| match effect {
                RowEffect::Node(row) => ("nodes", row.id.as_str()),
                RowEffect::Edge(row) => ("edges", row.id.as_str()),
            })
            .collect(),
        LogPayload::Tombstone(_) | LogPayload::DuplicateOf { .. } | LogPayload::Voided { .. } => {
            Vec::new()
        }
    }
}

/// How many of `rows` the store holds, live or superseded.
async fn rows_held(tx: &mut dyn BackendTx, rows: &[(&'static str, &str)]) -> Result<usize> {
    let mut held = 0;
    for (table, id) in rows {
        let sql = format!("SELECT 1 FROM {table} WHERE id = ?1");
        if !tx.query(&sql, &[Value::from(*id)]).await?.is_empty() {
            held += 1;
        }
    }
    Ok(held)
}

/// A refusal is the projection's own verdict on the event, so replaying it
/// again refuses again; anything else may be a passing backend fault.
fn is_refusal(error: &Error) -> bool {
    matches!(error, Error::RelateRefused(_) | Error::NodeNotFound(_))
}
