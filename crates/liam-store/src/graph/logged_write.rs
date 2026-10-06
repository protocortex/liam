// SPDX-License-Identifier: Apache-2.0
//! The log-first write path. A write is appended to the event log and then
//! projected into the store inside one transaction, with the log lock held
//! throughout, so the log's order is the commit order and a replay of the log
//! rebuilds the store.
//!
//! The log cursor tracks the last log record the store has accounted for,
//! whether that record was applied or voided. A write that is appended but
//! cannot be projected is cancelled with a `Voided` record, and the cursor
//! moves onto the void only if that append succeeds.
//!
//! The log is poisoned from the moment an append starts until the write is
//! reconciled, meaning committed or voided. A write that never reaches that
//! point, because its future was dropped, a task panicked, or the void could
//! not be appended, leaves the log poisoned, and it stays so until the store is
//! reopened and the log replayed.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use liam_log::dedup::{
    find_first_write, BloomConfig, HashBloom, HashIndex, IndexedWrite, PreCheck,
};
use liam_log::event::{LogEvent, LogPayload};
use liam_log::hash::content_hashes;
use liam_log::reader::LogReader;
use liam_log::{LogOffset, LogWriter};
use tokio::sync::OwnedMutexGuard;
use uuid::Uuid;

use super::log_cursor;
use super::logged_plan::{Dedup, Plan, Table};
use super::projection::{apply_steps, Step};
use super::Graph;
use crate::backend::{Backend, BackendTx};
use crate::error::{Error, Result};
use crate::ids::{EdgeId, Millis, NodeId};

const HASH_INDEX_UPSERT_SQL: &str =
    "INSERT INTO log_hash_index (content_hash, first_event_id, row_ids)
     VALUES (?1, ?2, ?3)
     ON CONFLICT(content_hash) DO UPDATE SET
       first_event_id = excluded.first_event_id, row_ids = excluded.row_ids";

pub(super) type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A plan that needs no read from the transaction.
pub(super) fn ready<'a>(plan: Result<Plan>) -> BoxFuture<'a, Result<Plan>> {
    Box::pin(std::future::ready(plan))
}

/// The log a `Graph` appends to, with the dedup filter and poison flag that
/// go with it. Graphs that share one `SharedLog` share all three.
pub struct EventLog {
    writer: Box<dyn LogWriter>,
    /// Reads the log back for replay; a log without one cannot be caught up from.
    reader: Option<Arc<dyn LogReader>>,
    bloom: HashBloom,
    /// What the operator asked for, kept apart from `bloom`'s own sizing so a
    /// rebuild that grew the filter does not raise the floor for the next one.
    configured: BloomConfig,
    poisoned: bool,
}

impl EventLog {
    pub fn new(writer: Box<dyn LogWriter>, bloom: HashBloom) -> Self {
        Self {
            writer,
            reader: None,
            configured: bloom.config().clone(),
            bloom,
            poisoned: false,
        }
    }

    /// Names the reader replay scans the log with.
    pub fn with_reader(mut self, reader: Arc<dyn LogReader>) -> Self {
        self.reader = Some(reader);
        self
    }

    pub(super) fn log_id(&self) -> Uuid {
        self.writer.log_id()
    }

    pub(super) fn head(&self) -> Option<LogOffset> {
        self.writer.head()
    }

    pub(super) fn reader(&self) -> Option<Arc<dyn LogReader>> {
        self.reader.clone()
    }

    /// Teaches the dedup filter hashes whose rows have committed.
    pub(super) fn remember(&mut self, hashes: &[[u8; 32]]) {
        hashes.iter().for_each(|hash| self.bloom.insert(hash));
    }

    /// The sizing the filter was configured with, not the size of the live
    /// filter, which a rebuild may have grown past it.
    pub(super) fn bloom_config(&self) -> &BloomConfig {
        &self.configured
    }

    pub(super) fn replace_bloom(&mut self, bloom: HashBloom) {
        self.bloom = bloom;
    }

    #[cfg(test)]
    pub(super) fn live_bloom_config(&self) -> &BloomConfig {
        self.bloom.config()
    }

    #[cfg(test)]
    pub(super) fn bloom_might_contain(&self, hash: &[u8; 32]) -> bool {
        self.bloom.might_contain(hash)
    }
}

/// One lock orders every append against the commit that follows it.
pub type SharedLog = Arc<tokio::sync::Mutex<EventLog>>;

/// What a write did.
pub(super) enum WriteOutcome {
    /// The rows were projected; the caller's ids are the rows' ids.
    Written,
    /// The content already has a live carrier, which is the id to return.
    Duplicate(String),
}

impl WriteOutcome {
    pub(super) fn node_id(&self, written: &NodeId) -> NodeId {
        match self {
            WriteOutcome::Written => written.clone(),
            WriteOutcome::Duplicate(first) => NodeId::from_raw(first),
        }
    }

    pub(super) fn edge_id(&self, written: &EdgeId) -> EdgeId {
        match self {
            WriteOutcome::Written => written.clone(),
            WriteOutcome::Duplicate(first) => EdgeId::from_raw(first),
        }
    }
}

impl<B: Backend> Graph<B> {
    /// The one path every write takes. `prepare` reads the transaction to build
    /// the write's `Plan` and may refuse the write before anything changes. It
    /// is given the write's timestamp, read once the transaction is open, so no
    /// write is stamped earlier than a commit it follows.
    pub(super) async fn project<F>(&self, prepare: F) -> Result<WriteOutcome>
    where
        F: for<'t> FnOnce(&'t mut dyn BackendTx, Millis) -> BoxFuture<'t, Result<Plan>> + Send,
    {
        match &self.log {
            Some(log) => self.transact(log, prepare).await,
            None => self.apply_unlogged(prepare).await,
        }
    }

    /// The same plan and projection without a log: nothing is recorded, indexed,
    /// or deduplicated, so a live twin the plan found is always a refusal.
    async fn apply_unlogged<F>(&self, prepare: F) -> Result<WriteOutcome>
    where
        F: for<'t> FnOnce(&'t mut dyn BackendTx, Millis) -> BoxFuture<'t, Result<Plan>> + Send,
    {
        let mut tx = self.backend.begin().await?;
        let now = self.clock.now();
        let applied = async {
            let plan = prepare(&mut *tx, now).await?;
            if let Some(refusal) = plan.dedup.and_then(|dedup| dedup.refusal_if_absent) {
                return Err(refusal);
            }
            apply_steps(&mut *tx, &plan.steps).await
        }
        .await;
        commit_or_abandon(tx, applied).await?;
        Ok(WriteOutcome::Written)
    }

    /// The logged path: poison check, log lock, `begin`, then plan, deduplicate,
    /// append, project, and commit.
    async fn transact<F>(&self, log: &SharedLog, prepare: F) -> Result<WriteOutcome>
    where
        F: for<'t> FnOnce(&'t mut dyn BackendTx, Millis) -> BoxFuture<'t, Result<Plan>> + Send,
    {
        let mut held = HeldLog::acquire(log).await?;
        let mut tx = self.backend.begin().await?;
        let now = self.clock.now();
        let resolved = match prepare(&mut *tx, now).await {
            Ok(plan) => {
                let bloom_hit = held.might_contain(&plan.event.content_hash);
                resolve_plan(&mut *tx, plan, bloom_hit).await
            }
            Err(error) => Err(error),
        };
        let (event, steps, logged) = match resolved {
            Ok(Resolved::Fresh { event, steps }) => (event, steps, WriteOutcome::Written),
            Ok(Resolved::Duplicate { event, first }) => {
                (event, Vec::new(), WriteOutcome::Duplicate(first.row_id))
            }
            Err(error) => {
                abandon(tx).await;
                return Err(error);
            }
        };
        tracing::debug!(event = %event.event_id, "log lock taken, transaction open");
        self.append_and_commit(tx, &mut held, event, &steps).await?;
        Ok(logged)
    }

    async fn append_and_commit(
        &self,
        mut tx: Box<dyn BackendTx + '_>,
        held: &mut HeldLog,
        event: LogEvent,
        steps: &[Step],
    ) -> Result<()> {
        let event_id = event.event_id.clone();
        let carried = content_hashes(&event);
        let void = follow_up(
            &event,
            LogPayload::Voided {
                target_event_id: event_id.clone(),
            },
        );
        let offset = match held.append(event).await {
            Ok(offset) => offset,
            Err(error) => {
                abandon(tx).await;
                return Err(error);
            }
        };
        tracing::debug!(%event_id, "event appended to the log");
        let projected = async {
            apply_steps(&mut *tx, steps).await?;
            index_rows(&mut *tx, &carried, &event_id).await?;
            log_cursor::advance_in_tx(&mut *tx, held.log_id, offset).await?;
            Ok(())
        }
        .await;
        match commit_or_abandon(tx, projected).await {
            Ok(()) => {
                let hashes: Vec<_> = carried.iter().map(|(hash, _)| *hash).collect();
                held.reconcile(&hashes);
                tracing::debug!(%event_id, "projection committed");
                Ok(())
            }
            Err(error) => {
                self.void_failed_write(held, void).await;
                Err(error)
            }
        }
    }

    /// Cancels an appended write the store failed to apply. Never fails: if
    /// the void cannot be appended the log stays poisoned instead.
    async fn void_failed_write(&self, held: &mut HeldLog, void: LogEvent) {
        match held.append(void).await {
            Ok(offset) => {
                let moved = log_cursor::advance(&self.backend, held.log_id, offset).await;
                if let Err(error) = moved {
                    tracing::warn!(%error, "voided write logged but the cursor did not advance");
                }
                held.reconcile(&[]);
            }
            Err(error) => {
                held.set_poisoned(true);
                tracing::error!(%error, "could not void a failed write, the log is poisoned until reopen");
            }
        }
    }
}

/// A record about `event`, carrying its own source, trust, and times.
fn follow_up(event: &LogEvent, payload: LogPayload) -> LogEvent {
    LogEvent {
        event_id: Uuid::now_v7().to_string(),
        content_hash: event.content_hash,
        source: event.source.clone(),
        trust_score: event.trust_score,
        observed_at: event.observed_at,
        ingested_at: event.ingested_at,
        encryption_key_id: event.encryption_key_id.clone(),
        schema_version: event.schema_version,
        payload,
    }
}

/// The log lock, held for the whole transaction. `append` blocks on fsync, so
/// it runs on the blocking pool with the guard moved in and back out.
struct HeldLog {
    guard: Option<OwnedMutexGuard<EventLog>>,
    log_id: Uuid,
}

impl HeldLog {
    async fn acquire(log: &SharedLog) -> Result<Self> {
        let guard = Arc::clone(log).lock_owned().await;
        if guard.poisoned {
            return Err(Error::LogPoisoned);
        }
        let log_id = guard.writer.log_id();
        Ok(Self {
            guard: Some(guard),
            log_id,
        })
    }

    async fn append(&mut self, event: LogEvent) -> Result<LogOffset> {
        let Some(mut guard) = self.guard.take() else {
            return Err(Error::LogPoisoned);
        };
        let joined = tokio::task::spawn_blocking(move || {
            // Set before the append and cleared only by `reconcile` or a
            // failed append, so a dropped future or a panic from here on
            // leaves it set, and a record that did land is never orphaned.
            guard.poisoned = true;
            let appended = guard.writer.append(&event);
            (guard, appended)
        })
        .await;
        let (guard, appended) = joined.map_err(|error| Error::LogTask(error.to_string()))?;
        self.guard = Some(guard);
        if appended.is_err() {
            self.set_poisoned(false);
        }
        Ok(appended?)
    }

    fn might_contain(&self, hash: &[u8; 32]) -> bool {
        self.guard
            .as_deref()
            .is_some_and(|log| log.bloom.might_contain(hash))
    }

    fn reconcile(&mut self, hashes: &[[u8; 32]]) {
        if let Some(log) = self.guard.as_deref_mut() {
            log.remember(hashes);
            log.poisoned = false;
        }
    }

    fn set_poisoned(&mut self, poisoned: bool) {
        if let Some(log) = self.guard.as_deref_mut() {
            log.poisoned = poisoned;
        }
    }
}

/// A plan after the dedup check.
enum Resolved {
    Fresh {
        event: LogEvent,
        steps: Vec<Step>,
    },
    /// The write's content is live already: log a `DuplicateOf` the first
    /// write instead of the write itself.
    Duplicate {
        event: LogEvent,
        first: Carrier,
    },
}

/// Checks a plan against the live carrier of its content, which is read in the
/// same transaction the write commits in.
async fn resolve_plan(tx: &mut dyn BackendTx, plan: Plan, bloom_hit: bool) -> Result<Resolved> {
    let Plan {
        event,
        steps,
        dedup,
    } = plan;
    let Some(Dedup {
        table,
        refusal_if_absent,
    }) = dedup
    else {
        return Ok(Resolved::Fresh { event, steps });
    };
    match live_duplicate(tx, bloom_hit, &event.content_hash, table).await? {
        Some(first) => {
            let duplicate = LogPayload::DuplicateOf {
                first_event_id: first.event_id.clone(),
            };
            let event = follow_up(&event, duplicate);
            Ok(Resolved::Duplicate { event, first })
        }
        None => match refusal_if_absent {
            Some(refusal) => Err(refusal),
            None => Ok(Resolved::Fresh { event, steps }),
        },
    }
}

/// The filter's answer for one hash, taken from the held log before the
/// transaction's reads begin.
struct Seen(bool);

impl PreCheck for Seen {
    fn might_contain(&self, _hash: &[u8; 32]) -> bool {
        self.0
    }
}

/// The live row that already carries a content hash.
struct Carrier {
    event_id: String,
    row_id: String,
}

/// A hit counts only while an indexed row is live, and the first live one is
/// the carrier: a hash repeated within one episode points at every row that
/// carries it, and an earlier one may be superseded by a later one. A
/// superseded or garbage-collected carrier is not a duplicate, and the write
/// that follows repoints the index at itself.
async fn live_duplicate(
    tx: &mut dyn BackendTx,
    bloom_hit: bool,
    hash: &[u8; 32],
    table: Table,
) -> Result<Option<Carrier>> {
    let hit = {
        let mut index = TxHashIndex { tx: &mut *tx };
        find_first_write(&Seen(bloom_hit), &mut index, hash).await?
    };
    let Some(IndexedWrite {
        first_event_id,
        row_ids,
    }) = hit
    else {
        return Ok(None);
    };
    for row_id in row_ids {
        if is_live(tx, table, &row_id).await? {
            return Ok(Some(Carrier {
                event_id: first_event_id,
                row_id,
            }));
        }
    }
    Ok(None)
}

async fn is_live(tx: &mut dyn BackendTx, table: Table, row_id: &str) -> Result<bool> {
    let (sql, params) = table.live_query(row_id);
    Ok(!tx.query(sql, &params).await?.is_empty())
}

/// The `log_hash_index` table, read inside the write transaction.
struct TxHashIndex<'a> {
    tx: &'a mut dyn BackendTx,
}

#[async_trait]
impl HashIndex for TxHashIndex<'_> {
    type Error = Error;

    async fn lookup(&mut self, hash: &[u8; 32]) -> Result<Option<IndexedWrite>> {
        let rows = self
            .tx
            .query(
                "SELECT first_event_id, row_ids FROM log_hash_index WHERE content_hash = ?1",
                &[hash.to_vec().into()],
            )
            .await?;
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        Ok(Some(IndexedWrite {
            first_event_id: row.get_string(0)?,
            row_ids: serde_json::from_str(&row.get_string(1)?)?,
        }))
    }
}

/// Points each carried hash at the rows that carry it, in write order, so the
/// first carrier of a hash repeated within one event stays first.
pub(super) async fn index_rows(
    tx: &mut dyn BackendTx,
    carried: &[([u8; 32], String)],
    event_id: &str,
) -> Result<()> {
    let mut by_hash: BTreeMap<[u8; 32], Vec<&str>> = BTreeMap::new();
    for (hash, row_id) in carried {
        by_hash.entry(*hash).or_default().push(row_id);
    }
    for (hash, row_ids) in by_hash {
        let row_ids = serde_json::to_string(&row_ids)?;
        tx.execute(
            HASH_INDEX_UPSERT_SQL,
            &[hash.to_vec().into(), event_id.into(), row_ids.into()],
        )
        .await?;
    }
    Ok(())
}

pub(super) async fn commit_or_abandon<T>(
    tx: Box<dyn BackendTx + '_>,
    applied: Result<T>,
) -> Result<T> {
    match applied {
        Ok(applied) => tx.commit().await.map(|()| applied),
        Err(error) => {
            abandon(tx).await;
            Err(error)
        }
    }
}

pub(super) async fn abandon(tx: Box<dyn BackendTx + '_>) {
    if let Err(error) = tx.rollback().await {
        tracing::warn!(%error, "rollback of an abandoned write failed");
    }
}
