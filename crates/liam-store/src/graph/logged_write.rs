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

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use liam_log::dedup::{find_first_write, HashBloom, HashIndex, IndexedWrite, PreCheck};
use liam_log::event::{LogEvent, LogPayload, NodeRow, CURRENT_SCHEMA_VERSION};
use liam_log::hash::node_row_hash;
use liam_log::{LogOffset, LogWriter};
use tokio::sync::OwnedMutexGuard;
use uuid::Uuid;

use super::{node_row_insert, resolve_node_row, Graph};
use crate::backend::{Backend, BackendTx};
use crate::error::{Error, Result};
use crate::ids::{Millis, NodeId, FOREVER};
use crate::types::NewNode;
use crate::value::Value;

const HASH_INDEX_UPSERT_SQL: &str =
    "INSERT INTO log_hash_index (content_hash, first_event_id, row_ids)
     VALUES (?1, ?2, ?3)
     ON CONFLICT(content_hash) DO UPDATE SET
       first_event_id = excluded.first_event_id, row_ids = excluded.row_ids";

// log_id is written once and never overwritten, so a database opened against a
// different log stays recognisable.
const CURSOR_UPSERT_SQL: &str = "INSERT INTO log_cursor (id, log_id, last_segment, last_index)
     VALUES (1, ?1, ?2, ?3)
     ON CONFLICT(id) DO UPDATE SET
       last_segment = excluded.last_segment, last_index = excluded.last_index";

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The log a `Graph` appends to, with the dedup filter and poison flag that
/// go with it. Graphs that share one `SharedLog` share all three.
pub struct EventLog {
    writer: Box<dyn LogWriter>,
    bloom: HashBloom,
    poisoned: bool,
}

impl EventLog {
    pub fn new(writer: Box<dyn LogWriter>, bloom: HashBloom) -> Self {
        Self {
            writer,
            bloom,
            poisoned: false,
        }
    }

    #[cfg(test)]
    pub(super) fn bloom_might_contain(&self, hash: &[u8; 32]) -> bool {
        self.bloom.might_contain(hash)
    }
}

/// One lock orders every append against the commit that follows it.
pub type SharedLog = Arc<tokio::sync::Mutex<EventLog>>;

/// What a logged node write did.
pub(super) enum Logged {
    /// The row was projected; the caller's id is the node's id.
    Written,
    /// The content already has a live carrier, which is the id to return.
    Duplicate(NodeId),
}

/// What a projection changed, for the log's bookkeeping after the commit.
#[derive(Default)]
struct RowEffects {
    /// Content hashes the dedup filter must now know.
    hashes: Vec<[u8; 32]>,
}

impl<B: Backend> Graph<B> {
    /// Logs and projects one node. The vector write stays with the caller
    /// because it takes the write lock the transaction holds.
    pub(super) async fn insert_node_logged(
        &self,
        log: &SharedLog,
        id: &NodeId,
        node: &NewNode,
        now: Millis,
    ) -> Result<Logged> {
        let row = resolve_node_row(id, node, now)?;
        let event = node_write_event(&row, now);
        let hash = event.content_hash;
        self.transact(log, event, true, move |tx, event_id, _| {
            Box::pin(async move {
                let (sql, params) = node_row_insert(&row);
                tx.execute(&sql, &params).await?;
                index_hash(tx, &hash, &row.id, event_id).await?;
                Ok(RowEffects { hashes: vec![hash] })
            })
        })
        .await
    }

    /// The one path every logged write takes: poison check, log lock, `begin`,
    /// then append, project, and commit. With `dedup`, content that already has
    /// a live carrier is logged as a `DuplicateOf` instead of projected.
    async fn transact<F>(
        &self,
        log: &SharedLog,
        event: LogEvent,
        dedup: bool,
        project: F,
    ) -> Result<Logged>
    where
        F: for<'t> FnOnce(
                &'t mut dyn BackendTx,
                &'t str,
                LogOffset,
            ) -> BoxFuture<'t, Result<RowEffects>>
            + Send,
    {
        let mut held = HeldLog::acquire(log).await?;
        let mut tx = self.backend.begin().await?;
        tracing::debug!(event = %event.event_id, "log lock taken, transaction open");
        if dedup {
            match live_duplicate(
                &mut *tx,
                held.might_contain(&event.content_hash),
                &event.content_hash,
            )
            .await
            {
                Ok(Some(first)) => {
                    let duplicate = follow_up(
                        &event,
                        LogPayload::DuplicateOf {
                            first_event_id: first.event_id,
                        },
                    );
                    self.append_and_commit(tx, &mut held, duplicate, |_, _, _| {
                        Box::pin(async { Ok(RowEffects::default()) })
                    })
                    .await?;
                    return Ok(Logged::Duplicate(first.row_id));
                }
                Ok(None) => {}
                Err(error) => {
                    abandon(tx).await;
                    return Err(error);
                }
            }
        }
        self.append_and_commit(tx, &mut held, event, project)
            .await?;
        Ok(Logged::Written)
    }

    async fn append_and_commit<F>(
        &self,
        mut tx: Box<dyn BackendTx + '_>,
        held: &mut HeldLog,
        event: LogEvent,
        project: F,
    ) -> Result<()>
    where
        F: for<'t> FnOnce(
                &'t mut dyn BackendTx,
                &'t str,
                LogOffset,
            ) -> BoxFuture<'t, Result<RowEffects>>
            + Send,
    {
        let event_id = event.event_id.clone();
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
            let effects = project(&mut *tx, &event_id, offset).await?;
            let moved = cursor_params(held.log_id, offset);
            tx.execute(CURSOR_UPSERT_SQL, &moved).await?;
            Ok(effects)
        }
        .await;
        match commit_or_abandon(tx, projected).await {
            Ok(effects) => {
                held.reconcile(&effects.hashes);
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
                let moved = self
                    .backend
                    .execute(CURSOR_UPSERT_SQL, &cursor_params(held.log_id, offset))
                    .await;
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

/// The log record for a new node: the row's producer is the source, its
/// confidence the trust, and its valid time the observation time.
fn node_write_event(row: &NodeRow, now: Millis) -> LogEvent {
    LogEvent {
        event_id: Uuid::now_v7().to_string(),
        content_hash: node_row_hash(row),
        source: row.producer.clone(),
        trust_score: row.confidence,
        observed_at: row.valid_from,
        ingested_at: now.0,
        encryption_key_id: None,
        schema_version: CURRENT_SCHEMA_VERSION,
        payload: LogPayload::NodeWrite(row.clone()),
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
            hashes.iter().for_each(|hash| log.bloom.insert(hash));
            log.poisoned = false;
        }
    }

    fn set_poisoned(&mut self, poisoned: bool) {
        if let Some(log) = self.guard.as_deref_mut() {
            log.poisoned = poisoned;
        }
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
    row_id: NodeId,
}

/// A hit counts only while the indexed row is live. A superseded or
/// garbage-collected carrier is not a duplicate, and the write that follows
/// repoints the index at itself.
async fn live_duplicate(
    tx: &mut dyn BackendTx,
    bloom_hit: bool,
    hash: &[u8; 32],
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
    let Some(row_id) = row_ids.into_iter().next() else {
        return Ok(None);
    };
    if !is_live(tx, &row_id).await? {
        return Ok(None);
    }
    Ok(Some(Carrier {
        event_id: first_event_id,
        row_id: NodeId::from_raw(row_id),
    }))
}

async fn is_live(tx: &mut dyn BackendTx, node_id: &str) -> Result<bool> {
    let rows = tx
        .query(
            "SELECT 1 FROM nodes WHERE id = ?1 AND tx_to = ?2",
            &[node_id.into(), FOREVER.into()],
        )
        .await?;
    Ok(!rows.is_empty())
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

async fn index_hash(
    tx: &mut dyn BackendTx,
    hash: &[u8; 32],
    row_id: &str,
    event_id: &str,
) -> Result<()> {
    let row_ids = serde_json::to_string(&[row_id])?;
    tx.execute(
        HASH_INDEX_UPSERT_SQL,
        &[hash.to_vec().into(), event_id.into(), row_ids.into()],
    )
    .await?;
    Ok(())
}

fn cursor_params(log_id: Uuid, offset: LogOffset) -> [Value; 3] {
    [
        log_id.to_string().into(),
        (offset.segment as i64).into(),
        (offset.index as i64).into(),
    ]
}

async fn commit_or_abandon<T>(tx: Box<dyn BackendTx + '_>, applied: Result<T>) -> Result<T> {
    match applied {
        Ok(applied) => tx.commit().await.map(|()| applied),
        Err(error) => {
            abandon(tx).await;
            Err(error)
        }
    }
}

async fn abandon(tx: Box<dyn BackendTx + '_>) {
    if let Err(error) = tx.rollback().await {
        tracing::warn!(%error, "rollback of an abandoned write failed");
    }
}
