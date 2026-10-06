// SPDX-License-Identifier: Apache-2.0
//! The log-first write path. A write is appended to the event log and then
//! projected into the store inside one transaction, with the log lock held
//! throughout, so the log's order is the commit order and a replay of the log
//! rebuilds the store.
//!
//! The log cursor tracks the last log record the store has accounted for,
//! whether that record was applied or voided. A write that is appended but
//! cannot be projected is cancelled with a `Voided` record, and the cursor
//! moves onto the void only if that append succeeds. If it does not, the log
//! holds a write the store never applied and cannot be reconciled in this
//! process, so the log is poisoned until the store is reopened.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex, PoisonError};

use async_trait::async_trait;
use liam_log::dedup::{
    find_first_write, BloomConfig, HashBloom, HashIndex, IndexedWrite, PreCheck,
};
use liam_log::event::{LogEvent, LogPayload, NodeRow, CURRENT_SCHEMA_VERSION};
use liam_log::hash::node_row_hash;
use liam_log::{LogOffset, LogWriter};
use tokio::sync::OwnedMutexGuard;
use uuid::Uuid;

use super::{opt_text, Graph, SharedLog};
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

/// What a logged node write did.
pub(super) enum Logged {
    /// The row was projected; the caller's id is the node's id.
    Written,
    /// The content already has a live carrier, which is the id to return.
    Duplicate(NodeId),
}

/// The log a `Graph` appends to, plus the state that goes with it.
pub(super) struct LogState {
    writer: SharedLog,
    bloom: StdMutex<HashBloom>,
    poisoned: AtomicBool,
}

impl LogState {
    pub(super) fn new(writer: SharedLog) -> Self {
        Self {
            writer,
            bloom: StdMutex::new(bloom_for_open()),
            poisoned: AtomicBool::new(false),
        }
    }

    fn refuse_if_poisoned(&self) -> Result<()> {
        if self.poisoned.load(Ordering::SeqCst) {
            return Err(Error::LogPoisoned);
        }
        Ok(())
    }

    fn poison(&self) {
        self.poisoned.store(true, Ordering::SeqCst);
    }

    fn remember(&self, hash: &[u8; 32]) {
        self.bloom
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(hash);
    }
}

/// The filter a graph starts with. The open-time rebuild from `log_hash_index`
/// belongs here: until it exists, a hash written before this process started
/// is not pre-checked, so such a duplicate is stored rather than deduplicated.
fn bloom_for_open() -> HashBloom {
    HashBloom::new(BloomConfig::default())
}

impl<B: Backend> Graph<B> {
    /// Logs and projects one node: dedup against the log's live carriers, then
    /// append, project, and commit under the log lock. The vector write stays
    /// with the caller because it takes the write lock the transaction holds.
    pub(super) async fn insert_node_logged(
        &self,
        log: &LogState,
        id: &NodeId,
        node: &NewNode,
        now: Millis,
    ) -> Result<Logged> {
        let pending = PendingNode::resolve(id, node, now)?;
        let mut held = HeldLog::acquire(&log.writer).await;
        log.refuse_if_poisoned()?;
        let mut tx = self.backend.begin().await?;
        match live_duplicate(&mut *tx, &log.bloom, &pending.hash).await {
            Ok(Some(first)) => self.record_duplicate(tx, &mut held, &pending, first).await,
            Ok(None) => self.record_write(log, tx, held, pending).await,
            Err(error) => {
                abandon(tx).await;
                Err(error)
            }
        }
    }

    async fn record_duplicate(
        &self,
        mut tx: Box<dyn BackendTx + '_>,
        held: &mut HeldLog,
        pending: &PendingNode,
        first: Carrier,
    ) -> Result<Logged> {
        let event = pending.event(LogPayload::DuplicateOf {
            first_event_id: first.event_id,
        });
        let offset = match held.append(event).await {
            Ok(offset) => offset,
            Err(error) => {
                abandon(tx).await;
                return Err(error);
            }
        };
        let moved = advance_cursor(&mut *tx, held.log_id, offset).await;
        commit_or_abandon(tx, moved).await?;
        tracing::debug!(
            node = first.row_id.as_str(),
            "duplicate write logged, no row written"
        );
        Ok(Logged::Duplicate(first.row_id))
    }

    async fn record_write(
        &self,
        log: &LogState,
        mut tx: Box<dyn BackendTx + '_>,
        mut held: HeldLog,
        pending: PendingNode,
    ) -> Result<Logged> {
        let event = pending.event(LogPayload::NodeWrite(pending.row.clone()));
        let event_id = event.event_id.clone();
        let offset = match held.append(event).await {
            Ok(offset) => offset,
            Err(error) => {
                abandon(tx).await;
                return Err(error);
            }
        };
        tracing::debug!(node = %pending.row.id, %event_id, "node write appended to the log");
        let projected = project_node(&mut *tx, &pending, &event_id, held.log_id, offset).await;
        match commit_or_abandon(tx, projected).await {
            Ok(()) => {
                log.remember(&pending.hash);
                Ok(Logged::Written)
            }
            Err(error) => {
                self.void_failed_write(log, &mut held, &pending, event_id)
                    .await;
                Err(error)
            }
        }
    }

    /// Cancels an appended write the store failed to apply. Never fails: if
    /// the void cannot be appended the log is poisoned instead.
    async fn void_failed_write(
        &self,
        log: &LogState,
        held: &mut HeldLog,
        pending: &PendingNode,
        target_event_id: String,
    ) {
        let void = pending.event(LogPayload::Voided { target_event_id });
        match held.append(void).await {
            Ok(offset) => {
                let moved = self
                    .backend
                    .execute(CURSOR_UPSERT_SQL, &cursor_params(held.log_id, offset))
                    .await;
                if let Err(error) = moved {
                    tracing::warn!(node = %pending.row.id, %error, "voided write logged but the cursor did not advance");
                }
            }
            Err(error) => {
                log.poison();
                tracing::error!(node = %pending.row.id, %error, "could not void a failed write, the log is poisoned until reopen");
            }
        }
    }
}

/// A node resolved to the row the store will hold, with its content hash.
struct PendingNode {
    row: NodeRow,
    hash: [u8; 32],
    now: Millis,
}

impl PendingNode {
    fn resolve(id: &NodeId, node: &NewNode, now: Millis) -> Result<Self> {
        let row = resolve_node_row(id, node, now)?;
        let hash = node_row_hash(&row);
        Ok(Self { row, hash, now })
    }

    /// The log record around `payload`: the row's producer is the source, its
    /// confidence the trust, and its valid time the observation time.
    fn event(&self, payload: LogPayload) -> LogEvent {
        LogEvent {
            event_id: Uuid::now_v7().to_string(),
            content_hash: self.hash,
            source: self.row.producer.clone(),
            trust_score: self.row.confidence,
            observed_at: self.row.valid_from,
            ingested_at: self.now.0,
            encryption_key_id: None,
            schema_version: CURRENT_SCHEMA_VERSION,
            payload,
        }
    }
}

/// The row a `NewNode` becomes once ids and times are minted and its scope and
/// attributes are in stored form. The log hashes and stores this, never the
/// raw request.
pub(super) fn resolve_node_row(id: &NodeId, node: &NewNode, now: Millis) -> Result<NodeRow> {
    Ok(NodeRow {
        id: id.as_str().to_string(),
        kind: node.kind.clone(),
        label: node.label.clone(),
        content: node.content.clone(),
        producer: node.producer.clone(),
        attributes: serde_json::to_string(&node.attributes)?,
        scope: node.scope.clone(),
        subject: node.subject.clone(),
        confidence: node.confidence,
        valid_from: node.valid_from.unwrap_or(now).0,
        valid_from_supplied: node.valid_from.is_some(),
        valid_until: FOREVER.0,
        tx_from: now.0,
        tx_to: FOREVER.0,
    })
}

pub(super) fn node_row_insert(row: &NodeRow) -> (String, Vec<Value>) {
    let sql = "INSERT INTO nodes
         (id, kind, label, content, producer, attributes, scope, subject, confidence,
          valid_from, valid_until, tx_from, tx_to)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)"
        .to_string();
    let params = vec![
        row.id.clone().into(),
        row.kind.clone().into(),
        row.label.clone().into(),
        row.content.clone().into(),
        row.producer.clone().into(),
        row.attributes.clone().into(),
        opt_text(row.scope.clone()),
        opt_text(row.subject.clone()),
        Value::Real(row.confidence),
        row.valid_from.into(),
        row.valid_until.into(),
        row.tx_from.into(),
        row.tx_to.into(),
    ];
    (sql, params)
}

/// The log writer, locked for the whole transaction. `append` blocks on fsync,
/// so it runs on the blocking pool with the guard moved in and back out.
struct HeldLog {
    guard: Option<OwnedMutexGuard<Box<dyn LogWriter>>>,
    log_id: Uuid,
}

impl HeldLog {
    async fn acquire(writer: &SharedLog) -> Self {
        let guard = Arc::clone(writer).lock_owned().await;
        let log_id = guard.log_id();
        Self {
            guard: Some(guard),
            log_id,
        }
    }

    async fn append(&mut self, event: LogEvent) -> Result<LogOffset> {
        let Some(mut guard) = self.guard.take() else {
            return Err(Error::LogPoisoned);
        };
        let joined = tokio::task::spawn_blocking(move || {
            let appended = guard.append(&event);
            (guard, appended)
        })
        .await;
        let (guard, appended) =
            joined.map_err(|error| Error::LogAppend(format!("append task failed: {error}")))?;
        self.guard = Some(guard);
        appended.map_err(|error| Error::LogAppend(error.to_string()))
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
    bloom: &StdMutex<HashBloom>,
    hash: &[u8; 32],
) -> Result<Option<Carrier>> {
    let hit = {
        let mut index = TxHashIndex { tx: &mut *tx };
        find_first_write(&BloomHandle(bloom), &mut index, hash).await?
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

/// The bloom filter as a `PreCheck`, locked only for the instant of each probe
/// so no std guard lives across an await.
struct BloomHandle<'a>(&'a StdMutex<HashBloom>);

impl PreCheck for BloomHandle<'_> {
    fn might_contain(&self, hash: &[u8; 32]) -> bool {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .might_contain(hash)
    }
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

/// Projection statements for a new node. The node insert comes first.
async fn project_node(
    tx: &mut dyn BackendTx,
    pending: &PendingNode,
    event_id: &str,
    log_id: Uuid,
    offset: LogOffset,
) -> Result<()> {
    let (sql, params) = node_row_insert(&pending.row);
    tx.execute(&sql, &params).await?;
    index_hash(tx, pending, event_id).await?;
    advance_cursor(tx, log_id, offset).await
}

async fn index_hash(tx: &mut dyn BackendTx, pending: &PendingNode, event_id: &str) -> Result<()> {
    let row_ids = serde_json::to_string(&[&pending.row.id])?;
    tx.execute(
        HASH_INDEX_UPSERT_SQL,
        &[
            pending.hash.to_vec().into(),
            event_id.into(),
            row_ids.into(),
        ],
    )
    .await?;
    Ok(())
}

async fn advance_cursor(tx: &mut dyn BackendTx, log_id: Uuid, offset: LogOffset) -> Result<()> {
    tx.execute(CURSOR_UPSERT_SQL, &cursor_params(log_id, offset))
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

async fn commit_or_abandon(tx: Box<dyn BackendTx + '_>, applied: Result<()>) -> Result<()> {
    match applied {
        Ok(()) => tx.commit().await,
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
