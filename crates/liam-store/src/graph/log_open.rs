// SPDX-License-Identifier: Apache-2.0
//! What opening a store on a log must settle before the first write: the log is
//! the one the store belongs to, the store has not applied records the log
//! lacks, the dedup filter knows every hash the store has indexed, and a store
//! whose backfill is owed is flagged so writes wait for it.

use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::LogOffset;
use uuid::Uuid;

use super::backfill::Progress;
use super::log_cursor::{self, Cursor};
use super::logged_write::SharedLog;
use super::rebuild::holds_rows;
use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::value::Row;

/// Hashes read per query, so rebuilding the filter never holds the whole index.
const HASH_PAGE: i64 = 500;

/// Runs the open-time checks in order, under the log lock so no write can
/// start between the checks and the filter swap. Nothing is written unless
/// the store has no cursor yet.
pub(super) async fn check_and_prime<B: Backend>(backend: &B, log: &SharedLog) -> Result<()> {
    let mut log = log.lock().await;
    let (log_id, head) = (log.log_id(), log.head());
    let cursor = match log_cursor::read(backend).await? {
        Some(cursor) => cursor,
        // A store with no cursor is fresh, or older than the log: its rows
        // predate the log and replay does not cover them. Either way the
        // cursor starts empty, since backfilling those rows is a separate step.
        // The row is read back because another handle may have created it first.
        None => {
            let created = log_cursor::create_if_absent(backend, log_id).await?;
            tracing::debug!(%log_id, "created the log cursor");
            created
        }
    };
    check_cursor(&cursor, log_id, head)?;
    tracing::debug!(%log_id, "log cursor accepted");
    let bloom = rebuild_bloom(backend, log.bloom_config()).await?;
    log.replace_bloom(bloom);
    tracing::debug!("dedup filter rebuilt from the hash index");
    let pending = backfill_pending(backend).await?;
    log.set_backfill_pending(pending);
    Ok(())
}

/// Whether the store holds rows whose backfill has not completed, so a write
/// to the log would land ahead of them.
async fn backfill_pending<B: Backend>(backend: &B) -> Result<bool> {
    let saved = Progress::load(backend).await?;
    let complete = saved.as_ref().is_some_and(Progress::is_complete);
    Ok(!complete && holds_rows(backend).await?)
}

/// A cursor behind the head is fine: the records between are replayed.
fn check_cursor(cursor: &Cursor, log_id: Uuid, head: Option<LogOffset>) -> Result<()> {
    if cursor.log_id != log_id {
        return Err(Error::LogIdMismatch {
            store: cursor.log_id,
            log: log_id,
        });
    }
    match cursor.last {
        Some(last) if head.is_none_or(|head| last > head) => {
            Err(Error::CursorBeyondLog { cursor: last, head })
        }
        _ => Ok(()),
    }
}

/// Inserts every indexed hash into a filter sized for the index with headroom,
/// reading it in key order a page at a time.
async fn rebuild_bloom<B: Backend>(backend: &B, configured: &BloomConfig) -> Result<HashBloom> {
    let rows = backend
        .query("SELECT COUNT(*) FROM log_hash_index", &[])
        .await?[0]
        .get_i64(0)?;
    let mut bloom = HashBloom::new(configured.grown_to(rows.max(0) as usize));
    // The empty blob sorts before every hash.
    let mut after: Vec<u8> = Vec::new();
    loop {
        let page = backend
            .query(
                "SELECT content_hash FROM log_hash_index
                 WHERE content_hash > ?1 ORDER BY content_hash LIMIT ?2",
                &[after.clone().into(), HASH_PAGE.into()],
            )
            .await?;
        for row in &page {
            bloom.insert(&hash_of(row)?);
        }
        match page.last() {
            Some(last) if page.len() as i64 == HASH_PAGE => after = last.get_blob(0)?.to_vec(),
            _ => return Ok(bloom),
        }
    }
}

fn hash_of(row: &Row) -> Result<[u8; 32]> {
    let stored = row.get_blob(0)?;
    <[u8; 32]>::try_from(stored).map_err(|_| {
        Error::CorruptLogState(format!(
            "log_hash_index holds a content_hash of {} bytes, not 32",
            stored.len()
        ))
    })
}
