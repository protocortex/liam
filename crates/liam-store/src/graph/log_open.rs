// SPDX-License-Identifier: Apache-2.0
//! What opening a store on a log must settle before the first write: the log is
//! the one the store belongs to, the store has not applied records the log
//! lacks, and the dedup filter knows every hash the store has indexed.

use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::LogOffset;
use uuid::Uuid;

use super::logged_write::SharedLog;
use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::value::{Row, Value};

/// Hashes read per query, so rebuilding the filter never holds the whole index.
const HASH_PAGE: i64 = 500;

const CREATE_CURSOR_SQL: &str = "INSERT INTO log_cursor (id, log_id, last_segment, last_index)
     VALUES (1, ?1, NULL, NULL) ON CONFLICT(id) DO NOTHING";

struct Cursor {
    log_id: String,
    last: Option<LogOffset>,
}

/// Runs the open-time checks in order, under the log lock so no write can
/// start between the checks and the filter swap. Nothing is written unless
/// the store has no cursor yet.
pub(super) async fn check_and_prime<B: Backend>(backend: &B, log: &SharedLog) -> Result<()> {
    let mut log = log.lock().await;
    let (log_id, head) = (log.log_id(), log.head());
    match read_cursor(backend).await? {
        // A store with no cursor is fresh, or older than the log: its rows
        // predate the log and replay does not cover them. Either way the
        // cursor starts empty, since backfilling those rows is a separate step.
        None => {
            backend
                .execute(CREATE_CURSOR_SQL, &[log_id.to_string().into()])
                .await?;
            tracing::debug!(%log_id, "created the log cursor");
        }
        Some(cursor) => {
            check_cursor(&cursor, log_id, head)?;
            tracing::debug!(%log_id, "log cursor accepted");
        }
    }
    let bloom = rebuild_bloom(backend, log.bloom_config()).await?;
    log.replace_bloom(bloom);
    tracing::debug!("dedup filter rebuilt from the hash index");
    Ok(())
}

async fn read_cursor<B: Backend>(backend: &B) -> Result<Option<Cursor>> {
    let rows = backend
        .query(
            "SELECT log_id, last_segment, last_index FROM log_cursor WHERE id = 1",
            &[],
        )
        .await?;
    rows.first()
        .map(|row| {
            Ok(Cursor {
                log_id: row.get_string(0)?,
                last: offset_of(row)?,
            })
        })
        .transpose()
}

fn offset_of(row: &Row) -> Result<Option<LogOffset>> {
    match (&row.0[1], &row.0[2]) {
        (Value::Null, Value::Null) => Ok(None),
        (Value::Int(segment), Value::Int(index)) => Ok(Some(LogOffset {
            segment: to_u64(*segment)?,
            index: to_u64(*index)?,
        })),
        _ => Err(Error::Backend(
            "log_cursor holds a segment without an index, or an index without a segment".into(),
        )),
    }
}

fn to_u64(value: i64) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| Error::Backend("log_cursor holds a negative offset".to_string()))
}

/// A cursor behind the head is fine: the records between are replayed.
fn check_cursor(cursor: &Cursor, log_id: Uuid, head: Option<LogOffset>) -> Result<()> {
    if cursor.log_id != log_id.to_string() {
        return Err(Error::LogIdMismatch {
            store: cursor.log_id.clone(),
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

/// Inserts every indexed hash into a filter sized for the index, reading it
/// in key order a page at a time.
async fn rebuild_bloom<B: Backend>(backend: &B, floor: &BloomConfig) -> Result<HashBloom> {
    let rows = backend
        .query("SELECT COUNT(*) FROM log_hash_index", &[])
        .await?[0]
        .get_i64(0)?;
    let expected = (rows.max(0) as usize).max(floor.expected_items());
    let mut bloom = HashBloom::new(BloomConfig::new(expected, floor.false_positive_rate())?);
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
    <[u8; 32]>::try_from(row.get_blob(0)?).map_err(|_| {
        Error::Backend("log_hash_index holds a content_hash that is not 32 bytes".into())
    })
}
