// SPDX-License-Identifier: Apache-2.0
//! The `log_cursor` table: one row naming the log the store belongs to and the
//! last log record the store has accounted for.

use liam_log::LogOffset;
use uuid::Uuid;

use crate::backend::{Backend, BackendTx};
use crate::error::{Error, Result};
use crate::value::Value;

const READ_SQL: &str = "SELECT log_id, last_segment, last_index FROM log_cursor WHERE id = 1";

const CREATE_SQL: &str = "INSERT INTO log_cursor (id, log_id, last_segment, last_index)
     VALUES (1, ?1, NULL, NULL) ON CONFLICT(id) DO NOTHING";

// Advancing never rewrites log_id, so a database opened against a different log
// stays recognisable. Only a rebuild's restart does.
const ADVANCE_SQL: &str = "INSERT INTO log_cursor (id, log_id, last_segment, last_index)
     VALUES (1, ?1, ?2, ?3)
     ON CONFLICT(id) DO UPDATE SET
       last_segment = excluded.last_segment, last_index = excluded.last_index";

const RESTART_SQL: &str = "INSERT INTO log_cursor (id, log_id, last_segment, last_index)
     VALUES (1, ?1, NULL, NULL)
     ON CONFLICT(id) DO UPDATE SET
       log_id = excluded.log_id, last_segment = NULL, last_index = NULL";

/// The stored cursor. `last` is `None` until a record has been accounted for.
pub(super) struct Cursor {
    pub(super) log_id: Uuid,
    pub(super) last: Option<LogOffset>,
}

/// The cursor row, or `None` when the store has none yet.
pub(super) async fn read<B: Backend>(backend: &B) -> Result<Option<Cursor>> {
    let rows = backend.query(READ_SQL, &[]).await?;
    let Some(row) = rows.first() else {
        return Ok(None);
    };
    let stored = row.get_string(0)?;
    let log_id = Uuid::parse_str(&stored).map_err(|_| {
        Error::CorruptLogState(format!(
            "log_cursor holds log_id {stored:?}, which is not a UUID"
        ))
    })?;
    Ok(Some(Cursor {
        log_id,
        last: offset_from_values(&row.0[1], &row.0[2])?,
    }))
}

/// Gives the store a cursor with no offsets for `log_id` unless it has one,
/// and returns the row that is stored: another handle may have created it
/// first, for a different log.
pub(super) async fn create_if_absent<B: Backend>(backend: &B, log_id: Uuid) -> Result<Cursor> {
    backend
        .execute(CREATE_SQL, &[log_id.to_string().into()])
        .await?;
    read(backend)
        .await?
        .ok_or_else(|| Error::Backend("log_cursor has no row right after it was created".into()))
}

/// Moves the cursor onto `offset` inside `tx`, so it commits with the write.
pub(super) async fn advance_in_tx(
    tx: &mut dyn BackendTx,
    log_id: Uuid,
    offset: LogOffset,
) -> Result<()> {
    tx.execute(ADVANCE_SQL, &advance_params(log_id, offset)?)
        .await?;
    Ok(())
}

/// Puts the cursor at the start of `log_id`'s log inside `tx`, taking over the
/// store for that log when it belonged to another.
pub(super) async fn restart_in_tx(tx: &mut dyn BackendTx, log_id: Uuid) -> Result<()> {
    tx.execute(RESTART_SQL, &[log_id.to_string().into()])
        .await?;
    Ok(())
}

/// Moves the cursor onto `offset` in a statement of its own.
pub(super) async fn advance<B: Backend>(
    backend: &B,
    log_id: Uuid,
    offset: LogOffset,
) -> Result<()> {
    backend
        .execute(ADVANCE_SQL, &advance_params(log_id, offset)?)
        .await?;
    Ok(())
}

fn advance_params(log_id: Uuid, offset: LogOffset) -> Result<[Value; 3]> {
    let (segment, index) = offset_to_values(offset)?;
    Ok([log_id.to_string().into(), segment, index])
}

/// The column values for an offset. SQLite integers are signed, so an offset
/// above `i64::MAX` cannot be stored.
pub(super) fn offset_to_values(offset: LogOffset) -> Result<(Value, Value)> {
    let column = |part: u64| {
        i64::try_from(part).map(Value::Int).map_err(|_| {
            Error::CorruptLogState(format!(
                "log offset {offset} does not fit the log_cursor columns"
            ))
        })
    };
    Ok((column(offset.segment)?, column(offset.index)?))
}

/// The offset held by two column values; both NULL means no offset yet.
pub(super) fn offset_from_values(segment: &Value, index: &Value) -> Result<Option<LogOffset>> {
    match (segment, index) {
        (Value::Null, Value::Null) => Ok(None),
        (Value::Int(segment), Value::Int(index)) => Ok(Some(LogOffset {
            segment: column_to_u64(*segment)?,
            index: column_to_u64(*index)?,
        })),
        _ => Err(Error::CorruptLogState(
            "log_cursor holds a segment without an index, or an index without a segment".into(),
        )),
    }
}

fn column_to_u64(value: i64) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| Error::CorruptLogState(format!("log_cursor holds the offset {value}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(segment: u64, index: u64) -> LogOffset {
        LogOffset { segment, index }
    }

    #[test]
    fn an_offset_survives_the_round_trip_through_column_values() {
        // Arrange
        let cases = [at(0, 0), at(3, 17), at(i64::MAX as u64, i64::MAX as u64)];

        for offset in cases {
            // Act
            let (segment, index) = offset_to_values(offset).expect("encode");
            let decoded = offset_from_values(&segment, &index).expect("decode");

            // Assert
            assert_eq!(decoded, Some(offset));
        }
    }

    #[test]
    fn an_offset_above_the_signed_range_is_refused_instead_of_wrapping() {
        // Arrange
        let cases = [at(i64::MAX as u64 + 1, 0), at(0, u64::MAX)];

        for offset in cases {
            // Act
            let encoded = offset_to_values(offset);

            // Assert
            assert!(
                matches!(encoded, Err(Error::CorruptLogState(_))),
                "{offset}: {encoded:?}"
            );
        }
    }

    #[test]
    fn a_negative_or_half_null_column_pair_is_corrupt_state() {
        // Arrange
        let cases = [
            (Value::Int(-1), Value::Int(0)),
            (Value::Int(0), Value::Int(-1)),
            (Value::Int(1), Value::Null),
            (Value::Null, Value::Int(1)),
        ];

        for (segment, index) in cases {
            // Act
            let decoded = offset_from_values(&segment, &index);

            // Assert
            assert!(
                matches!(decoded, Err(Error::CorruptLogState(_))),
                "{segment:?}, {index:?}: {decoded:?}"
            );
        }
    }

    #[test]
    fn both_columns_null_is_no_offset() {
        // Act
        let decoded = offset_from_values(&Value::Null, &Value::Null);

        // Assert
        assert_eq!(decoded.expect("decode"), None);
    }
}
