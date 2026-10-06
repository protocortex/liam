// SPDX-License-Identifier: Apache-2.0
//! One error type, backend-neutral. Backends map their native error into
//! `Backend(String)`, so the crate does not depend on any one engine's error.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("backend: {0}")]
    Backend(String),

    #[error("serialize attributes: {0}")]
    Attributes(#[from] serde_json::Error),

    #[error("embedding dimension mismatch: expected {expected}, got {got}")]
    Dimension { expected: usize, got: usize },

    #[error("node not found: {0}")]
    NodeNotFound(String),

    #[error("no live node matches handle {0}")]
    HandleNotFound(String),

    /// Two or more live nodes share the prefix the client sent. Carries every
    /// candidate in full because the client was shown a 13-character handle
    /// (ADR-0001 Amendment 3) and cannot lengthen it without being told what
    /// the alternatives are.
    #[error("handle {handle} matches more than one live node: {}", .candidates.join(", "))]
    AmbiguousHandle {
        handle: String,
        candidates: Vec<String>,
    },

    /// The conditional INSERT in `Graph::relate` wrote no row. The message
    /// names which of its three guards refused.
    #[error("relate refused: {0}")]
    RelateRefused(String),

    /// An `EpisodeRef::New(i)` passed to `Graph::ingest_episode` named an
    /// index outside the bounds of that call's own `nodes` list.
    #[error("invalid reference: {0}")]
    InvalidReference(String),

    /// `validate_scope` rejected a `scope` value: empty after trimming, over
    /// the length cap, an invalid character, or a malformed `/`-segment shape.
    /// Carries the specific reason.
    #[error("invalid scope: {0}")]
    InvalidScope(String),

    /// The event log refused an append. The source keeps the writer's own
    /// error, so a poisoned writer stays distinguishable from an I/O failure.
    #[error("event log append failed: {0}")]
    LogAppend(#[from] liam_log::wal::WalError),

    /// The blocking task that runs an append panicked or was cancelled, so
    /// whether the record reached the log is unknown.
    #[error("event log append task failed: {0}")]
    LogTask(String),

    /// A write's log record is not reconciled with the projection. The store
    /// refuses logged writes until it is reopened.
    #[error("a logged write is not reconciled with the log; reopen the store")]
    LogPoisoned,

    /// The dedup filter could not be sized for the hash index.
    #[error("dedup filter sizing is invalid: {0}")]
    BloomConfig(#[from] liam_log::dedup::BloomConfigError),

    /// The store was last written through a different log, so its cursor
    /// offsets mean nothing to this one.
    #[error(
        "the store belongs to log {store}, not to the log {log} it was opened with; \
         open the store with the log directory of log {store}, or rebuild it from this log"
    )]
    LogIdMismatch { store: uuid::Uuid, log: uuid::Uuid },

    /// The store has applied records the log does not hold, so the log is
    /// older than the store or lost its tail.
    #[error(
        "the store's cursor is at {cursor}, beyond {end}; \
         restore the missing log segments or rebuild the store from the log",
        end = log_end(.head)
    )]
    CursorBeyondLog {
        cursor: liam_log::LogOffset,
        head: Option<liam_log::LogOffset>,
    },

    /// The log was handed to the store without a reader, so replay cannot
    /// scan it.
    #[error(
        "the event log has no reader, so it cannot be replayed; \
         build the log with EventLog::with_reader before calling catch_up"
    )]
    LogReaderMissing,

    /// The log could not be scanned.
    #[error("event log read failed: {0}")]
    LogRead(#[from] liam_log::reader::ReaderError),

    /// A log table holds a value no write of this store produces, so the
    /// store was edited or damaged outside it.
    #[error("the log state in the store is corrupt: {0}")]
    CorruptLogState(String),
}

fn log_end(head: &Option<liam_log::LogOffset>) -> String {
    match head {
        Some(head) => format!("the log head at {head}"),
        None => "an empty log".to_string(),
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use liam_log::LogOffset;
    use uuid::Uuid;

    use super::*;

    #[test]
    fn a_cursor_beyond_an_empty_log_names_the_log_as_empty_and_says_what_to_do() {
        // Arrange
        let error = Error::CursorBeyondLog {
            cursor: LogOffset {
                segment: 1,
                index: 3,
            },
            head: None,
        };

        // Act
        let message = error.to_string();

        // Assert
        assert_eq!(
            message,
            "the store's cursor is at segment 1, record 3, beyond an empty log; \
             restore the missing log segments or rebuild the store from the log"
        );
    }

    #[test]
    fn a_cursor_beyond_the_head_names_both_positions() {
        // Arrange
        let error = Error::CursorBeyondLog {
            cursor: LogOffset {
                segment: 2,
                index: 0,
            },
            head: Some(LogOffset {
                segment: 1,
                index: 9,
            }),
        };

        // Act
        let message = error.to_string();

        // Assert
        assert!(
            message.contains("segment 2, record 0, beyond the log head at segment 1, record 9;"),
            "{message}"
        );
    }

    #[test]
    fn a_log_id_mismatch_ends_with_how_to_recover() {
        // Arrange
        let (store, log) = (Uuid::from_u128(1), Uuid::from_u128(2));

        // Act
        let message = Error::LogIdMismatch { store, log }.to_string();

        // Assert
        assert!(
            message.ends_with(&format!(
                "open the store with the log directory of log {store}, or rebuild it from this log"
            )),
            "{message}"
        );
    }
}
