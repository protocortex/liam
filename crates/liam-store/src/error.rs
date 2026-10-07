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

    /// A guard of the conditional edge INSERT flipped between the insert and
    /// the read that diagnoses it, so a retry would land. Kept apart from
    /// `RelateRefused` so a replay never mistakes it for a verdict on the data.
    #[error("a concurrent write took the row, retry")]
    ConcurrentWrite,

    /// The engine refused a statement on a constraint (foreign key, NOT NULL,
    /// UNIQUE). The same statement against the same data fails again.
    #[error("constraint violated: {0}")]
    Constraint(String),

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

    /// The log could not be scanned.
    #[error("event log read failed: {0}")]
    LogRead(#[from] liam_log::reader::ReaderError),

    /// Replay found some rows of an event in the store and others missing, which
    /// no write of this store leaves behind.
    #[error("event {0} is only partly in the store")]
    PartlyApplied(String),

    /// A rebuild or a log check needs an event log and the store has none.
    #[error("the store has no event log attached; open it with the log it was written through")]
    NoEventLog,

    /// A rebuild asked to start from an empty projection found rows in it.
    #[error(
        "the store already holds projected rows; a rebuild that clears them \
         has to be asked for, with a mode that does"
    )]
    ProjectionNotEmpty,

    /// The live nodes and edges are not the multiset of row hashes the log
    /// says they should be, even when the counts agree.
    #[error(
        "{subject} does not match the log: expected {expected} live rows, \
         found {found}, with {missing} missing and {unexpected} unexpected; {advice}",
        subject = .against.subject(),
        advice = .against.advice()
    )]
    RebuildMismatch {
        against: MismatchSource,
        expected: usize,
        found: usize,
        missing: usize,
        unexpected: usize,
    },

    /// A replacing rebuild would delete rows from a log that has no record at
    /// all, which is a log that was lost or pointed at the wrong place.
    #[error(
        "the log is empty but the store holds {rows_unknown_to_log} live rows \
         the log does not know; a rebuild would delete them all, so check the \
         log directory first"
    )]
    EmptyLogWouldWipe { rows_unknown_to_log: usize },

    /// A replacing rebuild would delete rows to take on a log other than the
    /// one the store was built from, which is only allowed onto an empty store.
    #[error(
        "the store belongs to log {store} and holds {rows_unknown_to_log} live \
         rows log {log} does not know; a rebuild would delete them, so open the \
         store with the log directory of log {store}, or start from an empty store"
    )]
    ForeignLogWouldWipe {
        store: uuid::Uuid,
        log: uuid::Uuid,
        rows_unknown_to_log: usize,
    },

    /// A log table holds a value no write of this store produces, so the
    /// store was edited or damaged outside it.
    #[error("the log state in the store is corrupt: {0}")]
    CorruptLogState(String),

    /// The store holds rows the log does not record yet, so a logged write
    /// would sit ahead of them and a replay could not apply it.
    #[error("the store has rows the log does not hold yet: run the backfill first")]
    BackfillRequired,
}

/// What the log's live rows were compared with when they did not match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MismatchSource {
    /// The rows the replay produced. The store is as the replay left it.
    Log,
    /// The rows the store held before the rebuild, which had not touched it.
    PriorProjection,
}

impl MismatchSource {
    fn subject(self) -> &'static str {
        match self {
            Self::Log => "the rebuilt projection",
            Self::PriorProjection => "the projection",
        }
    }

    fn advice(self) -> &'static str {
        match self {
            Self::Log => {
                "the log and the store disagree, so restore the missing log segments \
                 or the database from a backup before trusting this store"
            }
            Self::PriorProjection => {
                "nothing was changed; unexpected rows are what rows_unknown_to_log counts \
                 and the Replace mode drops them, and a store that is only behind the \
                 log is brought up to date with catch_up"
            }
        }
    }
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
    fn a_mismatch_gives_the_advice_that_fits_what_was_compared() {
        // Arrange
        let mismatch = |against| Error::RebuildMismatch {
            against,
            expected: 5,
            found: 3,
            missing: 3,
            unexpected: 1,
        };

        // Act
        let prior = mismatch(MismatchSource::PriorProjection).to_string();
        let rebuilt = mismatch(MismatchSource::Log).to_string();

        // Assert
        assert_eq!(
            prior,
            "the projection does not match the log: expected 5 live rows, found 3, with 3 \
             missing and 1 unexpected; nothing was changed; unexpected rows are what \
             rows_unknown_to_log counts and the Replace mode drops them, and a store that \
             is only behind the log is brought up to date with catch_up"
        );
        assert_eq!(
            rebuilt,
            "the rebuilt projection does not match the log: expected 5 live rows, found 3, \
             with 3 missing and 1 unexpected; the log and the store disagree, so restore \
             the missing log segments or the database from a backup before trusting this store"
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
