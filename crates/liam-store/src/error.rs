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
}

pub type Result<T> = std::result::Result<T, Error>;
