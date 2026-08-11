// SPDX-License-Identifier: MIT OR Apache-2.0
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
}

pub type Result<T> = std::result::Result<T, Error>;
