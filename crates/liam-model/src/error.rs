// SPDX-License-Identifier: Apache-2.0
//! Model-side errors: embedding and reranking failures.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("embedder: {0}")]
    Embed(String),

    #[error("reranker: {0}")]
    Rerank(String),

    /// The model directory lacks a pinned file, as before an install.
    #[error("reranker files not found: {0}")]
    RerankFilesNotFound(String),

    #[error("llm: {0}")]
    Llm(String),
}

pub type Result<T> = std::result::Result<T, ModelError>;
