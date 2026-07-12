//! Model-side errors: embedding and reranking failures.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("embedder: {0}")]
    Embed(String),

    #[error("reranker: {0}")]
    Rerank(String),
}

pub type Result<T> = std::result::Result<T, ModelError>;
