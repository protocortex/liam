// SPDX-License-Identifier: Apache-2.0
//! Model-side errors: embedding and generation failures.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("embedder: {0}")]
    Embed(String),

    #[error("llm: {0}")]
    Llm(String),
}

pub type Result<T> = std::result::Result<T, ModelError>;
