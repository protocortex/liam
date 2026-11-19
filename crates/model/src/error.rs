// SPDX-License-Identifier: Apache-2.0
//! Model-side errors: embedding and generation failures.
//! `Display` includes the cause on purpose, because most callers print the plain form.

use std::sync::Arc;

use thiserror::Error;

/// The underlying failure of a model call. Shared, so one error can be handed
/// to every caller that waited on the same load.
type Cause = Arc<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum ModelError {
    #[error("embedder: {0}")]
    Embed(String),

    #[error("llm: {0}")]
    Llm(String),

    /// An embedding step failed on an underlying error, kept as the source.
    #[error("embedder: {context}: {source}")]
    EmbedFailed {
        context: String,
        #[source]
        source: Cause,
    },

    /// A generation step failed on an underlying error, kept as the source.
    #[error("llm: {context}: {source}")]
    LlmFailed {
        context: String,
        #[source]
        source: Cause,
    },
}

impl ModelError {
    /// An embedding failure that keeps `source`. `context` says which step failed.
    pub fn embed_failed(
        context: impl Into<String>,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::EmbedFailed {
            context: context.into(),
            source: Arc::from(source.into()),
        }
    }

    /// A generation failure that keeps `source`. `context` says which step failed.
    pub fn llm_failed(
        context: impl Into<String>,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::LlmFailed {
            context: context.into(),
            source: Arc::from(source.into()),
        }
    }
}

pub type Result<T> = std::result::Result<T, ModelError>;

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn a_failure_with_a_cause_keeps_it_as_the_source_and_in_the_plain_text() {
        // Arrange
        let cause = std::io::Error::other("weights file is truncated");

        // Act
        let error = ModelError::embed_failed("load arctic", cause);

        // Assert
        let source = error.source().expect("the cause is the source");
        assert!(source.to_string().contains("weights file is truncated"));
        assert!(error.to_string().contains("weights file is truncated"));
    }

    #[test]
    fn a_failed_load_prints_the_text_it_always_did() {
        // Arrange
        let cause = std::io::Error::other("weights file is truncated");

        // Act
        let text = ModelError::embed_failed("load arctic", cause).to_string();

        // Assert
        assert_eq!(text, "embedder: load arctic: weights file is truncated");
    }

    #[test]
    fn a_cloned_failure_still_reaches_the_same_cause() {
        // Arrange
        let error = ModelError::llm_failed("load", std::io::Error::other("no memory"));

        // Act
        let copy = error.clone();

        // Assert
        assert!(copy.source().unwrap().to_string().contains("no memory"));
        assert_eq!(copy.to_string(), "llm: load: no memory");
    }
}
