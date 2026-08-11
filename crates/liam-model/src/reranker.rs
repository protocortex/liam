// SPDX-License-Identifier: MIT OR Apache-2.0
//! Reranking: a precision stage after retrieval. Given the query and candidate
//! texts, score them so the store's fused order can be refined. This is the
//! local cross-encoder step; it needs a model, so it lives here, not in the store.

use async_trait::async_trait;

use crate::error::Result;

#[async_trait]
pub trait Reranker: Send + Sync {
    /// Return one relevance score per document, aligned to `docs`. Higher is
    /// more relevant. The caller reorders by these scores.
    async fn scores(&self, query: &str, docs: &[String]) -> Result<Vec<f32>>;

    /// Convenience: indices of `docs` ordered most relevant first.
    async fn order(&self, query: &str, docs: &[String]) -> Result<Vec<usize>> {
        let scores = self.scores(query, docs).await?;
        let mut idx: Vec<usize> = (0..docs.len()).collect();
        idx.sort_by(|a, b| {
            scores[*b]
                .partial_cmp(&scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(idx)
    }
}

/// Keeps the incoming order (uniform scores). The default when no model is set,
/// so retrieval still works and reranking is a pure enhancement.
pub struct IdentityReranker;

#[async_trait]
impl Reranker for IdentityReranker {
    async fn scores(&self, _query: &str, docs: &[String]) -> Result<Vec<f32>> {
        Ok(vec![0.0; docs.len()])
    }
}

/// In-process cross-encoder reranker over fastembed-rs (feature `local`).
///
/// VERSION CHECK: confirm against fastembed v5: `TextRerank::try_new(..)` and
/// `rerank(query, docs, return_documents, batch) -> Vec<RerankResult>` with
/// `index` and `score` fields.
#[cfg(feature = "local")]
pub struct FastEmbedReranker {
    model: std::sync::Arc<std::sync::Mutex<fastembed::TextRerank>>,
}

#[cfg(feature = "local")]
impl FastEmbedReranker {
    /// Load the default cross-encoder reranker (a BGE reranker).
    pub fn load() -> Result<Self> {
        let model = fastembed::TextRerank::try_new(Default::default())
            .map_err(|e| crate::error::ModelError::Rerank(e.to_string()))?;
        Ok(Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(model)),
        })
    }
}

#[cfg(feature = "local")]
#[async_trait]
impl Reranker for FastEmbedReranker {
    async fn scores(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        let model = self.model.clone();
        let query = query.to_string();
        let docs = docs.to_vec();
        let n = docs.len();
        let results = tokio::task::spawn_blocking(move || {
            let mut m = model
                .lock()
                .map_err(|_| crate::error::ModelError::Rerank("model lock poisoned".into()))?;
            let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
            m.rerank(query.as_str(), refs, false, None)
                .map_err(|e| crate::error::ModelError::Rerank(e.to_string()))
        })
        .await
        .map_err(|e| crate::error::ModelError::Rerank(e.to_string()))??;

        let mut scores = vec![0.0f32; n];
        for r in results {
            if r.index < n {
                scores[r.index] = r.score;
            }
        }
        Ok(scores)
    }
}
