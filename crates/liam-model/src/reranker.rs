// SPDX-License-Identifier: Apache-2.0
//! Reranking: a precision stage after retrieval. Given the query and candidate
//! texts, score them so the store's fused order can be refined. This is the
//! local cross-encoder step; it needs a model, so it lives here, not in the store.

use async_trait::async_trait;

use crate::error::{ModelError, Result};

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

/// Scores documents with a model that one call may use at a time.
pub(crate) trait DocScorer: Send + 'static {
    fn score(&mut self, query: &str, docs: &[String]) -> Result<Vec<f32>>;
}

impl DocScorer for fastembed::TextRerank {
    fn score(&mut self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let results = self
            .rerank(query, refs, false, None)
            .map_err(|e| crate::error::ModelError::Rerank(e.to_string()))?;
        let mut scores = vec![0.0f32; docs.len()];
        for r in results {
            if r.index < docs.len() {
                scores[r.index] = r.score;
            }
        }
        Ok(scores)
    }
}

/// A reranker over one shared scorer, run on a blocking thread.
pub(crate) struct ModelReranker<S: DocScorer> {
    model: std::sync::Arc<tokio::sync::Mutex<S>>,
}

impl<S: DocScorer> ModelReranker<S> {
    pub(crate) fn new(scorer: S) -> Self {
        Self {
            model: std::sync::Arc::new(tokio::sync::Mutex::new(scorer)),
        }
    }
}

#[async_trait]
impl<S: DocScorer> Reranker for ModelReranker<S> {
    async fn scores(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        let query = query.to_string();
        let docs = docs.to_vec();
        // The owned guard is taken before the blocking job exists, so a caller
        // that gives up waiting never queues a scorer run behind the lock.
        let mut model = self.model.clone().lock_owned().await;
        tokio::task::spawn_blocking(move || model.score(&query, &docs))
            .await
            .map_err(|e| ModelError::Rerank(e.to_string()))?
    }
}

/// In-process cross-encoder reranker over fastembed-rs.
///
/// VERSION CHECK: confirm against fastembed v5: `TextRerank::try_new(..)` and
/// `rerank(query, docs, return_documents, batch) -> Vec<RerankResult>` with
/// `index` and `score` fields.
pub struct FastEmbedReranker {
    inner: ModelReranker<fastembed::TextRerank>,
}

impl FastEmbedReranker {
    /// Load the default cross-encoder reranker (a BGE reranker).
    pub fn load() -> Result<Self> {
        let model = fastembed::TextRerank::try_new(Default::default())
            .map_err(|e| crate::error::ModelError::Rerank(e.to_string()))?;
        Ok(Self {
            inner: ModelReranker::new(model),
        })
    }
}

#[async_trait]
impl Reranker for FastEmbedReranker {
    async fn scores(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        self.inner.scores(query, docs).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier, Mutex};
    use std::time::Duration;

    use super::*;

    /// Records each call's documents in order. The first call parks on
    /// `entered` and then `release`, so a test can hold the model busy.
    struct GatedScorer {
        calls: Arc<Mutex<Vec<Vec<String>>>>,
        entered: Arc<Barrier>,
        release: Arc<Barrier>,
    }

    impl DocScorer for GatedScorer {
        fn score(&mut self, _query: &str, docs: &[String]) -> Result<Vec<f32>> {
            let first = {
                let mut calls = self.calls.lock().unwrap();
                calls.push(docs.to_vec());
                calls.len() == 1
            };
            if first {
                self.entered.wait();
                self.release.wait();
            }
            Ok(docs.iter().map(|d| d.len() as f32).collect())
        }
    }

    fn docs(text: &str) -> Vec<String> {
        vec![text.to_string(), format!("{text}{text}")]
    }

    /// Waits on a barrier without stalling the single runtime thread.
    async fn wait_off_runtime(barrier: Arc<Barrier>) {
        tokio::task::spawn_blocking(move || {
            barrier.wait();
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn a_call_that_gives_up_waiting_for_a_busy_model_never_runs_its_scorer() {
        tokio::time::timeout(Duration::from_secs(2), async {
            // Arrange
            let calls = Arc::new(Mutex::new(Vec::new()));
            let entered = Arc::new(Barrier::new(2));
            let release = Arc::new(Barrier::new(2));
            let reranker = Arc::new(ModelReranker::new(GatedScorer {
                calls: calls.clone(),
                entered: entered.clone(),
                release: release.clone(),
            }));
            let (a, b, c) = (docs("a"), docs("b"), docs("c"));
            let call_a = {
                let reranker = reranker.clone();
                let a = a.clone();
                tokio::spawn(async move { reranker.scores("q", &a).await })
            };
            wait_off_runtime(entered).await;

            // Act: B waits for the lock behind A and gives up.
            let waited =
                tokio::time::timeout(Duration::from_millis(100), reranker.scores("q", &b)).await;
            let busy = reranker.model.try_lock().is_err();
            wait_off_runtime(release).await;
            call_a.await.unwrap().unwrap();
            let scores_c = reranker.scores("q", &c).await.unwrap();

            // Assert
            assert!(waited.is_err(), "B should still be waiting for the lock");
            assert!(busy, "the model should be locked while A runs");
            assert_eq!(scores_c, vec![1.0, 2.0]);
            assert_eq!(*calls.lock().unwrap(), vec![a, c]);
            assert!(reranker.model.try_lock().is_ok(), "the lock should be free");
        })
        .await
        .expect("the test hung");
    }
}
