// SPDX-License-Identifier: Apache-2.0
//! Re-embedding of live nodes and summaries that have no vector. A log record
//! carries a row and never its embedding, so a replay leaves them without one
//! until this runs.

use std::sync::Arc;

use async_trait::async_trait;

use super::Graph;
use crate::backend::Backend;
use crate::error::Result;
use crate::ids::NodeId;

/// Why an embedder could not embed a text.
pub type EmbedError = Box<dyn std::error::Error + Send + Sync>;

/// Turns a node's content into a vector. The store does not own an embedding
/// model, so the caller supplies one.
#[async_trait]
pub trait ContentEmbedder: Send + Sync {
    async fn embed(&self, text: &str) -> std::result::Result<Vec<f32>, EmbedError>;
}

/// What one `reembed_missing` pass did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReembedReport {
    /// Nodes and summaries that now have a vector.
    pub re_embedded: usize,
    /// Nodes and summaries that still have none: the embedder, the content
    /// read or the vector write failed, or the vector's size does not match the
    /// store's.
    pub failed: usize,
    /// Nodes and summaries that have none and were not embedded because no
    /// embedder is configured.
    pub pending: usize,
}

/// What became of one listed node.
enum Outcome {
    Stored,
    /// A concurrent write stored a vector first, and it is kept.
    AlreadyHeld,
    /// The node was removed after it was listed.
    Gone,
}

impl<B: Backend> Graph<B> {
    /// Names the embedder `reembed_missing` and `catch_up` use.
    pub fn with_embedder(mut self, embedder: Arc<dyn ContentEmbedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Embeds the content of every live node, and the text of every live
    /// summary, that has no vector and stores it unless a concurrent write
    /// stored one first. One that cannot be embedded is left as it is and
    /// counted instead of returned, so one bad row does not hide the rest; the
    /// next pass finds it again. Without an embedder it only counts the rows it
    /// would have embedded. Fails only when the rows cannot be listed.
    pub async fn reembed_missing(&self) -> Result<ReembedReport> {
        let mut report = ReembedReport::default();
        // A demoted node has no vector by design, so it is not missing one.
        let demoted = self.demoted_node_ids().await?;
        let mut missing = self.backend.nodes_missing_vectors().await?;
        missing.retain(|id| !demoted.contains(id.as_str()));
        let summaries = self.summaries_missing_embeddings().await?;
        let Some(embedder) = &self.embedder else {
            report.pending = missing.len() + summaries.len();
            if report.pending > 0 {
                tracing::warn!(
                    pending = report.pending,
                    "nodes or summaries have no vector and no embedder is configured"
                );
            }
            return Ok(report);
        };
        for id in missing {
            match self.reembed_one(embedder.as_ref(), &id).await {
                Ok(Outcome::Stored) => report.re_embedded += 1,
                Ok(Outcome::AlreadyHeld | Outcome::Gone) => {}
                Err(reason) => {
                    tracing::error!(node = id.as_str(), %reason, "re-embedding failed, the node stays without a vector");
                    report.failed += 1;
                }
            }
        }
        for (id, text) in summaries {
            match self.reembed_summary(embedder.as_ref(), &id, &text).await {
                Ok(Outcome::Stored) => report.re_embedded += 1,
                Ok(Outcome::AlreadyHeld | Outcome::Gone) => {}
                Err(reason) => {
                    tracing::error!(summary = id.as_str(), %reason, "re-embedding failed, the summary stays without a vector");
                    report.failed += 1;
                }
            }
        }
        // A node embedded for its signature may belong to a cold lineage.
        self.settle_cold_tiers().await?;
        Ok(report)
    }

    async fn reembed_summary(
        &self,
        embedder: &dyn ContentEmbedder,
        id: &str,
        text: &str,
    ) -> std::result::Result<Outcome, String> {
        let embedding = embedder.embed(text).await.map_err(|e| e.to_string())?;
        self.check_dims(&embedding).map_err(|e| e.to_string())?;
        let stored = self
            .store_summary_embedding_if_absent(id, &embedding)
            .await
            .map_err(|e| e.to_string())?;
        Ok(if stored {
            Outcome::Stored
        } else {
            Outcome::AlreadyHeld
        })
    }

    async fn reembed_one(
        &self,
        embedder: &dyn ContentEmbedder,
        id: &NodeId,
    ) -> std::result::Result<Outcome, String> {
        let Some(content) = self.node_content(id).await.map_err(|e| e.to_string())? else {
            return Ok(Outcome::Gone);
        };
        let embedding = embedder.embed(&content).await.map_err(|e| e.to_string())?;
        self.check_dims(&embedding).map_err(|e| e.to_string())?;
        self.ensure_quant(id, &embedding)
            .await
            .map_err(|e| e.to_string())?;
        let stored = self
            .backend
            .vector_insert_if_absent(id.as_str(), &embedding)
            .await
            .map_err(|e| e.to_string())?;
        Ok(if stored {
            Outcome::Stored
        } else {
            Outcome::AlreadyHeld
        })
    }

    /// `None` when the node was removed since it was listed.
    async fn node_content(&self, id: &NodeId) -> Result<Option<String>> {
        let rows = self
            .backend
            .query(
                "SELECT content FROM nodes WHERE id = ?1",
                &[id.as_str().into()],
            )
            .await?;
        rows.first().map(|row| row.get_string(0)).transpose()
    }
}
