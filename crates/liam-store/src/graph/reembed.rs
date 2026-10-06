// SPDX-License-Identifier: Apache-2.0
//! Re-embedding of live nodes that have no vector. A log record carries a
//! node's row and never its embedding, so a replay leaves its nodes without
//! one until this runs.

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
    /// Nodes that now have a vector.
    pub re_embedded: usize,
    /// Nodes that still have none: the embedder, the content read or the
    /// vector write failed, or the vector's size does not match the store's.
    pub failed: usize,
    /// Nodes that have none and were not embedded because no embedder is
    /// configured.
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

    /// Embeds the content of every live node that has no vector and stores it
    /// unless a concurrent write stored one first. A node that cannot be
    /// embedded is left as it is and counted instead of returned, so one bad
    /// node does not hide the rest; the next pass finds it again. Without an
    /// embedder it only counts the nodes it would have embedded. Fails only
    /// when the nodes cannot be listed.
    pub async fn reembed_missing(&self) -> Result<ReembedReport> {
        let mut report = ReembedReport::default();
        let missing = self.backend.nodes_missing_vectors().await?;
        let Some(embedder) = &self.embedder else {
            report.pending = missing.len();
            if report.pending > 0 {
                tracing::warn!(
                    pending = report.pending,
                    "nodes have no vector and no embedder is configured"
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
        Ok(report)
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
