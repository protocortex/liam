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
    /// Nodes that still have none: the embedder or the vector write failed.
    pub failed: usize,
}

impl<B: Backend> Graph<B> {
    /// Names the embedder `reembed_missing` and `catch_up` use.
    pub fn with_embedder(mut self, embedder: Arc<dyn ContentEmbedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// The live nodes with no stored vector, in id order.
    pub async fn nodes_missing_vectors(&self) -> Result<Vec<NodeId>> {
        self.backend.nodes_missing_vectors().await
    }

    /// Embeds the content of every live node that has no vector and stores it
    /// the way a live write does. Like a live write after its commit, a failed
    /// embed leaves the node as it is, but this pass counts the failure
    /// instead of returning it, so one bad node does not hide the rest; the
    /// node is found again by the next pass. Does nothing without an embedder.
    pub async fn reembed_missing(&self) -> Result<ReembedReport> {
        let mut report = ReembedReport::default();
        let Some(embedder) = &self.embedder else {
            return Ok(report);
        };
        for id in self.nodes_missing_vectors().await? {
            let Some(content) = self.node_content(&id).await? else {
                continue;
            };
            let stored = match embedder.embed(&content).await {
                Ok(embedding) => self
                    .put_vector(&id, &embedding)
                    .await
                    .map_err(|error| error.to_string()),
                Err(error) => Err(error.to_string()),
            };
            match stored {
                Ok(()) => report.re_embedded += 1,
                Err(reason) => {
                    tracing::error!(node = id.as_str(), %reason, "re-embedding failed, the node stays without a vector");
                    report.failed += 1;
                }
            }
        }
        Ok(report)
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
