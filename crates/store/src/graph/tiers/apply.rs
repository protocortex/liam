// SPDX-License-Identifier: Apache-2.0
//! Moving a lineage between tiers. The tier row decides: a move is logged and
//! committed first, and the vectors follow, so a crash between the two leaves a
//! state `reconcile_tier_vectors` repairs.

use super::quantize::quantize;
use super::state::Held;
use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::graph::logged_plan::Plan;
use crate::graph::logged_write::ready;
use crate::graph::projection::Step;
use crate::graph::Graph;
use crate::ids::NodeId;
use crate::types::{ApplyOutcome, Tier, TierMove};

impl<B: Backend> Graph<B> {
    /// Moves a lineage to the cold tier: one transaction sets the tier row,
    /// drops the int8 forms and appends `change` to the log, then the vectors
    /// are deleted. A move whose lineage is already cold or has no live node is
    /// stale, and a lineage with a vector that has no signature is left alone,
    /// since deleting the vector would lose the node's only embedding.
    ///
    /// A node written after `change.changed_at` keeps its vector. A vector that
    /// cannot be deleted is reported as `Partial`, and the next reconcile
    /// retries it.
    pub async fn demote_lineage(&self, change: &TierMove, source: &str) -> Result<ApplyOutcome> {
        if change.to != Tier::Cold {
            return Err(Error::UnexpectedTierMove { expected: "cold" });
        }
        let _one_move = self.tier_gate.lock().await;
        let nodes = self.lineage_nodes(&change.lineage_key).await?;
        if nodes.is_empty() || self.tier_of(&change.lineage_key).await? == Tier::Cold {
            return Ok(ApplyOutcome::Stale);
        }
        if nodes.iter().any(|n| n.has_vector && !n.has_signature) {
            return Ok(ApplyOutcome::NoSignature);
        }
        self.project(|_, now| ready(Ok(Plan::tier_move(change, source, now))))
            .await?;
        let committed = self.lineage_nodes(&change.lineage_key).await?;
        let held = Held::of(&committed, change.changed_at);
        for (deleted, node) in held.droppable.iter().enumerate() {
            if let Err(error) = self.backend.vector_delete(node.as_str()).await {
                let remaining = held.droppable.len() - deleted;
                tracing::warn!(%error, remaining, "demoted vectors not all deleted; reconcile retries");
                return Ok(ApplyOutcome::Partial { remaining });
            }
        }
        Ok(ApplyOutcome::Applied)
    }

    /// Moves a lineage to the hot tier. `vectors` holds the embeddings of the
    /// nodes that have none, computed by the caller because the store owns no
    /// model. They are stored first, then one transaction replaces the int8
    /// forms, sets the tier row and appends `change`. A crash between the two
    /// leaves a cold row with vectors, which reconcile deletes, so the
    /// promotion is retried.
    ///
    /// A node of the lineage that already has a vector, such as one a crashed
    /// promotion stored, gets its forms from that vector. A vector for a node
    /// outside the lineage is ignored.
    pub async fn promote_lineage(
        &self,
        change: &TierMove,
        vectors: &[(NodeId, Vec<f32>)],
        source: &str,
    ) -> Result<ApplyOutcome> {
        if change.to != Tier::Hot {
            return Err(Error::UnexpectedTierMove { expected: "hot" });
        }
        let _one_move = self.tier_gate.lock().await;
        let nodes = self.lineage_nodes(&change.lineage_key).await?;
        if nodes.is_empty() || self.tier_of(&change.lineage_key).await? == Tier::Hot {
            return Ok(ApplyOutcome::Stale);
        }
        let mut requants = Vec::new();
        let mut to_store = Vec::new();
        for node in &nodes {
            let given = vectors
                .iter()
                .find(|(id, _)| *id == node.node_id)
                .map(|(_, vector)| vector);
            let vector = match (given, node.has_vector) {
                (Some(vector), _) => {
                    self.check_dims(vector)?;
                    to_store.push((&node.node_id, vector));
                    vector.clone()
                }
                (None, true) => {
                    match self
                        .backend
                        .vector_get(node.node_id.as_str(), self.dims)
                        .await?
                    {
                        Some(vector) => vector,
                        None => continue,
                    }
                }
                (None, false) => continue,
            };
            requants.push(Step::Requant {
                node_id: node.node_id.as_str().to_string(),
                quantized: quantize(&vector)?,
            });
        }
        for (id, vector) in to_store {
            self.backend.vector_upsert(id.as_str(), vector).await?;
        }
        self.project(|_, now| ready(Ok(Plan::tier_move(change, source, now).with_steps(requants))))
            .await?;
        Ok(ApplyOutcome::Applied)
    }
}
