// SPDX-License-Identifier: Apache-2.0
//! Tiering state the store keeps beside its nodes: the lineage of each node, the
//! binary and int8 forms of its embedding, and the in-memory index over the
//! binary forms. Quantizing is model-free, so it lives here; deciding what to
//! move between tiers does not.

mod lineage;
mod quantize;

use std::sync::Arc;

pub(super) use lineage::assign_lineage;
use quantize::{bytes_to_words, words_to_bytes, QUANT_VERSION};
pub use quantize::{
    dequantize, hamming, quantize, IndexedEntry, QuantIndex, Quantized, BITS_PER_WORD,
};

use super::projection::Step;
use super::Graph;
use crate::backend::{Backend, BackendTx};
use crate::error::Result;
use crate::ids::{NodeId, FOREVER};
use crate::value::Value;
use liam_log::event::TombstoneTable;

/// Nodes the one-time signature backfill reads vectors for in one query.
const BACKFILL_CHUNK: usize = 200;

const QUANT_COLUMNS: &str = "node_id, bin, i8, scale, quant_version";

/// The shared handle to the lock that orders index updates against a reload.
pub(super) type IndexGate = Arc<tokio::sync::Mutex<()>>;

fn quant_params(node_id: &str, quantized: &Quantized) -> Vec<Value> {
    let int8 = quantized.i8.iter().map(|c| *c as u8).collect();
    vec![
        node_id.into(),
        words_to_bytes(&quantized.bin).into(),
        Value::Blob(int8),
        f64::from(quantized.scale).into(),
        QUANT_VERSION.into(),
    ]
}

/// Stores the forms of a node's embedding, failing when it already has them.
pub(super) async fn insert_quant(
    tx: &mut dyn BackendTx,
    node_id: &str,
    quantized: &Quantized,
) -> Result<()> {
    let sql = format!("INSERT INTO node_quant ({QUANT_COLUMNS}) VALUES (?1, ?2, ?3, ?4, ?5)");
    tx.execute(&sql, &quant_params(node_id, quantized)).await?;
    Ok(())
}

impl<B: Backend> Graph<B> {
    /// The binary signatures of the live nodes that have one.
    pub fn quant_index(&self) -> &QuantIndex {
        &self.quant
    }

    /// The statement that stores the forms of `embedding` with the node it
    /// belongs to. `None` for a node written without an embedding. The width
    /// is checked first so a bad vector fails before anything is written.
    pub(super) fn quant_step(
        &self,
        node_id: &NodeId,
        embedding: Option<&[f32]>,
    ) -> Result<Option<Step>> {
        let Some(embedding) = embedding else {
            return Ok(None);
        };
        self.check_dims(embedding)?;
        Ok(Some(Step::Quant {
            node_id: node_id.as_str().to_string(),
            quantized: quantize(embedding)?,
        }))
    }

    /// Rebuilds the index from `node_quant`, for a store that opens, replays,
    /// or restores, where memory and the tables can differ. Holds the index
    /// gate while it reads, so a write that commits meanwhile applies its own
    /// update after the reload and not before.
    pub async fn reload_quant_index(&self) -> Result<()> {
        let _gate = self.quant_gate.lock().await;
        let rows = self
            .backend
            .query(
                "SELECT q.node_id, q.bin FROM node_quant q
                 JOIN nodes n ON n.id = q.node_id WHERE n.tx_to = ?1",
                &[FOREVER.into()],
            )
            .await?;
        let mut entries = Vec::with_capacity(rows.len());
        for row in &rows {
            let id = row.get_string(0)?;
            match bytes_to_words(row.get_blob(1)?) {
                Some(bin) => entries.push((id, false, bin)),
                None => {
                    tracing::warn!(node = %id, "signature has a ragged length, left out of the index")
                }
            }
        }
        self.quant.replace_all(entries);
        Ok(())
    }

    /// Brings the index in line with a write that committed: a node it closed
    /// or removed leaves, a signature it stored joins.
    pub(super) async fn sync_quant(&self, steps: &[Step]) {
        let mut gone: Vec<&str> = Vec::new();
        let mut added = Vec::new();
        for step in steps {
            match step {
                Step::Close { id, .. } => gone.push(id),
                Step::Remove(target) if matches!(target.table, TombstoneTable::Nodes) => {
                    gone.push(&target.id)
                }
                Step::Quant { node_id, quantized } => added.push((node_id, &quantized.bin)),
                _ => {}
            }
        }
        if gone.is_empty() && added.is_empty() {
            return;
        }
        let _gate = self.quant_gate.lock().await;
        self.quant.remove(gone.iter().copied());
        for (id, bin) in added
            .into_iter()
            .filter(|(id, _)| !gone.contains(&id.as_str()))
        {
            self.quant.insert(id, false, bin.clone());
        }
    }

    /// Stores the forms of `embedding` for a live node that has none, so a
    /// vector is never written without its signature. A node that has them is
    /// left as it is.
    pub(super) async fn ensure_quant(&self, node_id: &NodeId, embedding: &[f32]) -> Result<()> {
        let quantized = quantize(embedding)?;
        let sql = format!(
            "INSERT OR IGNORE INTO node_quant ({QUANT_COLUMNS})
             SELECT ?1, ?2, ?3, ?4, ?5
             WHERE EXISTS (SELECT 1 FROM nodes WHERE id = ?1 AND tx_to = ?6)"
        );
        let mut params = quant_params(node_id.as_str(), &quantized);
        params.push(FOREVER.into());
        if self.backend.execute(&sql, &params).await? == 1 {
            let _gate = self.quant_gate.lock().await;
            self.quant.insert(node_id.as_str(), false, quantized.bin);
        }
        Ok(())
    }

    /// Gives every live node that has a vector and no signature one, once: a
    /// store from before tiering holds vectors only. A second call, or one on a
    /// store already done, changes nothing. Returns how many it wrote.
    pub async fn backfill_quant(&self) -> Result<usize> {
        if self.backfill_done("quant_backfill_state").await? {
            return Ok(0);
        }
        let mut filled = 0;
        let mut after = String::new();
        loop {
            let rows = self
                .backend
                .query(
                    "SELECT id FROM nodes n WHERE tx_to = ?1 AND id > ?2
                     AND NOT EXISTS (SELECT 1 FROM node_quant WHERE node_id = n.id)
                     ORDER BY id LIMIT ?3",
                    &[
                        FOREVER.into(),
                        after.as_str().into(),
                        (BACKFILL_CHUNK as i64).into(),
                    ],
                )
                .await?;
            let Some(last) = rows.last() else { break };
            after = last.get_string(0)?;
            for row in &rows {
                let id = NodeId::from_raw(row.get_string(0)?);
                if let Some(vector) = self.backend.vector_get(id.as_str(), self.dims).await? {
                    self.ensure_quant(&id, &vector).await?;
                    filled += 1;
                }
            }
        }
        self.backend
            .execute(
                "INSERT OR REPLACE INTO quant_backfill_state (id, completed_at, filled)
                 VALUES (1, ?1, ?2)",
                &[self.clock.now().into(), (filled as i64).into()],
            )
            .await?;
        tracing::info!(filled, "signatures written for vectors that had none");
        Ok(filled)
    }

    async fn backfill_done(&self, state_table: &str) -> Result<bool> {
        let rows = self
            .backend
            .query(&format!("SELECT 1 FROM {state_table} WHERE id = 1"), &[])
            .await?;
        Ok(!rows.is_empty())
    }
}
