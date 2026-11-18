// SPDX-License-Identifier: Apache-2.0
//! The cold arm of a query: a demoted item has no vector, so it is found by
//! the Hamming distance between its signature and the query's.

use std::collections::HashSet;

use super::quantize;
use crate::backend::Backend;
use crate::error::Result;
use crate::graph::{live_at, row_not_inferred, scope_within, Graph};
use crate::ids::{Millis, NodeId};
use crate::types::{GraphConfig, Tier};
use crate::value::Value;

/// Times the index is asked for twice as many candidates when the filters
/// leave too few of them.
const MAX_WIDENINGS: usize = 3;

/// Ids one hydration query binds, under SQLite's 999 parameter limit.
const HYDRATE_CHUNK: usize = 400;

/// The knobs of the arm, copied out of the store's configuration.
#[derive(Clone, Copy)]
pub(in crate::graph) struct ColdArm {
    pub weight: f64,
    max_hamming_ratio: f64,
    oversample: usize,
    pool: usize,
}

impl ColdArm {
    pub fn of(config: &GraphConfig) -> Self {
        Self {
            weight: config.cold_arm_weight,
            max_hamming_ratio: config.cold_max_hamming_ratio,
            oversample: config.cold_oversample.max(1),
            pool: config.cold_probe_pool,
        }
    }
}

/// What one query asks of the arm. `pool` is the length of the lists the other
/// arms contribute.
#[derive(Clone, Copy)]
pub(in crate::graph) struct ColdQuery<'a> {
    pub pool: usize,
    pub kind: Option<&'a str>,
    pub scope: Option<&'a str>,
    pub as_of: Millis,
    pub include_inferred: bool,
}

impl<B: Backend> Graph<B> {
    /// The cold items nearest `embedding`, nearest first and equal distances by
    /// id. Candidates beyond the Hamming cutoff are dropped, and so are those
    /// that are not live at `as_of` or fail the kind, scope and strict filters.
    /// The tier row decides whether an item is cold, whatever the index says
    /// and whether or not the lineage still has a vector, so a promotion caught
    /// mid-flight is still found. A query with no cold item in the index costs
    /// one read of a counter.
    pub(in crate::graph) async fn cold_probe(
        &self,
        embedding: &[f32],
        query: ColdQuery<'_>,
    ) -> Result<Vec<NodeId>> {
        let pool = query.pool.min(self.cold_arm.pool);
        if pool == 0 || self.quant.cold_count() == 0 {
            return Ok(Vec::new());
        }
        self.check_dims(embedding)?;
        let signature = quantize::quantize(embedding)?.bin;
        let bits = embedding.len() as f64;
        let cutoff = (self.cold_arm.max_hamming_ratio * bits).floor() as u32;
        let mut wanted = pool.saturating_mul(self.cold_arm.oversample);
        let mut passing = Vec::new();
        for _ in 0..=MAX_WIDENINGS {
            let nearest = self.quant.hamming_top(&signature, wanted, true)?;
            let exhausted = nearest.len() < wanted;
            let candidates = nearest.len();
            let near: Vec<NodeId> = nearest
                .into_iter()
                .take_while(|(_, distance)| *distance <= cutoff)
                .map(|(id, _)| id)
                .collect();
            // Past the cutoff nothing nearer can follow, so asking for more
            // only helps when every candidate was inside it.
            let cut_off = near.len() < candidates;
            let live = self.hydrate_cold(&near, query).await?;
            passing = near
                .into_iter()
                .filter(|id| live.contains(id.as_str()))
                .take(pool)
                .collect();
            if passing.len() >= pool || exhausted || cut_off {
                break;
            }
            wanted = wanted.saturating_mul(2);
        }
        Ok(passing)
    }

    /// The ids of `ids` that are cold by their tier row and pass the query's
    /// filters.
    async fn hydrate_cold(&self, ids: &[NodeId], query: ColdQuery<'_>) -> Result<HashSet<String>> {
        let mut found = HashSet::with_capacity(ids.len());
        for chunk in ids.chunks(HYDRATE_CHUNK) {
            let mut params: Vec<Value> = vec![query.as_of.into(), Tier::Cold.as_str().into()];
            let placeholders = chunk
                .iter()
                .map(|id| {
                    params.push(id.as_str().into());
                    format!("?{}", params.len())
                })
                .collect::<Vec<_>>()
                .join(",");
            let mut filters = String::new();
            if let Some(kind) = query.kind {
                params.push(kind.into());
                filters.push_str(&format!(" AND n.kind = ?{}", params.len()));
            }
            if let Some(scope) = query.scope {
                params.push(scope.into());
                filters.push_str(&format!(" AND {}", scope_within("n.scope", params.len())));
            }
            filters.push_str(&row_not_inferred("n", query.include_inferred));
            let sql = format!(
                "SELECT n.id FROM nodes n
                 JOIN node_lineage l ON l.node_id = n.id
                 JOIN item_tier t ON t.lineage_key = l.lineage_key AND t.tier = ?2
                 WHERE n.id IN ({placeholders}) AND {live}{filters}",
                live = live_at("n", 1),
            );
            for row in self.backend.query(&sql, &params).await? {
                found.insert(row.get_string(0)?);
            }
        }
        Ok(found)
    }
}
