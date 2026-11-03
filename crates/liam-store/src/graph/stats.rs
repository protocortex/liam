// SPDX-License-Identifier: Apache-2.0
//! Sizes and counters the store reports about itself.

use std::time::Duration;

use liam_log::dedup::BloomStats;

use super::Graph;
use crate::backend::Backend;
use crate::error::Result;
use crate::types::Tier;

/// Vector bytes a lineage with no tier row is counted under: it is hot.
const VECTOR_BYTES_SQL: &str = "SELECT COALESCE(t.tier, 'hot'), SUM(length(v.embedding))
     FROM node_vectors v
     LEFT JOIN node_lineage l ON l.node_id = v.node_id
     LEFT JOIN item_tier t ON t.lineage_key = l.lineage_key
     GROUP BY 1";

const QUANT_BYTES_SQL: &str =
    "SELECT COALESCE(t.tier, 'hot'), SUM(length(q.bin) + COALESCE(length(q.i8), 0))
     FROM node_quant q
     LEFT JOIN node_lineage l ON l.node_id = q.node_id
     LEFT JOIN item_tier t ON t.lineage_key = l.lineage_key
     GROUP BY 1";

/// Bytes the embeddings and their signatures take, by tier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TierBytes {
    pub hot: u64,
    pub cold: u64,
}

/// How long each retrieval arm of one query took, none for an arm the query
/// did not run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueryTimings {
    pub lexical: Option<Duration>,
    pub vector: Option<Duration>,
    pub cold: Option<Duration>,
}

impl<B: Backend> Graph<B> {
    /// How often the dedup filter passed a hash the log did not hold, and how
    /// often it rightly turned one away, since the store opened.
    pub fn bloom_stats(&self) -> BloomStats {
        self.bloom_counters.stats()
    }

    /// The bytes held in `node_vectors` and `node_quant`, split by the tier of
    /// each node's lineage.
    ///
    /// # Errors
    ///
    /// Fails when the store cannot be read.
    pub async fn storage_bytes(&self) -> Result<TierBytes> {
        let mut bytes = TierBytes::default();
        for sql in [VECTOR_BYTES_SQL, QUANT_BYTES_SQL] {
            for row in self.backend.query(sql, &[]).await? {
                let size = u64::try_from(row.get_i64(1)?).unwrap_or(0);
                if row.get_string(0)? == Tier::Cold.as_str() {
                    bytes.cold += size;
                } else {
                    bytes.hot += size;
                }
            }
        }
        Ok(bytes)
    }
}
