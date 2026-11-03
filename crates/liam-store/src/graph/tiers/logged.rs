// SPDX-License-Identifier: Apache-2.0
//! The writes that record tier state in the log. Each is appended before it is
//! applied, as every other write is, so a rebuild from the log restores usage,
//! verdicts and tiers.

use std::collections::BTreeMap;

use crate::backend::Backend;
use crate::error::Result;
use crate::graph::logged_plan::Plan;
use crate::graph::logged_write::ready;
use crate::graph::Graph;

use crate::types::{LineageKey, TierMove, UsageEntry};

impl<B: Backend> Graph<B> {
    /// Logs and applies counted accesses, one batch per producer, each written
    /// under its producer as the event source. An empty list is still written
    /// as one batch, because its epoch marks when counting began.
    pub async fn put_usage_logged(&self, day: i64, entries: &[UsageEntry]) -> Result<()> {
        let mut by_producer: BTreeMap<&str, Vec<UsageEntry>> = BTreeMap::new();
        for entry in entries {
            by_producer
                .entry(entry.producer.as_str())
                .or_default()
                .push(entry.clone());
        }
        if by_producer.is_empty() {
            by_producer.insert("", Vec::new());
        }
        for (producer, batch) in &by_producer {
            self.project(|_, now| ready(Ok(Plan::usage(day, producer, batch, now))))
                .await?;
        }
        Ok(())
    }

    /// Logs and applies a producer's verdict on the lineage of a result.
    pub async fn put_feedback_logged(
        &self,
        lineage_key: &LineageKey,
        producer: &str,
        correct: bool,
    ) -> Result<()> {
        self.project(|_, now| ready(Ok(Plan::feedback(lineage_key, producer, correct, now))))
            .await?;
        Ok(())
    }

    /// Logs and applies a tier move under `source`, which names the run that
    /// made it (`consolidation:<run id>`) so that the run's rollback cancels it.
    /// A lineage already in that tier is left alone and nothing is appended.
    /// Says whether the move was written.
    pub async fn put_tier_logged(&self, change: &TierMove, source: &str) -> Result<bool> {
        if self.tier_of(&change.lineage_key).await? == change.to {
            return Ok(false);
        }
        self.project(|_, now| ready(Ok(Plan::tier_move(change, source, now))))
            .await?;
        Ok(true)
    }
}
