// SPDX-License-Identifier: Apache-2.0
//! What a hosted call cost, from the usage the provider reported and the
//! versioned price table in `[hosted.prices]`.

use std::sync::Arc;

use crate::config::{PricesConfig, Provider, TokenUsage};
use crate::metrics::recorder::Metrics;

/// The cost of one call and the table that priced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostLine {
    pub micro_dollars: u64,
    pub table_version: String,
}

pub struct CostRecorder {
    metrics: Arc<Metrics>,
    prices: PricesConfig,
}

impl CostRecorder {
    pub fn new(metrics: Arc<Metrics>, prices: PricesConfig) -> Self {
        Self { metrics, prices }
    }

    /// Counts the call and its cost under the provider and model labels, and
    /// returns the cost with the table version for the caller's log. A model
    /// with no price counts as free and is logged, because the call has
    /// already been made.
    pub fn record(
        &self,
        provider: Provider,
        model: &str,
        tokens: TokenUsage,
        batch: bool,
    ) -> CostLine {
        let price = self.prices.models.get(model).copied().unwrap_or_else(|| {
            tracing::error!(
                provider = provider.name(),
                model,
                "hosted.prices has no entry for the model, so its cost is counted as zero"
            );
            Default::default()
        });
        self.metrics
            .model_call(provider, model, tokens, batch, price);
        let line = CostLine {
            micro_dollars: price.cost_micro(&tokens, batch),
            table_version: self.prices.version.clone(),
        };
        tracing::info!(
            provider = provider.name(),
            model,
            batch,
            input_tokens = tokens.input_tokens,
            output_tokens = tokens.output_tokens,
            cost_micro_dollars = line.micro_dollars,
            price_table = %line.table_version,
            "hosted call priced"
        );
        line
    }
}
