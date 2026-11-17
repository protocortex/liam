// SPDX-License-Identifier: Apache-2.0
//! The seam between summary generation and whatever model writes the text: the
//! local model here, a hosted provider in `hosted::summariser`.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use liam_model::Llm;

use crate::config::{Price, Provider, TokenUsage};
use crate::extraction::client::ModelIdentity;
use crate::metrics::recorder::Metrics;

#[derive(Debug, Clone)]
pub struct SummaryRequest {
    pub system: String,
    pub prompt: String,
    /// The log events the prompt cites, so a hosted send can be traced to them.
    pub source_event_ids: Vec<String>,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummaryResponse {
    pub text: String,
    /// What the provider reported using; the local model reports nothing.
    pub usage: Option<TokenUsage>,
}

#[async_trait]
pub trait Summariser: Send + Sync {
    /// The model's reply, which the caller still has to ground.
    async fn summarise(&self, req: &SummaryRequest) -> Result<SummaryResponse>;

    /// The model that answers, which keys the summary cache.
    fn identity(&self) -> &ModelIdentity;

    /// Tokens `text` takes for this model, when it can count them.
    fn count_tokens(&self, _text: &str) -> Option<usize> {
        None
    }
}

/// The in-process model behind the [`Summariser`] seam.
pub struct LocalSummariser {
    llm: Arc<dyn Llm>,
    identity: ModelIdentity,
    metrics: Option<Arc<Metrics>>,
}

impl LocalSummariser {
    pub fn new(llm: Arc<dyn Llm>, identity: ModelIdentity) -> Self {
        Self {
            llm,
            identity,
            metrics: None,
        }
    }

    /// Counts each call under the local provider, at no cost.
    pub fn with_metrics(mut self, metrics: Arc<Metrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }
}

#[async_trait]
impl Summariser for LocalSummariser {
    async fn summarise(&self, req: &SummaryRequest) -> Result<SummaryResponse> {
        tracing::debug!(events = req.source_event_ids.len(), "local summary request");
        let text = self
            .llm
            .complete_capped(&req.system, &req.prompt, req.max_tokens)
            .await?;
        if let Some(metrics) = &self.metrics {
            metrics.model_call(
                Provider::Local,
                &self.identity.model_id,
                TokenUsage::default(),
                false,
                Price::default(),
            );
        }
        Ok(SummaryResponse { text, usage: None })
    }

    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    fn count_tokens(&self, text: &str) -> Option<usize> {
        self.llm.count_tokens(text)
    }
}

#[cfg(test)]
mod tests;
