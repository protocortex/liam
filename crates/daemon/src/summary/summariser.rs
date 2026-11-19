// SPDX-License-Identifier: Apache-2.0
//! The seam between summary generation and whatever model writes the text: a
//! hosted provider, in `hosted::summariser`.

use anyhow::Result;
use async_trait::async_trait;

use crate::config::TokenUsage;
use crate::extraction::client::ModelIdentity;

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
    /// What the provider reported using, when it reported any.
    pub usage: Option<TokenUsage>,
}

#[async_trait]
pub trait Summariser: Send + Sync {
    /// The model's reply, which the caller still has to ground.
    async fn summarise(&self, req: &SummaryRequest) -> Result<SummaryResponse>;

    /// The model that answers, which keys the summary cache.
    fn identity(&self) -> &ModelIdentity;
}
