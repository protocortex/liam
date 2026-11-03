// SPDX-License-Identifier: Apache-2.0
//! The two views of a hosted model: it writes summaries, and it answers the
//! extraction prompt. Both go through [`HostedModel`], so the pin check, the
//! cost count and the host policy are the same for each.

use std::io;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use super::dialect::ChatCall;
use super::model::{opaque_request_id, HostedModel, ModelCallError};
use super::transport::HttpTransport;
use crate::config::ExtractionConfig;
use crate::extraction::client::{ExtractError, ExtractRequest, JsonExtractor, ModelIdentity};
use crate::extraction::extractor::{ExtractorProvider, ExtractorSession};
use crate::summary::summariser::{Summariser, SummaryRequest, SummaryResponse};

/// Summaries are written greedily so a rerun repeats.
const SUMMARY_TEMPERATURE: f32 = 0.0;
const SUMMARY_SEED: u32 = 0;

pub struct HostedSummariser<T: HttpTransport> {
    model: Arc<HostedModel<T>>,
}

impl<T: HttpTransport> HostedSummariser<T> {
    pub fn new(model: Arc<HostedModel<T>>) -> Self {
        Self { model }
    }
}

#[async_trait]
impl<T: HttpTransport + 'static> Summariser for HostedSummariser<T> {
    async fn summarise(&self, req: &SummaryRequest) -> Result<SummaryResponse> {
        let request_id = opaque_request_id(&req.system, &req.prompt);
        let reply = self
            .model
            .complete(&ChatCall {
                system: &req.system,
                prompt: &req.prompt,
                max_tokens: u32::try_from(req.max_tokens).unwrap_or(u32::MAX),
                temperature: SUMMARY_TEMPERATURE,
                seed: SUMMARY_SEED,
                schema: None,
                request_id: &request_id,
                source_event_ids: &req.source_event_ids,
            })
            .await?;
        Ok(SummaryResponse {
            text: reply.text,
            usage: Some(reply.usage),
        })
    }

    fn identity(&self) -> &ModelIdentity {
        self.model.identity()
    }
}

pub struct HostedExtractor<T: HttpTransport> {
    model: Arc<HostedModel<T>>,
}

#[async_trait]
impl<T: HttpTransport + 'static> JsonExtractor for HostedExtractor<T> {
    async fn extract_json(&self, req: &ExtractRequest) -> Result<String, ExtractError> {
        let request_id = opaque_request_id(&req.system, &req.prompt);
        let reply = self
            .model
            .complete(&ChatCall {
                system: &req.system,
                prompt: &req.prompt,
                max_tokens: req.max_tokens,
                temperature: req.temperature,
                seed: req.seed,
                schema: Some(&req.schema),
                request_id: &request_id,
                source_event_ids: &req.source_event_ids,
            })
            .await
            .map_err(extract_error)?;
        Ok(reply.text)
    }

    fn identity(&self) -> &ModelIdentity {
        self.model.identity()
    }

    /// A provider that does not answer is a timeout or a status, which the
    /// run already tells apart, so there is nothing separate to probe.
    async fn health(&self) -> bool {
        true
    }
}

/// How a failed hosted call reads to the extraction loop, which already knows
/// the sidecar's errors.
fn extract_error(error: ModelCallError) -> ExtractError {
    use super::client::HostedError;
    use crate::config::PinError;
    match error {
        ModelCallError::Status(status) => ExtractError::Status(status),
        ModelCallError::Malformed(why) => ExtractError::Malformed(why),
        ModelCallError::Pin(PinError::Retired { .. }) => ExtractError::ModelRetired,
        ModelCallError::Pin(_) => ExtractError::ModelMismatch,
        ModelCallError::Hosted(HostedError::Timeout { .. }) => ExtractError::Timeout,
        ModelCallError::Hosted(HostedError::Redirect { status }) => ExtractError::Redirect(status),
        ModelCallError::Hosted(HostedError::ResponseTooLarge { limit }) => {
            ExtractError::TooLarge { limit }
        }
        ModelCallError::Hosted(HostedError::NotHttps | HostedError::HostNotAllowed { .. }) => {
            ExtractError::InvalidRequest("the request is not allowed by [hosted]")
        }
        ModelCallError::Hosted(HostedError::Transport { .. }) | ModelCallError::Ledger => {
            ExtractError::Io(io::ErrorKind::Other)
        }
    }
}

/// Opens the hosted extractor when a run first misses the cache, so a run the
/// cache fully answers never reads the key.
pub struct HostedExtractorProvider<T: HttpTransport> {
    open: Box<dyn Fn() -> Result<Arc<HostedModel<T>>> + Send + Sync>,
}

impl<T: HttpTransport> HostedExtractorProvider<T> {
    pub fn new(open: impl Fn() -> Result<Arc<HostedModel<T>>> + Send + Sync + 'static) -> Self {
        Self {
            open: Box::new(open),
        }
    }
}

#[async_trait]
impl<T: HttpTransport + 'static> ExtractorProvider for HostedExtractorProvider<T> {
    async fn open(&self, _config: &ExtractionConfig) -> Result<ExtractorSession> {
        let model = (self.open)()?;
        Ok(ExtractorSession {
            extractor: Arc::new(HostedExtractor { model }),
            alive: Box::new(|| true),
            _keepalive: None,
        })
    }
}

#[cfg(test)]
mod tests;
