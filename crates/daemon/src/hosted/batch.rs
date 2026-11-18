// SPDX-License-Identifier: Apache-2.0
//! Hosted batches: many requests sent in one call, answered later at a lower
//! price. Each provider implements [`HostedBatch`] over the same policy layer
//! as a synchronous call. Endpoint paths and field names follow each
//! provider's public documentation as read when this was written and have not
//! been checked against a live account.

mod anthropic;
mod gemini;
mod openai;

use async_trait::async_trait;
use serde_json::Value;

pub use self::anthropic::AnthropicBatch;
pub use self::gemini::GeminiBatch;
pub use self::openai::OpenAiBatch;
use super::client::HostedClient;
use super::cost::CostRecorder;
use super::dialect::{ChatCall, ChatReply, Dialect};
use super::model::ModelCallError;
use super::secrets::SecretString;
use super::transport::{HttpResponse, HttpTransport};
use crate::config::ModelPin;
use crate::deletion::ProviderCopies;

const HTTP_OK_RANGE: std::ops::Range<u16> = 200..300;
const NOT_FOUND: u16 = 404;
/// A batch that already ended, or was already cancelled, answers a second stop
/// or delete with one of these.
const ALREADY_SETTLED: [u16; 2] = [400, 409];

/// One request in a batch. The cache key is both the provider's custom id and
/// the opaque request id.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchRequest {
    pub cache_key: String,
    pub system: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub seed: u32,
    pub schema: Option<Value>,
}

impl BatchRequest {
    fn call(&self) -> ChatCall<'_> {
        ChatCall {
            system: &self.system,
            prompt: &self.prompt,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            seed: self.seed,
            schema: self.schema.as_ref(),
            request_id: &self.cache_key,
            // The send ledger rows of a batch are written by its sender.
            source_event_ids: &[],
        }
    }
}

/// Where a batch is at its provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchStatus {
    InProgress,
    /// Results can be collected.
    Ended,
    /// The provider's window passed; whatever it finished can be collected.
    Expired,
    /// The provider refused the batch as a whole.
    Failed,
    Cancelled,
}

/// What became of one request.
#[derive(Debug, PartialEq)]
pub enum BatchOutcome {
    Answer(ChatReply),
    /// The request ran and its answer was refused: the provider reported an
    /// error, or the answer came from another model than the pin.
    Failed(ModelCallError),
    Expired,
    Cancelled,
}

#[derive(Debug, PartialEq)]
pub struct BatchResult {
    pub cache_key: String,
    pub outcome: BatchOutcome,
}

#[async_trait]
pub trait HostedBatch: Send + Sync {
    /// Sends `requests` and returns the provider's id for the batch.
    async fn submit(&self, requests: &[BatchRequest]) -> Result<String, ModelCallError>;

    async fn poll(&self, id: &str) -> Result<BatchStatus, ModelCallError>;

    /// Every result the provider holds for the batch, each priced at the batch
    /// rate. A request the provider holds no result for is absent.
    async fn collect(&self, id: &str) -> Result<Vec<BatchResult>, ModelCallError>;

    /// Stops the batch and deletes what the provider lets it. Safe to call
    /// again, and for a batch the provider no longer knows.
    async fn cancel(&self, id: &str) -> Result<ProviderCopies, ModelCallError>;
}

/// What every provider's batch code shares: the policy layer, the key, the pin
/// and the cost count.
struct Wire<T: HttpTransport> {
    client: HostedClient<T>,
    dialect: Box<dyn Dialect>,
    pin: ModelPin,
    key: SecretString,
    costs: CostRecorder,
}

impl<T: HttpTransport> Wire<T> {
    fn new(
        client: HostedClient<T>,
        dialect: Box<dyn Dialect>,
        pin: ModelPin,
        key: SecretString,
        costs: CostRecorder,
    ) -> Self {
        Self {
            client,
            dialect,
            pin,
            key,
            costs,
        }
    }

    fn headers(&self) -> Vec<(String, SecretString)> {
        self.dialect.auth_headers(&self.key)
    }

    fn model(&self) -> &str {
        &self.pin.model_id
    }

    async fn get(&self, url: &str) -> Result<HttpResponse, ModelCallError> {
        accepted(self.get_raw(url).await?)
    }

    /// A GET whose status the caller reads itself.
    async fn get_raw(&self, url: &str) -> Result<HttpResponse, ModelCallError> {
        self.client
            .get(url, self.headers())
            .await
            .map_err(ModelCallError::Hosted)
    }

    /// A POST whose URL or body names the pinned model, so a not-found answer
    /// means the pin was retired.
    async fn submit(&self, url: &str, body: &Value) -> Result<HttpResponse, ModelCallError> {
        let response = self.post_raw(url, body).await?;
        self.pin
            .check_status(response.status)
            .map_err(ModelCallError::Pin)?;
        accepted(response)
    }

    async fn post(&self, url: &str, body: &Value) -> Result<HttpResponse, ModelCallError> {
        accepted(self.post_raw(url, body).await?)
    }

    async fn post_raw(&self, url: &str, body: &Value) -> Result<HttpResponse, ModelCallError> {
        let body = serde_json::to_vec(body).map_err(|_| ModelCallError::Malformed("request"))?;
        self.client
            .post_json(url, self.headers(), body)
            .await
            .map_err(ModelCallError::Hosted)
    }

    async fn post_form(
        &self,
        url: &str,
        form: super::multipart::Form,
    ) -> Result<HttpResponse, ModelCallError> {
        let response = self
            .client
            .post_form(url, self.headers(), form.encode())
            .await
            .map_err(ModelCallError::Hosted)?;
        accepted(response)
    }

    /// Asks the provider to stop a batch. A batch that is unknown or already
    /// settled is as stopped as it will get.
    async fn stop(&self, url: &str) -> Result<(), ModelCallError> {
        let response = self
            .post_raw(url, &Value::Object(Default::default()))
            .await?;
        settled(response.status)
    }

    /// Deletes something at the provider. `Deleted` when it is gone, `Kept`
    /// when the provider refuses because the batch is not finished.
    async fn remove(&self, url: &str) -> Result<ProviderCopies, ModelCallError> {
        let response = self
            .client
            .delete(url, self.headers())
            .await
            .map_err(ModelCallError::Hosted)?;
        if HTTP_OK_RANGE.contains(&response.status) || response.status == NOT_FOUND {
            return Ok(ProviderCopies::Deleted);
        }
        settled(response.status).map(|()| ProviderCopies::Kept)
    }

    /// Prices an answer at the batch rate, then holds it to the pin. The call
    /// is billed either way.
    fn settle(&self, reply: ChatReply) -> BatchOutcome {
        self.costs
            .record(self.dialect.provider(), self.model(), reply.usage, true);
        match self.pin.check(&reply.model) {
            Ok(()) => BatchOutcome::Answer(reply),
            Err(error) => BatchOutcome::Failed(ModelCallError::Pin(error)),
        }
    }
}

fn accepted(response: HttpResponse) -> Result<HttpResponse, ModelCallError> {
    if HTTP_OK_RANGE.contains(&response.status) {
        Ok(response)
    } else {
        Err(ModelCallError::Status(response.status))
    }
}

/// Success, not-found and already-settled all leave nothing more to stop.
fn settled(status: u16) -> Result<(), ModelCallError> {
    if HTTP_OK_RANGE.contains(&status) || status == NOT_FOUND || ALREADY_SETTLED.contains(&status) {
        Ok(())
    } else {
        Err(ModelCallError::Status(status))
    }
}

fn json(response: &HttpResponse) -> Result<Value, ModelCallError> {
    serde_json::from_slice(&response.body)
        .map_err(|_| ModelCallError::Malformed("the body is not JSON"))
}

fn json_lines(response: &HttpResponse) -> Result<Vec<Value>, ModelCallError> {
    response
        .body
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| {
            serde_json::from_slice(line)
                .map_err(|_| ModelCallError::Malformed("a result line is not JSON"))
        })
        .collect()
}

/// A provider's batch id goes into a URL path, so it is held to the characters
/// a real id uses.
fn path_safe(id: &str) -> Result<&str, ModelCallError> {
    let allowed = |c: char| c.is_ascii_alphanumeric() || c == '-' || c == '_';
    if !id.is_empty() && id.chars().all(allowed) {
        Ok(id)
    } else {
        Err(ModelCallError::Malformed(
            "the batch id has unexpected characters",
        ))
    }
}

fn text_of<'a>(value: &'a Value, what: &'static str) -> Result<&'a str, ModelCallError> {
    value.as_str().ok_or(ModelCallError::Malformed(what))
}

/// The answer in `body`, parsed by `dialect` and settled against the pin.
fn answer<T: HttpTransport>(wire: &Wire<T>, body: &Value) -> BatchOutcome {
    let bytes = serde_json::to_vec(body).unwrap_or_default();
    match wire.dialect.parse(&bytes) {
        Ok(reply) => wire.settle(reply),
        Err(why) => BatchOutcome::Failed(ModelCallError::Malformed(why)),
    }
}

#[cfg(test)]
mod tests;
