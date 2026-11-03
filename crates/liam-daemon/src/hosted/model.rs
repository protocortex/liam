// SPDX-License-Identifier: Apache-2.0
//! One pinned hosted model behind a transport: shapes the request, enforces
//! the pin on the answer, and counts the cost. The extractor and the
//! summariser are thin views over it.

use std::fmt;
use std::sync::Arc;

use liam_store::Clock;
use sha2::{Digest, Sha256};

use super::client::{HostedClient, HostedError};
use super::cost::CostRecorder;
use super::dialect::{ChatCall, ChatReply, Dialect};
use super::secrets::SecretString;
use super::transport::HttpTransport;
use crate::config::{ModelPin, PinError};
use crate::deletion::send_ledger::{SendLedger, SendRow};
use crate::extraction::client::ModelIdentity;

const HTTP_OK_RANGE: std::ops::Range<u16> = 200..300;
/// Hex characters of the request digest sent as the opaque id.
const REQUEST_ID_CHARS: usize = 32;

/// Why a hosted call gave no answer. No variant carries a request, a reply or
/// a key.
#[derive(Debug, PartialEq, Eq)]
pub enum ModelCallError {
    Hosted(HostedError),
    /// The provider refused the request with this status.
    Status(u16),
    /// The pinned model is gone or the provider answered with another one.
    Pin(PinError),
    Malformed(&'static str),
    /// The send ledger could not be written, so nothing was sent.
    Ledger,
}

impl fmt::Display for ModelCallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hosted(error) => write!(f, "{error}"),
            Self::Status(status) => write!(f, "the provider answered with HTTP status {status}"),
            Self::Pin(error) => write!(f, "{error}"),
            Self::Malformed(why) => write!(f, "the provider sent a malformed reply: {why}"),
            Self::Ledger => write!(
                f,
                "the send ledger could not be written, so nothing was sent"
            ),
        }
    }
}

impl std::error::Error for ModelCallError {}

pub struct HostedModel<T: HttpTransport> {
    client: HostedClient<T>,
    dialect: Box<dyn Dialect>,
    pin: ModelPin,
    key: SecretString,
    identity: ModelIdentity,
    costs: CostRecorder,
    sends: Option<(SendLedger, Arc<dyn Clock>)>,
}

impl<T: HttpTransport> HostedModel<T> {
    pub fn new(
        client: HostedClient<T>,
        dialect: Box<dyn Dialect>,
        pin: ModelPin,
        key: SecretString,
        costs: CostRecorder,
    ) -> Self {
        let identity = hosted_identity(&pin);
        Self {
            client,
            dialect,
            pin,
            key,
            identity,
            costs,
            sends: None,
        }
    }

    /// Records every call in `ledger` before it is sent, so a delete can tell
    /// which providers hold text built from a deleted event.
    pub fn with_send_ledger(mut self, ledger: SendLedger, clock: Arc<dyn Clock>) -> Self {
        self.sends = Some((ledger, clock));
        self
    }

    pub fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    /// Sends one request. The model is billed even when its answer is then
    /// refused for naming another model, so the cost is counted first.
    pub async fn complete(&self, call: &ChatCall<'_>) -> Result<ChatReply, ModelCallError> {
        self.record_send(call)?;
        let url = self.dialect.url(&self.pin.model_id);
        let headers = self.dialect.headers(&self.key, call);
        let body = self.dialect.body(&self.pin.model_id, call);
        let body = serde_json::to_vec(&body).map_err(|_| ModelCallError::Malformed("request"))?;
        let response = self
            .client
            .post_json(&url, headers, body)
            .await
            .map_err(ModelCallError::Hosted)?;
        self.pin
            .check_status(response.status)
            .map_err(ModelCallError::Pin)?;
        if !HTTP_OK_RANGE.contains(&response.status) {
            return Err(ModelCallError::Status(response.status));
        }
        let reply = self
            .dialect
            .parse(&response.body)
            .map_err(ModelCallError::Malformed)?;
        self.costs.record(
            self.dialect.provider(),
            &self.pin.model_id,
            reply.usage,
            false,
        );
        self.pin.check(&reply.model).map_err(ModelCallError::Pin)?;
        Ok(reply)
    }
}

impl<T: HttpTransport> HostedModel<T> {
    fn record_send(&self, call: &ChatCall<'_>) -> Result<(), ModelCallError> {
        let Some((ledger, clock)) = &self.sends else {
            return Ok(());
        };
        ledger
            .append(&SendRow {
                provider: self.pin.provider.clone(),
                cache_key: call.request_id.to_owned(),
                source_event_ids: call.source_event_ids.to_vec(),
                sent_at: clock.now().0,
            })
            .map_err(|error| {
                tracing::error!(error = %format!("{error:#}"), "the send ledger could not be written");
                ModelCallError::Ledger
            })
    }
}

/// A hosted model has no file to hash, so its dated id names it.
pub fn hosted_identity(pin: &ModelPin) -> ModelIdentity {
    ModelIdentity {
        model_id: format!("{}:{}", pin.provider, pin.model_id),
        model_sha256: String::new(),
    }
}

/// An id for a request that says nothing about what it holds.
pub fn opaque_request_id(system: &str, prompt: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(system.as_bytes());
    digest.update([0]);
    digest.update(prompt.as_bytes());
    let hex: String = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    hex[..REQUEST_ID_CHARS].to_owned()
}

#[cfg(test)]
mod tests;
