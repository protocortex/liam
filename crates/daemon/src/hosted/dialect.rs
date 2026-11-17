// SPDX-License-Identifier: Apache-2.0
//! What differs between providers: where a request goes, how it is shaped, and
//! where the answer, the model and the token counts sit in the reply. Every
//! field name here follows the provider's public documentation as read when
//! this was written and has not been checked against a live account.

use serde_json::Value;

use super::secrets::SecretString;
use crate::config::{Provider, TokenUsage};

/// One model call, before a provider shapes it.
pub struct ChatCall<'a> {
    pub system: &'a str,
    pub prompt: &'a str,
    pub max_tokens: u32,
    pub temperature: f32,
    pub seed: u32,
    /// The JSON Schema the answer should follow, where the provider takes one.
    pub schema: Option<&'a Value>,
    /// An opaque id that ties a provider log line to a request without saying
    /// what it held.
    pub request_id: &'a str,
    /// The events the prompt was built from, for the send ledger.
    pub source_event_ids: &'a [String],
}

/// What a provider answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatReply {
    pub text: String,
    /// The model the provider says answered.
    pub model: String,
    pub usage: TokenUsage,
}

pub trait Dialect: Send + Sync {
    fn provider(&self) -> Provider;
    fn url(&self, model: &str) -> String;
    /// The headers that carry the key, and nothing about a request.
    fn auth_headers(&self, key: &SecretString) -> Vec<(String, SecretString)>;
    fn headers(&self, key: &SecretString, call: &ChatCall<'_>) -> Vec<(String, SecretString)>;
    fn body(&self, model: &str, call: &ChatCall<'_>) -> Value;
    /// Reads the reply, or says which part was missing.
    fn parse(&self, body: &[u8]) -> Result<ChatReply, &'static str>;
}

/// The key header alone; the transport adds the content type.
pub(super) fn base_headers(
    key_header: &str,
    key_value: SecretString,
) -> Vec<(String, SecretString)> {
    vec![(key_header.to_owned(), key_value)]
}

pub(super) fn count(value: &Value) -> u64 {
    value.as_u64().unwrap_or(0)
}
