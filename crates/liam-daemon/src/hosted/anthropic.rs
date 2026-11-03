// SPDX-License-Identifier: Apache-2.0
//! Anthropic's Messages API.

use serde_json::{json, Value};

use super::dialect::{base_headers, count, ChatCall, ChatReply, Dialect};
use super::secrets::SecretString;
use crate::config::{Provider, TokenUsage};

pub const HOST: &str = "api.anthropic.com";
const PATH: &str = "/v1/messages";
const KEY_HEADER: &str = "x-api-key";
const VERSION_HEADER: (&str, &str) = ("anthropic-version", "2023-06-01");

pub struct Anthropic;

impl Dialect for Anthropic {
    fn provider(&self) -> Provider {
        Provider::Anthropic
    }

    fn url(&self, _model: &str) -> String {
        format!("https://{HOST}{PATH}")
    }

    fn headers(&self, key: &SecretString, _call: &ChatCall<'_>) -> Vec<(String, SecretString)> {
        let mut headers = base_headers(KEY_HEADER, SecretString::new(key.expose()));
        headers.push((
            VERSION_HEADER.0.to_owned(),
            SecretString::new(VERSION_HEADER.1),
        ));
        headers
    }

    fn body(&self, model: &str, call: &ChatCall<'_>) -> Value {
        json!({
            "model": model,
            "max_tokens": call.max_tokens,
            "temperature": call.temperature,
            "system": call.system,
            "messages": [{ "role": "user", "content": call.prompt }],
            "metadata": { "user_id": call.request_id },
        })
    }

    fn parse(&self, body: &[u8]) -> Result<ChatReply, &'static str> {
        let value: Value = serde_json::from_slice(body).map_err(|_| "the body is not JSON")?;
        let text = value["content"]
            .as_array()
            .and_then(|blocks| blocks.iter().find(|block| block["type"] == "text"))
            .and_then(|block| block["text"].as_str())
            .ok_or("the reply has no text block")?;
        let model = value["model"].as_str().ok_or("the reply names no model")?;
        let usage = &value["usage"];
        // Anthropic reports cached reads beside, not inside, the input count.
        let cached = count(&usage["cache_read_input_tokens"]);
        Ok(ChatReply {
            text: text.to_owned(),
            model: model.to_owned(),
            usage: TokenUsage {
                input_tokens: count(&usage["input_tokens"]).saturating_add(cached),
                output_tokens: count(&usage["output_tokens"]),
                cached_input_tokens: cached,
            },
        })
    }
}
