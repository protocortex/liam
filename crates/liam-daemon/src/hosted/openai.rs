// SPDX-License-Identifier: Apache-2.0
//! OpenAI's Chat Completions API.

use serde_json::{json, Value};

use super::dialect::{base_headers, count, ChatCall, ChatReply, Dialect};
use super::secrets::SecretString;
use crate::config::{Provider, TokenUsage};

pub const HOST: &str = "api.openai.com";
const PATH: &str = "/v1/chat/completions";
const KEY_HEADER: &str = "authorization";
const BEARER: &str = "Bearer ";
const REQUEST_ID_HEADER: &str = "x-client-request-id";
const SCHEMA_NAME: &str = "answer";

pub struct OpenAi;

impl Dialect for OpenAi {
    fn provider(&self) -> Provider {
        Provider::OpenAi
    }

    fn url(&self, _model: &str) -> String {
        format!("https://{HOST}{PATH}")
    }

    fn auth_headers(&self, key: &SecretString) -> Vec<(String, SecretString)> {
        base_headers(
            KEY_HEADER,
            SecretString::new(format!("{BEARER}{}", key.expose())),
        )
    }

    fn headers(&self, key: &SecretString, call: &ChatCall<'_>) -> Vec<(String, SecretString)> {
        let mut headers = self.auth_headers(key);
        headers.push((
            REQUEST_ID_HEADER.to_owned(),
            SecretString::new(call.request_id),
        ));
        headers
    }

    fn body(&self, model: &str, call: &ChatCall<'_>) -> Value {
        let mut body = json!({
            "model": model,
            "max_completion_tokens": call.max_tokens,
            "temperature": call.temperature,
            "seed": call.seed,
            "messages": [
                { "role": "system", "content": call.system },
                { "role": "user", "content": call.prompt },
            ],
        });
        if let Some(schema) = call.schema {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": { "name": SCHEMA_NAME, "schema": schema },
            });
        }
        body
    }

    fn parse(&self, body: &[u8]) -> Result<ChatReply, &'static str> {
        let value: Value = serde_json::from_slice(body).map_err(|_| "the body is not JSON")?;
        let text = value["choices"][0]["message"]["content"]
            .as_str()
            .ok_or("the reply has no message content")?;
        let model = value["model"].as_str().ok_or("the reply names no model")?;
        let usage = &value["usage"];
        Ok(ChatReply {
            text: text.to_owned(),
            model: model.to_owned(),
            usage: TokenUsage {
                input_tokens: count(&usage["prompt_tokens"]),
                output_tokens: count(&usage["completion_tokens"]),
                cached_input_tokens: count(&usage["prompt_tokens_details"]["cached_tokens"]),
            },
        })
    }
}
