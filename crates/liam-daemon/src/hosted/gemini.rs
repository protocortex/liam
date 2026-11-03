// SPDX-License-Identifier: Apache-2.0
//! Google's Gemini `generateContent` API. It has no field for a caller's
//! request id, so none is sent.

use serde_json::{json, Value};

use super::dialect::{base_headers, count, ChatCall, ChatReply, Dialect};
use super::secrets::SecretString;
use crate::config::{Provider, TokenUsage};

pub const HOST: &str = "generativelanguage.googleapis.com";
const KEY_HEADER: &str = "x-goog-api-key";
const JSON_MIME: &str = "application/json";

pub struct Gemini;

impl Dialect for Gemini {
    fn provider(&self) -> Provider {
        Provider::Gemini
    }

    fn url(&self, model: &str) -> String {
        format!("https://{HOST}/v1beta/models/{model}:generateContent")
    }

    fn headers(&self, key: &SecretString, _call: &ChatCall<'_>) -> Vec<(String, SecretString)> {
        base_headers(KEY_HEADER, SecretString::new(key.expose()))
    }

    fn body(&self, _model: &str, call: &ChatCall<'_>) -> Value {
        let mut generation = json!({
            "temperature": call.temperature,
            "maxOutputTokens": call.max_tokens,
            "seed": call.seed,
        });
        if call.schema.is_some() {
            generation["responseMimeType"] = json!(JSON_MIME);
        }
        json!({
            "systemInstruction": { "parts": [{ "text": call.system }] },
            "contents": [{ "role": "user", "parts": [{ "text": call.prompt }] }],
            "generationConfig": generation,
        })
    }

    fn parse(&self, body: &[u8]) -> Result<ChatReply, &'static str> {
        let value: Value = serde_json::from_slice(body).map_err(|_| "the body is not JSON")?;
        let parts = value["candidates"][0]["content"]["parts"]
            .as_array()
            .ok_or("the reply has no candidate")?;
        let text: String = parts
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect();
        if text.is_empty() {
            return Err("the candidate has no text");
        }
        let model = value["modelVersion"]
            .as_str()
            .ok_or("the reply names no model")?;
        let usage = &value["usageMetadata"];
        Ok(ChatReply {
            text,
            model: model.to_owned(),
            usage: TokenUsage {
                input_tokens: count(&usage["promptTokenCount"]),
                output_tokens: count(&usage["candidatesTokenCount"]),
                cached_input_tokens: count(&usage["cachedContentTokenCount"]),
            },
        })
    }
}
