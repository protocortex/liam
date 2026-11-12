// SPDX-License-Identifier: Apache-2.0
//! The reply of `recall` and `ask`: a text body plus the reasons it is degraded.

use std::sync::Arc;

use rmcp::handler::server::common::schema_for_type;
use rmcp::model::{CallToolResult, ContentBlock, JsonObject};
use schemars::JsonSchema;
use serde::Serialize;

use crate::degraded::Degraded;

/// Opens the marker line that precedes the body of a degraded reply.
pub const DEGRADED_MARKER_PREFIX: &str = "(degraded: ";
const MARKER_SUFFIX: &str = ")";
const REASON_SEPARATOR: &str = "; ";

/// Shape of the structured content a `recall` or `ask` reply carries.
#[derive(Serialize, JsonSchema)]
struct DegradedOutput {
    degraded: Vec<Degraded>,
}

/// The JSON schema advertised as the output schema of `recall` and `ask`.
pub fn output_schema() -> Arc<JsonObject> {
    schema_for_type::<DegradedOutput>()
}

/// A tool's text body with the reasons it is degraded and whether it failed.
pub struct ToolReply {
    text: String,
    degraded: Vec<Degraded>,
    is_error: bool,
}

impl ToolReply {
    /// A successful reply carrying `text`.
    pub fn ok(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            degraded: Vec::new(),
            is_error: false,
        }
    }

    /// A tool-level failure carrying `text`.
    pub fn failed(text: impl Into<String>) -> Self {
        Self {
            is_error: true,
            ..Self::ok(text)
        }
    }

    /// Records a degradation reason, ignoring one already recorded.
    pub fn with_degraded(mut self, reason: Degraded) -> Self {
        if !self.degraded.contains(&reason) {
            self.degraded.push(reason);
        }
        self
    }

    /// The body text, without the marker line.
    #[cfg(test)]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The recorded degradation reasons in insertion order.
    pub fn degraded(&self) -> &[Degraded] {
        &self.degraded
    }

    /// Whether this reply reports a tool-level failure.
    #[cfg(test)]
    pub fn is_error(&self) -> bool {
        self.is_error
    }

    /// Converts to the wire result: marker plus body as text, reasons as structured content.
    pub fn into_call_result(self) -> CallToolResult {
        let text = if self.degraded.is_empty() {
            self.text
        } else {
            format!("{}\n{}", marker_line(&self.degraded), self.text)
        };
        let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
        result.structured_content = Some(
            serde_json::to_value(DegradedOutput {
                degraded: self.degraded,
            })
            .expect("a list of reason names always serializes"),
        );
        result.is_error = Some(self.is_error);
        result
    }
}

fn marker_line(reasons: &[Degraded]) -> String {
    let entries: Vec<String> = reasons
        .iter()
        .map(|reason| format!("{}, {}", reason.as_str(), reason.hint()))
        .collect();
    format!(
        "{DEGRADED_MARKER_PREFIX}{}{MARKER_SUFFIX}",
        entries.join(REASON_SEPARATOR)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::degraded::Degraded;
    use serde_json::json;

    const BODY: &str = "x";
    const FAILURE_TEXT: &str = "recall failed: store unavailable";

    fn text_of(result: &rmcp::model::CallToolResult) -> &str {
        assert_eq!(result.content.len(), 1, "reply must carry one text item");
        &result.content[0]
            .as_text()
            .expect("content item is text")
            .text
    }

    #[test]
    fn ok_reply_has_plain_text_and_empty_degraded_list() {
        // Arrange
        let reply = ToolReply::ok(BODY);

        // Act
        let result = reply.into_call_result();

        // Assert
        assert_eq!(text_of(&result), BODY);
        assert_eq!(result.structured_content, Some(json!({"degraded": []})));
        assert_ne!(result.is_error, Some(true));
    }

    #[test]
    fn two_reasons_are_marked_in_insertion_order_and_structured_in_snake_case() {
        // Arrange
        let reply = ToolReply::ok(BODY)
            .with_degraded(Degraded::RerankTimeout)
            .with_degraded(Degraded::SynthesisUnavailable);

        // Act
        let result = reply.into_call_result();

        // Assert
        let expected_text = format!(
            "{DEGRADED_MARKER_PREFIX}rerank_timeout, hits are in store order; \
             synthesis_unavailable, showing the retrieved evidence)\n{BODY}"
        );
        assert_eq!(text_of(&result), expected_text);
        assert_eq!(
            result.structured_content,
            Some(json!({"degraded": ["rerank_timeout", "synthesis_unavailable"]}))
        );
    }

    #[test]
    fn failed_reply_is_an_error_with_unchanged_text() {
        // Arrange
        let reply = ToolReply::failed(FAILURE_TEXT);

        // Act
        let result = reply.into_call_result();

        // Assert
        assert_eq!(result.is_error, Some(true));
        assert_eq!(text_of(&result), FAILURE_TEXT);
    }

    #[test]
    fn output_schema_requires_degraded_array_of_the_five_reasons() {
        // Arrange
        let expected_reasons: Vec<&str> = Degraded::ALL.iter().map(|d| d.as_str()).collect();

        // Act
        let schema = serde_json::Value::Object((*output_schema()).clone());

        // Assert
        assert_eq!(schema["required"], json!(["degraded"]));
        assert_eq!(schema["properties"]["degraded"]["type"], json!("array"));
        let schema_text = schema.to_string();
        for reason in expected_reasons {
            assert!(
                schema_text.contains(&format!("\"{reason}\"")),
                "schema must enumerate {reason}: {schema_text}"
            );
        }
    }

    #[test]
    fn adding_the_same_reason_twice_keeps_one_entry() {
        // Arrange
        let reply = ToolReply::ok(BODY)
            .with_degraded(Degraded::EmbedFailed)
            .with_degraded(Degraded::EmbedFailed);

        // Act
        let reasons = reply.degraded();

        // Assert
        assert_eq!(reasons, [Degraded::EmbedFailed]);
    }

    #[test]
    fn text_accessor_excludes_the_marker() {
        // Arrange
        let reply = ToolReply::ok(BODY).with_degraded(Degraded::RerankFailed);

        // Act
        let body = reply.text();

        // Assert
        assert_eq!(body, BODY);
    }

    #[test]
    fn each_reason_renders_its_own_marker_line_and_hint() {
        // Arrange
        let cases = [
            (
                Degraded::EmbedFailed,
                "embed_failed",
                "searched by keyword only",
            ),
            (
                Degraded::RerankSkipped,
                "rerank_skipped",
                "hits are in store order",
            ),
            (
                Degraded::RerankFailed,
                "rerank_failed",
                "hits are in store order",
            ),
            (
                Degraded::RerankTimeout,
                "rerank_timeout",
                "hits are in store order",
            ),
            (
                Degraded::SynthesisUnavailable,
                "synthesis_unavailable",
                "showing the retrieved evidence",
            ),
        ];
        assert_eq!(cases.len(), Degraded::ALL.len());

        for (reason, name, hint) in cases {
            // Act
            let result = ToolReply::ok(BODY).with_degraded(reason).into_call_result();

            // Assert
            assert_eq!(reason.as_str(), name);
            assert_eq!(reason.hint(), hint);
            assert_eq!(
                text_of(&result),
                format!("{DEGRADED_MARKER_PREFIX}{name}, {hint})\n{BODY}")
            );
            assert_eq!(result.structured_content, Some(json!({"degraded": [name]})));
        }
    }
}
