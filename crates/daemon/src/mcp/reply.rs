// SPDX-License-Identifier: Apache-2.0
//! The reply of `recall`: a text body plus the reasons it is degraded.

use std::sync::Arc;

use liam_store::{Basis, EventRef, Retrieved};
use rmcp::handler::server::common::schema_for_type;
use rmcp::model::{CallToolResult, ContentBlock, JsonObject};
use schemars::JsonSchema;
use serde::Serialize;

use crate::degraded::Degraded;

/// Opens the marker line that precedes the body of a degraded reply.
pub const DEGRADED_MARKER_PREFIX: &str = "(degraded: ";
const MARKER_SUFFIX: &str = ")";
const REASON_SEPARATOR: &str = "; ";

/// The `kind` an entity page carries in a `recall` reply. A memory has none.
pub const PAGE_KIND: &str = "entity_page";

/// An entity page among the results of a `recall`: dated lines cut from the
/// recorded memories of one entity, never rewritten, and not itself a memory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, JsonSchema)]
pub struct RecalledPage {
    /// The handle in the page's bracket.
    pub handle: String,
    /// Always `entity_page`.
    pub kind: String,
    /// Ids of the events of the memories the page was built from, the same ids
    /// those memories list as their events.
    pub basis: Vec<String>,
}

impl RecalledPage {
    /// The page for a hit with this `handle`, or `None` for a recorded memory.
    pub fn of(handle: &str, retrieved: &Retrieved) -> Option<Self> {
        let Retrieved::Inferred(hit) = retrieved else {
            return None;
        };
        let events: &[EventRef] = match &hit.basis {
            Basis::Known(events) => events,
            Basis::Unknown => &[],
        };
        Some(Self {
            handle: handle.to_string(),
            kind: PAGE_KIND.to_string(),
            basis: events.iter().map(|event| event.event_id.clone()).collect(),
        })
    }
}

/// Shape of the structured content a `recall` reply carries. `pages` is left
/// out when no result is an entity page, so a reply of memories keeps its shape.
#[derive(Serialize, JsonSchema)]
struct RecallOutput {
    degraded: Vec<Degraded>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pages: Vec<RecalledPage>,
}

/// The JSON schema advertised as the output schema of `recall`.
pub fn output_schema() -> Arc<JsonObject> {
    schema_for_type::<RecallOutput>()
}

/// A tool's text body with the reasons it is degraded and whether it failed.
pub struct ToolReply {
    text: String,
    degraded: Vec<Degraded>,
    pages: Vec<RecalledPage>,
    is_error: bool,
}

impl ToolReply {
    /// A successful reply carrying `text`.
    pub fn ok(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            degraded: Vec::new(),
            pages: Vec::new(),
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

    /// Lists the entity pages among the results the text shows, for a client
    /// that reads structure.
    pub fn with_pages(mut self, pages: Vec<RecalledPage>) -> Self {
        self.pages = pages;
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
            serde_json::to_value(RecallOutput {
                degraded: self.degraded,
                pages: self.pages,
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
    fn output_schema_requires_degraded_array_of_every_reason() {
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
        let reply = ToolReply::ok(BODY).with_degraded(Degraded::EmbedFailed);

        // Act
        let body = reply.text();

        // Assert
        assert_eq!(body, BODY);
    }

    #[test]
    fn each_reason_renders_its_own_marker_line_and_hint() {
        // Arrange
        let cases = [(
            Degraded::EmbedFailed,
            "embed_failed",
            "searched by keyword only",
        )];
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

    #[test]
    fn a_reply_with_pages_lists_them_and_one_without_keeps_its_shape() {
        // Arrange
        let page = RecalledPage {
            handle: "h1".to_string(),
            kind: PAGE_KIND.to_string(),
            basis: vec!["e1".to_string()],
        };

        // Act
        let with = ToolReply::ok(BODY)
            .with_pages(vec![page])
            .into_call_result();
        let without = ToolReply::ok(BODY).into_call_result();

        // Assert
        assert_eq!(
            with.structured_content,
            Some(json!({
                "degraded": [],
                "pages": [{"handle": "h1", "kind": "entity_page", "basis": ["e1"]}]
            }))
        );
        assert_eq!(without.structured_content, Some(json!({"degraded": []})));
    }

    #[test]
    fn output_schema_describes_pages_without_requiring_them() {
        // Arrange / Act
        let schema = serde_json::Value::Object((*output_schema()).clone());

        // Assert
        assert_eq!(schema["required"], json!(["degraded"]));
        assert_eq!(schema["properties"]["pages"]["type"], json!("array"));
    }

    #[test]
    fn only_an_inferred_hit_is_a_page_and_its_basis_is_its_event_ids() {
        // Arrange
        let recorded = Retrieved::Recorded(liam_store::RecordedHit {
            content: liam_store::RecordedText::for_test("said so"),
            events: Vec::new(),
            derivation: liam_store::Derivation::Recorded,
        });
        let page = |basis| {
            Retrieved::Inferred(liam_store::InferredHit {
                text: liam_store::InferredText::for_test("page"),
                basis,
                derivation: liam_store::Derivation::Inferred,
                summary_level: None,
            })
        };
        let event = |id: &str| EventRef {
            event_id: id.to_string(),
            synthetic: false,
        };

        // Act
        let none = RecalledPage::of("h", &recorded);
        let known = RecalledPage::of("h", &page(Basis::Known(vec![event("e1"), event("e2")])));
        let unknown = RecalledPage::of("h", &page(Basis::Unknown));

        // Assert
        assert_eq!(none, None);
        assert_eq!(known.unwrap().basis, ["e1", "e2"]);
        assert!(unknown.unwrap().basis.is_empty());
    }
}
