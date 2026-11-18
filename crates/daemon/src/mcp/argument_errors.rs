// SPDX-License-Identifier: Apache-2.0
//! Clear replies for a tool call whose arguments are missing a required field.
//!
//! rmcp rejects such a call before the tool runs, with the raw serde text
//! (`failed to deserialize parameters: missing field ...`). The message is
//! rewritten here so a client learns which field to add and, where the field
//! has a fixed set of values, what they are. What a tool accepts is unchanged.

use rmcp::model::{CallToolResponse, CallToolResult, ContentBlock};

/// How rmcp starts the text of an argument error.
const RMCP_ARGUMENT_ERROR_PREFIX: &str = "failed to deserialize parameters:";
/// How serde names a required field the input lacks, up to the field's name.
const MISSING_FIELD_MARKER: &str = "missing field `";

/// The common kinds `remember` documents on its `kind` field. Any label is
/// stored; a test keeps this list equal to the field's description.
pub(super) const REMEMBER_KINDS: [&str; 4] = ["decision", "fact", "symbol", "episode"];

/// Replaces the raw serde text of a missing-field reply from `tool`. Any other
/// response is returned as it came.
pub(super) fn clarify(tool: &str, response: CallToolResponse) -> CallToolResponse {
    let CallToolResponse::Complete(result) = &response else {
        return response;
    };
    let message = result
        .content
        .first()
        .and_then(|block| block.as_text())
        .and_then(|text| missing_field_message(tool, &text.text));
    match message {
        Some(message) => CallToolResult::error(vec![ContentBlock::text(message)]).into(),
        None => response,
    }
}

/// The message for a raw argument error that names a missing field, or `None`
/// for any other text.
fn missing_field_message(tool: &str, raw: &str) -> Option<String> {
    let after_prefix = raw.strip_prefix(RMCP_ARGUMENT_ERROR_PREFIX)?;
    let after_marker = after_prefix.split_once(MISSING_FIELD_MARKER)?.1;
    let field = after_marker.split_once('`')?.0;
    let hint = match (tool, field) {
        ("remember", "kind") => {
            format!(
                ", the kind of memory. The common kinds are {}, but any label is stored",
                REMEMBER_KINDS.join(", ")
            )
        }
        _ => String::new(),
    };
    Some(format!(
        "{tool} needs the field `{field}`{hint}. Add it to the arguments and call {tool} again."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_error(text: &str) -> CallToolResponse {
        CallToolResult::error(vec![ContentBlock::text(text)]).into()
    }

    fn text_of(response: &CallToolResponse) -> String {
        let CallToolResponse::Complete(result) = response else {
            panic!("expected a complete response");
        };
        result.content[0].as_text().expect("text").text.clone()
    }

    #[test]
    fn a_missing_kind_names_the_field_and_lists_the_accepted_kinds() {
        // Arrange
        let raw = raw_error("failed to deserialize parameters: missing field `kind`");

        // Act
        let text = text_of(&clarify("remember", raw));

        // Assert
        assert!(text.contains("`kind`"), "{text}");
        for kind in REMEMBER_KINDS {
            assert!(text.contains(kind), "{kind} missing from {text}");
        }
        assert!(!text.contains("deserialize"), "{text}");
    }

    #[test]
    fn another_missing_field_is_named_without_a_value_list() {
        // Arrange
        let raw = raw_error("failed to deserialize parameters: missing field `question`");

        // Act
        let text = text_of(&clarify("ask", raw));

        // Assert
        assert!(text.contains("ask needs the field `question`"), "{text}");
    }

    #[test]
    fn any_other_reply_is_returned_unchanged() {
        for original in [
            "failed to deserialize parameters: invalid type: integer `3`, expected a string",
            "remember failed: confidence must be within 0.0 and 1.0",
        ] {
            // Arrange
            let response = raw_error(original);

            // Act
            let text = text_of(&clarify("remember", response));

            // Assert
            assert_eq!(text, original);
        }
    }
}
