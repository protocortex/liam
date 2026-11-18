// SPDX-License-Identifier: Apache-2.0
//! What a client gets back for a call that leaves out a required field, over a
//! real MCP session.

use super::*;
use rmcp::ServiceExt;

/// Calls `tool` over an in-memory session and returns the text of the reply and
/// whether it was marked as an error.
async fn call(tool: &'static str, arguments: serde_json::Value) -> (String, bool) {
    let server = plain_server().await;
    let (server_io, client_io) = tokio::io::duplex(64 * 1024);
    let _running = tokio::spawn(async move { server.serve(server_io).await });
    let client = ().serve(client_io).await.expect("client handshake");
    let arguments = arguments.as_object().cloned().expect("an object");
    let result = client
        .peer()
        .call_tool(rmcp::model::CallToolRequestParams::new(tool).with_arguments(arguments))
        .await
        .expect("the call is answered");
    let text = result.content[0].as_text().expect("text").text.clone();
    (text, result.is_error == Some(true))
}

#[tokio::test]
async fn remember_without_a_kind_names_the_field_and_the_accepted_kinds() {
    // Arrange / Act
    let (text, is_error) = call("remember", json!({"label": "L", "content": "C"})).await;

    // Assert
    assert!(is_error, "{text}");
    assert!(text.contains("`kind`"), "{text}");
    assert!(text.contains("decision, fact, symbol, episode"), "{text}");
    assert!(!text.contains("deserialize"), "{text}");
}

#[tokio::test]
async fn ask_without_a_question_names_the_field() {
    // Arrange / Act
    let (text, is_error) = call("ask", json!({"query": "what do I work on"})).await;

    // Assert
    assert!(is_error, "{text}");
    assert!(text.contains("`question`"), "{text}");
}

#[tokio::test]
async fn remember_with_a_kind_still_stores_the_memory() {
    // Arrange / Act
    let (text, is_error) = call(
        "remember",
        json!({"kind": "fact", "label": "L", "content": "C"}),
    )
    .await;

    // Assert
    assert!(!is_error, "{text}");
    assert!(text.starts_with("remembered "), "{text}");
}

#[tokio::test]
async fn the_schema_marks_kind_as_required_as_the_code_treats_it() {
    // Arrange
    let server = plain_server().await;

    // Act
    let tool = server
        .tool_router
        .list_all()
        .into_iter()
        .find(|tool| tool.name == "remember")
        .expect("remember is listed");

    // Assert
    let required = tool.input_schema.get("required").expect("required list");
    assert!(
        required
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field == "kind"),
        "{required}"
    );
}
