// SPDX-License-Identifier: Apache-2.0
//! The `ask` tool is gone: it is not listed, and calling it is the same error
//! as calling any tool the server never had.

use super::*;
use rmcp::ServiceExt;

const REMOVED_TOOL: &str = "ask";
const NEVER_EXISTED_TOOL: &str = "no_such_tool";

/// The error text a client gets for calling `tool` over an in-memory session.
async fn unknown_tool_error(tool: &'static str) -> String {
    let server = plain_server().await;
    let (server_io, client_io) = tokio::io::duplex(64 * 1024);
    let _running = tokio::spawn(async move { server.serve(server_io).await });
    let client = ().serve(client_io).await.expect("client handshake");
    let arguments = json!({"question": "what do I work on"})
        .as_object()
        .cloned()
        .expect("an object");
    let error = client
        .peer()
        .call_tool(rmcp::model::CallToolRequestParams::new(tool).with_arguments(arguments))
        .await
        .expect_err("an unknown tool is refused");
    error.to_string()
}

#[tokio::test]
async fn the_tool_list_has_no_ask() {
    // Arrange
    let server = plain_server().await;

    // Act
    let names: Vec<String> = server
        .tool_router
        .list_all()
        .into_iter()
        .map(|tool| tool.name.to_string())
        .collect();

    // Assert
    assert!(!names.iter().any(|name| name == REMOVED_TOOL), "{names:?}");
    assert!(!server.tool_router.has_route(REMOVED_TOOL));
    assert!(names.iter().any(|name| name == "recall"), "{names:?}");
}

#[tokio::test]
async fn calling_ask_returns_the_standard_unknown_tool_error() {
    // Arrange / Act
    let removed = unknown_tool_error(REMOVED_TOOL).await;
    let never_existed = unknown_tool_error(NEVER_EXISTED_TOOL).await;

    // Assert: the same error text, apart from the tool's own name.
    assert_eq!(
        removed.replace(REMOVED_TOOL, "<tool>"),
        never_existed.replace(NEVER_EXISTED_TOOL, "<tool>"),
        "{removed} / {never_existed}"
    );
}
