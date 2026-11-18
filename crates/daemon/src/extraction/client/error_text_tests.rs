// SPDX-License-Identifier: Apache-2.0
//! What a failed model call says, and which failures are worth another try.

use std::net::Ipv4Addr;

use super::*;

const LIMIT: usize = 4096;

#[test]
fn each_failure_is_described_without_request_or_response_text() {
    // Arrange
    let cases = [
        (
            ExtractError::NotLoopback(Ipv4Addr::new(10, 0, 0, 1).into()),
            "the sidecar address 10.0.0.1 is not a loopback address",
        ),
        (
            ExtractError::Connect(io::ErrorKind::ConnectionRefused),
            "could not connect to the sidecar: connection refused",
        ),
        (
            ExtractError::Io(io::ErrorKind::BrokenPipe),
            "the connection to the sidecar failed: broken pipe",
        ),
        (ExtractError::Timeout, "the sidecar did not answer in time"),
        (
            ExtractError::Status(500),
            "the sidecar answered with HTTP status 500",
        ),
        (
            ExtractError::Redirect(302),
            "the sidecar answered with redirect status 302",
        ),
        (
            ExtractError::Truncated,
            "the sidecar's response ended early",
        ),
        (
            ExtractError::TooLarge { limit: LIMIT },
            "the sidecar's response is over 4096 bytes",
        ),
        (
            ExtractError::Malformed("no braces"),
            "the sidecar sent a malformed response: no braces",
        ),
        (
            ExtractError::InvalidRequest("empty prompt"),
            "the request is invalid: empty prompt",
        ),
        (
            ExtractError::BadEnvelope,
            "the sidecar's response has no completion text",
        ),
        (
            ExtractError::ModelMismatch,
            "the provider answered with a model that is not the pin",
        ),
        (
            ExtractError::ModelRetired,
            "the provider no longer has the pinned model",
        ),
    ];

    for (error, expected) in cases {
        // Act
        let text = error.to_string();

        // Assert
        assert_eq!(text, expected);
    }
}

#[test]
fn only_a_refused_connection_or_a_loading_server_is_worth_another_try() {
    // Act / Assert
    assert!(ExtractError::Connect(io::ErrorKind::ConnectionRefused).is_transient());
    assert!(ExtractError::Status(HTTP_SERVICE_UNAVAILABLE).is_transient());
    assert!(!ExtractError::Status(500).is_transient());
    assert!(!ExtractError::Timeout.is_transient());
    assert!(!ExtractError::Malformed("bad").is_transient());
}
