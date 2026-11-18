// SPDX-License-Identifier: Apache-2.0
//! What a failed model call says, and which failures are worth another try.

use std::net::Ipv4Addr;

use super::*;

const LIMIT: usize = 4096;

#[test]
fn each_failure_carries_its_variant_value_and_no_request_or_response_text() {
    // Arrange
    let cases = [
        (
            ExtractError::NotLoopback(Ipv4Addr::new(10, 0, 0, 1).into()),
            "10.0.0.1",
        ),
        (
            ExtractError::Connect(io::ErrorKind::ConnectionRefused),
            "connection refused",
        ),
        (ExtractError::Io(io::ErrorKind::BrokenPipe), "broken pipe"),
        (ExtractError::Status(500), "500"),
        (ExtractError::Redirect(302), "302"),
        (ExtractError::TooLarge { limit: LIMIT }, "4096"),
        (ExtractError::Malformed("no braces"), "no braces"),
        (ExtractError::InvalidRequest("empty prompt"), "empty prompt"),
    ];

    for (error, carried) in cases {
        // Act
        let text = error.to_string();

        // Assert
        assert!(text.contains(carried), "{text:?} should carry {carried:?}");
    }
}

#[test]
fn every_failure_has_a_distinct_message() {
    // Arrange
    let errors = [
        ExtractError::NotLoopback(Ipv4Addr::LOCALHOST.into()),
        ExtractError::Connect(io::ErrorKind::ConnectionRefused),
        ExtractError::Io(io::ErrorKind::BrokenPipe),
        ExtractError::Timeout,
        ExtractError::Status(500),
        ExtractError::Redirect(302),
        ExtractError::Truncated,
        ExtractError::TooLarge { limit: LIMIT },
        ExtractError::Malformed("why"),
        ExtractError::InvalidRequest("why"),
        ExtractError::BadEnvelope,
        ExtractError::ModelMismatch,
        ExtractError::ModelRetired,
    ];

    // Act
    let messages: std::collections::HashSet<String> =
        errors.iter().map(ToString::to_string).collect();

    // Assert
    assert_eq!(messages.len(), errors.len());
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
