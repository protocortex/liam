// SPDX-License-Identifier: Apache-2.0
//! What a refused or failed hosted request says: no URL, header or body.

use super::*;

const LIMIT: usize = 1_048_576;
const TIMEOUT_SECS: u64 = 30;

#[test]
fn each_refusal_names_what_the_operator_can_change() {
    // Arrange
    let cases = [
        (HostedError::NotHttps, "a hosted request must use https"),
        (
            HostedError::HostNotAllowed {
                host: "api.example.com".into(),
            },
            "host api.example.com is not in hosted.allowed_hosts",
        ),
        (
            HostedError::Redirect { status: 307 },
            "the provider answered with redirect 307, which is not followed",
        ),
        (
            HostedError::ResponseTooLarge { limit: LIMIT },
            "the response is larger than hosted.max_response_bytes (1048576)",
        ),
        (
            HostedError::Timeout { secs: TIMEOUT_SECS },
            "the provider did not answer within 30 seconds",
        ),
        (
            HostedError::Transport {
                reason: "the connection broke".into(),
            },
            "the request failed: the connection broke",
        ),
    ];

    for (error, expected) in cases {
        // Act
        let text = error.to_string();

        // Assert
        assert_eq!(text, expected);
    }
}
