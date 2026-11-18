// SPDX-License-Identifier: Apache-2.0
//! What a refused or failed hosted request says: no URL, header or body.

use super::*;

const LIMIT: usize = 1_048_576;
const TIMEOUT_SECS: u64 = 30;
const REDIRECT_STATUS: u16 = 307;
const HOST: &str = "api.example.com";

#[test]
fn each_refusal_carries_the_value_the_operator_can_change() {
    // Arrange
    let cases = [
        (HostedError::NotHttps, "https"),
        (HostedError::HostNotAllowed { host: HOST.into() }, HOST),
        (
            HostedError::HostNotAllowed { host: HOST.into() },
            "hosted.allowed_hosts",
        ),
        (
            HostedError::Redirect {
                status: REDIRECT_STATUS,
            },
            "307",
        ),
        (
            HostedError::ResponseTooLarge { limit: LIMIT },
            "hosted.max_response_bytes",
        ),
        (HostedError::ResponseTooLarge { limit: LIMIT }, "1048576"),
        (HostedError::Timeout { secs: TIMEOUT_SECS }, "30"),
        (
            HostedError::Transport {
                reason: "the connection broke".into(),
            },
            "the connection broke",
        ),
    ];

    for (error, carried) in cases {
        // Act
        let text = error.to_string();

        // Assert
        assert!(text.contains(carried), "{text:?} should carry {carried:?}");
    }
}
