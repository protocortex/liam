// SPDX-License-Identifier: Apache-2.0
//! How a failed call is described, and that every method reaches the wire.

use super::*;

const TIMEOUT: Duration = Duration::from_secs(2);
const LIMIT: u64 = 10;
/// Nothing listens on port 1, so a connection to it is refused at once.
const REFUSED_URL: &str = "https://127.0.0.1:1/v1?key=url-secret";
const PLAIN_HTTP_URL: &str = "http://127.0.0.1:1/v1";

fn request(method: Method, url: &str) -> HttpRequest {
    HttpRequest {
        method,
        url: url.into(),
        headers: vec![("x-api-key".into(), SecretString::new("header-secret"))],
        content_type: Some("application/json".into()),
        body: b"prompt-secret".to_vec(),
        max_response_bytes: 10,
    }
}

#[test]
fn each_failure_kind_has_its_own_plain_description() {
    // Arrange
    let cases = [
        (
            ureq::Error::Timeout(ureq::Timeout::Global),
            "the request timed out",
        ),
        (ureq::Error::HostNotFound, "the host could not be resolved"),
        (ureq::Error::ConnectionFailed, "the connection failed"),
        (
            ureq::Error::Io(std::io::Error::other("reset")),
            "the connection broke",
        ),
        (
            ureq::Error::BodyExceedsLimit(LIMIT),
            "the response is larger than the limit",
        ),
        (ureq::Error::TooManyRedirects, "the request failed"),
    ];

    for (error, expected) in cases {
        // Act
        let described = classify(error);

        // Assert
        assert_eq!(described, TransportError(expected.to_owned()));
    }
}

#[tokio::test]
async fn get_and_delete_requests_fail_by_kind_when_nothing_listens() {
    // Arrange
    let transport = UreqTransport::new(TIMEOUT);

    for method in [Method::Get, Method::Delete] {
        // Act
        let error = transport
            .send(request(method, REFUSED_URL))
            .await
            .unwrap_err();

        // Assert
        let text = error.to_string();
        assert!(
            ["the connection broke", "the connection failed"].contains(&text.as_str()),
            "{method:?}: {text}"
        );
        for secret in ["url-secret", "header-secret", "127.0.0.1"] {
            assert!(!text.contains(secret), "{text}");
        }
    }
}

#[tokio::test]
async fn a_plain_http_address_is_refused_for_every_method() {
    // Arrange
    let transport = UreqTransport::new(TIMEOUT);

    for method in [Method::Get, Method::Post, Method::Delete] {
        // Act
        let error = transport
            .send(request(method, PLAIN_HTTP_URL))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(error.to_string(), "the request failed", "{method:?}");
    }
}
