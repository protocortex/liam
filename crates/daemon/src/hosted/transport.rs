// SPDX-License-Identifier: Apache-2.0
//! The wire under the policy layer: one request and its response. The `ureq`
//! implementation is the only code that opens a socket.

use std::fmt;
use std::time::Duration;

use async_trait::async_trait;

use super::secrets::SecretString;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
    Delete,
}

/// One request. Its `Debug` shows no URL, header or body, because a URL can
/// carry a key and a body carries the prompt.
pub struct HttpRequest {
    pub method: Method,
    pub url: String,
    pub headers: Vec<(String, SecretString)>,
    /// Sent as the content type when the request has a body.
    pub content_type: Option<String>,
    pub body: Vec<u8>,
    /// The most bytes of the response to read; one more is read so the policy
    /// layer can tell a response over the limit from one at it.
    pub max_response_bytes: usize,
}

impl fmt::Debug for HttpRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpRequest")
            .field("method", &self.method)
            .field("headers", &self.headers.len())
            .field("body_bytes", &self.body.len())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

/// A request that did not produce a response. The text names the kind of
/// failure and never the URL, a header or a body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportError(pub String);

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TransportError {}

#[async_trait]
pub trait HttpTransport: Send + Sync {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError>;
}

/// `ureq` over rustls. It follows no redirect, treats every status as a
/// response, and ends a call at the configured time.
pub struct UreqTransport {
    agent: ureq::Agent,
}

/// The agent's settings, in one place so a test can read them back.
pub(crate) fn agent_config(timeout: Duration) -> ureq::config::Config {
    ureq::Agent::config_builder()
        .https_only(true)
        .max_redirects(0)
        .http_status_as_error(false)
        .timeout_global(Some(timeout))
        .build()
}

impl UreqTransport {
    pub fn new(timeout: Duration) -> Self {
        Self {
            agent: ureq::Agent::new_with_config(agent_config(timeout)),
        }
    }
}

#[async_trait]
impl HttpTransport for UreqTransport {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
        let agent = self.agent.clone();
        tokio::task::spawn_blocking(move || call(&agent, &request))
            .await
            .map_err(|_| TransportError("the request task stopped".to_owned()))?
    }
}

fn call(agent: &ureq::Agent, request: &HttpRequest) -> Result<HttpResponse, TransportError> {
    let mut response = match request.method {
        Method::Get => {
            let mut call = agent.get(&request.url);
            for (name, value) in &request.headers {
                call = call.header(name, value.expose());
            }
            call.call()
        }
        Method::Delete => {
            let mut call = agent.delete(&request.url);
            for (name, value) in &request.headers {
                call = call.header(name, value.expose());
            }
            call.call()
        }
        Method::Post => {
            let mut call = agent.post(&request.url);
            if let Some(content_type) = &request.content_type {
                call = call.header("content-type", content_type);
            }
            for (name, value) in &request.headers {
                call = call.header(name, value.expose());
            }
            call.send(&request.body[..])
        }
    }
    .map_err(classify)?;
    let status = response.status().as_u16();
    let limit = u64::try_from(request.max_response_bytes)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let body = response
        .body_mut()
        .with_config()
        .limit(limit)
        .read_to_vec()
        .map_err(classify)?;
    Ok(HttpResponse { status, body })
}

fn classify(error: ureq::Error) -> TransportError {
    let kind = match error {
        ureq::Error::Timeout(_) => "the request timed out",
        ureq::Error::HostNotFound => "the host could not be resolved",
        ureq::Error::ConnectionFailed => "the connection failed",
        ureq::Error::Io(_) => "the connection broke",
        ureq::Error::BodyExceedsLimit(_) => "the response is larger than the limit",
        _ => "the request failed",
    };
    TransportError(kind.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIMEOUT: Duration = Duration::from_secs(7);

    #[test]
    fn the_agent_follows_no_redirect_sends_https_only_and_ends_at_the_timeout() {
        // Act
        let config = agent_config(TIMEOUT);

        // Assert
        assert_eq!(config.max_redirects(), 0);
        assert!(config.https_only());
        assert!(!config.http_status_as_error());
        assert_eq!(config.timeouts().global, Some(TIMEOUT));
    }

    #[tokio::test]
    async fn a_refused_connection_is_reported_by_kind_without_the_url() {
        // Arrange
        let transport = UreqTransport::new(TIMEOUT);
        let request = HttpRequest {
            method: Method::Post,
            url: "https://127.0.0.1:1/v1?key=url-secret".into(),
            headers: vec![("x-api-key".into(), SecretString::new("header-secret"))],
            content_type: None,
            body: b"prompt-secret".to_vec(),
            max_response_bytes: 10,
        };

        // Act
        let error = transport.send(request).await.unwrap_err();

        // Assert
        let text = error.to_string();
        for secret in ["url-secret", "header-secret", "prompt-secret", "127.0.0.1"] {
            assert!(!text.contains(secret), "{text}");
        }
    }

    #[test]
    fn a_request_shows_no_url_header_or_body_when_printed() {
        // Arrange
        let request = HttpRequest {
            method: Method::Post,
            url: "https://api.example.com/v1?key=url-secret".into(),
            headers: vec![("x-api-key".into(), SecretString::new("header-secret"))],
            content_type: None,
            body: b"prompt-secret".to_vec(),
            max_response_bytes: 10,
        };

        // Act
        let shown = format!("{request:?}");

        // Assert
        for secret in ["url-secret", "header-secret", "prompt-secret", "example"] {
            assert!(!shown.contains(secret), "{shown}");
        }
    }
}
