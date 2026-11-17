// SPDX-License-Identifier: Apache-2.0
//! The policy every hosted request passes: https only, a host on the
//! allowlist, no redirect, a response size cap and a deadline. It wraps any
//! transport, so a test needs no socket.

use std::fmt;
use std::time::Duration;

use super::multipart::Encoded;
use super::secrets::SecretString;
use super::transport::{HttpRequest, HttpResponse, HttpTransport, Method};
use crate::config::HostedConfig;

const HTTPS_PREFIX: &str = "https://";
/// Where a URL's authority ends.
const AUTHORITY_END: [char; 4] = ['/', '?', '#', '\\'];
const JSON_CONTENT_TYPE: &str = "application/json";
const REDIRECT_STATUSES: std::ops::Range<u16> = 300..400;

/// Why a request was refused or did not complete. No variant carries a URL
/// path, a query, a header or a body.
#[derive(Debug, PartialEq, Eq)]
pub enum HostedError {
    NotHttps,
    HostNotAllowed { host: String },
    Redirect { status: u16 },
    ResponseTooLarge { limit: usize },
    Timeout { secs: u64 },
    Transport { reason: String },
}

impl fmt::Display for HostedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotHttps => write!(f, "a hosted request must use https"),
            Self::HostNotAllowed { host } => {
                write!(f, "host {host} is not in hosted.allowed_hosts")
            }
            Self::Redirect { status } => {
                write!(
                    f,
                    "the provider answered with redirect {status}, which is not followed"
                )
            }
            Self::ResponseTooLarge { limit } => {
                write!(
                    f,
                    "the response is larger than hosted.max_response_bytes ({limit})"
                )
            }
            Self::Timeout { secs } => {
                write!(f, "the provider did not answer within {secs} seconds")
            }
            Self::Transport { reason } => write!(f, "the request failed: {reason}"),
        }
    }
}

impl std::error::Error for HostedError {}

pub struct HostedClient<T: HttpTransport> {
    transport: T,
    allowed_hosts: Vec<String>,
    timeout: Duration,
    max_response_bytes: usize,
}

impl<T: HttpTransport> HostedClient<T> {
    pub fn new(transport: T, config: &HostedConfig) -> Self {
        Self {
            transport,
            allowed_hosts: config.allowed_hosts.clone(),
            timeout: Duration::from_secs(config.request_timeout_secs),
            max_response_bytes: config.max_response_bytes,
        }
    }

    /// Posts `body` to `url` if the policy allows it. The transport is not
    /// called for a URL the policy refuses.
    pub async fn post_json(
        &self,
        url: &str,
        headers: Vec<(String, SecretString)>,
        body: Vec<u8>,
    ) -> Result<HttpResponse, HostedError> {
        let content_type = Some(JSON_CONTENT_TYPE.to_owned());
        self.send(Method::Post, url, headers, content_type, body)
            .await
    }

    /// Posts a multipart form under the same policy as [`Self::post_json`].
    pub async fn post_form(
        &self,
        url: &str,
        headers: Vec<(String, SecretString)>,
        form: Encoded,
    ) -> Result<HttpResponse, HostedError> {
        self.send(
            Method::Post,
            url,
            headers,
            Some(form.content_type),
            form.body,
        )
        .await
    }

    pub async fn get(
        &self,
        url: &str,
        headers: Vec<(String, SecretString)>,
    ) -> Result<HttpResponse, HostedError> {
        self.send(Method::Get, url, headers, None, Vec::new()).await
    }

    pub async fn delete(
        &self,
        url: &str,
        headers: Vec<(String, SecretString)>,
    ) -> Result<HttpResponse, HostedError> {
        self.send(Method::Delete, url, headers, None, Vec::new())
            .await
    }

    async fn send(
        &self,
        method: Method,
        url: &str,
        headers: Vec<(String, SecretString)>,
        content_type: Option<String>,
        body: Vec<u8>,
    ) -> Result<HttpResponse, HostedError> {
        let host = self.check_url(url)?;
        let request = HttpRequest {
            method,
            url: url.to_owned(),
            headers,
            content_type,
            body,
            max_response_bytes: self.max_response_bytes,
        };
        let response = tokio::time::timeout(self.timeout, self.transport.send(request))
            .await
            .map_err(|_| HostedError::Timeout {
                secs: self.timeout.as_secs(),
            })?
            .map_err(|error| HostedError::Transport { reason: error.0 })?;
        tracing::debug!(host, status = response.status, "hosted request answered");
        self.check_response(&response)?;
        Ok(response)
    }

    /// The host of a URL that is https and listed. The host is compared as
    /// written: a listed name with a port, a trailing dot or capitals is a
    /// different string and is refused.
    fn check_url<'a>(&self, url: &'a str) -> Result<&'a str, HostedError> {
        let rest = url
            .strip_prefix(HTTPS_PREFIX)
            .ok_or(HostedError::NotHttps)?;
        let authority = rest.split(AUTHORITY_END).next().unwrap_or_default();
        let listed = self.allowed_hosts.iter().any(|host| host == authority);
        if listed && !authority.contains('@') && !url.chars().any(char::is_whitespace) {
            Ok(authority)
        } else {
            Err(HostedError::HostNotAllowed {
                host: authority.to_owned(),
            })
        }
    }

    fn check_response(&self, response: &HttpResponse) -> Result<(), HostedError> {
        if REDIRECT_STATUSES.contains(&response.status) {
            return Err(HostedError::Redirect {
                status: response.status,
            });
        }
        if response.body.len() > self.max_response_bytes {
            return Err(HostedError::ResponseTooLarge {
                limit: self.max_response_bytes,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;

    use super::*;
    use crate::hosted::transport::TransportError;

    const HOST: &str = "api.anthropic.com";
    const URL: &str = "https://api.anthropic.com/v1/messages";
    const MAX: usize = 8;
    const TIMEOUT_SECS: u64 = 5;

    /// Answers with a canned response and counts the calls.
    #[derive(Clone)]
    struct Fake {
        calls: Arc<AtomicUsize>,
        answer: Arc<Mutex<Option<Result<HttpResponse, TransportError>>>>,
        never: bool,
    }

    impl Fake {
        fn answering(response: Result<HttpResponse, TransportError>) -> Self {
            Self {
                calls: Arc::default(),
                answer: Arc::new(Mutex::new(Some(response))),
                never: false,
            }
        }

        fn ok(status: u16, body: &[u8]) -> Self {
            Self::answering(Ok(HttpResponse {
                status,
                body: body.to_vec(),
            }))
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl HttpTransport for Fake {
        async fn send(&self, _: HttpRequest) -> Result<HttpResponse, TransportError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.never {
                std::future::pending::<()>().await;
            }
            self.answer.lock().unwrap().take().expect("one answer")
        }
    }

    fn client(fake: &Fake) -> HostedClient<Fake> {
        let config = HostedConfig {
            enabled: true,
            allowed_hosts: vec![HOST.to_owned()],
            request_timeout_secs: TIMEOUT_SECS,
            max_response_bytes: MAX,
            ..HostedConfig::default()
        };
        HostedClient::new(fake.clone(), &config)
    }

    async fn post(client: &HostedClient<Fake>, url: &str) -> Result<HttpResponse, HostedError> {
        client.post_json(url, Vec::new(), b"{}".to_vec()).await
    }

    #[tokio::test]
    async fn a_listed_https_host_is_called_and_its_answer_returned() {
        // Arrange
        let fake = Fake::ok(200, b"{}");

        // Act
        let response = post(&client(&fake), URL).await.unwrap();

        // Assert
        assert_eq!(response.status, 200);
        assert_eq!(fake.calls(), 1);
    }

    #[tokio::test]
    async fn a_plain_http_url_is_refused_before_the_transport_is_called() {
        // Arrange
        let fake = Fake::ok(200, b"{}");

        // Act
        let result = post(&client(&fake), "http://api.anthropic.com/v1/messages").await;

        // Assert
        assert_eq!(result, Err(HostedError::NotHttps));
        assert_eq!(fake.calls(), 0);
    }

    #[tokio::test]
    async fn every_form_that_is_not_the_listed_host_is_refused_before_the_transport_is_called() {
        for url in [
            "https://api.anthropic.com.evil.com/v1",
            "https://evil.com@api.anthropic.com/v1",
            "https://api.anthropic.com@evil.com/v1",
            "https://API.ANTHROPIC.COM/v1",
            "https://api.anthropic.com./v1",
            "https://api.anthropic.com:443/v1",
            "https://93.184.216.34/v1",
            "https://[::1]/v1",
            "https://evil.com\\@api.anthropic.com/v1",
            "https://evil.com#@api.anthropic.com/v1",
            "https:///v1",
            "https://api.anthropic.com /v1",
        ] {
            // Arrange
            let fake = Fake::ok(200, b"{}");

            // Act
            let result = post(&client(&fake), url).await;

            // Assert
            assert!(
                matches!(result, Err(HostedError::HostNotAllowed { .. })),
                "{url}: {result:?}"
            );
            assert_eq!(fake.calls(), 0, "{url}");
        }
    }

    #[tokio::test]
    async fn a_refusal_names_the_host_and_never_the_path_or_query() {
        // Arrange
        let fake = Fake::ok(200, b"{}");

        // Act
        let error = post(&client(&fake), "https://evil.com/v1?key=secret-key")
            .await
            .unwrap_err();

        // Assert
        let text = error.to_string();
        assert!(text.contains("evil.com"), "{text}");
        assert!(
            !text.contains("secret-key") && !text.contains("/v1"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn every_redirect_status_is_a_typed_error() {
        for status in [301, 302, 303, 307, 308] {
            // Arrange
            let fake = Fake::ok(status, b"");

            // Act
            let result = post(&client(&fake), URL).await;

            // Assert
            assert_eq!(result, Err(HostedError::Redirect { status }));
        }
    }

    #[tokio::test]
    async fn a_response_of_exactly_the_limit_is_accepted_and_one_byte_over_is_not() {
        // Arrange
        let at_limit = Fake::ok(200, &[b'a'; MAX]);
        let over = Fake::ok(200, &[b'a'; MAX + 1]);

        // Act
        let accepted = post(&client(&at_limit), URL).await;
        let rejected = post(&client(&over), URL).await;

        // Assert
        assert!(accepted.is_ok());
        assert_eq!(rejected, Err(HostedError::ResponseTooLarge { limit: MAX }));
    }

    #[tokio::test(start_paused = true)]
    async fn a_transport_that_never_answers_ends_in_a_timeout() {
        // Arrange
        let mut fake = Fake::ok(200, b"{}");
        fake.never = true;
        let client = client(&fake);

        // Act
        let result = post(&client, URL).await;

        // Assert
        assert_eq!(result, Err(HostedError::Timeout { secs: TIMEOUT_SECS }));
    }

    #[tokio::test]
    async fn a_failing_request_reports_the_reason_without_header_or_body() {
        // Arrange
        let fake = Fake::answering(Err(TransportError("the connection failed".into())));
        let client = client(&fake);

        // Act
        let error = client
            .post_json(
                URL,
                vec![("x-api-key".into(), SecretString::new("header-secret"))],
                b"prompt-secret".to_vec(),
            )
            .await
            .unwrap_err();

        // Assert
        let text = format!("{error} {error:?}");
        assert!(text.contains("the connection failed"), "{text}");
        for secret in ["header-secret", "prompt-secret", "x-api-key"] {
            assert!(!text.contains(secret), "{text}");
        }
    }
}
