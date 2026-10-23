// SPDX-License-Identifier: Apache-2.0
//! The seam between extraction and whatever model answers it: the trait,
//! its request and error types, and a scripted mock for tests.

use std::fmt;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{debug, warn};

use super::http::{self, Limits, Request};

const COMPLETION_PATH: &str = "/v1/chat/completions";
const HEALTH_PATH: &str = "/health";
const HTTP_OK: u16 = 200;
const HTTP_SERVICE_UNAVAILABLE: u16 = 503;
const REDIRECT_STATUSES: std::ops::Range<u16> = 300..400;
/// A completion is at most a few thousand tokens of JSON; this only guards
/// against a runaway server.
const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
const DEFAULT_MAX_ATTEMPTS: u32 = 3;
const DEFAULT_INITIAL_BACKOFF: Duration = Duration::from_millis(250);
const DEFAULT_MAX_BACKOFF: Duration = Duration::from_secs(2);
const HEALTH_TIMEOUT: Duration = Duration::from_secs(2);

/// Why a model call failed. Carries no request or response text, since both
/// can hold event content.
// Most variants are produced by the HTTP client in later changes.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExtractError {
    NotLoopback(IpAddr),
    Connect(io::ErrorKind),
    Io(io::ErrorKind),
    Timeout,
    Status(u16),
    /// Redirects are never followed: the key must not leave the sidecar.
    Redirect(u16),
    Truncated,
    TooLarge {
        limit: usize,
    },
    Malformed(&'static str),
    InvalidRequest(&'static str),
    BadEnvelope,
}

impl ExtractError {
    /// Whether another attempt can help: a sidecar that is still loading
    /// refuses connections or answers 503. A timeout or a bad body repeats.
    fn is_transient(&self) -> bool {
        matches!(
            self,
            Self::Connect(_) | Self::Status(HTTP_SERVICE_UNAVAILABLE)
        )
    }
}

impl fmt::Display for ExtractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotLoopback(ip) => {
                write!(f, "the sidecar address {ip} is not a loopback address")
            }
            Self::Connect(kind) => write!(f, "could not connect to the sidecar: {kind}"),
            Self::Io(kind) => write!(f, "the connection to the sidecar failed: {kind}"),
            Self::Timeout => write!(f, "the sidecar did not answer in time"),
            Self::Status(status) => write!(f, "the sidecar answered with HTTP status {status}"),
            Self::Redirect(status) => {
                write!(f, "the sidecar answered with redirect status {status}")
            }
            Self::Truncated => write!(f, "the sidecar's response ended early"),
            Self::TooLarge { limit } => write!(f, "the sidecar's response is over {limit} bytes"),
            Self::Malformed(why) => write!(f, "the sidecar sent a malformed response: {why}"),
            Self::InvalidRequest(why) => write!(f, "the request is invalid: {why}"),
            Self::BadEnvelope => write!(f, "the sidecar's response has no completion text"),
        }
    }
}

impl std::error::Error for ExtractError {}

/// Which model answered, so a cache can key on it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelIdentity {
    pub model_id: String,
    pub model_sha256: String,
}

// Read by the sidecar client and the cache wrapper in later changes.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct ExtractRequest {
    pub system: String,
    pub prompt: String,
    /// JSON Schema the answer is constrained to.
    pub schema: Value,
    /// Events the prompt cites, for cache lineage.
    pub source_event_ids: Vec<String>,
    pub schema_hash: String,
    /// Always 0.0: extraction must repeat exactly.
    pub temperature: f32,
    pub seed: u32,
    pub max_tokens: u32,
}

#[async_trait]
pub trait JsonExtractor: Send + Sync {
    /// Returns the model's answer text, which the caller still has to parse.
    async fn extract_json(&self, req: &ExtractRequest) -> Result<String, ExtractError>;
    fn identity(&self) -> &ModelIdentity;
    async fn health(&self) -> bool;
}

/// Retries of transient failures: the delay doubles from `initial_backoff`
/// and never passes `max_backoff`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            initial_backoff: DEFAULT_INITIAL_BACKOFF,
            max_backoff: DEFAULT_MAX_BACKOFF,
        }
    }
}

impl RetryPolicy {
    fn backoff(&self, retry: u32) -> Duration {
        let factor = 1u32.checked_shl(retry).unwrap_or(u32::MAX);
        self.initial_backoff
            .saturating_mul(factor)
            .min(self.max_backoff)
    }
}

struct ApiKey(String);

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

/// Client for a `llama-server` on loopback.
#[derive(Debug)]
pub struct SidecarClient {
    addr: SocketAddr,
    api_key: ApiKey,
    timeout: Duration,
    identity: ModelIdentity,
    retry: RetryPolicy,
}

/// The slice of the OpenAI-style chat response that carries the answer.
#[derive(Deserialize)]
struct Completion {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Message {
    content: String,
}

impl SidecarClient {
    /// Refuses a non-loopback `addr`: the connection is plain HTTP and carries
    /// the bearer key and event text, so it must never leave the machine.
    pub fn new(
        addr: SocketAddr,
        api_key: String,
        timeout: Duration,
        identity: ModelIdentity,
    ) -> Result<Self, ExtractError> {
        if !addr.ip().is_loopback() {
            return Err(ExtractError::NotLoopback(addr.ip()));
        }
        http::check_header_value(&api_key)?;
        Ok(Self {
            addr,
            api_key: ApiKey(api_key),
            timeout,
            identity,
            retry: RetryPolicy::default(),
        })
    }

    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    async fn attempt(&self, body: &[u8]) -> Result<String, ExtractError> {
        let request = Request {
            method: "POST",
            path: COMPLETION_PATH,
            bearer: Some(&self.api_key.0),
            body,
        };
        let limits = Limits {
            timeout: self.timeout,
            max_body_bytes: MAX_RESPONSE_BYTES,
        };
        let response = http::send(self.addr, &request, limits).await?;
        debug!(status = response.status, "sidecar completion answered");
        match response.status {
            HTTP_OK => serde_json::from_slice::<Completion>(&response.body)
                .ok()
                .and_then(|completion| completion.choices.into_iter().next())
                .map(|choice| choice.message.content)
                .ok_or(ExtractError::BadEnvelope),
            status if REDIRECT_STATUSES.contains(&status) => Err(ExtractError::Redirect(status)),
            status => Err(ExtractError::Status(status)),
        }
    }
}

fn completion_body(req: &ExtractRequest) -> Vec<u8> {
    let body = json!({
        "messages": [
            { "role": "system", "content": req.system },
            { "role": "user", "content": req.prompt },
        ],
        "response_format": { "type": "json_object", "schema": req.schema },
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "seed": req.seed,
        // The server reuses cached prompt state by default, which can change
        // the logits and break repeatable answers.
        "cache_prompt": false,
        // A thinking model would otherwise spend the token budget reasoning
        // before it writes the JSON.
        "chat_template_kwargs": { "enable_thinking": false },
    });
    body.to_string().into_bytes()
}

#[async_trait]
impl JsonExtractor for SidecarClient {
    async fn extract_json(&self, req: &ExtractRequest) -> Result<String, ExtractError> {
        let body = completion_body(req);
        let mut attempt = 1;
        loop {
            match self.attempt(&body).await {
                Err(error) if error.is_transient() && attempt < self.retry.max_attempts => {
                    warn!(attempt, %error, "sidecar call failed, retrying");
                    tokio::time::sleep(self.retry.backoff(attempt - 1)).await;
                    attempt += 1;
                }
                result => return result,
            }
        }
    }

    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    async fn health(&self) -> bool {
        let request = Request {
            method: "GET",
            path: HEALTH_PATH,
            bearer: None,
            body: &[],
        };
        let limits = Limits {
            timeout: HEALTH_TIMEOUT.min(self.timeout),
            max_body_bytes: MAX_RESPONSE_BYTES,
        };
        matches!(
            http::send(self.addr, &request, limits).await,
            Ok(response) if response.status == HTTP_OK
        )
    }
}

#[cfg(test)]
pub use mock::MockExtractor;

#[cfg(test)]
mod mock {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;

    use super::*;

    /// Answers from a fixed script, one entry per call, in order.
    pub struct MockExtractor {
        identity: ModelIdentity,
        script: Mutex<VecDeque<Result<String, ExtractError>>>,
        calls: AtomicUsize,
        healthy: AtomicBool,
    }

    impl MockExtractor {
        pub fn new(identity: ModelIdentity, script: Vec<Result<String, ExtractError>>) -> Self {
            Self {
                identity,
                script: Mutex::new(script.into()),
                calls: AtomicUsize::new(0),
                healthy: AtomicBool::new(true),
            }
        }

        pub fn set_healthy(&self, healthy: bool) {
            self.healthy.store(healthy, Ordering::SeqCst);
        }

        pub fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl JsonExtractor for MockExtractor {
        async fn extract_json(&self, _req: &ExtractRequest) -> Result<String, ExtractError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.script
                .lock()
                .expect("script lock")
                .pop_front()
                .expect("the mock script ran out")
        }

        fn identity(&self) -> &ModelIdentity {
            &self.identity
        }

        async fn health(&self) -> bool {
            self.healthy.load(Ordering::SeqCst)
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::test_double::{ok, raw, spawn_double, Reply};
    use super::*;

    const SEED: u32 = 7;
    const MAX_TOKENS: u32 = 321;
    const API_KEY: &str = "test-key-1234";
    const REQUEST_TIMEOUT: Duration = Duration::from_millis(200);
    const STALL: Duration = Duration::from_secs(10);
    const FAST_BACKOFF: Duration = Duration::from_millis(1);

    fn identity() -> ModelIdentity {
        ModelIdentity {
            model_id: "test-model".into(),
            model_sha256: "abc123".into(),
        }
    }

    fn request() -> ExtractRequest {
        ExtractRequest {
            system: "SYSTEM TEXT".into(),
            prompt: "PROMPT TEXT".into(),
            schema: json!({ "type": "object", "title": "facts" }),
            source_event_ids: vec!["e1".into()],
            schema_hash: "schema-hash".into(),
            temperature: 0.0,
            seed: SEED,
            max_tokens: MAX_TOKENS,
        }
    }

    fn completion(content: &str) -> String {
        json!({ "choices": [{ "message": { "role": "assistant", "content": content } }] })
            .to_string()
    }

    fn client(addr: SocketAddr) -> SidecarClient {
        SidecarClient::new(addr, API_KEY.into(), REQUEST_TIMEOUT, identity())
            .unwrap()
            .with_retry(RetryPolicy {
                max_attempts: 1,
                ..RetryPolicy::default()
            })
    }

    #[tokio::test]
    async fn sends_one_authenticated_post_with_the_request_fields() {
        // Arrange
        let double = spawn_double(vec![ok(&completion("{}"))]).await;
        let client = client(double.addr);

        // Act
        let content = client.extract_json(&request()).await;

        // Assert
        assert_eq!(content.as_deref(), Ok("{}"));
        let requests = double.requests();
        assert_eq!(requests.len(), 1);
        let (head, body) = requests[0].split_once("\r\n\r\n").unwrap();
        assert!(head.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert!(head.contains(&format!("Authorization: Bearer {API_KEY}")));
        let body: Value = serde_json::from_str(body).unwrap();
        assert_eq!(body["response_format"]["schema"], request().schema);
        assert_eq!(body["max_tokens"], MAX_TOKENS);
        assert_eq!(body["cache_prompt"], false);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(body["seed"], SEED);
        assert_eq!(body["temperature"], 0.0);
        assert_eq!(
            body["messages"],
            json!([
                { "role": "system", "content": "SYSTEM TEXT" },
                { "role": "user", "content": "PROMPT TEXT" },
            ])
        );
    }

    #[tokio::test]
    async fn a_server_error_is_typed_with_its_status_and_not_retried() {
        // Arrange
        let reply = raw("HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n");
        let double = spawn_double(vec![reply]).await;
        let client = client(double.addr).with_retry(RetryPolicy::default());

        // Act
        let result = client.extract_json(&request()).await;

        // Assert
        assert_eq!(result, Err(ExtractError::Status(500)));
        assert_eq!(double.requests().len(), 1);
    }

    #[tokio::test]
    async fn a_body_that_is_not_the_envelope_is_a_typed_error() {
        for body in [
            "",
            "not json",
            r#"{"other":1}"#,
            r#"{"choices":[]}"#,
            r#"{"choices":[{"message":{"content":7}}]}"#,
        ] {
            // Arrange
            let double = spawn_double(vec![ok(body)]).await;
            let client = client(double.addr);

            // Act
            let result = client.extract_json(&request()).await;

            // Assert
            assert_eq!(result, Err(ExtractError::BadEnvelope), "body {body:?}");
        }
    }

    #[tokio::test]
    async fn a_redirect_is_reported_and_not_followed() {
        // Arrange
        let target = spawn_double(vec![ok(&completion("{}"))]).await;
        let redirect = format!(
            "HTTP/1.1 302 Found\r\nLocation: http://{}/v1/chat/completions\r\nContent-Length: 0\r\n\r\n",
            target.addr
        );
        let double = spawn_double(vec![raw(redirect)]).await;
        let client = client(double.addr);

        // Act
        let result = client.extract_json(&request()).await;

        // Assert
        assert_eq!(result, Err(ExtractError::Redirect(302)));
        assert!(target.requests().is_empty());
    }

    #[tokio::test]
    async fn a_loading_sidecar_is_retried_until_it_answers() {
        // Arrange
        let loading = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n";
        let double = spawn_double(vec![raw(loading), raw(loading), ok(&completion("{}"))]).await;
        let client = client(double.addr).with_retry(RetryPolicy {
            max_attempts: 3,
            initial_backoff: FAST_BACKOFF,
            max_backoff: FAST_BACKOFF,
        });

        // Act
        let result = client.extract_json(&request()).await;

        // Assert
        assert_eq!(result.as_deref(), Ok("{}"));
        assert_eq!(double.requests().len(), 3);
    }

    #[tokio::test]
    async fn retries_stop_at_the_attempt_budget() {
        // Arrange
        let loading = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n";
        let double = spawn_double(vec![raw(loading)]).await;
        let client = client(double.addr).with_retry(RetryPolicy {
            max_attempts: 2,
            initial_backoff: FAST_BACKOFF,
            max_backoff: FAST_BACKOFF,
        });

        // Act
        let result = client.extract_json(&request()).await;

        // Assert
        assert_eq!(result, Err(ExtractError::Status(503)));
        assert_eq!(double.requests().len(), 2);
    }

    #[test]
    fn backoff_doubles_and_stops_at_the_cap() {
        // Arrange
        let policy = RetryPolicy::default();

        // Act
        let delays: Vec<_> = [0, 1, 2, 3, 40].map(|retry| policy.backoff(retry)).into();

        // Assert
        assert_eq!(delays[0], DEFAULT_INITIAL_BACKOFF);
        assert_eq!(delays[1], DEFAULT_INITIAL_BACKOFF * 2);
        assert_eq!(delays[2], DEFAULT_INITIAL_BACKOFF * 4);
        assert_eq!(delays[3], DEFAULT_MAX_BACKOFF);
        assert_eq!(delays[4], DEFAULT_MAX_BACKOFF);
    }

    #[test]
    fn a_non_loopback_address_is_refused() {
        // Arrange
        let addr: SocketAddr = "192.0.2.1:8080".parse().unwrap();

        // Act
        let result = SidecarClient::new(addr, API_KEY.into(), REQUEST_TIMEOUT, identity());

        // Assert
        assert_eq!(result.unwrap_err(), ExtractError::NotLoopback(addr.ip()));
    }

    #[test]
    fn the_api_key_never_shows_in_debug_output() {
        // Arrange
        let client = client("127.0.0.1:1".parse().unwrap());

        // Act
        let shown = format!("{client:?}");

        // Assert
        assert!(!shown.contains(API_KEY));
    }

    #[tokio::test]
    async fn health_follows_the_health_endpoint() {
        // Arrange
        let healthy = spawn_double(vec![ok("{}")]).await;
        let failing = spawn_double(vec![raw("HTTP/1.1 500 X\r\nContent-Length: 0\r\n\r\n")]).await;

        // Act
        let (up, down) = (
            client(healthy.addr).health().await,
            client(failing.addr).health().await,
        );

        // Assert
        assert!(up && !down);
        assert!(healthy.requests()[0].starts_with("GET /health HTTP/1.1"));
        assert!(!healthy.requests()[0].contains("Authorization"));
    }

    #[tokio::test]
    async fn a_timeout_is_not_retried() {
        // Arrange
        let slow = Reply {
            delay: STALL,
            ..ok(&completion("{}"))
        };
        let double = spawn_double(vec![slow]).await;
        let client = client(double.addr).with_retry(RetryPolicy {
            max_attempts: 3,
            initial_backoff: FAST_BACKOFF,
            max_backoff: FAST_BACKOFF,
        });

        // Act
        let result = client.extract_json(&request()).await;

        // Assert
        assert_eq!(result, Err(ExtractError::Timeout));
        assert_eq!(double.requests().len(), 1);
    }

    #[test]
    fn only_a_refused_connection_or_a_503_is_transient() {
        // Arrange
        let transient = [
            ExtractError::Connect(io::ErrorKind::ConnectionRefused),
            ExtractError::Status(503),
        ];
        let permanent = [
            ExtractError::Timeout,
            ExtractError::Status(500),
            ExtractError::Redirect(302),
            ExtractError::Truncated,
            ExtractError::BadEnvelope,
            ExtractError::Io(io::ErrorKind::ConnectionReset),
        ];

        // Act and Assert
        assert!(transient.iter().all(ExtractError::is_transient));
        assert!(!permanent.iter().any(ExtractError::is_transient));
    }

    #[test]
    fn a_key_with_a_control_byte_is_refused() {
        // Arrange
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        // Act
        let result =
            SidecarClient::new(addr, "key\r\nX-Evil: 1".into(), REQUEST_TIMEOUT, identity());

        // Assert
        assert!(matches!(result, Err(ExtractError::InvalidRequest(_))));
    }

    #[tokio::test]
    async fn the_mock_fails_on_the_scripted_call_only() {
        // Arrange
        let mock = MockExtractor::new(
            identity(),
            vec![
                Ok("one".into()),
                Err(ExtractError::Status(500)),
                Ok("three".into()),
            ],
        );

        // Act
        let results = [
            mock.extract_json(&request()).await,
            mock.extract_json(&request()).await,
            mock.extract_json(&request()).await,
        ];

        // Assert
        assert_eq!(results[0].as_deref(), Ok("one"));
        assert_eq!(results[1], Err(ExtractError::Status(500)));
        assert_eq!(results[2].as_deref(), Ok("three"));
        assert_eq!(mock.calls(), 3);
        assert_eq!(mock.identity(), &identity());
    }

    #[tokio::test]
    async fn the_mock_health_is_scripted() {
        // Arrange
        let mock = MockExtractor::new(identity(), vec![]);
        mock.set_healthy(false);

        // Act
        let health = mock.health().await;

        // Assert
        assert!(!health);
    }
}
