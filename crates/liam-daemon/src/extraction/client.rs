// SPDX-License-Identifier: Apache-2.0
//! The seam between extraction and whatever model answers it: the trait,
//! its request and error types, and a scripted mock for tests.

use std::fmt;
use std::io;
use std::net::IpAddr;

use async_trait::async_trait;
use serde_json::Value;

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

    use super::*;

    const SEED: u32 = 7;
    const MAX_TOKENS: u32 = 321;

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
