// SPDX-License-Identifier: Apache-2.0
//! The summariser and the extractor over a pinned model: what they return and
//! how a failed call reads to the extraction loop.

use super::*;
use crate::config::{PinError, Provider};
use crate::hosted::client::HostedError;
use crate::hosted::openai::{OpenAi, HOST};
use crate::hosted::test_support::{model, FakeTransport};

const PIN: &str = "gpt-4o-2024-08-06";
const REPLY: &str = r#"{"model":"gpt-4o-2024-08-06","choices":[{"message":{"content":"a summary"}}],
    "usage":{"prompt_tokens":10,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":2}}}"#;

fn shared(transport: FakeTransport) -> Arc<HostedModel<FakeTransport>> {
    let (model, _) = model(
        Provider::OpenAi,
        Box::new(OpenAi),
        PIN,
        HOST,
        transport,
        64 * 1024,
    );
    Arc::new(model)
}

fn extract_request() -> ExtractRequest {
    ExtractRequest {
        system: "system".to_owned(),
        prompt: "prompt".to_owned(),
        schema: serde_json::json!({"type": "object"}),
        source_event_ids: vec!["e1".to_owned()],
        schema_hash: "hash".to_owned(),
        temperature: 0.0,
        seed: 0,
        max_tokens: 128,
    }
}

#[tokio::test]
async fn the_summariser_returns_the_text_and_the_reported_usage_under_the_pinned_identity() {
    // Arrange
    let transport = FakeTransport::answering(200, REPLY);
    let summariser = HostedSummariser::new(shared(transport.clone()));

    // Act
    let response = summariser
        .summarise(&SummaryRequest {
            system: "system".to_owned(),
            prompt: "prompt".to_owned(),
            source_event_ids: vec!["e1".to_owned()],
            max_tokens: 99,
        })
        .await
        .unwrap();

    // Assert
    let usage = response.usage.unwrap();
    assert_eq!(response.text, "a summary");
    assert_eq!(
        (
            usage.input_tokens,
            usage.output_tokens,
            usage.cached_input_tokens
        ),
        (10, 4, 2)
    );
    assert_eq!(summariser.identity().model_id, format!("openai:{PIN}"));
    let body = transport.seen()[0].body.to_string();
    assert!(body.contains("\"max_completion_tokens\":99"));
}

#[tokio::test]
async fn the_extractor_sends_the_schema_and_returns_the_answer_text() {
    // Arrange
    let transport = FakeTransport::answering(200, REPLY);
    let extractor = HostedExtractor {
        model: shared(transport.clone()),
    };

    // Act
    let text = extractor.extract_json(&extract_request()).await.unwrap();

    // Assert
    assert_eq!(text, "a summary");
    assert!(transport.seen()[0].body.to_string().contains("json_schema"));
    assert!(extractor.health().await);
}

#[test]
fn every_failure_reads_to_the_extraction_loop_as_the_error_it_already_handles() {
    let pin = |error| ModelCallError::Pin(error);
    for (error, expected) in [
        (ModelCallError::Status(429), ExtractError::Status(429)),
        (ModelCallError::Malformed("x"), ExtractError::Malformed("x")),
        (
            pin(PinError::Retired {
                provider: "openai".to_owned(),
                model_id: PIN.to_owned(),
            }),
            ExtractError::ModelRetired,
        ),
        (
            pin(PinError::Mismatch {
                pinned: PIN.to_owned(),
                reported: "other".to_owned(),
            }),
            ExtractError::ModelMismatch,
        ),
        (
            ModelCallError::Hosted(HostedError::Timeout { secs: 5 }),
            ExtractError::Timeout,
        ),
        (
            ModelCallError::Hosted(HostedError::Redirect { status: 302 }),
            ExtractError::Redirect(302),
        ),
        (
            ModelCallError::Hosted(HostedError::ResponseTooLarge { limit: 9 }),
            ExtractError::TooLarge { limit: 9 },
        ),
        (
            ModelCallError::Hosted(HostedError::NotHttps),
            ExtractError::InvalidRequest("the request is not allowed by [hosted]"),
        ),
        (
            ModelCallError::Hosted(HostedError::Transport {
                reason: "reset".to_owned(),
            }),
            ExtractError::Io(io::ErrorKind::Other),
        ),
    ] {
        // Act and Assert
        assert_eq!(extract_error(error), expected);
    }
}

#[tokio::test]
async fn the_provider_opens_the_model_only_when_asked_and_passes_its_failure_on() {
    // Arrange
    let opened = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = opened.clone();
    let transport = FakeTransport::answering(200, REPLY);
    let provider = HostedExtractorProvider::new(move || {
        counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(shared(transport.clone()))
    });
    let failing: HostedExtractorProvider<FakeTransport> =
        HostedExtractorProvider::new(|| anyhow::bail!("no key"));

    // Act
    let before = opened.load(std::sync::atomic::Ordering::SeqCst);
    let session = provider.open(&ExtractionConfig::default()).await.unwrap();
    let error = failing
        .open(&ExtractionConfig::default())
        .await
        .err()
        .unwrap();

    // Assert
    assert_eq!(before, 0);
    assert_eq!(opened.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!((session.alive)());
    assert_eq!(error.to_string(), "no key");
}
