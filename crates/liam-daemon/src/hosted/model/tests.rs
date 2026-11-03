// SPDX-License-Identifier: Apache-2.0
//! Each provider over a transport double: the request it sends, the reply it
//! reads, the pin it enforces and the cost it counts.

use super::*;
use crate::config::Provider;
use crate::hosted::dialect::ChatCall;
use crate::hosted::test_support::{model, FakeTransport, KEY};
use crate::hosted::{anthropic, gemini, openai};
use crate::metrics::sink::{MetricId, Reading};

const ANTHROPIC_PIN: &str = "claude-sonnet-4-5-20250929";
const OPENAI_PIN: &str = "gpt-4o-2024-08-06";
const GEMINI_PIN: &str = "gemini-2.0-flash-001";
const REQUEST_ID: &str = "0123456789abcdef0123456789abcdef";
const ROOMY: usize = 64 * 1024;

fn call<'a>(schema: Option<&'a serde_json::Value>) -> ChatCall<'a> {
    ChatCall {
        system: "be brief",
        prompt: "say hello",
        max_tokens: 64,
        temperature: 0.0,
        seed: 7,
        schema,
        request_id: REQUEST_ID,
    }
}

/// The provider, its dialect, pin, host, key header, a valid body for the pin,
/// and the same body naming another model.
struct Case {
    provider: Provider,
    dialect: fn() -> Box<dyn Dialect>,
    pin: &'static str,
    host: &'static str,
    key_header: &'static str,
    key_value: String,
    url: String,
    ok: String,
    other_model: String,
}

fn anthropic_body(model: &str) -> String {
    format!(
        r#"{{"model":"{model}","content":[{{"type":"text","text":"hello"}}],
        "usage":{{"input_tokens":100,"output_tokens":20,"cache_read_input_tokens":30}}}}"#
    )
}

fn openai_body(model: &str) -> String {
    format!(
        r#"{{"model":"{model}","choices":[{{"message":{{"content":"hello"}}}}],
        "usage":{{"prompt_tokens":130,"completion_tokens":20,
        "prompt_tokens_details":{{"cached_tokens":30}}}}}}"#
    )
}

fn gemini_body(model: &str) -> String {
    format!(
        r#"{{"modelVersion":"{model}","candidates":[{{"content":{{"parts":[{{"text":"hel"}},{{"text":"lo"}}]}}}}],
        "usageMetadata":{{"promptTokenCount":130,"candidatesTokenCount":20,"cachedContentTokenCount":30}}}}"#
    )
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            provider: Provider::Anthropic,
            dialect: || Box::new(anthropic::Anthropic),
            pin: ANTHROPIC_PIN,
            host: anthropic::HOST,
            key_header: "x-api-key",
            key_value: KEY.to_owned(),
            url: "https://api.anthropic.com/v1/messages".to_owned(),
            ok: anthropic_body(ANTHROPIC_PIN),
            other_model: anthropic_body("claude-sonnet-4-5-20251001"),
        },
        Case {
            provider: Provider::OpenAi,
            dialect: || Box::new(openai::OpenAi),
            pin: OPENAI_PIN,
            host: openai::HOST,
            key_header: "authorization",
            key_value: format!("Bearer {KEY}"),
            url: "https://api.openai.com/v1/chat/completions".to_owned(),
            ok: openai_body(OPENAI_PIN),
            other_model: openai_body("gpt-4o-2024-11-20"),
        },
        Case {
            provider: Provider::Gemini,
            dialect: || Box::new(gemini::Gemini),
            pin: GEMINI_PIN,
            host: gemini::HOST,
            key_header: "x-goog-api-key",
            key_value: KEY.to_owned(),
            url: format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_PIN}:generateContent"
            ),
            ok: gemini_body(GEMINI_PIN),
            other_model: gemini_body("gemini-2.0-flash-002"),
        },
    ]
}

fn open(
    case: &Case,
    transport: FakeTransport,
    max: usize,
) -> (HostedModel<FakeTransport>, Readings) {
    let (model, sink) = model(
        case.provider,
        (case.dialect)(),
        case.pin,
        case.host,
        transport,
        max,
    );
    (model, Readings(sink))
}

struct Readings(std::sync::Arc<crate::metrics::sink::MemorySink>);

impl Readings {
    fn all(&self) -> Vec<Reading> {
        use crate::metrics::sink::MetricsSink;
        self.0.snapshot().readings
    }

    fn value(&self, metric: MetricId) -> Option<f64> {
        self.all()
            .iter()
            .find(|r| r.metric == metric)
            .map(|r| r.value)
    }
}

#[tokio::test]
async fn each_provider_sends_one_post_with_the_pin_the_key_and_the_text_and_reads_the_reply() {
    for case in cases() {
        // Arrange
        let transport = FakeTransport::answering(200, &case.ok);
        let (model, _) = open(&case, transport.clone(), ROOMY);

        // Act
        let reply = model.complete(&call(None)).await.unwrap();

        // Assert
        let seen = transport.seen();
        assert_eq!(seen.len(), 1, "{:?}", case.provider);
        assert_eq!(seen[0].url, case.url);
        let header = seen[0]
            .headers
            .iter()
            .find(|(name, _)| name == case.key_header)
            .map(|(_, value)| value.as_str());
        assert_eq!(header, Some(case.key_value.as_str()), "{:?}", case.provider);
        let body = seen[0].body.to_string();
        assert!(body.contains("say hello") && body.contains("be brief"));
        assert_eq!(body.contains(case.pin), case.provider != Provider::Gemini);
        assert_eq!(reply.text, "hello");
        assert_eq!(reply.model, case.pin);
        assert_eq!(
            (
                reply.usage.input_tokens,
                reply.usage.output_tokens,
                reply.usage.cached_input_tokens
            ),
            (130, 20, 30)
        );
    }
}

#[tokio::test]
async fn the_opaque_request_id_travels_where_the_provider_has_a_place_for_it() {
    for case in cases() {
        // Arrange
        let transport = FakeTransport::answering(200, &case.ok);
        let (model, _) = open(&case, transport.clone(), ROOMY);

        // Act
        model.complete(&call(None)).await.unwrap();

        // Assert
        let seen = &transport.seen()[0];
        let in_body = seen.body.to_string().contains(REQUEST_ID);
        let in_header = seen.headers.iter().any(|(_, value)| value == REQUEST_ID);
        assert_eq!(in_body, case.provider == Provider::Anthropic);
        assert_eq!(in_header, case.provider == Provider::OpenAi);
    }
}

#[tokio::test]
async fn a_schema_is_sent_to_the_providers_that_take_one() {
    // Arrange
    let schema = serde_json::json!({"type": "object"});
    let mut sent = Vec::new();
    for case in cases() {
        let transport = FakeTransport::answering(200, &case.ok);
        let (model, _) = open(&case, transport.clone(), ROOMY);

        // Act
        model.complete(&call(Some(&schema))).await.unwrap();
        sent.push((case.provider, transport.seen()[0].body.to_string()));
    }

    // Assert
    let body = |provider| sent.iter().find(|(p, _)| *p == provider).unwrap().1.clone();
    assert!(body(Provider::OpenAi).contains("json_schema"));
    assert!(body(Provider::Gemini).contains("application/json"));
    assert!(!body(Provider::Anthropic).contains("json_schema"));
}

#[tokio::test]
async fn a_reply_from_another_model_is_a_mismatch_but_is_still_billed() {
    for case in cases() {
        // Arrange
        let (model, readings) = open(
            &case,
            FakeTransport::answering(200, &case.other_model),
            ROOMY,
        );

        // Act
        let error = model.complete(&call(None)).await.unwrap_err();

        // Assert
        assert!(
            matches!(error, ModelCallError::Pin(PinError::Mismatch { .. })),
            "{error}"
        );
        assert_eq!(readings.value(MetricId::ModelCalls), Some(1.0));
    }
}

#[tokio::test]
async fn failures_are_typed_and_nothing_is_billed() {
    for case in cases() {
        for (status, body, expected) in [
            (404, "{}", "retired"),
            (400, "{}", "status"),
            (500, "{}", "status"),
            (200, "not json", "malformed"),
            (200, "{}", "malformed"),
        ] {
            // Arrange
            let (model, readings) = open(&case, FakeTransport::answering(status, body), ROOMY);

            // Act
            let error = model.complete(&call(None)).await.unwrap_err();

            // Assert
            let kind = match &error {
                ModelCallError::Pin(PinError::Retired { .. }) => "retired",
                ModelCallError::Status(_) => "status",
                ModelCallError::Malformed(_) => "malformed",
                other => panic!("unexpected {other}"),
            };
            assert_eq!(kind, expected, "{:?} {status}", case.provider);
            assert!(readings.all().is_empty());
        }
    }
}

#[tokio::test]
async fn an_oversize_body_is_refused_before_it_is_read() {
    // Arrange
    let case = &cases()[0];
    let (model, readings) = open(case, FakeTransport::answering(200, &case.ok), 16);

    // Act
    let error = model.complete(&call(None)).await.unwrap_err();

    // Assert
    assert_eq!(
        error,
        ModelCallError::Hosted(HostedError::ResponseTooLarge { limit: 16 })
    );
    assert!(readings.all().is_empty());
}

#[tokio::test]
async fn the_cost_is_counted_from_the_reported_usage_under_the_pinned_model() {
    // Arrange
    let case = &cases()[1];
    let (model, readings) = open(case, FakeTransport::answering(200, &case.ok), ROOMY);

    // Act
    model.complete(&call(None)).await.unwrap();

    // Assert: 100 fresh input at 1.0, 30 cached at 0.1 and 20 output at 5.0
    assert_eq!(readings.value(MetricId::ModelCalls), Some(1.0));
    assert_eq!(readings.value(MetricId::ModelCostMicroDollars), Some(203.0));
}

#[tokio::test]
async fn the_key_is_in_no_error_and_no_debug_text() {
    // Arrange
    let case = &cases()[0];
    let (model, _) = open(case, FakeTransport::answering(500, "{}"), ROOMY);

    // Act
    let error = model.complete(&call(None)).await.unwrap_err();

    // Assert
    assert!(!format!("{error} {error:?}").contains(KEY));
}

#[test]
fn the_request_id_is_a_fixed_length_digest_that_does_not_show_the_text() {
    // Act
    let id = opaque_request_id("system text", "private prompt text");

    // Assert
    assert_eq!(id.len(), REQUEST_ID_CHARS);
    assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    assert_eq!(id, opaque_request_id("system text", "private prompt text"));
    assert_ne!(id, opaque_request_id("system text", "other"));
}
