// SPDX-License-Identifier: Apache-2.0
//! Each provider's batch calls over a transport double: the requests it sends,
//! the results it reads, and the copies it leaves.

use std::sync::Arc;

use super::*;
use crate::config::{HostedConfig, PinError, PricesConfig};
use crate::hosted::client::HostedError;
use crate::hosted::test_support::{metrics, price, FakeTransport, KEY, PRICE_VERSION};
use crate::hosted::transport::Method;
use crate::hosted::{anthropic, gemini, openai};
use crate::metrics::sink::{MemorySink, MetricId, MetricsSink};

const ANTHROPIC_PIN: &str = "claude-sonnet-4-5-20250929";
const OPENAI_PIN: &str = "gpt-4o-2024-08-06";
const GEMINI_PIN: &str = "gemini-2.0-flash-001";
const ROOMY: usize = 64 * 1024;
const KEY_ONE: &str = "0123456789abcdef0123456789abcdef";
const KEY_TWO: &str = "fedcba9876543210fedcba9876543210";
/// 100 fresh and 30 cached input tokens and 20 output tokens at the batch
/// rates in the test price table, in millionths of a dollar.
const BATCH_COST: f64 = 103.0;

fn wire_parts(
    host: &str,
    pin: &str,
    transport: FakeTransport,
    max: usize,
) -> (
    HostedClient<FakeTransport>,
    ModelPin,
    SecretString,
    CostRecorder,
    Arc<MemorySink>,
) {
    let config = HostedConfig {
        enabled: true,
        allowed_hosts: vec![host.to_owned()],
        max_response_bytes: max,
        ..HostedConfig::default()
    };
    let (metrics, sink) = metrics();
    let prices = PricesConfig {
        version: PRICE_VERSION.to_owned(),
        models: [(pin.to_owned(), price())].into(),
    };
    let provider = match host {
        anthropic::HOST => "anthropic",
        openai::HOST => "openai",
        _ => "gemini",
    };
    (
        HostedClient::new(transport, &config),
        ModelPin::new(provider, pin).unwrap(),
        SecretString::new(KEY),
        CostRecorder::new(metrics, prices),
        sink,
    )
}

fn anthropic_batch(
    transport: &FakeTransport,
    max: usize,
) -> (AnthropicBatch<FakeTransport>, Arc<MemorySink>) {
    let (client, pin, key, costs, sink) =
        wire_parts(anthropic::HOST, ANTHROPIC_PIN, transport.clone(), max);
    (AnthropicBatch::new(client, pin, key, costs), sink)
}

fn openai_batch(
    transport: &FakeTransport,
    max: usize,
) -> (OpenAiBatch<FakeTransport>, Arc<MemorySink>) {
    let (client, pin, key, costs, sink) =
        wire_parts(openai::HOST, OPENAI_PIN, transport.clone(), max);
    (OpenAiBatch::new(client, pin, key, costs), sink)
}

fn gemini_batch(
    transport: &FakeTransport,
    max: usize,
) -> (GeminiBatch<FakeTransport>, Arc<MemorySink>) {
    let (client, pin, key, costs, sink) =
        wire_parts(gemini::HOST, GEMINI_PIN, transport.clone(), max);
    (GeminiBatch::new(client, pin, key, costs), sink)
}

fn request(cache_key: &str) -> BatchRequest {
    BatchRequest {
        cache_key: cache_key.to_owned(),
        system: "be brief".to_owned(),
        prompt: format!("say hello to {cache_key}"),
        max_tokens: 64,
        temperature: 0.0,
        seed: 7,
        schema: None,
    }
}

fn cost_of(sink: &MemorySink) -> Option<f64> {
    sink.snapshot()
        .readings
        .iter()
        .find(|reading| reading.metric == MetricId::ModelCostMicroDollars)
        .map(|reading| reading.value)
}

/// The text as one line of JSON, which is how a results file holds it.
fn compact(text: String) -> String {
    serde_json::from_str::<serde_json::Value>(&text)
        .unwrap()
        .to_string()
}

fn anthropic_message(model: &str) -> String {
    compact(format!(
        r#"{{"model":"{model}","content":[{{"type":"text","text":"hello"}}],
        "usage":{{"input_tokens":100,"output_tokens":20,"cache_read_input_tokens":30}}}}"#
    ))
}

fn openai_completion(model: &str) -> String {
    compact(format!(
        r#"{{"model":"{model}","choices":[{{"message":{{"content":"hello"}}}}],
        "usage":{{"prompt_tokens":130,"completion_tokens":20,
        "prompt_tokens_details":{{"cached_tokens":30}}}}}}"#
    ))
}

fn gemini_response(model: &str) -> String {
    compact(format!(
        r#"{{"modelVersion":"{model}","candidates":[{{"content":{{"parts":[{{"text":"hello"}}]}}}}],
        "usageMetadata":{{"promptTokenCount":130,"candidatesTokenCount":20,"cachedContentTokenCount":30}}}}"#
    ))
}

fn is_answer(result: &BatchResult, cache_key: &str) -> bool {
    result.cache_key == cache_key
        && matches!(&result.outcome, BatchOutcome::Answer(reply) if reply.text == "hello")
}

#[tokio::test]
async fn anthropic_sends_one_batch_with_the_cache_key_as_custom_id_and_returns_the_batch_id() {
    // Arrange
    let transport = FakeTransport::answering(
        200,
        r#"{"id":"msgbatch_01","processing_status":"in_progress"}"#,
    );
    let (batch, _) = anthropic_batch(&transport, ROOMY);

    // Act
    let id = batch
        .submit(&[request(KEY_ONE), request(KEY_TWO)])
        .await
        .unwrap();

    // Assert
    let seen = transport.seen();
    assert_eq!(id, "msgbatch_01");
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].method, Method::Post);
    assert_eq!(seen[0].url, "https://api.anthropic.com/v1/messages/batches");
    assert!(seen[0]
        .headers
        .contains(&("x-api-key".to_owned(), KEY.to_owned())));
    let requests = seen[0].body["requests"].as_array().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0]["custom_id"], KEY_ONE);
    assert_eq!(requests[0]["params"]["model"], ANTHROPIC_PIN);
    assert_eq!(requests[0]["params"]["metadata"]["user_id"], KEY_ONE);
    assert_eq!(requests[1]["custom_id"], KEY_TWO);
}

#[tokio::test]
async fn anthropic_reports_each_status_and_reads_each_kind_of_result() {
    // Arrange
    let transport = FakeTransport::default();
    transport.push(200, r#"{"processing_status":"in_progress"}"#);
    transport.push(200, r#"{"processing_status":"canceling"}"#);
    transport.push(200, r#"{"processing_status":"ended"}"#);
    let succeeded = |key: &str, model: &str| {
        format!(
            r#"{{"custom_id":"{key}","result":{{"type":"succeeded","message":{}}}}}"#,
            anthropic_message(model)
        )
    };
    let results = [
        succeeded(KEY_ONE, ANTHROPIC_PIN),
        r#"{"custom_id":"b","result":{"type":"errored"}}"#.to_owned(),
        r#"{"custom_id":"c","result":{"type":"canceled"}}"#.to_owned(),
        r#"{"custom_id":"d","result":{"type":"expired"}}"#.to_owned(),
        succeeded("e", "claude-sonnet-4-5-20251001"),
    ]
    .join("\n");
    transport.push(200, &results);
    let (batch, sink) = anthropic_batch(&transport, ROOMY);

    // Act
    let polled = [
        batch.poll("msgbatch_01").await.unwrap(),
        batch.poll("msgbatch_01").await.unwrap(),
        batch.poll("msgbatch_01").await.unwrap(),
    ];
    let collected = batch.collect("msgbatch_01").await.unwrap();

    // Assert
    assert_eq!(
        polled,
        [
            BatchStatus::InProgress,
            BatchStatus::InProgress,
            BatchStatus::Ended
        ]
    );
    assert!(is_answer(&collected[0], KEY_ONE));
    assert!(matches!(
        &collected[1].outcome,
        BatchOutcome::Failed(ModelCallError::Malformed(_))
    ));
    assert_eq!(collected[2].outcome, BatchOutcome::Cancelled);
    assert_eq!(collected[3].outcome, BatchOutcome::Expired);
    assert!(matches!(
        &collected[4].outcome,
        BatchOutcome::Failed(ModelCallError::Pin(PinError::Mismatch { .. }))
    ));
    assert_eq!(
        cost_of(&sink),
        Some(BATCH_COST * 2.0),
        "both answers were billed at the batch rate"
    );
    let seen = transport.seen();
    assert_eq!(seen[3].method, Method::Get);
    assert_eq!(
        seen[3].url,
        "https://api.anthropic.com/v1/messages/batches/msgbatch_01/results"
    );
}

#[tokio::test]
async fn anthropic_cancel_stops_then_deletes_and_a_refused_delete_leaves_the_copy() {
    // Arrange
    let deleted = FakeTransport::default();
    deleted.push(200, "{}");
    deleted.push(200, "{}");
    let refused = FakeTransport::default();
    refused.push(200, "{}");
    refused.push(409, "{}");
    let gone = FakeTransport::default();
    gone.push(404, "{}");
    gone.push(404, "{}");

    // Act
    let after_delete = anthropic_batch(&deleted, ROOMY)
        .0
        .cancel("msgbatch_01")
        .await
        .unwrap();
    let after_refusal = anthropic_batch(&refused, ROOMY)
        .0
        .cancel("msgbatch_01")
        .await
        .unwrap();
    let unknown = anthropic_batch(&gone, ROOMY)
        .0
        .cancel("msgbatch_01")
        .await
        .unwrap();

    // Assert
    let seen = deleted.seen();
    assert_eq!(after_delete, ProviderCopies::Deleted);
    assert_eq!(after_refusal, ProviderCopies::Kept);
    assert_eq!(unknown, ProviderCopies::Deleted);
    assert_eq!(seen[0].method, Method::Post);
    assert!(seen[0].url.ends_with("/msgbatch_01/cancel"));
    assert_eq!(seen[1].method, Method::Delete);
    assert!(seen[1].url.ends_with("/msgbatch_01"));
}

#[tokio::test]
async fn openai_uploads_a_jsonl_file_then_creates_the_batch() {
    // Arrange
    let transport = FakeTransport::default();
    transport.push(200, r#"{"id":"file-in"}"#);
    transport.push(200, r#"{"id":"batch_01","status":"validating"}"#);
    let (batch, _) = openai_batch(&transport, ROOMY);

    // Act
    let id = batch
        .submit(&[request(KEY_ONE), request(KEY_TWO)])
        .await
        .unwrap();

    // Assert
    let seen = transport.seen();
    assert_eq!(id, "batch_01");
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].url, "https://api.openai.com/v1/files");
    let content_type = seen[0].content_type.clone().unwrap();
    assert!(
        content_type.starts_with("multipart/form-data; boundary="),
        "{content_type}"
    );
    let form = String::from_utf8(seen[0].raw_body.clone()).unwrap();
    assert!(form.contains("name=\"purpose\"\r\n\r\nbatch"), "{form}");
    let lines: Vec<serde_json::Value> = form
        .lines()
        .filter(|line| line.starts_with("{\"body\""))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 2, "{form}");
    assert_eq!(lines[0]["custom_id"], KEY_ONE);
    assert_eq!(lines[0]["url"], "/v1/chat/completions");
    assert_eq!(lines[0]["body"]["model"], OPENAI_PIN);
    assert_eq!(seen[1].url, "https://api.openai.com/v1/batches");
    assert_eq!(seen[1].body["input_file_id"], "file-in");
    assert_eq!(seen[1].body["completion_window"], "24h");
    assert!(seen[1]
        .headers
        .contains(&("authorization".to_owned(), format!("Bearer {KEY}"))));
}

#[tokio::test]
async fn openai_deletes_the_upload_when_the_batch_cannot_be_created() {
    // Arrange
    let transport = FakeTransport::default();
    transport.push(200, r#"{"id":"file-in"}"#);
    transport.push(400, "{}");
    transport.push(200, "{}");
    let (batch, _) = openai_batch(&transport, ROOMY);

    // Act
    let error = batch.submit(&[request(KEY_ONE)]).await.unwrap_err();

    // Assert
    let seen = transport.seen();
    assert_eq!(error, ModelCallError::Status(400));
    assert_eq!(seen.len(), 3);
    assert_eq!(seen[2].method, Method::Delete);
    assert_eq!(seen[2].url, "https://api.openai.com/v1/files/file-in");
}

#[tokio::test]
async fn openai_maps_each_status_and_reads_the_output_and_error_files() {
    // Arrange
    let transport = FakeTransport::default();
    for status in [
        "validating",
        "finalizing",
        "completed",
        "failed",
        "expired",
        "cancelled",
    ] {
        transport.push(200, &format!(r#"{{"status":"{status}"}}"#));
    }
    transport.push(
        200,
        r#"{"status":"completed","output_file_id":"file-out","error_file_id":"file-err"}"#,
    );
    transport.push(
        200,
        &format!(
            r#"{{"custom_id":"{KEY_ONE}","response":{{"status_code":200,"body":{}}}}}"#,
            openai_completion(OPENAI_PIN)
        ),
    );
    transport.push(
        200,
        r#"{"custom_id":"b","response":{"status_code":400,"body":{}}}"#,
    );
    let (batch, sink) = openai_batch(&transport, ROOMY);

    // Act
    let mut polled = Vec::new();
    for _ in 0..6 {
        polled.push(batch.poll("batch_01").await.unwrap());
    }
    let collected = batch.collect("batch_01").await.unwrap();

    // Assert
    assert_eq!(
        polled,
        [
            BatchStatus::InProgress,
            BatchStatus::InProgress,
            BatchStatus::Ended,
            BatchStatus::Failed,
            BatchStatus::Expired,
            BatchStatus::Cancelled
        ]
    );
    assert_eq!(collected.len(), 2);
    assert!(is_answer(&collected[0], KEY_ONE));
    assert_eq!(
        collected[1].outcome,
        BatchOutcome::Failed(ModelCallError::Status(400))
    );
    assert_eq!(cost_of(&sink), Some(BATCH_COST));
    let urls: Vec<String> = transport
        .seen()
        .iter()
        .skip(6)
        .map(|s| s.url.clone())
        .collect();
    assert_eq!(
        urls,
        [
            "https://api.openai.com/v1/batches/batch_01",
            "https://api.openai.com/v1/files/file-out/content",
            "https://api.openai.com/v1/files/file-err/content",
        ]
    );
}

#[tokio::test]
async fn openai_cancel_deletes_the_files_of_a_finished_batch_and_keeps_the_copy_of_a_running_one() {
    // Arrange
    let finished = FakeTransport::default();
    finished.push(
        200,
        r#"{"status":"completed","input_file_id":"fi","output_file_id":"fo"}"#,
    );
    finished.push(200, "{}");
    finished.push(200, "{}");
    let running = FakeTransport::default();
    running.push(200, r#"{"status":"in_progress","input_file_id":"fi"}"#);
    running.push(200, "{}");
    running.push(200, "{}");
    let unknown = FakeTransport::default();
    unknown.push(404, "{}");

    // Act
    let after_finished = openai_batch(&finished, ROOMY)
        .0
        .cancel("batch_01")
        .await
        .unwrap();
    let after_running = openai_batch(&running, ROOMY)
        .0
        .cancel("batch_01")
        .await
        .unwrap();
    let after_unknown = openai_batch(&unknown, ROOMY)
        .0
        .cancel("batch_01")
        .await
        .unwrap();

    // Assert
    let methods = |transport: &FakeTransport| -> Vec<Method> {
        transport.seen().iter().map(|seen| seen.method).collect()
    };
    assert_eq!(after_finished, ProviderCopies::Deleted);
    assert_eq!(
        methods(&finished),
        [Method::Get, Method::Delete, Method::Delete]
    );
    assert_eq!(after_running, ProviderCopies::Kept);
    assert_eq!(
        methods(&running),
        [Method::Get, Method::Post, Method::Delete]
    );
    assert!(running.seen()[1].url.ends_with("/batch_01/cancel"));
    assert_eq!(after_unknown, ProviderCopies::Deleted);
    assert_eq!(unknown.seen().len(), 1);
}

#[tokio::test]
async fn gemini_names_the_pin_in_the_url_and_returns_the_batch_name() {
    // Arrange
    let transport = FakeTransport::answering(200, r#"{"name":"batches/abc123"}"#);
    let (batch, _) = gemini_batch(&transport, ROOMY);

    // Act
    let name = batch.submit(&[request(KEY_ONE)]).await.unwrap();

    // Assert
    let seen = transport.seen();
    assert_eq!(name, "batches/abc123");
    assert_eq!(
        seen[0].url,
        format!("https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_PIN}:batchGenerateContent")
    );
    assert!(seen[0]
        .headers
        .contains(&("x-goog-api-key".to_owned(), KEY.to_owned())));
    let entries = &seen[0].body["batch"]["input_config"]["requests"]["requests"];
    assert_eq!(entries[0]["metadata"]["key"], KEY_ONE);
    assert_eq!(entries[0]["request"]["contents"][0]["role"], "user");
}

#[tokio::test]
async fn gemini_maps_each_state_reads_inline_results_and_cancels_then_deletes() {
    // Arrange
    let transport = FakeTransport::default();
    for state in [
        "PENDING",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
        "EXPIRED",
    ] {
        transport.push(
            200,
            &format!(r#"{{"metadata":{{"state":"BATCH_STATE_{state}"}}}}"#),
        );
    }
    transport.push(
        200,
        &format!(
            r#"{{"response":{{"inlinedResponses":{{"inlinedResponses":[
            {{"metadata":{{"key":"{KEY_ONE}"}},"response":{}}},
            {{"metadata":{{"key":"b"}},"error":{{"code":13}}}}]}}}}}}"#,
            gemini_response(GEMINI_PIN)
        ),
    );
    transport.push(200, "{}");
    transport.push(200, "{}");
    let (batch, sink) = gemini_batch(&transport, ROOMY);

    // Act
    let mut polled = Vec::new();
    for _ in 0..6 {
        polled.push(batch.poll("batches/abc123").await.unwrap());
    }
    let collected = batch.collect("batches/abc123").await.unwrap();
    let copies = batch.cancel("batches/abc123").await.unwrap();

    // Assert
    assert_eq!(
        polled,
        [
            BatchStatus::InProgress,
            BatchStatus::InProgress,
            BatchStatus::Ended,
            BatchStatus::Failed,
            BatchStatus::Cancelled,
            BatchStatus::Expired
        ]
    );
    assert!(is_answer(&collected[0], KEY_ONE));
    assert!(matches!(
        &collected[1].outcome,
        BatchOutcome::Failed(ModelCallError::Malformed(_))
    ));
    assert_eq!(cost_of(&sink), Some(BATCH_COST));
    assert_eq!(copies, ProviderCopies::Deleted);
    let seen = transport.seen();
    assert!(seen[7].url.ends_with("/v1beta/batches/abc123:cancel"));
    assert_eq!(seen[8].method, Method::Delete);
}

#[tokio::test]
async fn a_pinned_model_the_provider_no_longer_has_is_a_retired_error_on_submit() {
    for (name, error) in [
        (
            "anthropic",
            anthropic_batch(&FakeTransport::answering(404, "{}"), ROOMY)
                .0
                .submit(&[request(KEY_ONE)])
                .await,
        ),
        (
            "gemini",
            gemini_batch(&FakeTransport::answering(404, "{}"), ROOMY)
                .0
                .submit(&[request(KEY_ONE)])
                .await,
        ),
    ] {
        // Assert
        assert!(
            matches!(error, Err(ModelCallError::Pin(PinError::Retired { .. }))),
            "{name}: {error:?}"
        );
    }
}

#[tokio::test]
async fn a_batch_id_that_is_not_path_safe_is_refused_before_any_request() {
    for id in ["../files", "a/b", "a?x=1", "", "batches/..%2f"] {
        // Arrange
        let transport = FakeTransport::default();
        let (anthropic, _) = anthropic_batch(&transport, ROOMY);
        let (openai, _) = openai_batch(&transport, ROOMY);
        let (gemini, _) = gemini_batch(&transport, ROOMY);

        // Act
        let results = [
            anthropic.poll(id).await.map(|_| ()),
            openai.collect(id).await.map(|_| ()),
            gemini.cancel(id).await.map(|_| ()),
        ];

        // Assert
        for result in results {
            assert!(
                matches!(result, Err(ModelCallError::Malformed(_))),
                "{id:?}: {result:?}"
            );
        }
        assert!(transport.seen().is_empty(), "{id:?}");
    }
}

#[tokio::test]
async fn an_oversize_result_body_and_a_failing_status_are_typed_errors() {
    // Arrange
    let big = FakeTransport::answering(200, &"x".repeat(32));
    let down = FakeTransport::answering(500, "{}");
    let garbled = FakeTransport::answering(200, "not json\n");

    // Act
    let oversize = anthropic_batch(&big, 8).0.collect("msgbatch_01").await;
    let status = anthropic_batch(&down, ROOMY).0.poll("msgbatch_01").await;
    let malformed = anthropic_batch(&garbled, ROOMY)
        .0
        .collect("msgbatch_01")
        .await;

    // Assert
    assert_eq!(
        oversize,
        Err(ModelCallError::Hosted(HostedError::ResponseTooLarge {
            limit: 8
        }))
    );
    assert_eq!(status, Err(ModelCallError::Status(500)));
    assert!(matches!(malformed, Err(ModelCallError::Malformed(_))));
}
