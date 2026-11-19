// SPDX-License-Identifier: Apache-2.0
//! The retrieval deadline and cancellation of `recall`, on a paused
//! clock. Every sleeping double sleeps with `tokio::time::sleep`, so the
//! clock can advance past it; a `std::thread::sleep` would stall the test.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use liam_model::{Embedder, MockEmbedder, TextKind};
use liam_store::{GraphConfig, NewNode};
use tokio_util::sync::CancellationToken;

use super::reply::ToolReply;
use super::test_doubles::{ErroringEmbedder, GatedEmbedder};
use super::*;

const DIMS: usize = 8;
const QUERY: &str = "quarterly forecasts for the finance team";
const SLOW: Duration = Duration::from_secs(30);
const FACT_COUNT: usize = 5;
const K: usize = 2;
const SYNTHESIS_TIMEOUT_SECS: u64 = 30;
const NO_REASONS: &[&str] = &[];
const CANCELLED_TEXT: &str = "recall cancelled";
const TIMEOUT_TEXT: &str = "recall failed: retrieval took longer than 1 s";
const NO_MEMORY_TEXT: &str = "no relevant memory";
const HIT_PREFIX: &str = "[fact ";

/// Embeds like `MockEmbedder` after sleeping `delay`, counting its calls.
struct SleepingEmbedder {
    delay: Duration,
    calls: AtomicUsize,
    inner: MockEmbedder,
}

impl SleepingEmbedder {
    fn new(delay: Duration) -> Arc<Self> {
        Arc::new(Self {
            delay,
            calls: AtomicUsize::new(0),
            inner: MockEmbedder::new(DIMS),
        })
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl Embedder for SleepingEmbedder {
    fn dims(&self) -> usize {
        DIMS
    }
    async fn embed(&self, text: &str, kind: TextKind) -> liam_model::Result<Vec<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        self.inner.embed(text, kind).await
    }
}

/// Fresh in-memory server over the given doubles.
async fn server_with_embedder(embedder: Arc<dyn Embedder>) -> MemoryServer {
    let store = DefaultGraph::open(":memory:", GraphConfig::new(DIMS))
        .await
        .expect("open in-memory store");
    MemoryServer::new(Arc::new(store), embedder, SYNTHESIS_TIMEOUT_SECS)
}

async fn seed_facts(server: &MemoryServer) {
    let store = server.store().await.unwrap();
    for index in 0..FACT_COUNT {
        store
            .insert(NewNode::now(
                "fact",
                format!("forecast {index}"),
                format!("{QUERY} note {index}"),
            ))
            .await
            .unwrap();
    }
}

fn recall_args() -> RecallArgs {
    RecallArgs {
        query: QUERY.to_string(),
        kind: None,
        scope: None,
        k: Some(K),
        as_of: None,
        mode: None,
    }
}

fn cancelled() -> CancellationToken {
    let token = CancellationToken::new();
    token.cancel();
    token
}

/// Converts to the wire result, asserts the structured reasons and the error
/// flag, and returns the text.
fn wire_text(reply: ToolReply, degraded: &[&str], is_error: bool) -> String {
    let result = reply.into_call_result();
    assert_eq!(
        result.structured_content,
        Some(serde_json::json!({ "degraded": degraded }))
    );
    assert_eq!(result.is_error == Some(true), is_error);
    result.content[0]
        .as_text()
        .expect("content item is text")
        .text
        .clone()
}

fn marker_for(reason: &str) -> String {
    let hint = match reason {
        "embed_failed" => "searched by keyword only",
        other => panic!("no hint known for {other}"),
    };
    format!("(degraded: {reason}, {hint})")
}

fn block_count(out: &str) -> usize {
    out.matches("[fact ").count()
}

#[tokio::test(start_paused = true)]
async fn recall_replies_with_the_timeout_when_the_embedder_outlasts_the_deadline() {
    // Arrange
    let embedder = SleepingEmbedder::new(SLOW);
    let server = server_with_embedder(embedder).await.with_recall_timeout(1);

    // Act
    let out = server
        .recall_with(recall_args(), CancellationToken::new())
        .await;

    // Assert
    assert_eq!(wire_text(out, NO_REASONS, true), TIMEOUT_TEXT);
}

#[tokio::test(start_paused = true)]
async fn a_zero_recall_timeout_gives_a_one_second_deadline() {
    // Arrange
    let embedder = SleepingEmbedder::new(SLOW);
    let server = server_with_embedder(embedder).await.with_recall_timeout(0);

    // Act
    let out = server
        .recall_with(recall_args(), CancellationToken::new())
        .await;

    // Assert
    assert_eq!(wire_text(out, NO_REASONS, true), TIMEOUT_TEXT);
}

#[tokio::test(start_paused = true)]
async fn recall_replies_cancelled_without_embedding_when_the_token_is_already_cancelled() {
    // Arrange
    let embedder = SleepingEmbedder::new(Duration::ZERO);
    let server = server_with_embedder(embedder.clone())
        .await
        .with_recall_timeout(1);

    // Act
    let out = server.recall_with(recall_args(), cancelled()).await;

    // Assert
    assert_eq!(wire_text(out, NO_REASONS, false), CANCELLED_TEXT);
    assert_eq!(embedder.call_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn recall_returns_lexical_hits_with_a_marker_when_the_embedder_errors() {
    // Arrange
    let server = server_with_embedder(Arc::new(ErroringEmbedder::new(DIMS))).await;
    seed_facts(&server).await;

    // Act
    let out = server
        .recall_with(recall_args(), CancellationToken::new())
        .await;

    // Assert
    let text = wire_text(out, &["embed_failed"], false);
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some(marker_for("embed_failed").as_str()));
    assert!(lines.next().unwrap().starts_with(HIT_PREFIX), "{text}");
    assert_eq!(block_count(&text), K, "{text}");
}

#[tokio::test(start_paused = true)]
async fn recall_names_keyword_only_search_when_the_embedder_errors_and_nothing_matches() {
    // Arrange
    let server = server_with_embedder(Arc::new(ErroringEmbedder::new(DIMS))).await;

    // Act
    let out = server
        .recall_with(recall_args(), CancellationToken::new())
        .await;

    // Assert
    let text = wire_text(out, &["embed_failed"], false);
    assert_eq!(
        text,
        format!("(degraded: embed_failed, searched by keyword only)\n{NO_MEMORY_TEXT}")
    );
}

#[tokio::test(start_paused = true)]
async fn recall_does_not_call_an_erroring_embedder_when_already_cancelled() {
    // Arrange
    let embedder = Arc::new(ErroringEmbedder::new(DIMS));
    let server = server_with_embedder(embedder.clone()).await;

    // Act
    let out = server.recall_with(recall_args(), cancelled()).await;

    // Assert
    assert_eq!(wire_text(out, NO_REASONS, false), CANCELLED_TEXT);
    assert_eq!(embedder.call_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn recall_refuses_infer_mode_as_an_error() {
    // Arrange
    let server = server_with_embedder(Arc::new(MockEmbedder::new(DIMS))).await;
    let args = RecallArgs {
        mode: Some(Mode::Infer),
        ..recall_args()
    };

    // Act
    let out = server.recall_with(args, CancellationToken::new()).await;

    // Assert
    assert!(out.is_error());
    assert!(out.degraded().is_empty());
}

#[tokio::test(start_paused = true)]
async fn a_healthy_recall_has_no_degraded_reasons_and_opens_with_a_hit() {
    // Arrange
    let server = server_with_embedder(Arc::new(MockEmbedder::new(DIMS))).await;
    seed_facts(&server).await;

    // Act
    let out = server
        .recall_with(recall_args(), CancellationToken::new())
        .await;

    // Assert
    let text = wire_text(out, NO_REASONS, false);
    assert!(text.starts_with(HIT_PREFIX), "{text}");
}

#[tokio::test]
async fn the_recall_tool_declares_an_output_schema_requiring_degraded() {
    // Arrange
    let server = server_with_embedder(Arc::new(MockEmbedder::new(DIMS))).await;
    let expected = serde_json::Value::Object((*super::reply::output_schema()).clone());

    // Act
    let tools = server.tool_router.list_all();
    let recall = tools.iter().find(|t| t.name == "recall").unwrap();

    // Assert
    let schema = recall.output_schema.as_ref().expect("recall output schema");
    assert_eq!(serde_json::Value::Object((**schema).clone()), expected);
    assert_eq!(expected["required"], serde_json::json!(["degraded"]));
    assert_eq!(
        expected["properties"]["degraded"]["type"],
        serde_json::json!("array")
    );
}

#[tokio::test(start_paused = true)]
async fn recall_fails_without_degraded_reasons_when_the_store_is_unavailable() {
    // Arrange
    let server = server_with_embedder(Arc::new(MockEmbedder::new(DIMS))).await;
    *server.shared_store().exclusive().await = None;

    // Act
    let out = server
        .recall_with(recall_args(), CancellationToken::new())
        .await;

    // Assert
    let text = wire_text(out, NO_REASONS, true);
    assert!(text.starts_with("recall failed:"), "{text}");
}

#[tokio::test(start_paused = true)]
async fn recall_fails_without_degraded_reasons_when_the_query_is_too_long() {
    // Arrange
    let server = server_with_embedder(Arc::new(MockEmbedder::new(DIMS))).await;
    let args = RecallArgs {
        query: "a".repeat(MAX_QUERY_CHARS + 1),
        ..recall_args()
    };

    // Act
    let out = server.recall_with(args, CancellationToken::new()).await;

    // Assert
    let text = wire_text(out, NO_REASONS, true);
    assert!(text.starts_with("recall failed:"), "{text}");
}

#[tokio::test(start_paused = true)]
async fn a_recall_cancelled_while_waiting_on_the_embedder_never_takes_the_gate() {
    // Arrange
    let gate = Arc::new(tokio::sync::Mutex::new(()));
    let embedder = Arc::new(GatedEmbedder::new(DIMS, gate.clone()));
    let server = server_with_embedder(embedder.clone()).await;
    let held = gate.lock().await;
    let token = CancellationToken::new();
    let mut first = std::pin::pin!(server.recall_with(recall_args(), token.clone()));
    tokio::select! {
        biased;
        _ = &mut first => panic!("the recall finished while the gate was held"),
        _ = std::future::ready(()) => {}
    }

    // Act
    token.cancel();
    let cancelled_reply = first.await;
    drop(held);
    let second = server
        .recall_with(recall_args(), CancellationToken::new())
        .await;

    // Assert
    assert_eq!(
        wire_text(cancelled_reply, NO_REASONS, false),
        CANCELLED_TEXT
    );
    assert_eq!(embedder.call_count(), 1, "only the second recall embeds");
    wire_text(second, NO_REASONS, false);
}
