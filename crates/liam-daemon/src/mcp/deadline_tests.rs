// SPDX-License-Identifier: Apache-2.0
//! The retrieval deadline and cancellation of `recall` and `ask`, on a paused
//! clock. Every sleeping double sleeps with `tokio::time::sleep`, so the
//! clock can advance past it; a `std::thread::sleep` would stall the test.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use liam_model::{Embedder, IdentityReranker, Llm, MockEmbedder, MockLlm, Reranker, TextKind};
use liam_store::{EventRef, GraphConfig, Millis, NewNode, NewSummary, PutOutcome, SummaryChild};
use tokio_util::sync::CancellationToken;

use super::reply::ToolReply;
use super::test_doubles::{ErroringEmbedder, FailingReranker, GatedEmbedder};
use super::*;
use crate::summary::generate::SUMMARY_PROMPT_VERSION;

const DIMS: usize = 8;
const QUERY: &str = "quarterly forecasts for the finance team";
const SLOW: Duration = Duration::from_secs(30);
const FACT_COUNT: usize = 5;
const K: usize = 2;
const NO_REASONS: &[&str] = &[];
const CANCELLED_TEXT: &str = "recall cancelled";
const TIMEOUT_TEXT: &str = "recall failed: retrieval took longer than 1 s";
const NO_MEMORY_TEXT: &str = "no relevant memory";
const HIT_PREFIX: &str = "[fact ";
const BOUNDARY_RECALL_TIMEOUT_SECS: u64 = 1;
const BOUNDARY_BUDGET_MS: u64 = 200;
const ASK_CANCELLED_TEXT: &str = "ask cancelled";
const ASK_TIMEOUT_TEXT: &str = "ask failed: retrieval took longer than 1 s";
const ASK_TIMEOUT_SECS: u64 = 1;
const MARKER_PREFIX: &str = "(degraded: ";
const GROUNDED_ANSWER: &str = "The finance team has quarterly forecasts [1].";
const UNGROUNDED_ANSWER: &str =
    "Kubernetes clusters orchestrate containerized microservice deployments.";
const REASONS_EMBED_AND_TIMEOUT: &str = "(degraded: embed_failed, searched by keyword only; \
rerank_timeout, hits are in store order)";

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

/// Scores documents in reverse of their incoming order after sleeping `delay`,
/// so a reply that used it is told apart from one in store order.
struct SleepingReranker {
    delay: Duration,
    calls: AtomicUsize,
}

impl SleepingReranker {
    fn new(delay: Duration) -> Arc<Self> {
        Arc::new(Self {
            delay,
            calls: AtomicUsize::new(0),
        })
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl Reranker for SleepingReranker {
    async fn scores(&self, _query: &str, docs: &[String]) -> liam_model::Result<Vec<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        Ok((0..docs.len()).map(|i| i as f32).collect())
    }
}

/// Keeps the last prompt it was given.
#[derive(Default)]
struct PromptLlm {
    prompt: Mutex<String>,
}

#[async_trait::async_trait]
impl Llm for PromptLlm {
    async fn complete(&self, _system: &str, prompt: &str) -> liam_model::Result<String> {
        *self.prompt.lock().unwrap() = prompt.to_string();
        Ok("The finance team has quarterly forecasts [1].".to_string())
    }
}

/// Answers every call with `reply`, or fails when it is `None`.
struct CannedLlm {
    reply: Option<&'static str>,
}

#[async_trait::async_trait]
impl Llm for CannedLlm {
    async fn complete(&self, _system: &str, _prompt: &str) -> liam_model::Result<String> {
        self.reply
            .map(str::to_string)
            .ok_or_else(|| liam_model::ModelError::Llm("boom".into()))
    }
}

/// Never answers within any test budget.
struct SleepingLlm;

#[async_trait::async_trait]
impl Llm for SleepingLlm {
    async fn complete(&self, _system: &str, _prompt: &str) -> liam_model::Result<String> {
        tokio::time::sleep(SLOW).await;
        Ok(GROUNDED_ANSWER.to_string())
    }
}

/// Says the evidence cannot answer, so the sufficiency pre-pass refuses.
struct RefusingLlm;

#[async_trait::async_trait]
impl Llm for RefusingLlm {
    async fn complete(&self, _system: &str, _prompt: &str) -> liam_model::Result<String> {
        Ok("NO".to_string())
    }
}

/// Fresh in-memory server over the given doubles, with a 30 s ask timeout and
/// no sufficiency pre-pass.
async fn server_with_models(
    embedder: Arc<dyn Embedder>,
    reranker: Arc<dyn Reranker>,
    llm: Arc<dyn Llm>,
) -> MemoryServer {
    server_with_ask_timeout(embedder, reranker, llm, 30).await
}

/// As `server_with_models`, with the given ask timeout in seconds.
async fn server_with_ask_timeout(
    embedder: Arc<dyn Embedder>,
    reranker: Arc<dyn Reranker>,
    llm: Arc<dyn Llm>,
    ask_timeout_secs: u64,
) -> MemoryServer {
    let store = DefaultGraph::open(":memory:", GraphConfig::new(DIMS))
        .await
        .expect("open in-memory store");
    MemoryServer::new(
        Arc::new(store),
        embedder,
        reranker,
        llm,
        ask_timeout_secs,
        false,
        8192,
        1,
    )
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

async fn seed_summaries(server: &MemoryServer, count: usize) {
    let vector = MockEmbedder::new(DIMS)
        .embed(QUERY, TextKind::Document)
        .await
        .unwrap();
    let store = server.store().await.unwrap();
    for index in 0..count {
        let summary = NewSummary {
            level: 1,
            scope: String::new(),
            text: QUERY.to_string(),
            events_fed: 1,
            valid_min: Millis(0),
            valid_max: Millis(1),
            model_id: "model".to_string(),
            prompt_version: SUMMARY_PROMPT_VERSION.to_string(),
        };
        let children = [SummaryChild {
            id: format!("child-{index}"),
            kind: liam_store::ChildKind::Node,
        }];
        let source = EventRef {
            event_id: "event-one".to_string(),
            synthetic: false,
        };
        let outcome = store
            .put_summary(summary, &children, &[source], &vector, &[], "test")
            .await
            .unwrap();
        assert!(matches!(outcome, PutOutcome::Inserted(_)));
    }
}

/// The same server, but reranking keeps the store's order.
fn in_store_order(server: &MemoryServer) -> MemoryServer {
    MemoryServer {
        reranker: Arc::new(IdentityReranker),
        ..server.clone()
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

fn ask_args() -> AskArgs {
    AskArgs {
        question: QUERY.to_string(),
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

/// Cancels `token` once `after` has passed.
fn cancel_after(token: &CancellationToken, after: Duration) {
    let token = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(after).await;
        token.cancel();
    });
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
        "synthesis_unavailable" => "showing the retrieved evidence",
        _ => "hits are in store order",
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
    let server = server_with_models(embedder, Arc::new(IdentityReranker), Arc::new(MockLlm))
        .await
        .with_recall_timeout(1);

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
    let server = server_with_models(embedder, Arc::new(IdentityReranker), Arc::new(MockLlm))
        .await
        .with_recall_timeout(0);

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
    let server = server_with_models(
        embedder.clone(),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
    )
    .await
    .with_recall_timeout(1);

    // Act
    let out = server.recall_with(recall_args(), cancelled()).await;

    // Assert
    assert_eq!(wire_text(out, NO_REASONS, false), CANCELLED_TEXT);
    assert_eq!(embedder.call_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn recall_returns_k_hits_in_store_order_when_the_reranker_outlasts_the_deadline() {
    // Arrange
    let reranker = SleepingReranker::new(SLOW);
    let server = server_with_models(
        SleepingEmbedder::new(Duration::ZERO),
        reranker,
        Arc::new(MockLlm),
    )
    .await
    .with_recall_timeout(1);
    seed_facts(&server).await;
    let expected = in_store_order(&server)
        .recall_with(recall_args(), CancellationToken::new())
        .await;

    // Act
    let out = server
        .recall_with(recall_args(), CancellationToken::new())
        .await;

    // Assert
    assert_eq!(block_count(out.text()), K, "{}", out.text());
    assert_eq!(out.text(), expected.text());
}

#[tokio::test(start_paused = true)]
async fn recall_skips_the_reranker_when_too_little_of_the_budget_is_left() {
    // Arrange
    let reranker = SleepingReranker::new(Duration::ZERO);
    let server = server_with_models(
        SleepingEmbedder::new(Duration::from_millis(900)),
        reranker.clone(),
        Arc::new(MockLlm),
    )
    .await
    .with_recall_timeout(1);
    seed_facts(&server).await;
    let expected = in_store_order(&server)
        .recall_with(recall_args(), CancellationToken::new())
        .await;

    // Act
    let out = server
        .recall_with(recall_args(), CancellationToken::new())
        .await;

    // Assert
    assert_eq!(reranker.call_count(), 0);
    assert_eq!(block_count(out.text()), K, "{}", out.text());
    assert_eq!(out.text(), expected.text());
}

#[tokio::test(start_paused = true)]
async fn recall_replies_cancelled_when_the_token_is_cancelled_during_the_rerank() {
    // Arrange
    let reranker = SleepingReranker::new(SLOW);
    let server = server_with_models(
        SleepingEmbedder::new(Duration::ZERO),
        reranker.clone(),
        Arc::new(MockLlm),
    )
    .await
    .with_recall_timeout(10);
    seed_facts(&server).await;
    let token = CancellationToken::new();
    cancel_after(&token, Duration::from_millis(100));

    // Act
    let out = server.recall_with(recall_args(), token).await;

    // Assert
    assert_eq!(wire_text(out, NO_REASONS, false), CANCELLED_TEXT);
    assert_eq!(
        reranker.call_count(),
        1,
        "the cancel should land in the rerank"
    );
}

#[tokio::test(start_paused = true)]
async fn ask_replies_with_the_timeout_when_the_embedder_outlasts_the_deadline() {
    // Arrange
    let server = server_with_models(
        SleepingEmbedder::new(SLOW),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
    )
    .await
    .with_recall_timeout(1);

    // Act
    let out = server.ask_with(ask_args(), CancellationToken::new()).await;

    // Assert
    assert_eq!(wire_text(out, NO_REASONS, true), ASK_TIMEOUT_TEXT);
}

#[tokio::test(start_paused = true)]
async fn ask_replies_cancelled_without_embedding_when_the_token_is_already_cancelled() {
    // Arrange
    let embedder = SleepingEmbedder::new(Duration::ZERO);
    let server = server_with_models(
        embedder.clone(),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
    )
    .await
    .with_recall_timeout(1);

    // Act
    let out = server.ask_with(ask_args(), cancelled()).await;

    // Assert
    assert_eq!(wire_text(out, NO_REASONS, false), ASK_CANCELLED_TEXT);
    assert_eq!(embedder.call_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn ask_keeps_store_order_hits_then_summaries_when_the_reranker_outlasts_the_deadline() {
    // Arrange
    let llm = Arc::new(PromptLlm::default());
    let server = server_with_models(
        SleepingEmbedder::new(Duration::ZERO),
        SleepingReranker::new(SLOW),
        llm.clone(),
    )
    .await
    .with_recall_timeout(1);
    seed_facts(&server).await;
    seed_summaries(&server, 2).await;
    in_store_order(&server)
        .ask_with(ask_args(), CancellationToken::new())
        .await;
    let expected = llm.prompt.lock().unwrap().clone();

    // Act
    let out = server.ask_with(ask_args(), CancellationToken::new()).await;

    // Assert
    let text = wire_text(out, &["rerank_timeout"], false);
    assert_eq!(
        text.lines().next(),
        Some(marker_for("rerank_timeout").as_str())
    );
    let prompt = llm.prompt.lock().unwrap().clone();
    assert_eq!(prompt.matches("] (fact) ").count(), K, "{prompt}");
    assert_eq!(prompt.matches("] (summary) ").count(), 2, "{prompt}");
    assert_eq!(prompt, expected);
}

#[tokio::test(start_paused = true)]
async fn recall_marks_each_rerank_degradation_and_keeps_store_order() {
    // Arrange
    let cases: [(&str, Duration, Arc<dyn Reranker>); 3] = [
        (
            "rerank_timeout",
            Duration::ZERO,
            SleepingReranker::new(SLOW),
        ),
        (
            "rerank_skipped",
            Duration::from_millis(900),
            SleepingReranker::new(Duration::ZERO),
        ),
        ("rerank_failed", Duration::ZERO, Arc::new(FailingReranker)),
    ];

    for (reason, embed_delay, reranker) in cases {
        let server = server_with_models(
            SleepingEmbedder::new(embed_delay),
            reranker,
            Arc::new(MockLlm),
        )
        .await
        .with_recall_timeout(1);
        seed_facts(&server).await;
        let expected = in_store_order(&server)
            .recall_with(recall_args(), CancellationToken::new())
            .await;

        // Act
        let out = server
            .recall_with(recall_args(), CancellationToken::new())
            .await;

        // Assert
        let text = wire_text(out, &[reason], false);
        let expected_text = format!("{}\n{}", marker_for(reason), expected.text());
        assert_eq!(text, expected_text, "{reason}");
        assert_eq!(block_count(&text), K, "{reason}");
    }
}

#[tokio::test(start_paused = true)]
async fn recall_returns_lexical_hits_with_a_marker_when_the_embedder_errors() {
    // Arrange
    let server = server_with_models(
        Arc::new(ErroringEmbedder::new(DIMS)),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
    )
    .await;
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
    let server = server_with_models(
        Arc::new(ErroringEmbedder::new(DIMS)),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
    )
    .await;

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
    let server = server_with_models(
        embedder.clone(),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
    )
    .await;

    // Act
    let out = server.recall_with(recall_args(), cancelled()).await;

    // Assert
    assert_eq!(wire_text(out, NO_REASONS, false), CANCELLED_TEXT);
    assert_eq!(embedder.call_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn recall_refuses_infer_mode_as_an_error() {
    // Arrange
    let server = server_with_models(
        Arc::new(MockEmbedder::new(DIMS)),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
    )
    .await;
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
    let server = server_with_models(
        Arc::new(MockEmbedder::new(DIMS)),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
    )
    .await;
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
    let server = server_with_models(
        Arc::new(MockEmbedder::new(DIMS)),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
    )
    .await;
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
async fn recall_skips_the_reranker_only_below_the_minimum_budget() {
    // Arrange
    let cases: [(u64, usize, &[&str]); 2] = [
        (BOUNDARY_BUDGET_MS, 1, NO_REASONS),
        (BOUNDARY_BUDGET_MS - 1, 0, &["rerank_skipped"]),
    ];

    for (remaining_ms, reranker_calls, reasons) in cases {
        // The embedder consumes all but `remaining_ms` of the deadline, on the
        // paused clock, so the budget left at the rerank is exact.
        let embed_delay =
            Duration::from_secs(BOUNDARY_RECALL_TIMEOUT_SECS) - Duration::from_millis(remaining_ms);
        let reranker = SleepingReranker::new(Duration::ZERO);
        let server = server_with_models(
            SleepingEmbedder::new(embed_delay),
            reranker.clone(),
            Arc::new(MockLlm),
        )
        .await
        .with_recall_timeout(BOUNDARY_RECALL_TIMEOUT_SECS);
        seed_facts(&server).await;

        // Act
        let out = server
            .recall_with(recall_args(), CancellationToken::new())
            .await;

        // Assert
        wire_text(out, reasons, false);
        assert_eq!(reranker.call_count(), reranker_calls, "{remaining_ms} ms");
    }
}

#[tokio::test(start_paused = true)]
async fn recall_fails_without_degraded_reasons_when_the_store_is_unavailable() {
    // Arrange
    let server = server_with_models(
        Arc::new(MockEmbedder::new(DIMS)),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
    )
    .await;
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
    let server = server_with_models(
        Arc::new(MockEmbedder::new(DIMS)),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
    )
    .await;
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
async fn recall_names_both_reasons_when_the_embed_errors_and_the_reranker_times_out() {
    // Arrange
    let server = server_with_models(
        Arc::new(ErroringEmbedder::new(DIMS)),
        SleepingReranker::new(SLOW),
        Arc::new(MockLlm),
    )
    .await
    .with_recall_timeout(1);
    seed_facts(&server).await;

    // Act
    let out = server
        .recall_with(recall_args(), CancellationToken::new())
        .await;

    // Assert
    let text = wire_text(out, &["embed_failed", "rerank_timeout"], false);
    assert_eq!(text.lines().next(), Some(REASONS_EMBED_AND_TIMEOUT));
    assert_eq!(block_count(&text), K, "{text}");
}

#[tokio::test(start_paused = true)]
async fn a_recall_cancelled_while_waiting_on_the_embedder_never_takes_the_gate() {
    // Arrange
    let gate = Arc::new(tokio::sync::Mutex::new(()));
    let embedder = Arc::new(GatedEmbedder::new(DIMS, gate.clone()));
    let server = server_with_models(
        embedder.clone(),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
    )
    .await;
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

#[tokio::test(start_paused = true)]
async fn ask_marks_each_fallback_as_synthesis_unavailable_and_keeps_the_evidence() {
    // Arrange
    let cases: [(&str, Arc<dyn Llm>, bool); 5] = [
        ("synthesis timeout", Arc::new(SleepingLlm), false),
        ("model error", Arc::new(CannedLlm { reply: None }), false),
        (
            "empty answer",
            Arc::new(CannedLlm { reply: Some("   ") }),
            false,
        ),
        (
            "ungrounded answer",
            Arc::new(CannedLlm {
                reply: Some(UNGROUNDED_ANSWER),
            }),
            false,
        ),
        (
            "permit wait timeout",
            Arc::new(CannedLlm {
                reply: Some(GROUNDED_ANSWER),
            }),
            true,
        ),
    ];

    for (name, llm, hold_permit) in cases {
        let server = server_with_ask_timeout(
            Arc::new(MockEmbedder::new(DIMS)),
            Arc::new(IdentityReranker),
            llm,
            ASK_TIMEOUT_SECS,
        )
        .await;
        seed_facts(&server).await;
        let _held = hold_permit.then(|| {
            server
                .generation_permits_handle()
                .try_acquire_owned()
                .expect("the sole permit is free")
        });

        // Act
        let out = server.ask_with(ask_args(), CancellationToken::new()).await;

        // Assert
        let text = wire_text(out, &["synthesis_unavailable"], false);
        assert_eq!(
            text.lines().next(),
            Some(marker_for("synthesis_unavailable").as_str()),
            "{name}"
        );
        assert_eq!(text.matches(" note ").count(), K, "{name}: {text}");
    }
}

#[tokio::test(start_paused = true)]
async fn ask_marks_the_embed_error_and_still_answers() {
    // Arrange
    let server = server_with_models(
        Arc::new(ErroringEmbedder::new(DIMS)),
        Arc::new(IdentityReranker),
        Arc::new(CannedLlm {
            reply: Some(GROUNDED_ANSWER),
        }),
    )
    .await;
    seed_facts(&server).await;

    // Act
    let out = server.ask_with(ask_args(), CancellationToken::new()).await;

    // Assert
    let text = wire_text(out, &["embed_failed"], false);
    assert_eq!(
        text.lines().next(),
        Some(marker_for("embed_failed").as_str())
    );
    assert!(text.contains(GROUNDED_ANSWER), "{text}");
}

#[tokio::test(start_paused = true)]
async fn a_healthy_ask_has_no_degraded_reasons_and_no_marker() {
    // Arrange
    let server = server_with_models(
        Arc::new(MockEmbedder::new(DIMS)),
        Arc::new(IdentityReranker),
        Arc::new(CannedLlm {
            reply: Some(GROUNDED_ANSWER),
        }),
    )
    .await;
    seed_facts(&server).await;

    // Act
    let out = server.ask_with(ask_args(), CancellationToken::new()).await;

    // Assert
    let text = wire_text(out, NO_REASONS, false);
    assert!(!text.starts_with(MARKER_PREFIX), "{text}");
    assert!(text.contains(GROUNDED_ANSWER), "{text}");
}

#[tokio::test(start_paused = true)]
async fn ask_fails_without_degraded_reasons_when_the_question_is_empty() {
    // Arrange
    let server = server_with_models(
        Arc::new(MockEmbedder::new(DIMS)),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
    )
    .await;
    let args = AskArgs {
        question: String::new(),
        ..ask_args()
    };

    // Act
    let out = server.ask_with(args, CancellationToken::new()).await;

    // Assert
    let text = wire_text(out, NO_REASONS, true);
    assert!(text.starts_with("ask failed:"), "{text}");
}

#[tokio::test(start_paused = true)]
async fn ask_fails_without_degraded_reasons_when_the_store_is_unavailable() {
    // Arrange
    let server = server_with_models(
        Arc::new(MockEmbedder::new(DIMS)),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
    )
    .await;
    *server.shared_store().exclusive().await = None;

    // Act
    let out = server.ask_with(ask_args(), CancellationToken::new()).await;

    // Assert
    let text = wire_text(out, NO_REASONS, true);
    assert!(text.starts_with("ask failed:"), "{text}");
}

#[tokio::test(start_paused = true)]
async fn ask_replies_cancelled_when_the_token_is_cancelled_during_the_rerank() {
    // Arrange
    let reranker = SleepingReranker::new(SLOW);
    let server = server_with_models(
        SleepingEmbedder::new(Duration::ZERO),
        reranker.clone(),
        Arc::new(MockLlm),
    )
    .await
    .with_recall_timeout(10);
    seed_facts(&server).await;
    let token = CancellationToken::new();
    cancel_after(&token, Duration::from_millis(100));

    // Act
    let out = server.ask_with(ask_args(), token).await;

    // Assert
    assert_eq!(wire_text(out, NO_REASONS, false), ASK_CANCELLED_TEXT);
    assert_eq!(
        reranker.call_count(),
        1,
        "the cancel should land in the rerank"
    );
}

#[tokio::test(start_paused = true)]
async fn ask_names_each_retrieval_reason_once_when_the_embed_and_the_reranker_fail() {
    // Arrange
    let server = server_with_models(
        Arc::new(ErroringEmbedder::new(DIMS)),
        Arc::new(FailingReranker),
        Arc::new(CannedLlm {
            reply: Some(GROUNDED_ANSWER),
        }),
    )
    .await;
    seed_facts(&server).await;

    // Act
    let out = server.ask_with(ask_args(), CancellationToken::new()).await;

    // Assert
    let text = wire_text(out, &["embed_failed", "rerank_failed"], false);
    let expected_marker = "(degraded: embed_failed, searched by keyword only; \
rerank_failed, hits are in store order)";
    assert_eq!(text.lines().next(), Some(expected_marker));
}

#[tokio::test(start_paused = true)]
async fn ask_names_the_rerank_timeout_and_the_synthesis_timeout_together() {
    // Arrange
    let server = server_with_ask_timeout(
        SleepingEmbedder::new(Duration::ZERO),
        SleepingReranker::new(SLOW),
        Arc::new(SleepingLlm),
        ASK_TIMEOUT_SECS,
    )
    .await
    .with_recall_timeout(1);
    seed_facts(&server).await;

    // Act
    let out = server.ask_with(ask_args(), CancellationToken::new()).await;

    // Assert
    let text = wire_text(out, &["rerank_timeout", "synthesis_unavailable"], false);
    let expected_marker = "(degraded: rerank_timeout, hits are in store order; \
synthesis_unavailable, showing the retrieved evidence)";
    assert_eq!(text.lines().next(), Some(expected_marker));
}

#[tokio::test(start_paused = true)]
async fn ask_names_keyword_only_search_when_the_embedder_errors_and_nothing_matches() {
    // Arrange
    let server = server_with_models(
        Arc::new(ErroringEmbedder::new(DIMS)),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
    )
    .await;

    // Act
    let out = server.ask_with(ask_args(), CancellationToken::new()).await;

    // Assert
    let text = wire_text(out, &["embed_failed"], false);
    assert_eq!(
        text,
        format!("{}\n{NO_MEMORY_TEXT}", marker_for("embed_failed"))
    );
}

#[tokio::test(start_paused = true)]
async fn ask_keeps_the_rerank_reason_in_strict_mode() {
    // Arrange
    let server = server_with_models(
        Arc::new(MockEmbedder::new(DIMS)),
        Arc::new(FailingReranker),
        Arc::new(MockLlm),
    )
    .await;
    seed_facts(&server).await;
    let args = AskArgs {
        mode: Some(Mode::Strict),
        ..ask_args()
    };

    // Act
    let out = server.ask_with(args, CancellationToken::new()).await;

    // Assert
    let text = wire_text(out, &["rerank_failed"], false);
    assert_eq!(
        text.lines().next(),
        Some(marker_for("rerank_failed").as_str())
    );
}

#[tokio::test(start_paused = true)]
async fn ask_keeps_the_rerank_reason_when_the_pre_pass_refuses() {
    // Arrange
    let server = server_with_models(
        Arc::new(MockEmbedder::new(DIMS)),
        Arc::new(FailingReranker),
        Arc::new(RefusingLlm),
    )
    .await;
    let server = MemoryServer {
        ask_sufficiency_check: true,
        ..server
    };
    seed_facts(&server).await;

    // Act
    let out = server.ask_with(ask_args(), CancellationToken::new()).await;

    // Assert
    let text = wire_text(out, &["rerank_failed"], false);
    assert_eq!(
        text.lines().next(),
        Some(marker_for("rerank_failed").as_str())
    );
    assert!(text.contains("does not contain an answer"), "{text}");
}
