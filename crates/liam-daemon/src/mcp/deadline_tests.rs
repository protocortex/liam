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

use super::*;
use crate::summary::generate::SUMMARY_PROMPT_VERSION;

const DIMS: usize = 8;
const QUERY: &str = "quarterly forecasts for the finance team";
const SLOW: Duration = Duration::from_secs(30);
const FACT_COUNT: usize = 5;
const K: usize = 2;

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

/// Fresh in-memory server over the given doubles, with a 30 s ask timeout and
/// no sufficiency pre-pass.
async fn server_with_models(
    embedder: Arc<dyn Embedder>,
    reranker: Arc<dyn Reranker>,
    llm: Arc<dyn Llm>,
) -> MemoryServer {
    let store = DefaultGraph::open(":memory:", GraphConfig::new(DIMS))
        .await
        .expect("open in-memory store");
    MemoryServer::new(Arc::new(store), embedder, reranker, llm, 30, false, 8192, 1)
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
    assert_eq!(out, "recall failed: retrieval took longer than 1 s");
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
    assert_eq!(out, "recall failed: retrieval took longer than 1 s");
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
    assert_eq!(out, "recall cancelled");
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
    assert_eq!(block_count(&out), K, "{out}");
    assert_eq!(out, expected);
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
    assert_eq!(block_count(&out), K, "{out}");
    assert_eq!(out, expected);
}

#[tokio::test(start_paused = true)]
async fn recall_replies_cancelled_when_the_token_is_cancelled_during_the_rerank() {
    // Arrange
    let server = server_with_models(
        SleepingEmbedder::new(Duration::ZERO),
        SleepingReranker::new(SLOW),
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
    assert_eq!(out, "recall cancelled");
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
    assert_eq!(out, "ask failed: retrieval took longer than 1 s");
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
    assert_eq!(out, "ask cancelled");
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
    server.ask_with(ask_args(), CancellationToken::new()).await;

    // Assert
    let prompt = llm.prompt.lock().unwrap().clone();
    assert_eq!(prompt.matches("] (fact) ").count(), K, "{prompt}");
    assert_eq!(prompt.matches("] (summary) ").count(), 2, "{prompt}");
    assert_eq!(prompt, expected);
}
