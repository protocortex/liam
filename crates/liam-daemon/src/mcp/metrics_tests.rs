// SPDX-License-Identifier: Apache-2.0
//! The `metrics` tool: what it shows, its filter and its limit.

use std::sync::Arc;
use std::time::Duration;

use liam_model::{Llm, Reranker};
use liam_store::{DefaultGraph, FixedClock, GraphConfig, Millis, NewNode};
use rmcp::handler::server::wrapper::Parameters;
use tokio_util::sync::CancellationToken;

use super::metrics::{label_suffix, MetricsArgs, MAX_LIMIT};
use super::{AskArgs, MemoryServer, RecallArgs};
use crate::degraded::Degraded;
use crate::log_startup::test_support::{embedder, DIMS};
use crate::metrics::alerts::{Alert, Rule};
use crate::metrics::monotonic::TickingMonotonic;
use crate::metrics::recorder::Metrics;
use crate::metrics::sink::Tool;
use crate::metrics::sink::{Labels, MemorySink, MetricId, MetricsSink, Sample};

const AT: Millis = Millis(1_000);
const TICK: Duration = Duration::from_micros(250);

const QUERY: &str = "quarterly forecasts for the finance team";
const SLOW: Duration = Duration::from_secs(30);
const RECALL_TIMEOUT_SECS: u64 = 1;
const FACT_COUNT: usize = 3;
const DEGRADED_RECALLS: usize = 3;
const DEGRADED_REPLIES: &str = "degraded_replies";

/// Outlasts any retrieval deadline, so recall falls back to store order.
struct SleepingReranker;

#[async_trait::async_trait]
impl Reranker for SleepingReranker {
    async fn scores(&self, _query: &str, docs: &[String]) -> liam_model::Result<Vec<f32>> {
        tokio::time::sleep(SLOW).await;
        Ok(vec![0.0; docs.len()])
    }
}

/// Embeds after a delay no deadline outlasts, so retrieval fails on time.
struct SleepingEmbedder;

#[async_trait::async_trait]
impl liam_model::Embedder for SleepingEmbedder {
    fn dims(&self) -> usize {
        DIMS
    }
    async fn embed(
        &self,
        _text: &str,
        _kind: liam_model::TextKind,
    ) -> liam_model::Result<Vec<f32>> {
        tokio::time::sleep(SLOW).await;
        Ok(vec![0.0; DIMS])
    }
}

struct FailingLlm;

#[async_trait::async_trait]
impl Llm for FailingLlm {
    async fn complete(&self, _system: &str, _prompt: &str) -> liam_model::Result<String> {
        Err(liam_model::ModelError::Llm("boom".into()))
    }
}

async fn server(metrics: Option<Arc<Metrics>>) -> MemoryServer {
    server_with(
        metrics,
        Arc::new(liam_model::IdentityReranker),
        Arc::new(liam_model::MockLlm),
    )
    .await
}

async fn server_with(
    metrics: Option<Arc<Metrics>>,
    reranker: Arc<dyn Reranker>,
    llm: Arc<dyn Llm>,
) -> MemoryServer {
    let store = DefaultGraph::open(":memory:", GraphConfig::new(DIMS))
        .await
        .unwrap();
    let server = MemoryServer::new(
        Arc::new(store),
        embedder(),
        reranker,
        llm,
        30,
        false,
        8192,
        1,
    );
    let server = server.with_recall_timeout(RECALL_TIMEOUT_SECS);
    match metrics {
        Some(metrics) => server.with_metrics(metrics),
        None => server,
    }
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
        k: None,
        as_of: None,
        mode: None,
    }
}

fn ask_args() -> AskArgs {
    AskArgs {
        question: QUERY.to_string(),
        kind: None,
        scope: None,
        k: None,
        as_of: None,
        mode: None,
    }
}

async fn recall_reply(server: &MemoryServer) -> rmcp::model::CallToolResult {
    server
        .recall_tool(Parameters(recall_args()), CancellationToken::new())
        .await
}

/// The `metrics` tool's lines for `degraded_replies`.
async fn degraded_lines(server: &MemoryServer) -> String {
    ask(server, Some(DEGRADED_REPLIES), None).await
}

fn recorder() -> (Arc<Metrics>, Arc<MemorySink>) {
    let clock = Arc::new(FixedClock::new(AT));
    let sink = Arc::new(MemorySink::new(clock.clone()));
    let metrics =
        Metrics::new(sink.clone(), clock, 2).with_monotonic(Arc::new(TickingMonotonic::new(TICK)));
    (Arc::new(metrics), sink)
}

fn record(sink: &MemorySink, metric: MetricId, value: f64) {
    sink.record(Sample {
        metric,
        value,
        labels: Labels::None,
        at: AT,
    });
}

async fn ask(server: &MemoryServer, metric: Option<&str>, limit: Option<usize>) -> String {
    server
        .metrics(Parameters(MetricsArgs {
            metric: metric.map(str::to_owned),
            limit,
        }))
        .await
}

#[tokio::test]
async fn the_tool_lists_each_reading_and_the_active_alerts() {
    // Arrange
    let (metrics, sink) = recorder();
    record(&sink, MetricId::ProbeRecallAt10, 0.5);
    record(&sink, MetricId::RollbackCount, 3.0);
    sink.set_alerts(vec![Alert {
        metric: MetricId::RollbackCount,
        rule: Rule::Absolute { ceiling: 1.0 },
        value: 3.0,
    }]);
    let server = server(Some(metrics)).await;

    // Act
    let text = ask(&server, None, None).await;

    // Assert
    assert!(text.contains("probe_recall_at_10 = 0.5"), "{text}");
    assert!(text.contains("rollback_count = 3"), "{text}");
    assert!(
        text.contains("alerts:\nrollback_count = 3: reached 1"),
        "{text}"
    );
}

#[tokio::test]
async fn a_filter_shows_only_that_metric_and_its_alerts() {
    // Arrange
    let (metrics, sink) = recorder();
    record(&sink, MetricId::ProbeRecallAt10, 0.5);
    record(&sink, MetricId::RollbackCount, 3.0);
    sink.set_alerts(vec![Alert {
        metric: MetricId::RollbackCount,
        rule: Rule::Absolute { ceiling: 1.0 },
        value: 3.0,
    }]);
    let server = server(Some(metrics)).await;

    // Act
    let text = ask(&server, Some("probe_recall_at_10"), None).await;

    // Assert
    assert!(text.contains("probe_recall_at_10"), "{text}");
    assert!(!text.contains("rollback_count"), "{text}");
    assert!(text.ends_with("alerts: none"), "{text}");
}

#[tokio::test]
async fn an_unknown_metric_is_named_with_the_known_ones() {
    // Arrange
    let (metrics, _sink) = recorder();
    let server = server(Some(metrics)).await;

    // Act
    let text = ask(&server, Some("speed"), None).await;

    // Assert
    assert!(
        text.starts_with("metrics failed: unknown metric \"speed\""),
        "{text}"
    );
    assert!(text.contains("probe_recall_at_10"), "{text}");
}

#[tokio::test]
async fn a_limit_of_zero_returns_nothing_and_a_large_one_is_capped() {
    // Arrange
    let (metrics, sink) = recorder();
    for (index, metric) in MetricId::ALL.iter().enumerate() {
        record(&sink, *metric, index as f64);
    }
    let server = server(Some(metrics)).await;

    // Act
    let none = ask(&server, None, Some(0)).await;
    let capped = ask(&server, None, Some(MAX_LIMIT * 10)).await;
    let two = ask(&server, None, Some(2)).await;

    // Assert
    assert_eq!(none, "no readings requested: limit is 0");
    assert_eq!(
        capped.matches(" = ").count(),
        MetricId::ALL.len().min(MAX_LIMIT)
    );
    assert_eq!(two.matches("(samples").count(), 2);
}

#[tokio::test]
async fn a_server_without_metrics_says_so() {
    // Arrange
    let server = server(None).await;

    // Act
    let text = ask(&server, None, None).await;

    // Assert
    assert!(text.starts_with("metrics failed:"), "{text}");
}

#[tokio::test(start_paused = true)]
async fn degraded_recalls_and_asks_are_counted_by_tool_and_reason() {
    // Arrange
    let (metrics, _sink) = recorder();
    let recalling = server_with(
        Some(metrics.clone()),
        Arc::new(SleepingReranker),
        Arc::new(liam_model::MockLlm),
    )
    .await;
    seed_facts(&recalling).await;
    let asking = MemoryServer {
        llm: Arc::new(FailingLlm),
        reranker: Arc::new(liam_model::IdentityReranker),
        ..recalling.clone()
    };

    // Act
    for _ in 0..DEGRADED_RECALLS {
        recall_reply(&recalling).await;
    }
    asking
        .ask_tool(Parameters(ask_args()), CancellationToken::new())
        .await;
    let text = degraded_lines(&recalling).await;

    // Assert
    assert!(
        text.contains("degraded_replies{tool=recall,reason=rerank_timeout} = 3"),
        "{text}"
    );
    assert!(
        text.contains("degraded_replies{tool=ask,reason=synthesis_unavailable} = 1"),
        "{text}"
    );
}

#[tokio::test]
async fn a_recall_reports_a_latency_above_zero() {
    // Arrange
    let (metrics, sink) = recorder();
    let server = server(Some(metrics)).await;
    seed_facts(&server).await;

    // Act
    recall_reply(&server).await;

    // Assert
    let latency = sink
        .snapshot()
        .readings
        .into_iter()
        .find(|reading| reading.metric == MetricId::RecallLatencyP50Ms)
        .unwrap();
    assert!(latency.value > 0.0, "{}", latency.value);
}

#[tokio::test]
async fn no_degraded_reply_leaves_the_counter_without_readings() {
    // Arrange
    let (metrics, _sink) = recorder();
    let server = server(Some(metrics)).await;
    seed_facts(&server).await;

    // Act
    recall_reply(&server).await;
    let text = degraded_lines(&server).await;

    // Assert
    assert!(text.starts_with("no readings yet"), "{text}");
}

#[tokio::test(start_paused = true)]
async fn a_deadline_failure_and_a_cancel_are_not_counted() {
    // Arrange
    let (metrics, _sink) = recorder();
    let slow = MemoryServer {
        embedder: Arc::new(SleepingEmbedder),
        ..server(Some(metrics)).await
    };
    let cancelled = CancellationToken::new();
    cancelled.cancel();

    // Act
    let failed = slow
        .recall_tool(Parameters(recall_args()), CancellationToken::new())
        .await;
    slow.recall_tool(Parameters(recall_args()), cancelled).await;
    let text = degraded_lines(&slow).await;

    // Assert
    assert_eq!(failed.is_error, Some(true));
    assert!(text.starts_with("no readings yet"), "{text}");
}

#[tokio::test(start_paused = true)]
async fn a_server_without_metrics_replies_the_same_for_a_degraded_recall() {
    // Arrange
    let (metrics, _sink) = recorder();
    let with = server_with(
        Some(metrics),
        Arc::new(SleepingReranker),
        Arc::new(liam_model::MockLlm),
    )
    .await;
    let without = MemoryServer {
        metrics: None,
        ..with.clone()
    };
    seed_facts(&with).await;

    // Act
    let counted = recall_reply(&with).await;
    let uncounted = recall_reply(&without).await;

    // Assert
    assert_eq!(uncounted.structured_content, counted.structured_content);
    assert_eq!(uncounted.content, counted.content);
    assert_eq!(
        uncounted.structured_content,
        Some(serde_json::json!({ "degraded": [Degraded::RerankTimeout] }))
    );
}

#[test]
fn a_degraded_label_renders_its_tool_and_reason() {
    for tool in [Tool::Recall, Tool::Ask] {
        for reason in Degraded::ALL {
            // Arrange
            let labels = Labels::Degraded { tool, reason };
            let tool_name = match tool {
                Tool::Recall => "recall",
                Tool::Ask => "ask",
            };

            // Act
            let suffix = label_suffix(labels);

            // Assert
            assert_eq!(
                suffix,
                format!("{{tool={tool_name},reason={}}}", reason.as_str())
            );
        }
    }
}
