// SPDX-License-Identifier: Apache-2.0
//! The `metrics` tool: what it shows, its filter and its limit.

use std::sync::Arc;

use liam_store::{DefaultGraph, FixedClock, GraphConfig, Millis};
use rmcp::handler::server::wrapper::Parameters;

use super::metrics::{MetricsArgs, MAX_LIMIT};
use super::MemoryServer;
use crate::log_startup::test_support::{embedder, DIMS};
use crate::metrics::alerts::{Alert, Rule};
use crate::metrics::recorder::Metrics;
use crate::metrics::sink::{Labels, MemorySink, MetricId, MetricsSink, Sample};

const AT: Millis = Millis(1_000);

async fn server(metrics: Option<Arc<Metrics>>) -> MemoryServer {
    let store = DefaultGraph::open(":memory:", GraphConfig::new(DIMS))
        .await
        .unwrap();
    let server = MemoryServer::new(
        Arc::new(store),
        embedder(),
        Arc::new(liam_model::IdentityReranker),
        Arc::new(liam_model::MockLlm),
        30,
        false,
        8192,
        1,
    );
    match metrics {
        Some(metrics) => server.with_metrics(metrics),
        None => server,
    }
}

fn recorder() -> (Arc<Metrics>, Arc<MemorySink>) {
    let clock = Arc::new(FixedClock::new(AT));
    let sink = Arc::new(MemorySink::new(clock.clone()));
    (Arc::new(Metrics::new(sink.clone(), clock, 2)), sink)
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
