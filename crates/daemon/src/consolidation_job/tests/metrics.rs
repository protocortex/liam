// SPDX-License-Identifier: Apache-2.0
//! What a consolidation run reports to the metrics.

use liam_store::Millis;

use super::*;
use crate::metrics::sink::{Labels, MemorySink, MetricId, MetricsSink};
use liam_store::FixedClock;

const MIN_PAIRS: usize = 2;

fn measured() -> (Arc<Metrics>, Arc<MemorySink>) {
    let clock = Arc::new(FixedClock::new(Millis(1_000)));
    let sink = Arc::new(MemorySink::new(clock.clone()));
    (Arc::new(Metrics::new(sink.clone(), clock, MIN_PAIRS)), sink)
}

fn value(sink: &MemorySink, metric: MetricId) -> Option<f64> {
    sink.snapshot()
        .readings
        .into_iter()
        .find(|reading| reading.metric == metric && reading.labels == Labels::None)
        .map(|reading| reading.value)
}

#[tokio::test]
async fn a_committed_run_reports_its_duration_the_probe_recall_and_the_disk_sizes() {
    // Arrange
    let harness = Harness::new();
    let (store, _clock) = harness.store_at(Millis(1000)).await;
    keep_one_fact(&store).await;
    let server = harness.server(store, Arc::new(liam_model::MockLlm));
    let (metrics, sink) = measured();
    let job = harness.job_measured(
        &server,
        honest(),
        None,
        ConsolidationHookSet::default(),
        Some(metrics),
    );

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert!(value(&sink, MetricId::ConsolidationSeconds).is_some());
    assert_eq!(value(&sink, MetricId::RollbackCount), Some(0.0));
    assert_eq!(value(&sink, MetricId::ProbeRecallAt10), Some(1.0));
    assert!(value(&sink, MetricId::LogBytes).unwrap() > 0.0);
}

#[tokio::test]
async fn a_rolled_back_run_adds_one_to_the_rollback_count() {
    // Arrange
    let harness = Harness::new();
    let (store, _clock) = harness.store_at(Millis(100 * DAY)).await;
    keep_one_fact(&store).await;
    let server = harness.server(store, Arc::new(liam_model::MockLlm));
    let (metrics, sink) = measured();
    let job = harness.job_measured(
        &server,
        regresses(),
        None,
        ConsolidationHookSet::default(),
        Some(metrics),
    );

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::RolledBack));
    assert_eq!(value(&sink, MetricId::RollbackCount), Some(1.0));
}

#[tokio::test]
async fn a_real_rollback_raises_the_rollback_alert_without_an_injected_sample() {
    // Arrange
    let mut harness = Harness::new();
    harness.config.metrics.rollback_ceiling = 1.0;
    let (store, _clock) = harness.store_at(Millis(100 * DAY)).await;
    keep_one_fact(&store).await;
    let server = harness.server(store, Arc::new(liam_model::MockLlm));
    let metrics = crate::metrics::build(
        &harness.config.metrics,
        &database_in(&harness.root),
        harness.config.deletion.deadline_days,
    );
    let job = harness.job_measured(
        &server,
        regresses(),
        None,
        ConsolidationHookSet::default(),
        Some(metrics.clone()),
    );

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::RolledBack));
    let alerts = metrics.snapshot().await.alerts;
    assert!(
        alerts
            .iter()
            .any(|alert| alert.metric == MetricId::RollbackCount),
        "{alerts:?}"
    );
}

#[tokio::test]
async fn a_committed_first_run_raises_no_alert() {
    // Arrange
    let harness = Harness::new();
    let (store, _clock) = harness.store_at(Millis(1000)).await;
    keep_one_fact(&store).await;
    let server = harness.server(store, Arc::new(liam_model::MockLlm));
    let metrics = crate::metrics::build(
        &harness.config.metrics,
        &database_in(&harness.root),
        harness.config.deletion.deadline_days,
    );
    let job = harness.job_measured(
        &server,
        honest(),
        None,
        ConsolidationHookSet::default(),
        Some(metrics.clone()),
    );

    // Act
    job.tick().await;

    // Assert
    assert!(metrics.snapshot().await.alerts.is_empty());
}
