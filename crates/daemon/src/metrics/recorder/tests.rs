// SPDX-License-Identifier: Apache-2.0

use liam_store::{FixedClock, Millis};

use super::*;
use crate::metrics::monotonic::TickingMonotonic;
use crate::metrics::sink::{MemorySink, Reading};
use crate::tiering::AppliedMoves;

const MIN_PAIRS: usize = 2;
/// Under a millisecond, to show timings keep their fraction.
const TICK: Duration = Duration::from_micros(250);

fn recorder() -> (Arc<Metrics>, Arc<MemorySink>) {
    let clock = Arc::new(FixedClock::new(Millis(5_000)));
    let sink = Arc::new(MemorySink::new(clock.clone()));
    let metrics = Arc::new(
        Metrics::new(sink.clone(), clock, MIN_PAIRS)
            .with_monotonic(Arc::new(TickingMonotonic::new(TICK))),
    );
    (metrics, sink)
}

fn reading(sink: &MemorySink, metric: MetricId, labels: Labels) -> Option<Reading> {
    sink.snapshot()
        .readings
        .into_iter()
        .find(|reading| reading.metric == metric && reading.labels == labels)
}

fn ms(millis: u64) -> Duration {
    Duration::from_millis(millis)
}

#[test]
fn a_timed_tool_call_lands_in_its_own_percentiles_only() {
    // Arrange
    let (metrics, sink) = recorder();

    // Act
    for millis in 1..=10 {
        metrics.record_tool(Tool::Ask, ms(millis));
    }

    // Assert
    let p95 = reading(&sink, MetricId::AskLatencyP95Ms, Labels::None).unwrap();
    assert_eq!(p95.value, 10.0);
    assert!(reading(&sink, MetricId::RecallLatencyP95Ms, Labels::None).is_none());
}

#[test]
fn dropping_a_timer_records_the_call() {
    // Arrange
    let (metrics, sink) = recorder();

    // Act
    drop(metrics.time(Tool::Recall));

    // Assert
    let p50 = reading(&sink, MetricId::RecallLatencyP50Ms, Labels::None).unwrap();
    assert!(p50.value > 0.0 && p50.value < 1.0, "{}", p50.value);
}

#[test]
fn a_stage_keeps_its_sub_millisecond_time() {
    // Arrange
    let (metrics, sink) = recorder();

    // Act
    let started = metrics.now();
    metrics.record_stage(Stage::RecordUsage, started);

    // Assert
    let p50 = reading(&sink, MetricId::StageRecordUsageP50Ms, Labels::None).unwrap();
    assert!(p50.value > 0.0 && p50.value < 1.0, "{}", p50.value);
}

#[test]
fn a_query_times_only_the_arms_it_ran() {
    // Arrange
    let (metrics, sink) = recorder();
    let timings = QueryTimings {
        lexical: Some(ms(4)),
        vector: None,
        cold: Some(ms(9)),
    };

    // Act
    metrics.record_arms(timings);

    // Assert
    let lexical = reading(
        &sink,
        MetricId::ArmLatencyP95Ms,
        Labels::Arm(RetrievalArm::Lexical),
    );
    assert_eq!(lexical.unwrap().value, 4.0);
    let vector = reading(
        &sink,
        MetricId::ArmLatencyP95Ms,
        Labels::Arm(RetrievalArm::F32),
    );
    assert!(vector.is_none());
    let cold = reading(
        &sink,
        MetricId::ArmLatencyP95Ms,
        Labels::Arm(RetrievalArm::ColdProbe),
    );
    assert_eq!(cold.unwrap().value, 9.0);
}

#[test]
fn the_tier_report_gives_the_moves_and_the_calibration() {
    // Arrange
    let (metrics, sink) = recorder();
    let applied = AppliedMoves {
        promoted: 3,
        reembedded: 5,
        ..AppliedMoves::default()
    };
    let calibration = [(0.95, true), (0.95, true), (0.05, false)];

    // Act
    metrics.tier_report(&applied, &calibration);

    // Assert
    let value = |metric| reading(&sink, metric, Labels::None).unwrap().value;
    assert_eq!(value(MetricId::ItemsPromoted), 3.0);
    assert_eq!(value(MetricId::ItemsReembedded), 5.0);
    assert!(value(MetricId::Ece) < 0.1);
}

#[test]
fn too_few_verdicts_give_no_calibration_error() {
    // Arrange
    let (metrics, sink) = recorder();

    // Act
    metrics.tier_report(&AppliedMoves::default(), &[(0.5, true)]);

    // Assert
    assert!(reading(&sink, MetricId::Ece, Labels::None).is_none());
}

#[test]
fn the_rollback_count_grows_with_each_rolled_back_run_and_not_with_a_commit() {
    // Arrange
    let (metrics, sink) = recorder();

    // Act
    metrics.run_finished(Some(RunOutcome::RolledBack), ms(1500));
    metrics.run_finished(Some(RunOutcome::Committed), ms(500));
    metrics.run_finished(Some(RunOutcome::RestoreFailed), ms(500));

    // Assert
    let count = reading(&sink, MetricId::RollbackCount, Labels::None).unwrap();
    assert_eq!(count.value, 2.0);
    let seconds = reading(&sink, MetricId::ConsolidationSeconds, Labels::None).unwrap();
    assert_eq!(seconds.value, 0.5);
}

#[test]
fn a_probe_report_gives_the_mean_recall() {
    // Arrange
    let (metrics, sink) = recorder();
    let mut report = ProbeReport::default();
    report.scores.insert(
        "a".into(),
        liam_daemon::consolidation::probes::ProbeScore {
            recall: 0.5,
            ndcg: 1.0,
        },
    );

    // Act
    metrics.probe_report(&report);

    // Assert
    let recall = reading(&sink, MetricId::ProbeRecallAt10, Labels::None).unwrap();
    assert_eq!(recall.value, 0.5);
}

const HOUR: i64 = 3_600_000;
const SAMPLES_ENOUGH: usize = 3;

/// A recorder whose clock the test sets, judging under the default rules.
fn judging(dir: &std::path::Path) -> (Arc<Metrics>, Arc<MemorySink>, Arc<FixedClock>) {
    let clock = Arc::new(FixedClock::new(Millis(0)));
    let sink = Arc::new(MemorySink::new(clock.clone()));
    let config = crate::config::MetricsConfig {
        min_samples: 1,
        rollback_ceiling: 1.0,
        extraction_backlog_ceiling: 100.0,
        model_calls_per_hour: 10.0,
        delta_pct: 20.0,
        ..crate::config::MetricsConfig::default()
    };
    let metrics = Metrics::new(sink.clone(), clock.clone(), SAMPLES_ENOUGH).with_alerts(
        crate::metrics::alerts::state_path_for(&dir.join("liam.db")),
        crate::metrics::rules(&config, 30),
    );
    (Arc::new(metrics), sink, clock)
}

fn sample(metric: MetricId, value: f64, at: i64) -> Sample {
    Sample {
        metric,
        value,
        labels: Labels::None,
        at: Millis(at),
    }
}

async fn run(
    metrics: &Metrics,
    clock: &FixedClock,
    at: i64,
    outcome: RunOutcome,
    readings: Vec<Sample>,
) {
    clock.set(Millis(at));
    metrics.finish_run(Some(outcome), ms(1), readings).await;
}

fn alerting(sink: &MemorySink) -> Vec<MetricId> {
    sink.snapshot()
        .alerts
        .iter()
        .map(|alert| alert.metric)
        .collect()
}

#[tokio::test]
async fn each_kind_of_rule_raises_its_alert_on_a_staged_excursion() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let (metrics, sink, clock) = judging(dir.path());
    let healthy = vec![
        sample(MetricId::ProbeRecallAt10, 1.0, 0),
        sample(MetricId::DeletionBacklogAgeDays, 1.0, 0),
        sample(MetricId::RulesExtractionBacklog, 5.0, 0),
    ];
    run(&metrics, &clock, 0, RunOutcome::Committed, healthy).await;
    assert!(alerting(&sink).is_empty(), "the first run raises nothing");
    let mut model_calls = Sample {
        labels: Labels::Model {
            provider: crate::metrics::sink::Provider::Local,
            model: crate::metrics::sink::ModelCatalog::default()
                .intern("local-model")
                .unwrap(),
        },
        ..sample(MetricId::ModelCalls, 0.0, 0)
    };
    metrics.sink.record(model_calls);
    run(&metrics, &clock, HOUR, RunOutcome::Committed, vec![]).await;
    model_calls.value = 50.0;

    // Act
    let excursion = vec![
        sample(MetricId::ProbeRecallAt10, 0.5, 2 * HOUR),
        sample(MetricId::DeletionBacklogAgeDays, 30.0, 2 * HOUR),
        sample(MetricId::RulesExtractionBacklog, 100.0, 2 * HOUR),
        model_calls,
    ];
    run(&metrics, &clock, 2 * HOUR, RunOutcome::Committed, excursion).await;

    // Assert
    let raised = alerting(&sink);
    for metric in [
        MetricId::ProbeRecallAt10,
        MetricId::DeletionBacklogAgeDays,
        MetricId::RulesExtractionBacklog,
        MetricId::ModelCalls,
    ] {
        assert!(raised.contains(&metric), "{metric} missing from {raised:?}");
    }
    let age_alerts = raised
        .iter()
        .filter(|metric| **metric == MetricId::DeletionBacklogAgeDays)
        .count();
    assert_eq!(age_alerts, 2, "the warning and the deadline both fire");
}

#[tokio::test]
async fn an_alert_clears_on_the_run_after_the_recovery() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let (metrics, sink, clock) = judging(dir.path());
    let backlog = |value| vec![sample(MetricId::ModelExtractionBacklog, value, 0)];
    run(&metrics, &clock, 0, RunOutcome::Committed, backlog(1.0)).await;
    run(
        &metrics,
        &clock,
        HOUR,
        RunOutcome::Committed,
        backlog(500.0),
    )
    .await;
    assert_eq!(alerting(&sink), [MetricId::ModelExtractionBacklog]);

    // Act
    run(
        &metrics,
        &clock,
        2 * HOUR,
        RunOutcome::Committed,
        backlog(2.0),
    )
    .await;

    // Assert
    assert!(alerting(&sink).is_empty());
}

#[tokio::test]
async fn a_failed_restore_neither_judges_nor_remembers_nor_clears() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let (metrics, sink, clock) = judging(dir.path());
    let backlog = |value| vec![sample(MetricId::RulesExtractionBacklog, value, 0)];
    run(&metrics, &clock, 0, RunOutcome::Committed, backlog(500.0)).await;
    let state = crate::metrics::alerts::state_path_for(&dir.path().join("liam.db"));
    let before = std::fs::read(&state).unwrap();

    // Act
    run(
        &metrics,
        &clock,
        HOUR,
        RunOutcome::RestoreFailed,
        backlog(1.0),
    )
    .await;

    // Assert
    assert_eq!(std::fs::read(&state).unwrap(), before);
    assert_eq!(alerting(&sink), [MetricId::RulesExtractionBacklog]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_reader_never_sees_one_runs_readings_with_the_previous_runs_alerts() {
    // Arrange
    const ROUNDS: i64 = 200;
    let dir = tempfile::tempdir().unwrap();
    let (metrics, _sink, clock) = judging(dir.path());
    let writer = {
        let (metrics, clock) = (metrics.clone(), clock.clone());
        tokio::spawn(async move {
            for round in 1..=ROUNDS {
                let backlog = sample(
                    MetricId::RulesExtractionBacklog,
                    100.0 + round as f64,
                    round,
                );
                run(
                    &metrics,
                    &clock,
                    round,
                    RunOutcome::Committed,
                    vec![backlog],
                )
                .await;
                tokio::task::yield_now().await;
            }
        })
    };

    // Act and Assert
    while !writer.is_finished() {
        let snapshot = metrics.snapshot().await;
        let reading = snapshot
            .readings
            .iter()
            .find(|reading| reading.metric == MetricId::RulesExtractionBacklog);
        if let Some(reading) = reading {
            let alert = snapshot
                .alerts
                .first()
                .expect("every round is over the ceiling");
            assert_eq!(alert.value, reading.value);
        }
        tokio::task::yield_now().await;
    }
    writer.await.unwrap();
}

#[test]
fn a_model_call_is_counted_with_its_cost_under_provider_and_model_labels() {
    // Arrange
    let (metrics, sink) = recorder();
    let price = crate::metrics::collect::Price {
        input_micro_per_mtok: 2_000_000,
        output_micro_per_mtok: 10_000_000,
        ..Default::default()
    };
    let tokens = crate::metrics::sink::TokenUsage {
        input_tokens: 1_000_000,
        output_tokens: 100_000,
        cached_input_tokens: 0,
    };

    // Act
    metrics.model_call(
        crate::metrics::sink::Provider::OpenAi,
        "gpt-4o-2024-08-06",
        tokens,
        false,
        price,
    );
    metrics.model_call(
        crate::metrics::sink::Provider::OpenAi,
        "gpt-4o-2024-08-06",
        tokens,
        false,
        price,
    );
    metrics.model_call(
        crate::metrics::sink::Provider::OpenAi,
        "not a model id",
        tokens,
        false,
        price,
    );

    // Assert
    let readings = sink.snapshot().readings;
    let value = |metric| readings.iter().find(|r| r.metric == metric).unwrap().value;
    assert_eq!(value(MetricId::ModelCalls), 2.0);
    assert_eq!(value(MetricId::ModelCostMicroDollars), 6_000_000.0);
    assert_eq!(readings.len(), 2);
}
