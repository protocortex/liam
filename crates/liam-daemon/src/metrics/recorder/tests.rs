// SPDX-License-Identifier: Apache-2.0

use liam_store::{FixedClock, Millis};

use super::*;
use crate::metrics::sink::{MemorySink, Reading};
use crate::tiering::AppliedMoves;

const MIN_PAIRS: usize = 2;

fn recorder() -> (Arc<Metrics>, Arc<MemorySink>) {
    let clock = Arc::new(FixedClock::new(Millis(5_000)));
    let sink = Arc::new(MemorySink::new(clock.clone()));
    let metrics = Arc::new(Metrics::new(sink.clone(), clock, MIN_PAIRS));
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
    assert!(reading(&sink, MetricId::RecallLatencyP50Ms, Labels::None).is_some());
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
