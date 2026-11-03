// SPDX-License-Identifier: Apache-2.0
//! The metrics exit criterion: a staged failure over runs that are judged and
//! remembered on disk, read back through the `metrics` tool, with the alerts
//! cleared by a healthy run after it.

use std::sync::Arc;
use std::time::Duration;

use liam_daemon::consolidation::manifest::RunOutcome;
use liam_store::{DefaultGraph, FixedClock, GraphConfig, Millis};
use rmcp::handler::server::wrapper::Parameters;

use super::metrics::MetricsArgs;
use super::MemoryServer;
use crate::config::MetricsConfig;
use crate::log_startup::test_support::{embedder, DIMS};
use crate::metrics::alerts::state_path_for;
use crate::metrics::recorder::Metrics;
use crate::metrics::sink::{
    Labels, MemorySink, MetricId, MetricsSink, ModelCatalog, Provider, Sample,
};

const HOUR: i64 = 3_600_000;
const MIN_SAMPLES: usize = 3;
const DEADLINE_DAYS: u32 = 30;
const RUN_TIME: Duration = Duration::from_millis(1);
const ROLLBACK_CEILING: f64 = 1.0;
const BACKLOG_CEILING: f64 = 100.0;
const CALLS_PER_HOUR: f64 = 10.0;
const DELTA_PCT: f64 = 20.0;

struct Staged {
    metrics: Arc<Metrics>,
    sink: Arc<MemorySink>,
    clock: Arc<FixedClock>,
    server: MemoryServer,
    _dir: tempfile::TempDir,
}

async fn staged() -> Staged {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(FixedClock::new(Millis(0)));
    let sink = Arc::new(MemorySink::new(clock.clone()));
    let config = MetricsConfig {
        min_samples: 1,
        rollback_ceiling: ROLLBACK_CEILING,
        extraction_backlog_ceiling: BACKLOG_CEILING,
        model_calls_per_hour: CALLS_PER_HOUR,
        delta_pct: DELTA_PCT,
        ..MetricsConfig::default()
    };
    let metrics = Arc::new(
        Metrics::new(sink.clone(), clock.clone(), MIN_SAMPLES).with_alerts(
            state_path_for(&dir.path().join("liam.db")),
            crate::metrics::rules(&config, DEADLINE_DAYS),
        ),
    );
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
    )
    .with_metrics(metrics.clone());
    Staged {
        metrics,
        sink,
        clock,
        server,
        _dir: dir,
    }
}

fn sample(metric: MetricId, value: f64, labels: Labels) -> Sample {
    Sample {
        metric,
        value,
        labels,
        at: Millis(0),
    }
}

impl Staged {
    async fn run(&self, at: i64, outcome: RunOutcome, readings: Vec<Sample>) {
        self.clock.set(Millis(at));
        self.metrics
            .finish_run(Some(outcome), RUN_TIME, readings)
            .await;
    }

    async fn tool(&self) -> String {
        self.server
            .metrics(Parameters(MetricsArgs {
                metric: None,
                limit: None,
            }))
            .await
    }
}

#[tokio::test]
async fn a_staged_failure_raises_every_kind_of_alert_in_the_tool_and_a_healthy_run_clears_them() {
    // Arrange
    let staged = staged().await;
    let local = Labels::Model {
        provider: Provider::Local,
        model: ModelCatalog::default().intern("local-model").unwrap(),
    };
    let healthy = || {
        vec![
            sample(MetricId::ProbeRecallAt10, 1.0, Labels::None),
            sample(MetricId::DeletionBacklogAgeDays, 1.0, Labels::None),
            sample(MetricId::RulesExtractionBacklog, 5.0, Labels::None),
            sample(MetricId::ModelCalls, 0.0, local),
        ]
    };
    staged.run(0, RunOutcome::Committed, healthy()).await;
    let first = staged.tool().await;
    staged.run(HOUR, RunOutcome::Committed, healthy()).await;
    let excursion = vec![
        sample(MetricId::ProbeRecallAt10, 0.5, Labels::None),
        sample(
            MetricId::DeletionBacklogAgeDays,
            f64::from(DEADLINE_DAYS),
            Labels::None,
        ),
        sample(
            MetricId::RulesExtractionBacklog,
            BACKLOG_CEILING,
            Labels::None,
        ),
        sample(MetricId::ModelCalls, 50.0, local),
    ];

    // Act
    staged
        .run(2 * HOUR, RunOutcome::RolledBack, excursion)
        .await;
    let failing = staged.tool().await;
    let alerted: Vec<MetricId> = staged
        .sink
        .snapshot()
        .alerts
        .iter()
        .map(|alert| alert.metric)
        .collect();
    // The probe recall holds at its lower level, so it has stopped moving; the
    // rollback count is kept since the daemon started and does not fall.
    let mut recovery = healthy();
    recovery[0].value = 0.5;
    staged.run(3 * HOUR, RunOutcome::Committed, recovery).await;
    let recovered = staged.tool().await;

    // Assert
    assert!(
        first.ends_with("alerts: none"),
        "a first run raises nothing: {first}"
    );
    for metric in [
        MetricId::ProbeRecallAt10,
        MetricId::DeletionBacklogAgeDays,
        MetricId::RulesExtractionBacklog,
        MetricId::ModelCalls,
        MetricId::RollbackCount,
    ] {
        assert!(alerted.contains(&metric), "{metric} not in {alerted:?}");
        assert!(
            failing.contains(&format!("\n{metric} = ")),
            "{metric} not in the tool output: {failing}"
        );
    }
    let (_, still) = recovered.split_once("alerts:").unwrap();
    assert_eq!(still.lines().count(), 2, "{recovered}");
    assert!(still.contains("rollback_count"), "{recovered}");
}
