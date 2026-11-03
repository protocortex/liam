// SPDX-License-Identifier: Apache-2.0
//! The daemon's side of the metrics: what the handlers and a consolidation run
//! report, turned into samples in the sink. Nothing here writes to disk, so
//! the read path stays read-only.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use liam_daemon::consolidation::manifest::RunOutcome;
use liam_daemon::consolidation::probes::ProbeReport;
use liam_store::{Clock, QueryTimings};

use super::alerts::{evaluate_run, Observation, RuleSet};
use super::collect::{self, LatencyWindow, ModelMeter, Price};
use super::sink::{
    Labels, MetricId, MetricsSink, ModelCatalog, Provider, RetrievalArm, Sample, Snapshot,
    TokenUsage, Usage,
};
use crate::tiering::AppliedMoves;

/// Timings kept per window, the size `ask` already keeps for its own.
const LATENCY_WINDOW: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tool {
    Recall,
    Ask,
}

impl Tool {
    const fn metrics(self) -> [MetricId; 2] {
        match self {
            Self::Recall => [MetricId::RecallLatencyP50Ms, MetricId::RecallLatencyP95Ms],
            Self::Ask => [MetricId::AskLatencyP50Ms, MetricId::AskLatencyP95Ms],
        }
    }
}

const ARM_METRICS: [MetricId; 2] = [MetricId::ArmLatencyP50Ms, MetricId::ArmLatencyP95Ms];

/// The model ids a run has seen and the running totals of calls made to them.
#[derive(Default)]
struct Models {
    catalog: ModelCatalog,
    meter: ModelMeter,
}

struct Windows {
    tools: BTreeMap<Tool, LatencyWindow>,
    arms: BTreeMap<RetrievalArm, LatencyWindow>,
}

/// Where the alert state is kept between runs, and what raises an alert.
struct Alerting {
    state_path: PathBuf,
    rules: RuleSet,
}

pub struct Metrics {
    sink: Arc<dyn MetricsSink>,
    clock: Arc<dyn Clock>,
    min_samples: usize,
    windows: Mutex<Windows>,
    models: Mutex<Models>,
    rollbacks: AtomicU64,
    alerting: Option<Alerting>,
    /// Held while a run publishes its readings and alerts, and by a reader of
    /// the snapshot, so a reader never sees one run's readings with the
    /// previous run's alerts.
    publication: tokio::sync::Mutex<()>,
}

impl Metrics {
    pub fn new(sink: Arc<dyn MetricsSink>, clock: Arc<dyn Clock>, min_samples: usize) -> Self {
        Self {
            sink,
            clock,
            min_samples,
            windows: Mutex::new(Windows {
                tools: BTreeMap::new(),
                arms: BTreeMap::new(),
            }),
            models: Mutex::new(Models::default()),
            rollbacks: AtomicU64::new(0),
            alerting: None,
            publication: tokio::sync::Mutex::new(()),
        }
    }

    /// Judges each run against the previous one under `rules`, keeping the
    /// previous values in `state_path`.
    pub fn with_alerts(mut self, state_path: PathBuf, rules: RuleSet) -> Self {
        self.alerting = Some(Alerting { state_path, rules });
        self
    }

    /// The readings and active alerts as they stood between two runs' publications.
    pub async fn snapshot(&self) -> Snapshot {
        let _published = self.publication.lock().await;
        self.sink.snapshot()
    }

    /// Publishes a finished run: its outcome, the readings taken at its end,
    /// and the alerts they raise. A failed restore keeps the alerts as they
    /// were, because the numbers of such a run are not judged.
    pub async fn finish_run(
        &self,
        outcome: Option<RunOutcome>,
        elapsed: Duration,
        readings: Vec<Sample>,
    ) {
        let _publishing = self.publication.lock().await;
        self.run_finished(outcome, elapsed);
        readings
            .into_iter()
            .for_each(|sample| self.sink.record(sample));
        self.raise_alerts(outcome == Some(RunOutcome::RestoreFailed));
    }

    fn raise_alerts(&self, restore_failed: bool) {
        let Some(alerting) = &self.alerting else {
            return;
        };
        let observations = observations_of(&self.sink.snapshot());
        let at = self.clock.now();
        match evaluate_run(
            &alerting.state_path,
            &observations,
            &alerting.rules,
            at,
            restore_failed,
        ) {
            Ok(_) if restore_failed => {}
            Ok(alerts) => self.sink.set_alerts(alerts),
            Err(error) => {
                tracing::error!(error = %format!("{error:#}"), "metrics: the alerts could not be judged");
            }
        }
    }

    pub fn record(&self, sample: Option<Sample>) {
        if let Some(sample) = sample {
            self.sink.record(sample);
        }
    }

    /// One finished model call. The samples carry the provider and the model
    /// id as labels and nothing else.
    pub fn model_call(
        &self,
        provider: Provider,
        model: &str,
        tokens: TokenUsage,
        batch: bool,
        price: Price,
    ) {
        let at = self.clock.now();
        let samples = {
            // A panic while counting leaves a total one call short at worst.
            let mut models = self.models.lock().unwrap_or_else(PoisonError::into_inner);
            match models.catalog.intern(model) {
                Ok(model) => models.meter.record(
                    Usage {
                        provider,
                        model,
                        tokens,
                        batch,
                    },
                    price,
                    at,
                ),
                Err(reason) => {
                    tracing::warn!(%reason, "metrics: a model call is not counted");
                    return;
                }
            }
        };
        samples
            .into_iter()
            .for_each(|sample| self.sink.record(sample));
    }

    fn windows(&self) -> std::sync::MutexGuard<'_, Windows> {
        // A panic while timing leaves a window one timing short at worst.
        self.windows.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// One finished `recall` or `ask`.
    pub fn record_tool(&self, tool: Tool, elapsed: Duration) {
        let now = self.clock.now();
        let mut windows = self.windows();
        let window = windows
            .tools
            .entry(tool)
            .or_insert_with(|| LatencyWindow::new(LATENCY_WINDOW));
        window.record(millis(elapsed));
        for sample in window.samples(tool.metrics(), Labels::None, now) {
            self.sink.record(sample);
        }
    }

    /// The per-arm timings of one query, for the arms it ran.
    pub fn record_arms(&self, timings: QueryTimings) {
        let now = self.clock.now();
        let mut windows = self.windows();
        let arms = [
            (RetrievalArm::Lexical, timings.lexical),
            (RetrievalArm::F32, timings.vector),
            (RetrievalArm::ColdProbe, timings.cold),
        ];
        for (arm, elapsed) in arms {
            let Some(elapsed) = elapsed else { continue };
            let window = windows
                .arms
                .entry(arm)
                .or_insert_with(|| LatencyWindow::new(LATENCY_WINDOW));
            window.record(millis(elapsed));
            for sample in window.samples(ARM_METRICS, Labels::Arm(arm), now) {
                self.sink.record(sample);
            }
        }
    }

    /// What the tier step moved, and how well utility predicted the verdicts.
    pub fn tier_report(&self, applied: &AppliedMoves, calibration: &[(f64, bool)]) {
        let now = self.clock.now();
        for (metric, count) in [
            (MetricId::ItemsPromoted, applied.promoted),
            (MetricId::ItemsReembedded, applied.reembedded),
        ] {
            self.sink.record(Sample {
                metric,
                value: count as f64,
                labels: Labels::None,
                at: now,
            });
        }
        self.record(collect::ece(calibration, self.min_samples, now));
    }

    pub fn probe_report(&self, report: &ProbeReport) {
        self.record(collect::probe_recall(report, self.clock.now()));
    }

    /// The end of a consolidation run. The rollback count only grows.
    fn run_finished(&self, outcome: Option<RunOutcome>, elapsed: Duration) {
        let now = self.clock.now();
        if matches!(
            outcome,
            Some(RunOutcome::RolledBack | RunOutcome::RestoreFailed)
        ) {
            self.rollbacks.fetch_add(1, Ordering::Relaxed);
        }
        for (metric, value) in [
            (MetricId::ConsolidationSeconds, elapsed.as_secs_f64()),
            (
                MetricId::RollbackCount,
                self.rollbacks.load(Ordering::Relaxed) as f64,
            ),
        ] {
            self.sink.record(Sample {
                metric,
                value,
                labels: Labels::None,
                at: now,
            });
        }
    }
}

/// Records how long a tool call took when it is dropped, so every early
/// return of a handler is timed.
pub struct ToolTimer {
    metrics: Arc<Metrics>,
    tool: Tool,
    started: std::time::Instant,
}

impl Drop for ToolTimer {
    fn drop(&mut self) {
        self.metrics.record_tool(self.tool, self.started.elapsed());
    }
}

impl Metrics {
    pub fn time(self: &Arc<Self>, tool: Tool) -> ToolTimer {
        ToolTimer {
            metrics: Arc::clone(self),
            tool,
            started: std::time::Instant::now(),
        }
    }
}

/// One observation per metric. Where a metric has a reading per label, the
/// largest value stands for it and the busiest reading gives the sample count.
fn observations_of(snapshot: &Snapshot) -> Vec<Observation> {
    let mut by_metric: BTreeMap<MetricId, Observation> = BTreeMap::new();
    for reading in &snapshot.readings {
        let samples = u32::try_from(reading.samples).unwrap_or(u32::MAX);
        by_metric
            .entry(reading.metric)
            .and_modify(|seen| {
                seen.value = seen.value.max(reading.value);
                seen.samples = seen.samples.max(samples);
            })
            .or_insert(Observation {
                metric: reading.metric,
                value: reading.value,
                samples,
            });
    }
    by_metric.into_values().collect()
}

fn millis(elapsed: Duration) -> f64 {
    elapsed.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests;
