// SPDX-License-Identifier: Apache-2.0
//! The daemon's side of the metrics: what the handlers and a consolidation run
//! report, turned into samples in the sink. Nothing here writes to disk, so
//! the read path stays read-only.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use liam_daemon::consolidation::manifest::RunOutcome;
use liam_daemon::consolidation::probes::ProbeReport;
use liam_store::{Clock, QueryTimings};

use super::collect::{self, LatencyWindow};
use super::sink::{Labels, MetricId, MetricsSink, RetrievalArm, Sample};
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

struct Windows {
    tools: BTreeMap<Tool, LatencyWindow>,
    arms: BTreeMap<RetrievalArm, LatencyWindow>,
}

pub struct Metrics {
    sink: Arc<dyn MetricsSink>,
    clock: Arc<dyn Clock>,
    min_samples: usize,
    windows: Mutex<Windows>,
    rollbacks: AtomicU64,
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
            rollbacks: AtomicU64::new(0),
        }
    }

    pub fn sink(&self) -> &Arc<dyn MetricsSink> {
        &self.sink
    }

    pub fn record(&self, sample: Option<Sample>) {
        if let Some(sample) = sample {
            self.sink.record(sample);
        }
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
    pub fn run_finished(&self, outcome: Option<RunOutcome>, elapsed: Duration) {
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

fn millis(elapsed: Duration) -> f64 {
    elapsed.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests;
