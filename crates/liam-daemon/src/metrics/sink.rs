// SPDX-License-Identifier: Apache-2.0
//! Samples, the closed labels they carry, and the sinks that hold them.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use liam_store::{Clock, Millis};
use serde::{Deserialize, Serialize};

use super::alerts::Alert;
pub use crate::config::{Provider, TokenUsage};
use crate::degraded::Degraded;

/// A tool whose replies the daemon times and counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tool {
    Recall,
    Ask,
}

impl Tool {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Recall => "recall",
            Self::Ask => "ask",
        }
    }

    pub(crate) const fn metrics(self) -> [MetricId; 2] {
        match self {
            Self::Recall => [MetricId::RecallLatencyP50Ms, MetricId::RecallLatencyP95Ms],
            Self::Ask => [MetricId::AskLatencyP50Ms, MetricId::AskLatencyP95Ms],
        }
    }
}

/// A step of `recall` and `ask` that is timed on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    Embed,
    StoreSearch,
    Rerank,
    RecordUsage,
    IssueHandles,
}

impl Stage {
    /// Every stage, in the order a report lists them.
    #[cfg(test)]
    pub const ALL: [Self; 5] = [
        Self::Embed,
        Self::StoreSearch,
        Self::Rerank,
        Self::RecordUsage,
        Self::IssueHandles,
    ];

    /// The name its metrics carry between `stage_` and `_p50_ms`.
    #[cfg(test)]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::StoreSearch => "store_search",
            Self::Rerank => "rerank",
            Self::RecordUsage => "record_usage",
            Self::IssueHandles => "issue_handles",
        }
    }

    pub(crate) const fn metrics(self) -> [MetricId; 2] {
        match self {
            Self::Embed => [MetricId::StageEmbedP50Ms, MetricId::StageEmbedP95Ms],
            Self::StoreSearch => [
                MetricId::StageStoreSearchP50Ms,
                MetricId::StageStoreSearchP95Ms,
            ],
            Self::Rerank => [MetricId::StageRerankP50Ms, MetricId::StageRerankP95Ms],
            Self::RecordUsage => [
                MetricId::StageRecordUsageP50Ms,
                MetricId::StageRecordUsageP95Ms,
            ],
            Self::IssueHandles => [
                MetricId::StageIssueHandlesP50Ms,
                MetricId::StageIssueHandlesP95Ms,
            ],
        }
    }
}

/// Every metric the daemon keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MetricId {
    ProbeRecallAt10,
    ArmLatencyP50Ms,
    ArmLatencyP95Ms,
    RecallLatencyP50Ms,
    RecallLatencyP95Ms,
    AskLatencyP50Ms,
    AskLatencyP95Ms,
    TierBytes,
    LogBytes,
    ArchiveBytes,
    PeakRssBytes,
    ConsolidationSeconds,
    RollbackCount,
    ItemsPromoted,
    ItemsReembedded,
    BloomFalsePositiveRate,
    Ece,
    RulesExtractionBacklog,
    ModelExtractionBacklog,
    ModelCalls,
    ModelCostMicroDollars,
    CacheHitRate,
    DeletionBacklogCount,
    DeletionBacklogAgeDays,
    DegradedReplies,
    StageEmbedP50Ms,
    StageEmbedP95Ms,
    StageStoreSearchP50Ms,
    StageStoreSearchP95Ms,
    StageRerankP50Ms,
    StageRerankP95Ms,
    StageRecordUsageP50Ms,
    StageRecordUsageP95Ms,
    StageIssueHandlesP50Ms,
    StageIssueHandlesP95Ms,
}

impl MetricId {
    /// Every metric, in declaration order.
    pub const ALL: [Self; 35] = [
        Self::ProbeRecallAt10,
        Self::ArmLatencyP50Ms,
        Self::ArmLatencyP95Ms,
        Self::RecallLatencyP50Ms,
        Self::RecallLatencyP95Ms,
        Self::AskLatencyP50Ms,
        Self::AskLatencyP95Ms,
        Self::TierBytes,
        Self::LogBytes,
        Self::ArchiveBytes,
        Self::PeakRssBytes,
        Self::ConsolidationSeconds,
        Self::RollbackCount,
        Self::ItemsPromoted,
        Self::ItemsReembedded,
        Self::BloomFalsePositiveRate,
        Self::Ece,
        Self::RulesExtractionBacklog,
        Self::ModelExtractionBacklog,
        Self::ModelCalls,
        Self::ModelCostMicroDollars,
        Self::CacheHitRate,
        Self::DeletionBacklogCount,
        Self::DeletionBacklogAgeDays,
        Self::DegradedReplies,
        Self::StageEmbedP50Ms,
        Self::StageEmbedP95Ms,
        Self::StageStoreSearchP50Ms,
        Self::StageStoreSearchP95Ms,
        Self::StageRerankP50Ms,
        Self::StageRerankP95Ms,
        Self::StageRecordUsageP50Ms,
        Self::StageRecordUsageP95Ms,
        Self::StageIssueHandlesP50Ms,
        Self::StageIssueHandlesP95Ms,
    ];

    /// The metric called `name`.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|metric| metric.name() == name)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::ProbeRecallAt10 => "probe_recall_at_10",
            Self::ArmLatencyP50Ms => "arm_latency_p50_ms",
            Self::ArmLatencyP95Ms => "arm_latency_p95_ms",
            Self::RecallLatencyP50Ms => "recall_latency_p50_ms",
            Self::RecallLatencyP95Ms => "recall_latency_p95_ms",
            Self::AskLatencyP50Ms => "ask_latency_p50_ms",
            Self::AskLatencyP95Ms => "ask_latency_p95_ms",
            Self::TierBytes => "tier_bytes",
            Self::LogBytes => "log_bytes",
            Self::ArchiveBytes => "archive_bytes",
            Self::PeakRssBytes => "peak_rss_bytes",
            Self::ConsolidationSeconds => "consolidation_seconds",
            Self::RollbackCount => "rollback_count",
            Self::ItemsPromoted => "items_promoted",
            Self::ItemsReembedded => "items_reembedded",
            Self::BloomFalsePositiveRate => "bloom_false_positive_rate",
            Self::Ece => "ece",
            Self::RulesExtractionBacklog => "rules_extraction_backlog",
            Self::ModelExtractionBacklog => "model_extraction_backlog",
            Self::ModelCalls => "model_calls",
            Self::ModelCostMicroDollars => "model_cost_micro_dollars",
            Self::CacheHitRate => "cache_hit_rate",
            Self::DeletionBacklogCount => "deletion_backlog_count",
            Self::DeletionBacklogAgeDays => "deletion_backlog_age_days",
            Self::DegradedReplies => "degraded_replies",
            Self::StageEmbedP50Ms => "stage_embed_p50_ms",
            Self::StageEmbedP95Ms => "stage_embed_p95_ms",
            Self::StageStoreSearchP50Ms => "stage_store_search_p50_ms",
            Self::StageStoreSearchP95Ms => "stage_store_search_p95_ms",
            Self::StageRerankP50Ms => "stage_rerank_p50_ms",
            Self::StageRerankP95Ms => "stage_rerank_p95_ms",
            Self::StageRecordUsageP50Ms => "stage_record_usage_p50_ms",
            Self::StageRecordUsageP95Ms => "stage_record_usage_p95_ms",
            Self::StageIssueHandlesP50Ms => "stage_issue_handles_p50_ms",
            Self::StageIssueHandlesP95Ms => "stage_issue_handles_p95_ms",
        }
    }
}

impl fmt::Display for MetricId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TierLabel {
    Hot,
    Cold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RetrievalArm {
    F32,
    Lexical,
    ColdProbe,
}

/// A model id from the pinned set, held as an index into a [`ModelCatalog`], so
/// a label stays `Copy` and cannot carry free text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelId(u16);

/// The model ids a run has seen, each interned once.
#[derive(Debug, Default)]
pub struct ModelCatalog {
    names: Vec<String>,
}

const MAX_MODEL_NAME_CHARS: usize = 64;
const MAX_MODELS: usize = 64;

impl ModelCatalog {
    /// The id of `name`, added when new.
    ///
    /// # Errors
    ///
    /// Fails for a name that is empty, long, or has a character outside
    /// letters, digits and `.-_:/`, and when the catalog is full.
    pub fn intern(&mut self, name: &str) -> Result<ModelId, String> {
        let valid = !name.is_empty()
            && name.chars().count() <= MAX_MODEL_NAME_CHARS
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || ".-_:/".contains(c));
        if !valid {
            return Err(format!(
                "a model id is 1 to {MAX_MODEL_NAME_CHARS} letters, digits or .-_:/, got {name:?}"
            ));
        }
        if let Some(index) = self.names.iter().position(|known| known == name) {
            return Ok(ModelId(index as u16));
        }
        if self.names.len() >= MAX_MODELS {
            return Err(format!("the catalog already holds {MAX_MODELS} model ids"));
        }
        self.names.push(name.to_owned());
        Ok(ModelId((self.names.len() - 1) as u16))
    }

    #[cfg(test)]
    pub fn name(&self, id: ModelId) -> Option<&str> {
        self.names.get(usize::from(id.0)).map(String::as_str)
    }
}

/// What a model call used, as an adapter reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    pub provider: Provider,
    pub model: ModelId,
    pub tokens: TokenUsage,
    /// Billed at the provider's batch rates.
    pub batch: bool,
}

/// The only dimensions a sample may carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Labels {
    None,
    Tier(TierLabel),
    Arm(RetrievalArm),
    Model { provider: Provider, model: ModelId },
    Degraded { tool: Tool, reason: Degraded },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub metric: MetricId,
    pub value: f64,
    pub labels: Labels,
    pub at: Millis,
}

/// The latest sample of one metric and label set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reading {
    pub metric: MetricId,
    pub labels: Labels,
    pub value: f64,
    pub at: Millis,
    /// Samples recorded for it since the sink was made.
    pub samples: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    pub taken_at: Millis,
    pub readings: Vec<Reading>,
    pub alerts: Vec<Alert>,
    /// Samples refused for a value that was not finite.
    pub rejected: u64,
}

pub trait MetricsSink: Send + Sync {
    /// Keeps `sample`, or counts it as rejected when its value is not finite.
    fn record(&self, sample: Sample);
    /// Replaces the active alerts with `alerts`.
    fn set_alerts(&self, alerts: Vec<Alert>);
    fn snapshot(&self) -> Snapshot;
}

#[derive(Default)]
struct Held {
    readings: BTreeMap<(MetricId, Labels), Reading>,
    alerts: Vec<Alert>,
    rejected: u64,
}

/// Keeps the latest sample of every metric and label set.
pub struct MemorySink {
    clock: Arc<dyn Clock>,
    held: Mutex<Held>,
}

impl MemorySink {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            clock,
            held: Mutex::new(Held::default()),
        }
    }

    fn held(&self) -> MutexGuard<'_, Held> {
        // A panic while recording leaves a reading at worst one sample behind.
        self.held.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl MetricsSink for MemorySink {
    fn record(&self, sample: Sample) {
        let mut held = self.held();
        if !sample.value.is_finite() {
            held.rejected += 1;
            return;
        }
        let key = (sample.metric, sample.labels);
        let samples = held.readings.get(&key).map_or(0, |reading| reading.samples) + 1;
        held.readings.insert(
            key,
            Reading {
                metric: sample.metric,
                labels: sample.labels,
                value: sample.value,
                at: sample.at,
                samples,
            },
        );
    }

    fn set_alerts(&self, alerts: Vec<Alert>) {
        self.held().alerts = alerts;
    }

    fn snapshot(&self) -> Snapshot {
        let held = self.held();
        Snapshot {
            taken_at: self.clock.now(),
            readings: held.readings.values().copied().collect(),
            alerts: held.alerts.clone(),
            rejected: held.rejected,
        }
    }
}

/// A [`MemorySink`] that also logs each active alert at error level.
pub struct LogSink {
    inner: MemorySink,
}

impl LogSink {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            inner: MemorySink::new(clock),
        }
    }
}

impl MetricsSink for LogSink {
    fn record(&self, sample: Sample) {
        self.inner.record(sample);
    }

    fn set_alerts(&self, alerts: Vec<Alert>) {
        for alert in &alerts {
            tracing::error!(
                metric = alert.metric.name(),
                value = alert.value,
                rule = ?alert.rule,
                "metric alert"
            );
        }
        self.inner.set_alerts(alerts);
    }

    fn snapshot(&self) -> Snapshot {
        self.inner.snapshot()
    }
}

#[cfg(test)]
mod tests;
