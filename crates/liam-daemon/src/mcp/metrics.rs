// SPDX-License-Identifier: Apache-2.0
//! The `metrics` tool: the latest reading of each metric and the alerts that
//! are active. It reads the in-memory sink and changes nothing.

use std::fmt::Write;

use schemars::JsonSchema;
use serde::Deserialize;

use super::MemoryServer;
use crate::metrics::alerts::{Alert, Rule};
use crate::metrics::sink::{Labels, MetricId, Reading, Snapshot};

/// The most readings one call returns.
pub const MAX_LIMIT: usize = 200;
const NOT_ENABLED: &str = "metrics are not enabled on this server";

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MetricsArgs {
    /// Only this metric, by name, for example `probe_recall_at_10`.
    pub metric: Option<String>,
    /// The most readings to return. Absent or above 200 means 200; 0 returns
    /// none.
    pub limit: Option<usize>,
}

pub(super) async fn read(server: &MemoryServer, args: MetricsArgs) -> String {
    let Some(metrics) = &server.metrics else {
        return format!("metrics failed: {NOT_ENABLED}");
    };
    let filter = match args.metric.as_deref().map(|name| {
        MetricId::from_name(name).ok_or_else(|| {
            let known: Vec<&str> = MetricId::ALL.iter().map(|metric| metric.name()).collect();
            format!(
                "metrics failed: unknown metric {name:?}; known metrics are {}",
                known.join(", ")
            )
        })
    }) {
        Some(Ok(metric)) => Some(metric),
        Some(Err(message)) => return message,
        None => None,
    };
    let limit = args.limit.unwrap_or(MAX_LIMIT).min(MAX_LIMIT);
    render(&metrics.snapshot().await, filter, limit)
}

fn render(snapshot: &Snapshot, filter: Option<MetricId>, limit: usize) -> String {
    if limit == 0 {
        return "no readings requested: limit is 0".to_owned();
    }
    let wanted = |metric: MetricId| filter.is_none_or(|only| only == metric);
    let readings: Vec<&Reading> = snapshot
        .readings
        .iter()
        .filter(|reading| wanted(reading.metric))
        .take(limit)
        .collect();
    let alerts: Vec<&Alert> = snapshot
        .alerts
        .iter()
        .filter(|alert| wanted(alert.metric))
        .collect();
    let mut text = String::new();
    if readings.is_empty() {
        text.push_str("no readings yet\n");
    }
    for reading in readings {
        let _ = writeln!(
            text,
            "{}{} = {} (samples {})",
            reading.metric,
            label_suffix(reading.labels),
            reading.value,
            reading.samples
        );
    }
    if alerts.is_empty() {
        text.push_str("alerts: none");
    } else {
        text.push_str("alerts:");
        for alert in alerts {
            let _ = write!(
                text,
                "\n{} = {}: {}",
                alert.metric,
                alert.value,
                describe(alert.rule)
            );
        }
    }
    text
}

fn label_suffix(labels: Labels) -> String {
    match labels {
        Labels::None => String::new(),
        Labels::Tier(tier) => format!("{{tier={tier:?}}}").to_lowercase(),
        Labels::Arm(arm) => format!("{{arm={arm:?}}}").to_lowercase(),
        Labels::Provider(provider) => format!("{{provider={provider:?}}}").to_lowercase(),
        Labels::Model { provider, model } => {
            format!("{{provider={provider:?},model={model:?}}}").to_lowercase()
        }
    }
}

fn describe(rule: Rule) -> String {
    match rule {
        Rule::Delta { pct, .. } => format!("moved at least {pct} percent since the last run"),
        Rule::Absolute { ceiling } => format!("reached {ceiling}"),
        Rule::Rate { per_hour, .. } => format!("grew at least {per_hour} an hour"),
    }
}
