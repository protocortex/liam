// SPDX-License-Identifier: Apache-2.0
//! The `[metrics]` section: whether the daemon keeps metrics, where an active
//! alert is reported, and the thresholds that raise one. The numeric defaults
//! are starting values, not measured on real data.

use serde::Deserialize;
#[cfg(test)]
use serde::Serialize;

const DEFAULT_MIN_SAMPLES: u32 = 5;
const DEFAULT_DELTA_PCT: f64 = 20.0;
const DEFAULT_ROLLBACK_CEILING: f64 = 1.0;
const DEFAULT_EXTRACTION_BACKLOG_CEILING: f64 = 10_000.0;
const DEFAULT_MODEL_CALLS_PER_HOUR: f64 = 1_000.0;

/// Where metrics and active alerts go besides the snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(rename_all = "lowercase")]
pub enum SinkKind {
    /// Held in memory for the `metrics` tool.
    Memory,
    /// Held in memory, and each active alert is also logged at error level.
    Log,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(default, deny_unknown_fields)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub sink: SinkKind,
    /// Samples a metric needs before a relative or rate rule can raise an alert.
    pub min_samples: u32,
    /// Percent a probe recall, latency, `ece` or Bloom rate may move between
    /// runs before an alert.
    pub delta_pct: f64,
    /// Rollbacks in the run window at which an alert is raised.
    pub rollback_ceiling: f64,
    /// Events past an extraction cursor at which an alert is raised.
    pub extraction_backlog_ceiling: f64,
    /// Model calls per hour at which an alert is raised.
    pub model_calls_per_hour: f64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sink: SinkKind::Memory,
            min_samples: DEFAULT_MIN_SAMPLES,
            delta_pct: DEFAULT_DELTA_PCT,
            rollback_ceiling: DEFAULT_ROLLBACK_CEILING,
            extraction_backlog_ceiling: DEFAULT_EXTRACTION_BACKLOG_CEILING,
            model_calls_per_hour: DEFAULT_MODEL_CALLS_PER_HOUR,
        }
    }
}

impl MetricsConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.min_samples > 0, "metrics.min_samples must be above 0");
        for (key, value) in [
            ("delta_pct", self.delta_pct),
            ("rollback_ceiling", self.rollback_ceiling),
            (
                "extraction_backlog_ceiling",
                self.extraction_backlog_ceiling,
            ),
            ("model_calls_per_hour", self.model_calls_per_hour),
        ] {
            anyhow::ensure!(
                value.is_finite() && value >= 0.0,
                "metrics.{key} must be 0 or more, got {value}"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn load(toml: &str) -> anyhow::Result<Config> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("liam.toml");
        std::fs::write(&path, toml).unwrap();
        Config::load(&path)
    }

    #[test]
    fn a_config_without_the_section_keeps_metrics_in_memory() {
        // Act
        let config = load("[gc]\n").unwrap();

        // Assert
        assert_eq!(config.metrics, MetricsConfig::default());
        assert!(config.metrics.enabled);
        assert_eq!(config.metrics.sink, SinkKind::Memory);
    }

    #[test]
    fn a_negative_threshold_a_zero_sample_count_and_an_unknown_sink_or_key_fail() {
        for (toml, mention) in [
            ("delta_pct = -1.0", "metrics.delta_pct"),
            ("rollback_ceiling = -0.5", "metrics.rollback_ceiling"),
            ("model_calls_per_hour = nan", "metrics.model_calls_per_hour"),
            ("min_samples = 0", "metrics.min_samples"),
            ("sink = \"file\"", "file"),
            ("colour = \"red\"", "colour"),
        ] {
            // Act
            let error = load(&format!("[metrics]\n{toml}\n")).expect_err(toml);

            // Assert
            assert!(format!("{error:#}").contains(mention), "{toml}: {error:#}");
        }
    }

    #[test]
    fn the_log_sink_is_accepted() {
        // Act
        let config = load("[metrics]\nsink = \"log\"\n").unwrap();

        // Assert
        assert_eq!(config.metrics.sink, SinkKind::Log);
    }
}
