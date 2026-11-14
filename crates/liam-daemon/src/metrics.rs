// SPDX-License-Identifier: Apache-2.0
//! Metrics the daemon keeps about itself, and the alerts raised from them. A
//! sample carries a closed set of labels and never a subject, a producer or a
//! lineage key, so keeping metrics cannot defeat a delete.

pub mod alerts;
pub mod collect;
pub mod memory;
pub mod monotonic;
pub mod recorder;
pub mod sink;
pub mod sources;

use std::path::Path;
use std::sync::Arc;

use alerts::{Rule, RuleSet};
use liam_store::SystemClock;
use recorder::Metrics;
use sink::{LogSink, MemorySink, MetricId, MetricsSink};

use crate::config::{MetricsConfig, SinkKind};

/// The recorder over the sink `config` names, judging each run against the
/// previous one with the state kept beside `database_path`.
pub fn build(config: &MetricsConfig, database_path: &Path, deadline_days: u32) -> Arc<Metrics> {
    let clock = Arc::new(SystemClock);
    let sink: Arc<dyn MetricsSink> = match config.sink {
        SinkKind::Memory => Arc::new(MemorySink::new(clock.clone())),
        SinkKind::Log => Arc::new(LogSink::new(clock.clone())),
    };
    let metrics = Metrics::new(sink, clock, config.min_samples as usize).with_alerts(
        alerts::state_path_for(database_path),
        rules(config, deadline_days),
    );
    Arc::new(metrics)
}

/// The share of a delete's deadline after which the backlog is warned about.
const DELETION_WARNING_SHARE: f64 = 0.8;

/// The metrics that alert on a relative move between runs.
const DELTA_METRICS: [MetricId; 6] = [
    MetricId::ProbeRecallAt10,
    MetricId::ArmLatencyP95Ms,
    MetricId::RecallLatencyP95Ms,
    MetricId::AskLatencyP95Ms,
    MetricId::Ece,
    MetricId::BloomFalsePositiveRate,
];

/// The alert rules `config` asks for. The oldest delete job alerts at 80
/// percent of `deadline_days` and again at the deadline.
pub fn rules(config: &MetricsConfig, deadline_days: u32) -> RuleSet {
    let mut rules = RuleSet::new();
    for metric in DELTA_METRICS {
        rules.insert(
            metric,
            vec![Rule::Delta {
                pct: config.delta_pct,
                min_samples: config.min_samples,
            }],
        );
    }
    rules.insert(
        MetricId::RollbackCount,
        vec![Rule::Absolute {
            ceiling: config.rollback_ceiling,
        }],
    );
    for metric in [
        MetricId::RulesExtractionBacklog,
        MetricId::ModelExtractionBacklog,
    ] {
        rules.insert(
            metric,
            vec![Rule::Absolute {
                ceiling: config.extraction_backlog_ceiling,
            }],
        );
    }
    rules.insert(
        MetricId::ModelCalls,
        vec![Rule::Rate {
            per_hour: config.model_calls_per_hour,
            min_samples: config.min_samples,
        }],
    );
    let deadline = f64::from(deadline_days);
    rules.insert(
        MetricId::DeletionBacklogAgeDays,
        vec![
            Rule::Absolute {
                ceiling: deadline * DELETION_WARNING_SHARE,
            },
            Rule::Absolute { ceiling: deadline },
        ],
    );
    rules
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_deletion_backlog_warns_at_80_percent_of_the_deadline_and_alerts_at_it() {
        // Act
        let rules = rules(&MetricsConfig::default(), 30);

        // Assert
        assert_eq!(
            rules[&MetricId::DeletionBacklogAgeDays],
            [
                Rule::Absolute { ceiling: 24.0 },
                Rule::Absolute { ceiling: 30.0 }
            ]
        );
    }

    #[test]
    fn each_kind_of_rule_comes_from_its_config_key() {
        // Arrange
        let config = MetricsConfig {
            delta_pct: 7.0,
            min_samples: 3,
            rollback_ceiling: 2.0,
            extraction_backlog_ceiling: 50.0,
            model_calls_per_hour: 9.0,
            ..MetricsConfig::default()
        };

        // Act
        let rules = rules(&config, 30);

        // Assert
        assert_eq!(
            rules[&MetricId::ProbeRecallAt10],
            [Rule::Delta {
                pct: 7.0,
                min_samples: 3
            }]
        );
        assert_eq!(
            rules[&MetricId::RollbackCount],
            [Rule::Absolute { ceiling: 2.0 }]
        );
        assert_eq!(
            rules[&MetricId::ModelExtractionBacklog],
            [Rule::Absolute { ceiling: 50.0 }]
        );
        assert_eq!(
            rules[&MetricId::ModelCalls],
            [Rule::Rate {
                per_hour: 9.0,
                min_samples: 3
            }]
        );
    }
}
