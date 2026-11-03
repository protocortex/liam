// SPDX-License-Identifier: Apache-2.0
//! Alert rules, their evaluation against the previous run, and the state file
//! that remembers the previous run.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use liam_log::wal::{write_atomically, OsFileOps};
use liam_store::Millis;
use serde::{Deserialize, Serialize};

use super::sink::MetricId;

/// Appended to the database path to name the state file.
const STATE_FILE_SUFFIX: &str = ".metrics-state.json";
const FORMAT_VERSION: u32 = 1;
const PERCENT: f64 = 100.0;
const MILLIS_PER_HOUR: f64 = 3_600_000.0;

pub fn state_path_for(database_path: &Path) -> PathBuf {
    let mut path = database_path.as_os_str().to_owned();
    path.push(STATE_FILE_SUFFIX);
    PathBuf::from(path)
}

/// When a metric raises an alert. Every bound is inclusive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rule {
    /// The value moved at least `pct` percent from the previous run's.
    Delta { pct: f64, min_samples: u32 },
    /// The value reached `ceiling`, whatever the previous run held.
    Absolute { ceiling: f64 },
    /// The value grew at least `per_hour` an hour since the previous run.
    Rate { per_hour: f64, min_samples: u32 },
}

/// A metric's value at the end of a run and the samples behind it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    pub metric: MetricId,
    pub value: f64,
    pub samples: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Alert {
    pub metric: MetricId,
    pub rule: Rule,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Previous {
    value: f64,
    at: i64,
}

/// What the previous run left, to compare the next with.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertState {
    version: u32,
    values: BTreeMap<MetricId, Previous>,
}

/// A loaded state and how many warnings loading it cost.
#[derive(Debug)]
pub struct LoadedState {
    pub state: AlertState,
    pub warnings: u32,
}

impl AlertState {
    /// The state saved at `path`. A missing file is a first run. A file that
    /// cannot be read or is of another version is dropped for a fresh state
    /// and counted as a warning, because an alert state must never stop a run.
    pub fn load(path: &Path) -> LoadedState {
        let fresh = |warnings| LoadedState {
            state: Self {
                version: FORMAT_VERSION,
                values: BTreeMap::new(),
            },
            warnings,
        };
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return fresh(0),
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "the metrics state could not be read");
                return fresh(1);
            }
        };
        match serde_json::from_str::<Self>(&text) {
            Ok(state) if state.version == FORMAT_VERSION => LoadedState { state, warnings: 0 },
            Ok(state) => {
                tracing::warn!(path = %path.display(), version = state.version, "the metrics state is of another version");
                fresh(1)
            }
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "the metrics state is corrupt");
                fresh(1)
            }
        }
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let bytes = serde_json::to_vec_pretty(self)?;
        write_atomically(path, &bytes, &mut OsFileOps).map_err(|source| {
            anyhow::anyhow!(
                "could not save the metrics state {}: {source}",
                path.display()
            )
        })
    }

    /// Takes `observations` as the state the next run compares with.
    fn remember(&mut self, observations: &[Observation], at: Millis) {
        self.version = FORMAT_VERSION;
        for observation in observations {
            if observation.value.is_finite() {
                self.values.insert(
                    observation.metric,
                    Previous {
                        value: observation.value,
                        at: at.0,
                    },
                );
            }
        }
    }
}

/// The rules of each metric.
pub type RuleSet = BTreeMap<MetricId, Vec<Rule>>;

/// The alerts `observations` raise against `previous` under `rules`. A metric
/// with no previous value, a previous value of 0 (a relative change from 0 is
/// undefined), too few samples, or a value that is not finite raises nothing
/// from a rule that needs them.
pub fn evaluate(
    previous: &AlertState,
    observations: &[Observation],
    rules: &RuleSet,
    at: Millis,
) -> Vec<Alert> {
    let mut alerts = Vec::new();
    for observation in observations {
        if !observation.value.is_finite() {
            continue;
        }
        let before = previous.values.get(&observation.metric);
        for rule in rules.get(&observation.metric).into_iter().flatten() {
            if fires(*rule, observation, before, at) {
                alerts.push(Alert {
                    metric: observation.metric,
                    rule: *rule,
                    value: observation.value,
                });
            }
        }
    }
    alerts
}

fn fires(rule: Rule, now: &Observation, before: Option<&Previous>, at: Millis) -> bool {
    match rule {
        Rule::Absolute { ceiling } => now.value >= ceiling,
        Rule::Delta { pct, min_samples } => {
            let Some(before) = before.filter(|before| before.value != 0.0) else {
                return false;
            };
            now.samples >= min_samples
                && (now.value - before.value).abs() * PERCENT >= pct * before.value.abs()
        }
        Rule::Rate {
            per_hour,
            min_samples,
        } => {
            let Some(before) = before else {
                return false;
            };
            let hours = (at.0 - before.at) as f64 / MILLIS_PER_HOUR;
            now.samples >= min_samples
                && hours > 0.0
                && (now.value - before.value) / hours >= per_hour
        }
    }
}

/// Evaluates a finished run and, unless it ended in a failed restore, saves
/// `observations` as the next run's previous values. A failed restore leaves
/// the store in an unknown state, so its numbers are neither judged nor kept.
///
/// # Errors
///
/// Fails when the new state cannot be written.
pub fn evaluate_run(
    state_path: &Path,
    observations: &[Observation],
    rules: &RuleSet,
    at: Millis,
    restore_failed: bool,
) -> anyhow::Result<Vec<Alert>> {
    if restore_failed {
        return Ok(Vec::new());
    }
    let LoadedState { mut state, .. } = AlertState::load(state_path);
    let alerts = evaluate(&state, observations, rules, at);
    state.remember(observations, at);
    state.save(state_path)?;
    Ok(alerts)
}

#[cfg(test)]
mod tests;
