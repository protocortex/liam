// SPDX-License-Identifier: Apache-2.0
//! Turns what the daemon already knows into samples. Each collector is a pure
//! function of counts and timings, and returns nothing when it has no data, so
//! an empty window or a zero denominator never reports a made-up zero.

use std::collections::{BTreeMap, VecDeque};
use std::io;
use std::path::Path;

use liam_store::{Millis, TierBytes};

use super::memory::ProcessMemory;
use super::sink::{Labels, MetricId, Sample, TierLabel, Usage};
use crate::deletion::DeleteJob;
use liam_daemon::consolidation::probes::ProbeReport;

const MILLIS_PER_DAY: f64 = 86_400_000.0;
const CALIBRATION_BINS: usize = 10;
const P50: u32 = 50;
const P95: u32 = 95;

fn sample(metric: MetricId, value: f64, at: Millis) -> Sample {
    Sample {
        metric,
        value,
        labels: Labels::None,
        at,
    }
}

fn ratio(part: u64, whole: u64) -> Option<f64> {
    (whole > 0).then(|| part as f64 / whole as f64)
}

/// The most recent timings of one operation, in milliseconds.
#[derive(Debug)]
pub struct LatencyWindow {
    capacity: usize,
    timings: VecDeque<f64>,
}

impl LatencyWindow {
    /// Keeps at least one timing, whatever `capacity` says.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            timings: VecDeque::with_capacity(capacity),
        }
    }

    pub fn record(&mut self, millis: f64) {
        if self.timings.len() == self.capacity {
            self.timings.pop_front();
        }
        self.timings.push_back(millis);
    }

    /// The nearest-rank percentile: the value at rank `ceil(p / 100 * n)`.
    pub fn percentile(&self, percent: u32) -> Option<f64> {
        if self.timings.is_empty() {
            return None;
        }
        let mut sorted: Vec<f64> = self.timings.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let rank = (usize::try_from(percent).ok()? * sorted.len()).div_ceil(100);
        sorted.get(rank.max(1) - 1).copied()
    }

    /// The p50 and p95 samples under `labels`, none for an empty window.
    pub fn samples(&self, metrics: [MetricId; 2], labels: Labels, at: Millis) -> Vec<Sample> {
        [(metrics[0], P50), (metrics[1], P95)]
            .into_iter()
            .filter_map(|(metric, percent)| {
                Some(Sample {
                    labels,
                    ..sample(metric, self.percentile(percent)?, at)
                })
            })
            .collect()
    }
}

/// Expected calibration error: the bin-weighted gap between the confidence
/// a prediction stated and how often it was right. `None` below `min_pairs`
/// pairs, where a bin average says nothing.
pub fn ece(pairs: &[(f64, bool)], min_pairs: usize, at: Millis) -> Option<Sample> {
    if pairs.is_empty() || pairs.len() < min_pairs {
        return None;
    }
    let mut bins = [(0u64, 0.0f64, 0u64); CALIBRATION_BINS];
    for &(confidence, correct) in pairs {
        let last = CALIBRATION_BINS - 1;
        let index = ((confidence.clamp(0.0, 1.0) * CALIBRATION_BINS as f64) as usize).min(last);
        bins[index].0 += 1;
        bins[index].1 += confidence;
        bins[index].2 += u64::from(correct);
    }
    let total = pairs.len() as f64;
    let error: f64 = bins
        .iter()
        .filter(|(count, ..)| *count > 0)
        .map(|&(count, confidence, correct)| {
            let count = count as f64;
            count / total * (correct as f64 / count - confidence / count).abs()
        })
        .sum();
    Some(sample(MetricId::Ece, error, at))
}

/// Events the log holds past an extractor's cursor.
pub fn extraction_backlog(metric: MetricId, unread: u64, at: Millis) -> Sample {
    sample(metric, unread as f64, at)
}

/// How many delete jobs are open and how old the oldest is. A failed job is
/// open; a finished or cancelled one is not.
pub fn deletion_backlog(jobs: &[DeleteJob], now: Millis) -> Vec<Sample> {
    let open: Vec<&DeleteJob> = jobs.iter().filter(|job| !job.state.is_final()).collect();
    let mut samples = vec![sample(
        MetricId::DeletionBacklogCount,
        open.len() as f64,
        now,
    )];
    if let Some(oldest) = open.iter().map(|job| job.requested_at).min() {
        let age = (now.0 - oldest).max(0) as f64 / MILLIS_PER_DAY;
        samples.push(sample(MetricId::DeletionBacklogAgeDays, age, now));
    }
    samples
}

#[allow(dead_code, reason = "the hosted adapters report model usage")]
/// The price of a model's tokens, in millionths of a dollar per million tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Price {
    pub input_micro_per_mtok: u64,
    pub output_micro_per_mtok: u64,
}

#[allow(dead_code, reason = "the hosted adapters report model usage")]
const TOKENS_PER_MTOK: u64 = 1_000_000;

#[allow(dead_code, reason = "the hosted adapters report model usage")]
/// Running totals of calls and cost per provider and model. The alert rules
/// read growth between runs, so a sample carries the total, not one call.
#[derive(Debug, Default)]
pub struct ModelMeter {
    totals: BTreeMap<Labels, (u64, u64)>,
}

#[allow(dead_code, reason = "the hosted adapters report model usage")]
impl ModelMeter {
    pub fn record(&mut self, usage: Usage, price: Price, at: Millis) -> [Sample; 2] {
        let labels = Labels::Model {
            provider: usage.provider,
            model: usage.model,
        };
        let cost = (u128::from(usage.input_tokens) * u128::from(price.input_micro_per_mtok)
            + u128::from(usage.output_tokens) * u128::from(price.output_micro_per_mtok))
            / u128::from(TOKENS_PER_MTOK);
        let total = self.totals.entry(labels).or_default();
        total.0 += 1;
        total.1 = total
            .1
            .saturating_add(u64::try_from(cost).unwrap_or(u64::MAX));
        [
            Sample {
                metric: MetricId::ModelCalls,
                value: total.0 as f64,
                labels,
                at,
            },
            Sample {
                metric: MetricId::ModelCostMicroDollars,
                value: total.1 as f64,
                labels,
                at,
            },
        ]
    }
}

pub fn cache_hit_rate(hits: u64, misses: u64, at: Millis) -> Option<Sample> {
    Some(sample(
        MetricId::CacheHitRate,
        ratio(hits, hits + misses)?,
        at,
    ))
}

/// Lookups the Bloom filter passed that the log then showed absent, over those
/// plus the lookups it rightly turned away.
pub fn bloom_false_positive_rate(
    confirmed_false: u64,
    true_negatives: u64,
    at: Millis,
) -> Option<Sample> {
    Some(sample(
        MetricId::BloomFalsePositiveRate,
        ratio(confirmed_false, confirmed_false + true_negatives)?,
        at,
    ))
}

/// The mean recall at 10 over the probes, none when there are no probes.
pub fn probe_recall(report: &ProbeReport, at: Millis) -> Option<Sample> {
    let scores = &report.scores;
    if scores.is_empty() {
        return None;
    }
    let mean = scores.values().map(|score| score.recall).sum::<f64>() / scores.len() as f64;
    Some(sample(MetricId::ProbeRecallAt10, mean, at))
}

/// The bytes the hot and the cold tier hold.
pub fn tier_bytes(bytes: TierBytes, at: Millis) -> [Sample; 2] {
    [(TierLabel::Hot, bytes.hot), (TierLabel::Cold, bytes.cold)].map(|(tier, size)| Sample {
        metric: MetricId::TierBytes,
        value: size as f64,
        labels: Labels::Tier(tier),
        at,
    })
}

/// The size of every file under `dir`, as the sample of `metric`. A directory
/// that does not exist holds nothing.
///
/// # Errors
///
/// Fails when a file or directory in the tree cannot be read.
pub fn directory_bytes(metric: MetricId, dir: &Path, at: Millis) -> io::Result<Sample> {
    fn walk(dir: &Path) -> io::Result<u64> {
        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(error),
        };
        let mut total = 0;
        for entry in entries {
            let entry = entry?;
            let meta = entry.metadata()?;
            total += if meta.is_dir() {
                walk(&entry.path())?
            } else {
                meta.len()
            };
        }
        Ok(total)
    }
    Ok(sample(metric, walk(dir)? as f64, at))
}

/// The resident memory of the daemon and its children together.
pub fn peak_rss(memory: &mut dyn ProcessMemory, at: Millis) -> Sample {
    let total: u64 = memory.resident_bytes().iter().sum();
    sample(MetricId::PeakRssBytes, total as f64, at)
}

#[cfg(test)]
mod tests;
