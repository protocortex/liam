// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::deletion::{DeleteTarget, JobState};
use crate::metrics::memory::SystemMemory;
use crate::metrics::sink::{ModelCatalog, Provider, TierLabel, TokenUsage};
use liam_daemon::consolidation::probes::ProbeScore;
use liam_store::TierBytes;

const AT: Millis = Millis(1_000);
const DAY_MS: i64 = 86_400_000;
const MB: u64 = 1 << 20;
const P_METRICS: [MetricId; 2] = [MetricId::AskLatencyP50Ms, MetricId::AskLatencyP95Ms];

fn value_of(samples: &[Sample], metric: MetricId) -> Option<f64> {
    samples
        .iter()
        .find(|sample| sample.metric == metric)
        .map(|sample| sample.value)
}

fn window_of(timings: impl IntoIterator<Item = u32>) -> LatencyWindow {
    let mut window = LatencyWindow::new(50);
    for millis in timings {
        window.record(f64::from(millis));
    }
    window
}

fn job(id: &str, requested_at: i64, failed: bool, done: bool) -> DeleteJob {
    let mut job = DeleteJob::new(
        id.into(),
        DeleteTarget::Event {
            event_id: "e".into(),
        },
        requested_at,
        30,
    );
    if failed {
        job.state = JobState::Failed;
    }
    if done {
        job.state = JobState::Done;
    }
    job
}

#[test]
fn the_nearest_rank_percentiles_of_ten_timings_are_the_fifth_and_the_tenth() {
    // Arrange
    let window = window_of((1..=10).rev());

    // Act
    let samples = window.samples(P_METRICS, Labels::None, AT);

    // Assert
    assert_eq!(value_of(&samples, MetricId::AskLatencyP50Ms), Some(5.0));
    assert_eq!(value_of(&samples, MetricId::AskLatencyP95Ms), Some(10.0));
}

#[test]
fn an_empty_window_reports_no_value() {
    // Arrange
    let window = LatencyWindow::new(50);

    // Act
    let samples = window.samples(P_METRICS, Labels::None, AT);

    // Assert
    assert!(samples.is_empty());
}

#[test]
fn one_timing_is_every_percentile_so_the_alert_rule_must_wait_for_more() {
    // Arrange
    let window = window_of([7]);

    // Act
    let samples = window.samples(P_METRICS, Labels::None, AT);

    // Assert
    assert_eq!(value_of(&samples, MetricId::AskLatencyP50Ms), Some(7.0));
    assert_eq!(value_of(&samples, MetricId::AskLatencyP95Ms), Some(7.0));
}

#[test]
fn a_full_window_drops_its_oldest_timing() {
    // Arrange
    let mut window = LatencyWindow::new(3);

    // Act
    for millis in [1000.0, 1.0, 2.0, 3.0] {
        window.record(millis);
    }

    // Assert
    assert_eq!(window.percentile(95), Some(3.0));
}

#[test]
fn a_confident_model_that_is_always_right_is_calibrated_to_the_bin() {
    // Arrange
    let pairs = [(0.95, true), (0.95, true), (0.05, false), (0.05, false)];

    // Act
    let sample = ece(&pairs, 4, AT).expect("enough data");

    // Assert
    assert!((sample.value - 0.05).abs() < 1e-9, "{}", sample.value);
}

#[test]
fn ece_waits_for_enough_pairs() {
    // Arrange
    let pairs = [(0.9, true), (0.9, false)];

    // Act and Assert
    assert!(ece(&pairs, 3, AT).is_none());
    assert!(ece(&[], 0, AT).is_none());
}

#[test]
fn the_backlog_sample_carries_the_unread_count_under_its_own_metric() {
    // Act
    let sample = extraction_backlog(MetricId::ModelExtractionBacklog, 7, AT);

    // Assert
    assert_eq!(
        (sample.metric, sample.value),
        (MetricId::ModelExtractionBacklog, 7.0)
    );
}

#[test]
fn the_deletion_backlog_ages_from_the_oldest_open_job_and_counts_a_failed_one() {
    // Arrange
    let now = Millis(10 * DAY_MS);
    let jobs = [
        job("old-done", 0, false, true),
        job("failed", 4 * DAY_MS, true, false),
        job("fresh", 9 * DAY_MS, false, false),
    ];

    // Act
    let samples = deletion_backlog(&jobs, now);

    // Assert
    assert_eq!(
        value_of(&samples, MetricId::DeletionBacklogCount),
        Some(2.0)
    );
    assert_eq!(
        value_of(&samples, MetricId::DeletionBacklogAgeDays),
        Some(6.0)
    );
}

#[test]
fn no_open_jobs_report_a_zero_count_and_no_age() {
    // Act
    let samples = deletion_backlog(&[job("done", 0, false, true)], AT);

    // Assert
    assert_eq!(
        value_of(&samples, MetricId::DeletionBacklogCount),
        Some(0.0)
    );
    assert_eq!(value_of(&samples, MetricId::DeletionBacklogAgeDays), None);
}

#[test]
fn model_calls_and_cost_add_up_under_provider_and_model_labels_only() {
    // Arrange
    let mut catalog = ModelCatalog::default();
    let model = catalog.intern("claude-haiku").unwrap();
    let usage = Usage {
        provider: Provider::Anthropic,
        model,
        tokens: TokenUsage {
            input_tokens: 2_000_000,
            output_tokens: 500_000,
            cached_input_tokens: 0,
        },
        batch: false,
    };
    let price = Price {
        input_micro_per_mtok: 1_000_000,
        output_micro_per_mtok: 5_000_000,
        ..Price::default()
    };
    let mut meter = ModelMeter::default();
    meter.record(usage, price, AT);

    // Act
    let [calls, cost] = meter.record(usage, price, AT);

    // Assert
    assert_eq!(calls.value, 2.0);
    assert_eq!(cost.value, 9_000_000.0);
    assert_eq!(
        cost.labels,
        Labels::Model {
            provider: Provider::Anthropic,
            model
        }
    );
}

#[test]
fn cached_input_and_batch_calls_are_charged_at_their_own_rates() {
    // Arrange
    let model = catalog_id(&mut ModelCatalog::default());
    let price = Price {
        input_micro_per_mtok: 1_000_000,
        output_micro_per_mtok: 5_000_000,
        cached_input_micro_per_mtok: 100_000,
        batch_input_micro_per_mtok: 500_000,
        batch_output_micro_per_mtok: 2_500_000,
    };
    let call = |batch| Usage {
        provider: Provider::OpenAi,
        model,
        tokens: TokenUsage {
            input_tokens: 2_000_000,
            output_tokens: 500_000,
            cached_input_tokens: 1_000_000,
        },
        batch,
    };

    // Act
    let live = {
        let u = call(false);
        price.cost_micro(&u.tokens, u.batch)
    };
    let batched = {
        let u = call(true);
        price.cost_micro(&u.tokens, u.batch)
    };

    // Assert: 1M fresh at 1.0, 1M cached at 0.1, 0.5M out at 5.0, and the
    // same with the batch input and output rates
    assert_eq!(live, 3_600_000);
    assert_eq!(batched, 1_850_000);
}

#[test]
fn more_cached_tokens_than_input_tokens_cannot_make_the_cost_negative() {
    // Arrange
    let model = catalog_id(&mut ModelCatalog::default());
    let price = Price {
        cached_input_micro_per_mtok: 100_000,
        ..Price::default()
    };
    let usage = Usage {
        provider: Provider::Gemini,
        model,
        tokens: TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 0,
            cached_input_tokens: 5_000_000,
        },
        batch: false,
    };

    // Act and Assert
    assert_eq!(price.cost_micro(&usage.tokens, usage.batch), 100_000);
}

fn catalog_id(catalog: &mut ModelCatalog) -> crate::metrics::sink::ModelId {
    catalog.intern("gpt-4o-2024-08-06").unwrap()
}

#[test]
fn three_hits_and_a_miss_make_a_rate_of_three_quarters() {
    // Act and Assert
    assert_eq!(cache_hit_rate(3, 1, AT).unwrap().value, 0.75);
    assert!(cache_hit_rate(0, 0, AT).is_none());
}

#[test]
fn the_bloom_rate_is_confirmed_false_over_those_plus_true_negatives() {
    // Act
    let sample = bloom_false_positive_rate(3, 90, AT).unwrap();

    // Assert
    assert_eq!(sample.value, 3.0 / 93.0);
    assert!(bloom_false_positive_rate(0, 0, AT).is_none());
}

#[test]
fn the_probe_recall_is_the_mean_over_the_probes() {
    // Arrange
    let mut report = ProbeReport::default();
    for (name, recall) in [("a", 0.8), ("b", 0.6)] {
        report
            .scores
            .insert(name.into(), ProbeScore { recall, ndcg: 1.0 });
    }

    // Act and Assert
    let sample = probe_recall(&report, AT).unwrap();
    assert!((sample.value - 0.7).abs() < 1e-12);
    assert!(probe_recall(&ProbeReport::default(), AT).is_none());
}

struct Fixed(Vec<u64>);

impl ProcessMemory for Fixed {
    fn resident_bytes(&mut self) -> Vec<u64> {
        self.0.clone()
    }
}

#[test]
fn the_peak_adds_the_daemon_and_its_children() {
    // Arrange
    let mut memory = Fixed(vec![100 * MB, 4000 * MB]);

    // Act
    let sample = peak_rss(&mut memory, AT);

    // Assert
    assert_eq!(sample.value, (4100 * MB) as f64);
}

#[test]
fn the_real_reader_sees_this_process() {
    // Act
    let sample = peak_rss(&mut SystemMemory::new(), AT);

    // Assert
    assert!(sample.value > 0.0);
}

#[test]
fn the_tier_bytes_carry_a_tier_label_each() {
    // Act
    let [hot, cold] = tier_bytes(TierBytes { hot: 40, cold: 8 }, AT);

    // Assert
    assert_eq!(
        (hot.labels, hot.value),
        (Labels::Tier(TierLabel::Hot), 40.0)
    );
    assert_eq!(
        (cold.labels, cold.value),
        (Labels::Tier(TierLabel::Cold), 8.0)
    );
}

#[test]
fn a_directory_adds_up_every_file_under_it_and_a_missing_one_is_empty() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a"), [0u8; 100]).unwrap();
    std::fs::create_dir(dir.path().join("nested")).unwrap();
    std::fs::write(dir.path().join("nested").join("b"), [0u8; 23]).unwrap();

    // Act
    let present = directory_bytes(MetricId::LogBytes, dir.path(), AT).unwrap();
    let missing = directory_bytes(MetricId::ArchiveBytes, &dir.path().join("gone"), AT).unwrap();

    // Assert
    assert_eq!(present.value, 123.0);
    assert_eq!(missing.value, 0.0);
}
