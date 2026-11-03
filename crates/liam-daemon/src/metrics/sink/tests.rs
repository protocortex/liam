// SPDX-License-Identifier: Apache-2.0

use std::sync::{Barrier, Mutex};

use liam_store::FixedClock;

use super::*;
use crate::metrics::alerts::Rule;

const NOW: Millis = Millis(1_000);
const THREADS: usize = 8;
const RECORDS_PER_THREAD: usize = 200;

fn clock() -> Arc<dyn Clock> {
    Arc::new(FixedClock::new(NOW))
}

fn sample(metric: MetricId, value: f64) -> Sample {
    Sample {
        metric,
        value,
        labels: Labels::None,
        at: NOW,
    }
}

fn alert() -> Alert {
    Alert {
        metric: MetricId::RollbackCount,
        rule: Rule::Absolute { ceiling: 1.0 },
        value: 2.0,
    }
}

fn assert_copy<T: Copy>() {}

#[test]
fn every_label_is_copy_and_none_holds_text() {
    // Assert: the exhaustive match lists exactly the closed variants, and a
    // variant that held a `String` would not be `Copy`
    assert_copy::<Labels>();
    assert_copy::<TierLabel>();
    assert_copy::<RetrievalArm>();
    assert_copy::<Provider>();
    assert_copy::<ModelId>();
    let label = |labels: Labels| match labels {
        Labels::None => 0,
        Labels::Tier(TierLabel::Hot | TierLabel::Cold) => 1,
        Labels::Arm(RetrievalArm::F32 | RetrievalArm::Lexical | RetrievalArm::ColdProbe) => 2,
        Labels::Model {
            provider: Provider::Local | Provider::Anthropic | Provider::OpenAi | Provider::Gemini,
            ..
        } => 3,
    };
    assert_eq!(label(Labels::None), 0);
}

#[test]
fn a_model_id_is_validated_and_interned_once() {
    // Arrange
    let mut catalog = ModelCatalog::default();

    // Act
    let first = catalog.intern("claude-sonnet-4-20250514").unwrap();
    let again = catalog.intern("claude-sonnet-4-20250514").unwrap();
    let other = catalog.intern("gpt-5.1-2025-11-13").unwrap();

    // Assert
    assert_eq!(first, again);
    assert_ne!(first, other);
    assert_eq!(catalog.name(first), Some("claude-sonnet-4-20250514"));
    for bad in ["", "has space", "semi;colon", &"x".repeat(65)] {
        assert!(catalog.intern(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn the_memory_sink_keeps_the_latest_sample_per_metric_and_label_set() {
    // Arrange
    let sink = MemorySink::new(clock());

    // Act
    sink.record(sample(MetricId::Ece, 0.5));
    sink.record(sample(MetricId::Ece, 0.25));
    sink.record(Sample {
        labels: Labels::Tier(TierLabel::Hot),
        ..sample(MetricId::TierBytes, 10.0)
    });

    // Assert
    let snapshot = sink.snapshot();
    assert_eq!(snapshot.taken_at, NOW);
    let ece = snapshot
        .readings
        .iter()
        .find(|reading| reading.metric == MetricId::Ece)
        .unwrap();
    assert_eq!((ece.value, ece.samples), (0.25, 2));
    assert_eq!(snapshot.readings.len(), 2);
}

#[test]
fn a_value_that_is_not_finite_is_rejected_and_counted() {
    // Arrange
    let sink = MemorySink::new(clock());

    // Act
    for value in [f64::NAN, f64::INFINITY] {
        sink.record(sample(MetricId::Ece, value));
    }

    // Assert
    let snapshot = sink.snapshot();
    assert!(snapshot.readings.is_empty());
    assert_eq!(snapshot.rejected, 2);
}

/// Captures what a subscriber would write, for the lines of one scope.
#[derive(Clone, Default)]
struct Lines(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Lines {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn both_sinks_hold_the_same_snapshot_and_the_log_sink_writes_one_error_line_per_alert() {
    // Arrange
    let memory = MemorySink::new(clock());
    let log = LogSink::new(clock());
    let lines = Lines::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(move || writer.clone())
        .with_max_level(tracing::Level::ERROR)
        .finish();

    // Act
    tracing::subscriber::with_default(subscriber, || {
        for sink in [&memory as &dyn MetricsSink, &log] {
            sink.record(sample(MetricId::RollbackCount, 2.0));
            sink.set_alerts(vec![alert(), alert()]);
        }
    });

    // Assert
    assert_eq!(memory.snapshot(), log.snapshot());
    let written = String::from_utf8(lines.0.lock().unwrap().clone()).unwrap();
    assert_eq!(written.lines().count(), 2, "{written}");
    assert!(written.contains("rollback_count"), "{written}");
}

#[test]
fn concurrent_records_lose_no_sample() {
    // Arrange
    let sink = Arc::new(MemorySink::new(clock()));
    let barrier = Arc::new(Barrier::new(THREADS));

    // Act
    let threads: Vec<_> = (0..THREADS)
        .map(|_| {
            let (sink, barrier) = (Arc::clone(&sink), Arc::clone(&barrier));
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..RECORDS_PER_THREAD {
                    sink.record(sample(MetricId::ModelCalls, 1.0));
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }

    // Assert
    let snapshot = sink.snapshot();
    assert_eq!(
        snapshot.readings[0].samples,
        (THREADS * RECORDS_PER_THREAD) as u64
    );
}
