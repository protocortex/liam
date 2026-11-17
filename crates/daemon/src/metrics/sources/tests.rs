// SPDX-License-Identifier: Apache-2.0

use futures_util::StreamExt;
use liam_daemon::consolidation::hooks::CacheLookups;
use liam_store::{FixedClock, Millis, NewNode};

use super::*;
use crate::log_startup::test_support::{config_in, database_in, embedder, log_dir_in};
use crate::log_startup::{open_store_with_log, StartedStore};
use crate::metrics::sink::{Labels, MemorySink, MetricsSink, Reading};
use crate::shutdown::Shutdown;

const NODES: usize = 4;

async fn started_with_nodes(dir: &Path) -> StartedStore {
    let started = open_store_with_log(
        &config_in(dir),
        &database_in(dir),
        &log_dir_in(dir),
        embedder(),
        &Shutdown::default(),
    )
    .await
    .expect("start");
    for index in 0..NODES {
        let note = NewNode::now("fact", format!("n{index}"), format!("content {index}"));
        started.store.insert(note).await.expect("insert");
    }
    started
}

fn reading(sink: &MemorySink, metric: MetricId) -> Option<Reading> {
    sink.snapshot()
        .readings
        .into_iter()
        .find(|reading| reading.metric == metric)
}

fn sources(dir: &Path, store: liam_store::DefaultGraph) -> RunSources {
    RunSources::new(
        Arc::new(SharedStore::new(store)),
        database_in(dir),
        log_dir_in(dir),
        None,
        None,
        Arc::new(FixedClock::new(Millis(1_000))),
    )
}

#[tokio::test]
async fn the_unread_events_are_those_after_the_cursor() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let _started = started_with_nodes(dir.path()).await;
    let log_dir = log_dir_in(dir.path());
    let reader = SequentialScanReader::local(&log_dir).unwrap();
    let first = reader.scan(None).next().await.unwrap().unwrap().offset;
    let total = events_after(&log_dir, None).await.unwrap();

    // Act
    let after_first = events_after(&log_dir, Some(first)).await.unwrap();

    // Assert
    assert!(total >= NODES as u64);
    assert_eq!(after_first, total - 1);
}

#[tokio::test]
async fn a_run_records_the_sizes_the_store_and_the_disk_hold() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let started = started_with_nodes(dir.path()).await;
    let clock = Arc::new(FixedClock::new(Millis(1_000)));
    let sink = Arc::new(MemorySink::new(clock.clone()));

    // Act
    let readings = sources(dir.path(), started.store).read().await;
    readings.into_iter().for_each(|sample| sink.record(sample));

    // Assert
    let value = |metric| reading(&sink, metric).map(|reading| reading.value);
    assert!(value(MetricId::LogBytes).unwrap() > 0.0);
    assert!(value(MetricId::PeakRssBytes).unwrap() > 0.0);
    assert_eq!(value(MetricId::DeletionBacklogCount), Some(0.0));
    assert_eq!(value(MetricId::DeletionBacklogAgeDays), None);
    assert_eq!(value(MetricId::ArchiveBytes), None);
    assert_eq!(value(MetricId::RulesExtractionBacklog), None);
    let tiers = sink
        .snapshot()
        .readings
        .iter()
        .filter(|reading| reading.metric == MetricId::TierBytes)
        .count();
    assert_eq!(tiers, 2);
    assert_eq!(
        reading(&sink, MetricId::BloomFalsePositiveRate).map(|r| r.labels),
        Some(Labels::None)
    );
}

struct CachedExtraction;

#[async_trait::async_trait]
impl ExtractionHook for CachedExtraction {
    async fn extract(
        &self,
        _: &liam_daemon::consolidation::hooks::ExtractCtx<'_>,
    ) -> anyhow::Result<liam_daemon::consolidation::hooks::ExtractionReport> {
        Ok(Default::default())
    }

    fn cache_lookups(&self) -> Option<CacheLookups> {
        Some(CacheLookups { hits: 3, misses: 1 })
    }
}

#[tokio::test]
async fn with_extraction_on_a_run_records_both_backlogs_and_the_cache_hit_rate() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let started = started_with_nodes(dir.path()).await;
    let clock = Arc::new(FixedClock::new(Millis(1_000)));
    let sink = Arc::new(MemorySink::new(clock.clone()));
    let mut sources = sources(dir.path(), started.store);
    sources.extraction = Some(Arc::new(CachedExtraction));
    let total = events_after(&log_dir_in(dir.path()), None).await.unwrap();

    // Act
    sources
        .read()
        .await
        .into_iter()
        .for_each(|sample| sink.record(sample));

    // Assert
    let value = |metric| reading(&sink, metric).map(|reading| reading.value);
    assert_eq!(value(MetricId::RulesExtractionBacklog), Some(total as f64));
    assert_eq!(value(MetricId::ModelExtractionBacklog), Some(total as f64));
    assert_eq!(value(MetricId::CacheHitRate), Some(0.75));
}
