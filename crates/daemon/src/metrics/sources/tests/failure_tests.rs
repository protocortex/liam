// SPDX-License-Identifier: Apache-2.0
//! A reading that cannot be taken is skipped, and the others still land.

use super::*;
use crate::deletion::job_file_path_for;

fn metrics_of(samples: &[Sample]) -> Vec<MetricId> {
    samples.iter().map(|sample| sample.metric).collect()
}

#[tokio::test]
async fn with_the_store_slot_empty_only_the_readings_that_need_no_store_are_taken() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let started = started_with_nodes(dir.path()).await;
    let mut sources = sources(dir.path(), started.store);
    sources.extraction = Some(Arc::new(CachedExtraction));
    drop(sources.store.exclusive().await.take());

    // Act
    let samples = sources.read().await;

    // Assert
    let taken = metrics_of(&samples);
    assert!(taken.contains(&MetricId::LogBytes));
    assert!(taken.contains(&MetricId::PeakRssBytes));
    assert!(taken.contains(&MetricId::CacheHitRate));
    for needs_the_store in [
        MetricId::TierBytes,
        MetricId::BloomFalsePositiveRate,
        MetricId::RulesExtractionBacklog,
        MetricId::ModelExtractionBacklog,
    ] {
        assert!(!taken.contains(&needs_the_store), "{needs_the_store:?}");
    }
}

#[tokio::test]
async fn a_log_directory_that_cannot_be_read_costs_only_its_own_size_reading() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let started = started_with_nodes(dir.path()).await;
    let mut sources = sources(dir.path(), started.store);
    let not_a_directory = dir.path().join("log-as-a-file");
    std::fs::write(&not_a_directory, b"").unwrap();
    sources.log_dir = not_a_directory;

    // Act
    let samples = sources.read().await;

    // Assert
    let taken = metrics_of(&samples);
    assert!(!taken.contains(&MetricId::LogBytes));
    assert!(taken.contains(&MetricId::PeakRssBytes));
    assert!(taken.contains(&MetricId::DeletionBacklogCount));
}

#[tokio::test]
async fn a_delete_job_file_that_cannot_be_read_costs_only_the_deletion_readings() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let started = started_with_nodes(dir.path()).await;
    let sources = sources(dir.path(), started.store);
    std::fs::write(
        job_file_path_for(&database_in(dir.path())),
        b"not a job file",
    )
    .unwrap();

    // Act
    let samples = sources.read().await;

    // Assert
    let taken = metrics_of(&samples);
    assert!(!taken.contains(&MetricId::DeletionBacklogCount));
    assert!(taken.contains(&MetricId::LogBytes));
}

#[tokio::test]
async fn a_log_that_cannot_be_counted_leaves_out_the_backlog_readings() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let started = started_with_nodes(dir.path()).await;
    let mut sources = sources(dir.path(), started.store);
    sources.extraction = Some(Arc::new(CachedExtraction));
    sources.log_dir = dir.path().join("no-such-log-dir");

    // Act
    let samples = sources.read().await;

    // Assert
    let taken = metrics_of(&samples);
    assert!(!taken.contains(&MetricId::RulesExtractionBacklog));
    assert!(!taken.contains(&MetricId::ModelExtractionBacklog));
    assert!(taken.contains(&MetricId::CacheHitRate));
}
