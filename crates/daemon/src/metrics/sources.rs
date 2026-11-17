// SPDX-License-Identifier: Apache-2.0
//! What the daemon measures about itself at the end of a consolidation run:
//! sizes, backlogs and counters that no handler reports as it goes. Each
//! reading is independent, so one that fails is logged and the rest still land.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use futures_util::StreamExt;
use liam_daemon::consolidation::hooks::ExtractionHook;
use liam_daemon::consolidation::shared_store::SharedStore;
use liam_log::reader::{LogReader, SequentialScanReader};
use liam_log::LogOffset;
use liam_store::{Clock, ExtractorName};

use super::collect;
use super::memory::SystemMemory;
use super::sink::{MetricId, Sample};

pub struct RunSources {
    pub store: Arc<SharedStore>,
    pub database_path: PathBuf,
    pub log_dir: PathBuf,
    pub archive_dir: Option<PathBuf>,
    /// Present when extraction is on, which is what gives the backlogs and the
    /// response cache a meaning.
    pub extraction: Option<Arc<dyn ExtractionHook>>,
    pub clock: Arc<dyn Clock>,
    memory: Mutex<SystemMemory>,
}

impl RunSources {
    pub fn new(
        store: Arc<SharedStore>,
        database_path: PathBuf,
        log_dir: PathBuf,
        archive_dir: Option<PathBuf>,
        extraction: Option<Arc<dyn ExtractionHook>>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            store,
            database_path,
            log_dir,
            archive_dir,
            extraction,
            clock,
            memory: Mutex::new(SystemMemory::new()),
        }
    }

    /// Takes every reading. The caller records them together, so a reader
    /// never sees half of a run's numbers.
    pub async fn read(&self) -> Vec<Sample> {
        let now = self.clock.now();
        let mut samples = Vec::new();
        self.read_store(&mut samples).await;
        self.read_files(&mut samples);
        samples.push(collect::peak_rss(
            &mut *self.memory.lock().unwrap_or_else(PoisonError::into_inner),
            now,
        ));
        self.read_deletions(&mut samples);
        if let Some(extraction) = &self.extraction {
            self.read_backlogs(&mut samples).await;
            if let Some(lookups) = extraction.cache_lookups() {
                samples.extend(collect::cache_hit_rate(lookups.hits, lookups.misses, now));
            }
        }
        samples
    }

    async fn read_store(&self, samples: &mut Vec<Sample>) {
        let now = self.clock.now();
        let store = match self.store.get().await {
            Ok(store) => store,
            Err(error) => {
                tracing::warn!(%error, "metrics: the store is unavailable, so its readings are skipped");
                return;
            }
        };
        match store.storage_bytes().await {
            Ok(bytes) => samples.extend(collect::tier_bytes(bytes, now)),
            Err(error) => tracing::warn!(%error, "metrics: the tier sizes could not be read"),
        }
        let bloom = store.bloom_stats();
        samples.extend(collect::bloom_false_positive_rate(
            bloom.confirmed_false,
            bloom.true_negatives,
            now,
        ));
    }

    fn read_files(&self, samples: &mut Vec<Sample>) {
        let now = self.clock.now();
        let dirs = [
            (MetricId::LogBytes, Some(self.log_dir.as_path())),
            (MetricId::ArchiveBytes, self.archive_dir.as_deref()),
        ];
        for (metric, dir) in dirs {
            let Some(dir) = dir else { continue };
            match collect::directory_bytes(metric, dir, now) {
                Ok(sample) => samples.push(sample),
                Err(error) => {
                    tracing::warn!(%error, metric = metric.name(), "metrics: a directory size could not be read")
                }
            }
        }
    }

    fn read_deletions(&self, samples: &mut Vec<Sample>) {
        match crate::deletion::open_jobs(&self.database_path) {
            Ok(jobs) => samples.extend(collect::deletion_backlog(&jobs, self.clock.now())),
            Err(error) => {
                tracing::warn!(error = %format!("{error:#}"), "metrics: the delete jobs could not be read")
            }
        }
    }

    async fn read_backlogs(&self, samples: &mut Vec<Sample>) {
        let now = self.clock.now();
        let store = match self.store.get().await {
            Ok(store) => store,
            Err(_) => return,
        };
        for (name, metric) in [
            (ExtractorName::Rules, MetricId::RulesExtractionBacklog),
            (ExtractorName::Llm, MetricId::ModelExtractionBacklog),
        ] {
            let cursor = match store.extraction_cursor(name).await {
                Ok(cursor) => cursor,
                Err(error) => {
                    tracing::warn!(%error, metric = metric.name(), "metrics: a cursor could not be read");
                    continue;
                }
            };
            match events_after(&self.log_dir, cursor).await {
                Ok(count) => samples.push(collect::extraction_backlog(metric, count, now)),
                Err(error) => {
                    tracing::warn!(%error, metric = metric.name(), "metrics: the log could not be counted")
                }
            }
        }
    }
}

/// The records written after `cursor`, or all of them when it is `None`.
async fn events_after(log_dir: &Path, cursor: Option<LogOffset>) -> anyhow::Result<u64> {
    let reader = SequentialScanReader::local(log_dir)?;
    let mut stream = reader.scan(cursor);
    let mut count = 0;
    while let Some(record) = stream.next().await {
        record?;
        count += 1;
    }
    Ok(count)
}

#[cfg(test)]
mod tests;
