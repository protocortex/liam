// SPDX-License-Identifier: Apache-2.0
//! The extraction part of a consolidation run. It reads the log from the
//! model pass's own cursor, up to what the store has applied, and hands the
//! window to the model loop, which writes the facts under the run's source so a
//! rollback of the run takes them away again.
//!
//! The cursor moves only after the facts are written, and every write is
//! idempotent, so a run that stops anywhere is finished by the next one.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use futures_util::StreamExt;
use liam_log::reader::LogReader;
use liam_log::LogOffset;
use liam_store::ExtractorName;
use tokio::sync::Semaphore;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use super::cache::CacheStore;
use super::client::ModelIdentity;
use super::extractor::ExtractionReport as ModelRun;
use super::extractor::{extract_events, ExtractionDeps, ExtractorProvider};
use super::inputs::events_to_inputs;
use super::sidecar::FreeRam;
use crate::config::ExtractionConfig;
use liam_daemon::consolidation::hooks::{ExtractCtx, ExtractionHook, ExtractionReport};

/// What the extraction part is built from.
pub struct ExtractionHooks {
    pub config: ExtractionConfig,
    pub reader: Arc<dyn LogReader>,
    pub provider: Arc<dyn ExtractorProvider>,
    /// The model the provider is expected to start, known before it starts.
    pub identity: ModelIdentity,
    pub cache: Arc<CacheStore>,
    /// The daemon's generation permits, shared with the other model users.
    pub permits: Arc<Semaphore>,
    pub free_ram: Arc<dyn FreeRam + Sync>,
    /// Cancelled when the daemon is stopping.
    pub stopping: CancellationToken,
}

#[async_trait]
impl ExtractionHook for ExtractionHooks {
    async fn extract(&self, ctx: &ExtractCtx<'_>) -> anyhow::Result<ExtractionReport> {
        if !self.config.enabled || !self.config.llm_enabled {
            return Ok(ExtractionReport::default());
        }
        self.model_pass(ctx).await
    }
}

impl ExtractionHooks {
    async fn model_pass(&self, ctx: &ExtractCtx<'_>) -> anyhow::Result<ExtractionReport> {
        let Some(applied) = ctx.store.last_applied_offset().await? else {
            return Ok(ExtractionReport::default());
        };
        let from = ctx.store.extraction_cursor(ExtractorName::Llm).await?;
        let events = self.read_window(from, applied).await?;
        let Some(&(window_end, _)) = events.last() else {
            return Ok(ExtractionReport::default());
        };
        let inputs = events_to_inputs(&events, self.config.min_event_chars);
        info!(
            events = events.len(),
            inputs = inputs.len(),
            "extraction: model pass starting"
        );
        if inputs.is_empty() {
            advance(ctx, window_end).await?;
            return Ok(ExtractionReport::default());
        }
        let deps = ExtractionDeps {
            provider: self.provider.as_ref(),
            identity: &self.identity,
            cache: Arc::clone(&self.cache),
            store: ctx.store,
            config: &self.config,
            permits: &self.permits,
            free_ram: self.free_ram.as_ref(),
            write_source: &ctx.source,
            deadline: Instant::now() + Duration::from_secs(self.config.run_deadline_secs),
            window_end,
        };
        // A run dropped at any point is safe to repeat, so stopping is a drop.
        let run = tokio::select! {
            run = extract_events(&inputs, deps) => run?,
            () = self.stopping.cancelled() => {
                warn!("extraction: the daemon is stopping, the model pass is left for the next run");
                return Ok(ExtractionReport { partial: true });
            }
        };
        if let Some(done) = run.completed_offset {
            advance(ctx, done).await?;
        }
        Ok(ExtractionReport {
            partial: is_partial(&run),
        })
    }

    /// The oldest `max_events_per_run` events after `from`, up to `through`.
    async fn read_window(
        &self,
        from: Option<LogOffset>,
        through: LogOffset,
    ) -> anyhow::Result<Vec<(LogOffset, liam_log::event::LogEvent)>> {
        let mut records = self.reader.scan_through(from, through);
        let mut events = Vec::new();
        while events.len() < self.config.max_events_per_run {
            match records.next().await {
                Some(record) => {
                    let record = record.context("could not read the event log")?;
                    events.push((record.offset, record.event));
                }
                None => break,
            }
        }
        Ok(events)
    }
}

async fn advance(ctx: &ExtractCtx<'_>, offset: LogOffset) -> anyhow::Result<()> {
    ctx.store
        .advance_extraction_cursor(ExtractorName::Llm, offset)
        .await
        .context("could not move the extraction cursor")?;
    Ok(())
}

/// Work was left for a later run: a batch failed or was quarantined, the model
/// was lost, or the deadline came first.
fn is_partial(run: &ModelRun) -> bool {
    run.batches_failed > 0 || run.batches_quarantined > 0 || run.sidecar_lost || run.deadline_hit
}
