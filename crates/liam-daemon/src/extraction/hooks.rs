// SPDX-License-Identifier: Apache-2.0
//! The extraction part of a consolidation run. Two passes, each with its own
//! log cursor: the rules pass, which needs no model, and the model pass. Each
//! reads the log up to what the store has applied and writes its facts under
//! the run's source, so a rollback of the run takes them away again.
//!
//! A cursor moves only after its facts are written, and every write is
//! idempotent, so a run that stops anywhere is finished by the next one.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use futures_util::StreamExt;
use liam_daemon::consolidation::hooks::{ExtractCtx, ExtractionHook, ExtractionReport};
use liam_daemon::consolidation::manifest::ExtractionModel;
use liam_log::event::LogEvent;
use liam_log::reader::LogReader;
use liam_log::LogOffset;
use liam_store::ExtractorName;
use tokio::sync::Semaphore;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use super::apply::{apply_rules, ApplyContext};
use super::cache::CacheStore;
use super::client::ModelIdentity;
use super::extractor::ExtractionReport as ModelRun;
use super::extractor::{extract_events, ExtractionDeps, ExtractorProvider};
use super::inputs::events_to_inputs;
use super::prompt::PROMPT_VERSION;
use super::sidecar::FreeRam;
use crate::config::{resolve_vocabulary, ExtractionConfig, MAX_RUN_DEADLINE_SECS};

/// Entity pages the rules look for in each event. The cap bounds the memory
/// and scan time of a store with a very large number of them.
const MAX_KNOWN_ENTITIES: usize = 10_000;

/// What the extraction part is built from.
pub struct ExtractionHooks {
    pub config: ExtractionConfig,
    pub reader: Arc<dyn LogReader>,
    pub provider: Arc<dyn ExtractorProvider>,
    /// The model the provider is expected to start, known before it starts.
    /// `None` when the model file could not be read, which fails the model
    /// pass on every run until it can.
    pub identity: Option<ModelIdentity>,
    pub cache: Arc<CacheStore>,
    /// The daemon's generation permits, shared with the other model users.
    pub permits: Arc<Semaphore>,
    pub free_ram: Arc<dyn FreeRam + Sync>,
    /// Cancelled when the daemon is stopping.
    pub stopping: CancellationToken,
}

/// The oldest events a pass has not read, in log order.
struct Window {
    events: Vec<(LogOffset, LogEvent)>,
    end: LogOffset,
}

#[async_trait]
impl ExtractionHook for ExtractionHooks {
    async fn extract(&self, ctx: &ExtractCtx<'_>) -> anyhow::Result<ExtractionReport> {
        if !self.config.enabled {
            return Ok(ExtractionReport::default());
        }
        // The passes do not depend on each other: a model that cannot start
        // must not stall the rules, and a rules failure must not stall the model.
        let rules = self.rules_pass(ctx).await;
        let model = if self.config.llm_enabled {
            self.model_pass(ctx).await
        } else {
            Ok(ModelEnd::default())
        };
        let mut report = ExtractionReport::default();
        if let Err(error) = rules {
            warn!(error = %format!("{error:#}"), "extraction: the rules pass failed");
            report.partial = true;
        }
        match model {
            Ok(end) => {
                report.partial |= end.partial;
                report.model = end.model.map(|model| ExtractionModel {
                    model_id: model.model_id,
                    model_sha256: model.model_sha256,
                    prompt_version: PROMPT_VERSION.to_owned(),
                });
            }
            Err(error) => {
                warn!(error = %format!("{error:#}"), "extraction: the model pass failed");
                report.partial = true;
            }
        }
        Ok(report)
    }
}

/// How the model pass ended.
#[derive(Default)]
struct ModelEnd {
    /// Work was left for a later run.
    partial: bool,
    /// The model that answered, when one did.
    model: Option<ModelIdentity>,
}

impl ExtractionHooks {
    async fn rules_pass(&self, ctx: &ExtractCtx<'_>) -> anyhow::Result<()> {
        let Some(window) = self.window(ctx, ExtractorName::Rules).await? else {
            return Ok(());
        };
        let inputs = events_to_inputs(&window.events, self.config.min_event_chars);
        if !inputs.is_empty() {
            let known = ctx.store.live_entity_subjects(MAX_KNOWN_ENTITIES).await?;
            if known.len() >= MAX_KNOWN_ENTITIES {
                warn!(
                    limit = MAX_KNOWN_ENTITIES,
                    "extraction: more entity pages than the rules look for, the rest are not matched"
                );
            }
            let vocabulary = resolve_vocabulary(&self.config);
            let apply = ApplyContext {
                store: ctx.store,
                config: &self.config,
                vocabulary: &vocabulary,
                write_source: &ctx.source,
            };
            let counts = apply_rules(&apply, &inputs, &known).await?;
            info!(
                ?counts,
                inputs = inputs.len(),
                "extraction: rules pass done"
            );
        }
        advance(ctx, ExtractorName::Rules, window.end).await
    }

    async fn model_pass(&self, ctx: &ExtractCtx<'_>) -> anyhow::Result<ModelEnd> {
        let Some(window) = self.window(ctx, ExtractorName::Llm).await? else {
            return Ok(ModelEnd::default());
        };
        let inputs = events_to_inputs(&window.events, self.config.min_event_chars);
        info!(
            events = window.events.len(),
            inputs = inputs.len(),
            "extraction: model pass starting"
        );
        if inputs.is_empty() {
            advance(ctx, ExtractorName::Llm, window.end).await?;
            return Ok(ModelEnd::default());
        }
        // Validation caps the setting; the cap here keeps a hand built config
        // from overflowing the deadline.
        let deadline_secs = self.config.run_deadline_secs.min(MAX_RUN_DEADLINE_SECS);
        let identity = self.identity.as_ref().context(
            "the model file could not be read, so the model pass cannot tell which model it runs",
        )?;
        let deps = ExtractionDeps {
            provider: self.provider.as_ref(),
            identity,
            cache: Arc::clone(&self.cache),
            store: ctx.store,
            config: &self.config,
            permits: &self.permits,
            free_ram: self.free_ram.as_ref(),
            write_source: &ctx.source,
            deadline: Instant::now() + Duration::from_secs(deadline_secs),
            stopping: &self.stopping,
            window_end: window.end,
        };
        // The loop stops between batches when the daemon is stopping; dropping
        // it mid write would leave the event log poisoned.
        let run = extract_events(&inputs, deps).await?;
        if run.stopped {
            warn!("extraction: the daemon is stopping, the rest of the window is left for the next run");
        }
        if let Some(done) = run.completed_offset {
            advance(ctx, ExtractorName::Llm, done).await?;
        }
        Ok(ModelEnd {
            partial: is_partial(&run),
            model: run.model,
        })
    }

    /// The oldest `max_events_per_run` events after the cursor of `name`, up
    /// to what the store has applied, or `None` when there are none.
    async fn window(
        &self,
        ctx: &ExtractCtx<'_>,
        name: ExtractorName,
    ) -> anyhow::Result<Option<Window>> {
        let Some(applied) = ctx.store.last_applied_offset().await? else {
            return Ok(None);
        };
        let from = ctx.store.extraction_cursor(name).await?;
        let mut records = self.reader.scan_through(from, applied);
        let mut events = Vec::new();
        while events.len() < self.config.max_events_per_run {
            let Some(record) = records.next().await else {
                break;
            };
            let record = record.context("could not read the event log")?;
            events.push((record.offset, record.event));
        }
        Ok(events
            .last()
            .map(|&(end, _)| end)
            .map(|end| Window { events, end }))
    }
}

async fn advance(
    ctx: &ExtractCtx<'_>,
    name: ExtractorName,
    offset: LogOffset,
) -> anyhow::Result<()> {
    ctx.store
        .advance_extraction_cursor(name, offset)
        .await
        .context("could not move the extraction cursor")?;
    Ok(())
}

/// Work was left for a later run: a batch failed or was quarantined, the model
/// was lost, or the deadline came first.
fn is_partial(run: &ModelRun) -> bool {
    run.batches_failed > 0
        || run.batches_quarantined > 0
        || run.sidecar_lost
        || run.deadline_hit
        || run.stopped
}
