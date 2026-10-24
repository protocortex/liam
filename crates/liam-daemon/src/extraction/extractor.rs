// SPDX-License-Identifier: Apache-2.0
//! The model half of extraction: batches the inputs of a scan window per
//! scope, asks the model (or the response cache) about each batch, and writes
//! the facts that survive grounding.
//!
//! Nothing here holds a store lock across a model call; each fact is its own
//! short write. A batch is cached before its facts are applied and every write
//! is idempotent, so a run that dies at any point is finished by the next one
//! without duplicating a fact or calling the model again.

use std::any::Any;
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use liam_log::LogOffset;
use liam_store::{DefaultGraph, Vocabulary};
use serde_json::Value;
use tracing::{debug, info, warn};

use super::apply::{apply_answer, ApplyContext, ApplyError};
use super::cache::{cache_key, CacheStore, CachedExtractor, KeyParts};
use super::client::{ExtractRequest, JsonExtractor, ModelIdentity};
use super::inputs::ExtractionInput;
use super::prompt::{build_prompt, fact_json_schema, prompt_fingerprint, prompt_template};
use super::prompt::{PromptEvent, PROMPT_VERSION};
use super::response::parse_response;
use crate::config::{declared_predicates, resolve_vocabulary, ExtractionConfig};

/// Extraction must repeat exactly, so sampling is pinned.
const EXTRACTION_SEED: u32 = 0;
const EXTRACTION_TEMPERATURE: f32 = 0.0;
/// The answer may use this share of the context window; the prompt has the rest.
const ANSWER_CONTEXT_DIVISOR: usize = 4;

/// A running model and what must stay alive to keep it running.
pub struct ExtractorSession {
    pub extractor: Arc<dyn JsonExtractor>,
    /// Dropped when the run ends, which stops the model (the sidecar guard in
    /// production, nothing in tests).
    pub _keepalive: Option<Box<dyn Any + Send>>,
}

/// Starts a model on demand, so a run that the cache fully answers never does.
#[async_trait]
pub trait ExtractorProvider: Send + Sync {
    async fn open(&self, config: &ExtractionConfig) -> Result<ExtractorSession>;
}

/// What a run needs from the outside world.
pub struct ExtractionDeps<'a> {
    pub provider: &'a dyn ExtractorProvider,
    /// Which model the run expects, so the cache can answer before one starts.
    pub identity: &'a ModelIdentity,
    pub cache: Arc<CacheStore>,
    pub store: &'a DefaultGraph,
    pub config: &'a ExtractionConfig,
    /// The log source the facts are written under.
    pub write_source: &'a str,
    /// The last offset of the scanned window, where a fully done run ends.
    pub window_end: LogOffset,
}

/// What a run did. The caller moves the cursor to `completed_offset`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExtractionReport {
    /// Everything up to here is done for every scope; `None` when some scope
    /// has not finished even its first input.
    pub completed_offset: Option<LogOffset>,
    pub facts_written: usize,
    pub facts_dropped: usize,
    pub facts_dropped_ungrounded: usize,
    pub batches_failed: usize,
    pub batches_quarantined: usize,
    pub ended_no_match: usize,
    pub ended_outranked: usize,
    pub events_truncated: usize,
    pub sidecar_lost: bool,
    pub deadline_hit: bool,
    /// The model that answered, for the run manifest.
    pub model: Option<ModelIdentity>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Pending,
    Done,
    Failed,
}

struct Batch {
    scope: Option<String>,
    /// Positions in the run's inputs, in offset order.
    indices: Vec<usize>,
}

enum BatchEnd {
    Done,
    Failed,
    /// The run cannot go on: the deadline passed or the model is gone.
    Halted,
}

enum Attempt {
    Done,
    Failed,
    Halted,
}

enum Generation {
    Text(String),
    Stop(Attempt),
}

struct Opened {
    cached: Arc<CachedExtractor>,
    _keepalive: Option<Box<dyn Any + Send>>,
}

struct Run<'a> {
    deps: ExtractionDeps<'a>,
    inputs: Arc<Vec<ExtractionInput>>,
    status: Vec<Status>,
    vocabulary: Vocabulary,
    predicates: Vec<String>,
    schema: Value,
    schema_hash: String,
    session: Option<Opened>,
    report: ExtractionReport,
}

/// Batches `inputs` per scope, extracts each batch, and writes the facts.
pub async fn extract_events(
    inputs: &[ExtractionInput],
    deps: ExtractionDeps<'_>,
) -> Result<ExtractionReport> {
    let mut run = Run::new(inputs, deps);
    run.execute().await?;
    Ok(run.report)
}

fn is_parseable(response: &str) -> bool {
    parse_response(response, usize::MAX, &[]).is_ok()
}

impl<'a> Run<'a> {
    fn new(inputs: &[ExtractionInput], deps: ExtractionDeps<'a>) -> Self {
        let mut report = ExtractionReport::default();
        let max_chars = deps.config.max_event_chars();
        let mut inputs: Vec<ExtractionInput> = inputs.to_vec();
        inputs.sort_by_key(|input| input.offset);
        for input in &mut inputs {
            if let Some((cut, _)) = input.text.char_indices().nth(max_chars) {
                input.text.truncate(cut);
                report.events_truncated += 1;
            }
        }
        let predicates = declared_predicates(deps.config);
        let schema = fact_json_schema(&predicates);
        Self {
            status: vec![Status::Pending; inputs.len()],
            inputs: Arc::new(inputs),
            vocabulary: resolve_vocabulary(deps.config),
            schema_hash: prompt_fingerprint(&prompt_template(), &schema),
            schema,
            predicates,
            session: None,
            report,
            deps,
        }
    }

    async fn execute(&mut self) -> Result<()> {
        let batches = self.plan_batches();
        info!(
            inputs = self.inputs.len(),
            batches = batches.len(),
            "extracting facts from the scan window"
        );
        let mut stopped_scopes: HashSet<Option<String>> = HashSet::new();
        for batch in &batches {
            if stopped_scopes.contains(&batch.scope) {
                continue;
            }
            match self.run_batch(&batch.indices).await? {
                BatchEnd::Done => {}
                BatchEnd::Failed => {
                    self.report.batches_failed += 1;
                    stopped_scopes.insert(batch.scope.clone());
                }
                BatchEnd::Halted => break,
            }
        }
        self.report.completed_offset = self.completed_offset();
        info!(
            written = self.report.facts_written,
            dropped = self.report.facts_dropped,
            failed = self.report.batches_failed,
            deadline_hit = self.report.deadline_hit,
            sidecar_lost = self.report.sidecar_lost,
            "extraction run finished"
        );
        Ok(())
    }

    fn plan_batches(&self) -> Vec<Batch> {
        let mut by_scope: BTreeMap<&Option<String>, Vec<usize>> = BTreeMap::new();
        for (position, input) in self.inputs.iter().enumerate() {
            by_scope.entry(&input.scope).or_default().push(position);
        }
        let mut batches: Vec<Batch> = by_scope
            .into_iter()
            .flat_map(|(scope, positions)| {
                positions
                    .chunks(self.deps.config.batch_events)
                    .map(|chunk| Batch {
                        scope: scope.clone(),
                        indices: chunk.to_vec(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        batches.sort_by_key(|batch| batch.indices[0]);
        batches
    }

    async fn run_batch(&mut self, batch: &[usize]) -> Result<BatchEnd> {
        match self.attempt(batch).await? {
            Attempt::Done => {
                self.mark(batch, Status::Done);
                Ok(BatchEnd::Done)
            }
            Attempt::Failed => {
                self.mark(batch, Status::Failed);
                Ok(BatchEnd::Failed)
            }
            Attempt::Halted => Ok(BatchEnd::Halted),
        }
    }

    fn mark(&mut self, positions: &[usize], status: Status) {
        for &position in positions {
            self.status[position] = status;
        }
    }

    async fn attempt(&mut self, piece: &[usize]) -> Result<Attempt> {
        let inputs = Arc::clone(&self.inputs);
        let batch: Vec<&ExtractionInput> = piece.iter().map(|&i| &inputs[i]).collect();
        let request = self.request(&batch);
        let key = cache_key(&KeyParts {
            prompt_version: PROMPT_VERSION,
            identity: self.deps.identity,
            request: &request,
        });
        let response = match self.deps.cache.get(&key).filter(|text| is_parseable(text)) {
            Some(text) => {
                debug!(events = batch.len(), "answered from the response cache");
                text
            }
            None => match self.generate(&request).await? {
                Generation::Text(text) => text,
                Generation::Stop(stop) => return Ok(stop),
            },
        };
        self.report.model = Some(self.deps.identity.clone());
        let parsed = match parse_response(&response, batch.len(), &self.predicates) {
            Ok(parsed) => parsed,
            Err(error) => {
                warn!(%error, events = batch.len(), "the model's answer could not be read");
                return Ok(Attempt::Failed);
            }
        };
        let ctx = ApplyContext {
            store: self.deps.store,
            config: self.deps.config,
            vocabulary: &self.vocabulary,
            write_source: self.deps.write_source,
        };
        match apply_answer(&ctx, &batch, parsed).await {
            Ok(counts) => {
                self.report.facts_written += counts.written;
                self.report.facts_dropped += counts.dropped;
                self.report.facts_dropped_ungrounded += counts.dropped_ungrounded;
                self.report.ended_no_match += counts.ended_no_match;
                self.report.ended_outranked += counts.ended_outranked;
                debug!(
                    written = counts.written,
                    dropped = counts.dropped,
                    "batch applied"
                );
                Ok(Attempt::Done)
            }
            Err(ApplyError::Rejected(why)) => {
                warn!(%why, events = batch.len(), "the store refused the batch's facts");
                Ok(Attempt::Failed)
            }
            Err(ApplyError::Store(error)) => Err(error.into()),
        }
    }

    fn request(&self, batch: &[&ExtractionInput]) -> ExtractRequest {
        let events: Vec<PromptEvent> = batch
            .iter()
            .map(|input| PromptEvent { text: &input.text })
            .collect();
        let (system, prompt) = build_prompt(&events);
        let answer_tokens = self.deps.config.ctx_tokens / ANSWER_CONTEXT_DIVISOR;
        ExtractRequest {
            system,
            prompt,
            schema: self.schema.clone(),
            source_event_ids: batch.iter().map(|input| input.event_id.clone()).collect(),
            schema_hash: self.schema_hash.clone(),
            temperature: EXTRACTION_TEMPERATURE,
            seed: EXTRACTION_SEED,
            max_tokens: u32::try_from(answer_tokens).unwrap_or(u32::MAX),
        }
    }

    /// Asks the model, starting it first if this is the run's first miss.
    async fn generate(&mut self, request: &ExtractRequest) -> Result<Generation> {
        if !self.ensure_session().await {
            return Ok(Generation::Stop(Attempt::Halted));
        }
        let cached = Arc::clone(&self.session.as_ref().expect("session was opened").cached);
        match cached.extract_json(request).await {
            Ok(text) => Ok(Generation::Text(text)),
            Err(error) => {
                warn!(%error, "the model call failed");
                Ok(Generation::Stop(Attempt::Failed))
            }
        }
    }

    /// False when the model could not be started, which ends the run.
    async fn ensure_session(&mut self) -> bool {
        if self.session.is_some() {
            return true;
        }
        match self.deps.provider.open(self.deps.config).await {
            Ok(session) => {
                let cached = CachedExtractor::new(
                    session.extractor,
                    Arc::clone(&self.deps.cache),
                    PROMPT_VERSION.to_string(),
                    is_parseable,
                );
                self.session = Some(Opened {
                    cached: Arc::new(cached),
                    _keepalive: session._keepalive,
                });
                true
            }
            Err(error) => {
                warn!(%error, "could not start the extraction model");
                self.report.sidecar_lost = true;
                false
            }
        }
    }

    /// Per scope, the last input before the first one not done, or the window
    /// end when all are. The run's offset is the lowest of them.
    fn completed_offset(&self) -> Option<LogOffset> {
        let mut by_scope: BTreeMap<&Option<String>, Vec<usize>> = BTreeMap::new();
        for (position, input) in self.inputs.iter().enumerate() {
            by_scope.entry(&input.scope).or_default().push(position);
        }
        let mut lowest = self.deps.window_end;
        for positions in by_scope.values() {
            let Some(unfinished) = positions
                .iter()
                .position(|&p| self.status[p] != Status::Done)
            else {
                continue;
            };
            let before = unfinished.checked_sub(1)?;
            lowest = lowest.min(self.inputs[positions[before]].offset);
        }
        Some(lowest)
    }
}

#[cfg(test)]
mod tests;
