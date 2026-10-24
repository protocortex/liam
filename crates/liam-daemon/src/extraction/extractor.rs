// SPDX-License-Identifier: Apache-2.0
//! The model half of extraction: batches the inputs of a scan window per
//! scope, asks the model (or the response cache) about each batch, and writes
//! the facts that survive grounding.
//!
//! Nothing here holds a store lock across a model call; a batch's facts are
//! one short write. An answer is cached before its facts are applied, when the
//! cache file can be written, and every write is idempotent. A run that dies at
//! any point is finished by the next one without duplicating a fact; it calls
//! the model again only for an answer the cache could not keep.

use std::any::Any;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Result};
use async_trait::async_trait;
use liam_log::LogOffset;
use liam_store::{DefaultGraph, Vocabulary};
use serde_json::Value;
use tokio::sync::Semaphore;
use tokio::time::Instant;
use tracing::{debug, info, warn};

use super::apply::{apply_answer, ApplyContext, ApplyError};
use super::cache::{cache_key, CacheStore, CachedExtractor, KeyParts};
use super::client::{ExtractError, ExtractRequest, JsonExtractor, ModelIdentity};
use super::inputs::ExtractionInput;
use super::prompt::{build_prompt, fact_json_schema, prompt_fingerprint, prompt_template};
use super::prompt::{PromptEvent, PROMPT_VERSION};
use super::response::parse_response;
use super::sidecar::FreeRam;
use crate::config::{declared_predicates, resolve_vocabulary, ExtractionConfig};

/// Extraction must repeat exactly, so sampling is pinned.
const EXTRACTION_SEED: u32 = 0;
const EXTRACTION_TEMPERATURE: f32 = 0.0;
/// The answer may use this share of the context window; the prompt has the rest.
const ANSWER_CONTEXT_DIVISOR: usize = 4;
/// Replies that say the request itself is unacceptable (a prompt over the
/// context window is a 400 from llama-server), so a retry repeats the failure.
const CONTENT_REJECTION_STATUSES: [u16; 3] = [400, 413, 422];

/// A running model and what must stay alive to keep it running.
pub struct ExtractorSession {
    pub extractor: Arc<dyn JsonExtractor>,
    /// False once the model process is gone.
    pub alive: Box<dyn Fn() -> bool + Send + Sync>,
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
    /// The daemon's shared generation permits.
    pub permits: &'a Semaphore,
    pub free_ram: &'a (dyn FreeRam + Sync),
    /// The log source the facts are written under.
    pub write_source: &'a str,
    pub deadline: Instant,
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
    /// Skipped for good after repeated failures; counts as passed.
    Quarantined,
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
    /// The answer or the facts are bad, which asking again repeats.
    Content(String),
    /// The call outlasted its limit while the model was still answering health checks.
    Timeout,
    /// The call failed in a way that says nothing about the events.
    Transport,
    Halted,
}

enum Generation {
    Text(String),
    Stop(Attempt),
}

struct Opened {
    cached: Arc<CachedExtractor>,
    alive: Arc<dyn Fn() -> bool + Send + Sync>,
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
    /// Inputs with failures recorded in the store, to clear when they succeed.
    failing: HashSet<usize>,
    /// Single events whose call timed out, settled once the run's evidence is in.
    timeouts: Vec<usize>,
    /// Inputs a real model call was made for.
    attempted: HashSet<usize>,
    /// The model answered a call this run (or a bisection sibling of a timed
    /// out call was done), so a timeout is about the event, not the limit.
    model_answered: bool,
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

/// Halves a piece of two or more inputs; a single input cannot be split.
fn split(piece: &[usize]) -> Option<(Vec<usize>, Vec<usize>)> {
    (piece.len() > 1).then(|| {
        let (first, second) = piece.split_at(piece.len() / 2);
        (first.to_vec(), second.to_vec())
    })
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
            failing: HashSet::new(),
            timeouts: Vec::new(),
            attempted: HashSet::new(),
            model_answered: false,
            deps,
        }
    }

    async fn execute(&mut self) -> Result<()> {
        self.skip_quarantined().await?;
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
            if self.expired() {
                self.report.deadline_hit = true;
                break;
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
        self.settle_timeouts().await?;
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
            if self.status[position] != Status::Quarantined {
                by_scope.entry(&input.scope).or_default().push(position);
            }
        }
        let mut batches: Vec<Batch> = by_scope
            .into_iter()
            .flat_map(|(scope, positions)| self.chunk(scope, &positions))
            .collect();
        batches.sort_by_key(|batch| batch.indices[0]);
        batches
    }

    /// Cuts one scope's inputs into batches of at most `batch_events` events
    /// and `max_batch_chars` characters, so a batch fits the context window. An
    /// event alone over the character bound still gets a batch of its own.
    fn chunk(&self, scope: &Option<String>, positions: &[usize]) -> Vec<Batch> {
        let (max_events, max_chars) = (
            self.deps.config.batch_events,
            self.deps.config.max_batch_chars(),
        );
        let mut batches: Vec<Batch> = Vec::new();
        let mut chars = 0;
        for &position in positions {
            let length = self.inputs[position].text.chars().count();
            let fits = batches.last().is_some_and(|batch| {
                batch.indices.len() < max_events && chars + length <= max_chars
            });
            if fits {
                batches
                    .last_mut()
                    .expect("a batch was open")
                    .indices
                    .push(position);
                chars += length;
            } else {
                batches.push(Batch {
                    scope: scope.clone(),
                    indices: vec![position],
                });
                chars = length;
            }
        }
        batches
    }

    fn expired(&self) -> bool {
        Instant::now() >= self.deps.deadline
    }

    fn remaining(&self) -> Duration {
        self.deps.deadline.saturating_duration_since(Instant::now())
    }

    /// Runs one batch. A batch the model cannot answer is split in half and
    /// each half asked again, so one bad event costs only itself.
    async fn run_batch(&mut self, batch: &[usize]) -> Result<BatchEnd> {
        let mut queue: VecDeque<(Vec<usize>, bool)> = VecDeque::from([(batch.to_vec(), false)]);
        let mut failed = false;
        while let Some((piece, after_timeout)) = queue.pop_front() {
            match self.attempt(&piece).await? {
                Attempt::Done => {
                    self.mark(&piece, Status::Done);
                    self.clear_failures(&piece).await?;
                    self.model_answered |= after_timeout;
                }
                Attempt::Halted => return Ok(BatchEnd::Halted),
                Attempt::Transport => return Ok(BatchEnd::Failed),
                Attempt::Content(reason) => match split(&piece) {
                    Some((first, second)) => {
                        queue.push_front((second, after_timeout));
                        queue.push_front((first, after_timeout));
                    }
                    None => failed |= self.fail_single(piece[0], &reason).await?,
                },
                Attempt::Timeout => match split(&piece) {
                    Some((first, second)) => {
                        queue.push_front((second, true));
                        queue.push_front((first, true));
                    }
                    None => {
                        self.timeouts.push(piece[0]);
                        self.mark(&piece, Status::Failed);
                        failed = true;
                    }
                },
            }
        }
        Ok(if failed {
            BatchEnd::Failed
        } else {
            BatchEnd::Done
        })
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
            None => match self.generate(&request, piece).await? {
                Generation::Text(text) => text,
                Generation::Stop(stop) => return Ok(stop),
            },
        };
        self.report.model = Some(self.deps.identity.clone());
        let parsed = match parse_response(&response, batch.len(), &self.predicates) {
            Ok(parsed) => parsed,
            Err(error) => {
                warn!(%error, events = batch.len(), "the model's answer could not be read");
                return Ok(Attempt::Content(error.to_string()));
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
                Ok(Attempt::Content(why))
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
    async fn generate(&mut self, request: &ExtractRequest, piece: &[usize]) -> Result<Generation> {
        if self.expired() {
            self.report.deadline_hit = true;
            return Ok(Generation::Stop(Attempt::Halted));
        }
        if !self.ensure_session().await {
            return Ok(Generation::Stop(Attempt::Halted));
        }
        let opened = self.session.as_ref().expect("the session was just opened");
        let (cached, alive) = (Arc::clone(&opened.cached), Arc::clone(&opened.alive));
        let free_mb = self.deps.free_ram.free_mb();
        if !alive() || free_mb < self.deps.config.ram_budget_mb {
            warn!(
                free_mb,
                "the model is gone or memory ran out, stopping the run"
            );
            self.report.sidecar_lost = true;
            return Ok(Generation::Stop(Attempt::Halted));
        }
        let Ok(permit) = tokio::time::timeout(self.remaining(), self.deps.permits.acquire()).await
        else {
            self.report.deadline_hit = true;
            return Ok(Generation::Stop(Attempt::Halted));
        };
        let Ok(_permit) = permit else {
            bail!("the generation permits are closed");
        };
        // The wait for a permit can be long; the model may have died in it.
        if !alive() {
            warn!("the model died while the run waited for a generation permit");
            self.report.sidecar_lost = true;
            return Ok(Generation::Stop(Attempt::Halted));
        }
        let limit = self
            .remaining()
            .min(Duration::from_secs(self.deps.config.request_timeout_secs));
        self.attempted.extend(piece);
        match tokio::time::timeout(limit, cached.extract_json(request)).await {
            Ok(Ok(text)) => {
                self.model_answered = true;
                Ok(Generation::Text(text))
            }
            // The deadline cut the call short, which is the run ending, not a failure.
            Err(_) | Ok(Err(ExtractError::Timeout)) if self.expired() => {
                self.report.deadline_hit = true;
                Ok(Generation::Stop(Attempt::Halted))
            }
            Err(_) | Ok(Err(ExtractError::Timeout)) => {
                // A model that is gone also times out; it must not be blamed on the events.
                if cached.health().await {
                    warn!("the model call timed out");
                    Ok(Generation::Stop(Attempt::Timeout))
                } else {
                    warn!("the model call timed out and the model does not answer");
                    self.report.sidecar_lost = true;
                    Ok(Generation::Stop(Attempt::Halted))
                }
            }
            Ok(Err(ExtractError::Status(status)))
                if CONTENT_REJECTION_STATUSES.contains(&status) =>
            {
                warn!(status, "the model rejected the request");
                Ok(Generation::Stop(Attempt::Content(format!(
                    "the model rejected the request with HTTP {status}"
                ))))
            }
            Ok(Err(error)) => {
                warn!(%error, "the model call failed");
                Ok(Generation::Stop(Attempt::Transport))
            }
        }
    }

    /// False when the model could not be started, which ends the run.
    async fn ensure_session(&mut self) -> bool {
        if self.session.is_some() {
            return true;
        }
        match self.deps.provider.open(self.deps.config).await {
            Ok(session) if session.extractor.identity() != self.deps.identity => {
                warn!("the model that started is not the one the run expects");
                self.report.sidecar_lost = true;
                false
            }
            Ok(session) => {
                let cached = CachedExtractor::new(
                    session.extractor,
                    Arc::clone(&self.deps.cache),
                    PROMPT_VERSION.to_string(),
                    is_parseable,
                );
                self.session = Some(Opened {
                    cached: Arc::new(cached),
                    alive: Arc::from(session.alive),
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
                .position(|&p| !matches!(self.status[p], Status::Done | Status::Quarantined))
            else {
                continue;
            };
            let before = unfinished.checked_sub(1)?;
            lowest = lowest.min(self.inputs[positions[before]].offset);
        }
        Some(lowest)
    }
}

mod failures;

#[cfg(test)]
mod tests;
