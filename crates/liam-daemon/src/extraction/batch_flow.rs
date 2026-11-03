// SPDX-License-Identifier: Apache-2.0
//! Submit and collect for a provider that answers extraction requests in
//! batches. A run collects every batch that has ended into the response cache
//! (never into facts), then sends only the requests the cache cannot answer
//! and the table does not already hold. The table and the cache sit outside
//! the database snapshot, so a rolled back run leaves both as they were and the
//! next run replays from them.
//!
//! Every step writes before it moves on, so a run that dies anywhere resumes
//! from the table: the only send that can repeat is one whose answer from the
//! provider never reached the table, and that is reported.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use liam_store::Clock;
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

use super::cache::{CacheStore, NewEntry};
use super::client::{ExtractRequest, ModelIdentity};
use super::extractor::is_parseable;
use super::prompt::PROMPT_VERSION;
use crate::deletion::batch_store::{
    BatchEntry, BatchState, BatchStore, EntryFailure, FIRST_ATTEMPT,
};
use crate::deletion::send_ledger::{SendLedger, SendRow};

/// A request is sent again once after it expires, and no more.
const MAX_ATTEMPTS: u32 = 2;

/// Where a batch is at the provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "hosted"), allow(dead_code))]
pub enum RemoteState {
    InProgress,
    Ended,
    /// The provider's window passed; what it finished can still be collected.
    Expired,
    /// The provider refused the batch as a whole.
    Failed,
    Cancelled,
}

/// What the provider holds for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(not(feature = "hosted"), allow(dead_code))]
pub enum RemoteResult {
    Answer(String),
    /// The provider refused the request or answered as another model than the
    /// pin. Nothing about the events is to blame.
    Refused,
    Expired,
    Cancelled,
}

/// The provider side of a batch, in the terms the extraction loop needs.
#[async_trait]
pub trait BatchApi: Send + Sync {
    /// The provider's name, as `[hosted.models]` spells it.
    fn provider(&self) -> &str;

    /// Sends the requests, each under its cache key, and returns the batch id.
    async fn submit(&self, requests: &[(String, ExtractRequest)]) -> Result<String>;

    async fn poll(&self, batch_id: &str) -> Result<RemoteState>;

    /// Every result the provider holds, by cache key.
    async fn collect(&self, batch_id: &str) -> Result<Vec<(String, RemoteResult)>>;
}

/// A step of the flow where a test stops it, standing in for a crash. The step
/// before it has been written and the step after it has not started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchFault {
    BeforeSubmit,
    AfterSubmitBeforeTable,
    AfterTable,
    AfterCollectBeforeCache,
    /// After `n` results are in the cache.
    PartwayThroughCache(usize),
    AfterCacheBeforeCollected,
}

#[cfg(test)]
#[derive(Debug)]
pub struct Crashed(pub BatchFault);

#[cfg(test)]
impl std::fmt::Display for Crashed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stopped at {:?}", self.0)
    }
}

#[cfg(test)]
impl std::error::Error for Crashed {}

/// What collecting did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Collected {
    /// Results validated and written to the cache.
    pub cached: usize,
}

/// What submitting did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Submitted {
    pub requests: usize,
    /// Requests an earlier send may already have put with the provider, so
    /// they may be billed twice.
    pub possible_duplicates: usize,
}

/// What the table says about a request the cache cannot answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Standing {
    /// The provider holds it; wait.
    Pending,
    /// The provider's answer failed validation.
    Invalid,
    /// The provider's answer is unusable for a reason no event is to blame for.
    Refused,
    /// Nothing is in flight: send it.
    Unsent,
}

pub struct BatchFlow {
    api: Arc<dyn BatchApi>,
    batches: BatchStore,
    ledger: SendLedger,
    cache: Arc<CacheStore>,
    clock: Arc<dyn Clock>,
    #[cfg(test)]
    fault: Option<BatchFault>,
}

impl BatchFlow {
    pub fn new(
        api: Arc<dyn BatchApi>,
        batches: BatchStore,
        ledger: SendLedger,
        cache: Arc<CacheStore>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            api,
            batches,
            ledger,
            cache,
            clock,
            #[cfg(test)]
            fault: None,
        }
    }

    #[cfg(test)]
    pub fn stopping_at(mut self, fault: BatchFault) -> Self {
        self.fault = Some(fault);
        self
    }

    fn crash_at(&self, point: BatchFault) -> Result<()> {
        #[cfg(test)]
        if self.fault == Some(point) {
            return Err(Crashed(point).into());
        }
        let _ = point;
        Ok(())
    }

    /// Collects every batch the provider has finished into the cache. A batch
    /// the provider cannot be asked about now is left for the next run.
    pub async fn collect_ended(&self, identity: &ModelIdentity) -> Result<Collected> {
        let mut entries = self.batches.load()?;
        let live: BTreeSet<String> = entries
            .iter()
            .filter(|entry| entry.state.is_live())
            .map(|entry| entry.batch_id.clone())
            .collect();
        let mut collected = Collected::default();
        for batch_id in live {
            let outcome = self
                .collect_batch(&mut entries, &batch_id, identity, &mut collected)
                .await;
            match outcome {
                Ok(()) => {}
                Err(error) if is_crash(&error) => return Err(error),
                Err(error) => {
                    warn!(error = %format!("{error:#}"), "a hosted batch could not be collected, it is tried again next run");
                }
            }
        }
        Ok(collected)
    }

    async fn collect_batch(
        &self,
        entries: &mut [BatchEntry],
        batch_id: &str,
        identity: &ModelIdentity,
        collected: &mut Collected,
    ) -> Result<()> {
        let of_batch = |entry: &BatchEntry| entry.batch_id == batch_id && entry.state.is_live();
        // The cache already answers every request: a run died after writing
        // the answers and before marking them, so no provider call is needed.
        if entries
            .iter()
            .filter(|entry| of_batch(entry))
            .all(|entry| self.cache.contains(&entry.cache_key))
        {
            for entry in entries.iter_mut().filter(|entry| of_batch(entry)) {
                entry.state = BatchState::Collected;
            }
            self.batches.replace(entries.to_vec())?;
            return Ok(());
        }
        let state = self.api.poll(batch_id).await?;
        let results = match state {
            RemoteState::InProgress => return Ok(()),
            RemoteState::Failed => {
                for entry in entries.iter_mut().filter(|entry| of_batch(entry)) {
                    entry.state = BatchState::Failed;
                    entry.failure = Some(EntryFailure::Refused);
                }
                return self.batches.replace(entries.to_vec());
            }
            RemoteState::Cancelled => {
                for entry in entries.iter_mut().filter(|entry| of_batch(entry)) {
                    entry.state = BatchState::Cancelled;
                }
                return self.batches.replace(entries.to_vec());
            }
            RemoteState::Ended | RemoteState::Expired => {
                for entry in entries.iter_mut().filter(|entry| of_batch(entry)) {
                    entry.state = BatchState::Ended;
                }
                self.batches.replace(entries.to_vec())?;
                self.api.collect(batch_id).await?
            }
        };
        self.crash_at(BatchFault::AfterCollectBeforeCache)?;
        let mut answered: HashSet<String> = HashSet::new();
        for (cache_key, result) in results {
            // A result for a request the table no longer holds as live was
            // cancelled or deleted, and is dropped.
            let Some(entry) = entries
                .iter_mut()
                .find(|entry| entry.cache_key == cache_key && of_batch(entry))
            else {
                debug!("a hosted result for a request that is no longer live was dropped");
                continue;
            };
            answered.insert(cache_key.clone());
            match result {
                RemoteResult::Answer(text) if is_parseable(&text) => {
                    self.cache
                        .put(NewEntry {
                            key: &cache_key,
                            response: &text,
                            model_id: &identity.model_id,
                            model_sha256: &identity.model_sha256,
                            prompt_version: PROMPT_VERSION,
                            source_event_ids: &entry.source_event_ids,
                        })
                        .context("could not write a hosted result to the cache")?;
                    collected.cached += 1;
                    self.crash_at(BatchFault::PartwayThroughCache(collected.cached))?;
                    entry.state = BatchState::Collected;
                }
                RemoteResult::Answer(_) => {
                    entry.state = BatchState::Failed;
                    entry.failure = Some(EntryFailure::Invalid);
                }
                RemoteResult::Refused => {
                    entry.state = BatchState::Failed;
                    entry.failure = Some(EntryFailure::Refused);
                }
                RemoteResult::Expired => expire(entry),
                RemoteResult::Cancelled => entry.state = BatchState::Cancelled,
            }
        }
        // Requests the provider returned nothing for: an ended batch lost
        // them, an expired one never ran them.
        for entry in entries
            .iter_mut()
            .filter(|entry| entry.batch_id == batch_id && entry.state == BatchState::Ended)
        {
            if answered.contains(&entry.cache_key) {
                continue;
            }
            if state == RemoteState::Expired {
                expire(entry);
            } else {
                entry.state = BatchState::Failed;
                entry.failure = Some(EntryFailure::Refused);
            }
        }
        self.crash_at(BatchFault::AfterCacheBeforeCollected)?;
        self.batches.replace(entries.to_vec())?;
        info!(batch = batch_id, "a hosted batch was collected");
        Ok(())
    }

    /// What the table says about each request key.
    pub fn standings(&self) -> Result<BTreeMap<String, Standing>> {
        Ok(self
            .batches
            .load()?
            .into_iter()
            .map(|entry| {
                let standing = match entry.state {
                    BatchState::Submitted | BatchState::Ended => Standing::Pending,
                    BatchState::Failed if entry.failure == Some(EntryFailure::Invalid) => {
                        Standing::Invalid
                    }
                    BatchState::Failed => Standing::Refused,
                    BatchState::Collected | BatchState::Expired | BatchState::Cancelled => {
                        Standing::Unsent
                    }
                };
                (entry.cache_key, standing)
            })
            .collect())
    }

    /// Drops the failed entry of `cache_key` once the run has counted it, so a
    /// later run may send the request afresh.
    pub fn discard(&self, cache_key: &str) -> Result<()> {
        let mut entries = self.batches.load()?;
        let before = entries.len();
        entries
            .retain(|entry| !(entry.cache_key == cache_key && entry.state == BatchState::Failed));
        if entries.len() != before {
            self.batches.replace(entries)?;
        }
        Ok(())
    }

    /// Sends `requests` in one batch. Each request's ledger row is written
    /// first, so a delete finds it whatever happens next, and the table is
    /// written as soon as the provider names the batch.
    pub async fn submit(&self, requests: Vec<(String, ExtractRequest)>) -> Result<Submitted> {
        if requests.is_empty() {
            return Ok(Submitted::default());
        }
        let mut entries = self.batches.load()?;
        let sent = self.ledger.load()?;
        let provider = self.api.provider().to_owned();
        let mut possible_duplicates = 0;
        for (cache_key, request) in &requests {
            let known = entries.iter().any(|entry| &entry.cache_key == cache_key);
            let recorded = sent.iter().any(|row| &row.cache_key == cache_key);
            if recorded && !known {
                possible_duplicates += 1;
            } else if !recorded {
                self.ledger.append(&SendRow {
                    provider: provider.clone(),
                    cache_key: cache_key.clone(),
                    source_event_ids: request.source_event_ids.clone(),
                    sent_at: self.clock.now().0,
                })?;
            }
        }
        if possible_duplicates > 0 {
            warn!(
                requests = possible_duplicates,
                provider = %provider,
                "an earlier send may have reached the provider, so these requests may be billed twice"
            );
        }
        self.crash_at(BatchFault::BeforeSubmit)?;
        let batch_id = self.api.submit(&requests).await?;
        self.crash_at(BatchFault::AfterSubmitBeforeTable)?;
        let now = self.clock.now().0;
        for (cache_key, request) in &requests {
            let previous = entries
                .iter()
                .position(|entry| &entry.cache_key == cache_key);
            let attempts = match previous.map(|at| &entries[at]) {
                Some(entry) if entry.state == BatchState::Expired => entry.attempts + 1,
                _ => FIRST_ATTEMPT,
            };
            let entry = BatchEntry {
                cache_key: cache_key.clone(),
                provider: provider.clone(),
                batch_id: batch_id.clone(),
                state: BatchState::Submitted,
                submitted_at: now,
                input_hash: input_hash(request),
                source_event_ids: request.source_event_ids.clone(),
                attempts,
                failure: None,
            };
            match previous {
                Some(at) => entries[at] = entry,
                None => entries.push(entry),
            }
        }
        self.batches.replace(entries)?;
        self.crash_at(BatchFault::AfterTable)?;
        info!(requests = requests.len(), provider = %provider, "a hosted batch was submitted");
        Ok(Submitted {
            requests: requests.len(),
            possible_duplicates,
        })
    }
}

/// An expired request is sent once more, and a second expiry fails it.
fn expire(entry: &mut BatchEntry) {
    if entry.attempts < MAX_ATTEMPTS {
        entry.state = BatchState::Expired;
    } else {
        entry.state = BatchState::Failed;
        entry.failure = Some(EntryFailure::Refused);
    }
}

/// A digest of what was asked, so the table can tell two sends apart without
/// holding the text.
fn input_hash(request: &ExtractRequest) -> String {
    let mut digest = Sha256::new();
    digest.update(request.system.as_bytes());
    digest.update([0]);
    digest.update(request.prompt.as_bytes());
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
pub(super) fn is_crash(error: &anyhow::Error) -> bool {
    error.downcast_ref::<Crashed>().is_some()
}

#[cfg(not(test))]
pub(super) fn is_crash(_error: &anyhow::Error) -> bool {
    false
}
