// SPDX-License-Identifier: Apache-2.0
//! Doubles for the summary tests. Replies are scripted, so the grounding and
//! injection tests control exactly what the model said.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use async_trait::async_trait;

use super::generate::SummaryCache;
use super::summariser::{Summariser, SummaryRequest, SummaryResponse};
use crate::extraction::client::ModelIdentity;

/// Replies in order, counts its calls and keeps every prompt it was given.
#[derive(Default)]
pub struct ScriptedReplies {
    replies: Mutex<VecDeque<std::result::Result<String, String>>>,
    calls: AtomicUsize,
    prompts: Mutex<Vec<(String, String)>>,
}

impl ScriptedReplies {
    pub fn replying(replies: &[&str]) -> Self {
        Self {
            replies: Mutex::new(replies.iter().map(|r| Ok((*r).to_string())).collect()),
            ..Self::default()
        }
    }

    pub fn failing(error: &str) -> Self {
        Self {
            replies: Mutex::new(VecDeque::from([Err(error.to_string())])),
            ..Self::default()
        }
    }

    /// Each call takes the next outcome, an error text for a failed call.
    pub fn scripted(outcomes: Vec<std::result::Result<String, String>>) -> Self {
        Self {
            replies: Mutex::new(outcomes.into()),
            ..Self::default()
        }
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// The (system, user) pair of every call so far.
    pub fn prompts(&self) -> Vec<(String, String)> {
        self.prompts.lock().unwrap().clone()
    }
}

/// A [`Summariser`] that answers from a [`ScriptedReplies`].
pub struct ScriptedSummariser {
    replies: std::sync::Arc<ScriptedReplies>,
    identity: ModelIdentity,
}

impl ScriptedSummariser {
    pub fn new(replies: std::sync::Arc<ScriptedReplies>, identity: ModelIdentity) -> Self {
        Self { replies, identity }
    }
}

#[async_trait]
impl Summariser for ScriptedSummariser {
    async fn summarise(&self, req: &SummaryRequest) -> Result<SummaryResponse> {
        let replies = &self.replies;
        replies.calls.fetch_add(1, Ordering::SeqCst);
        replies
            .prompts
            .lock()
            .unwrap()
            .push((req.system.clone(), req.prompt.clone()));
        match replies.replies.lock().unwrap().pop_front() {
            Some(Ok(text)) => Ok(SummaryResponse { text, usage: None }),
            Some(Err(error)) => Err(anyhow!(error)),
            None => Err(anyhow!("no scripted reply left")),
        }
    }

    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }
}

/// A cache in memory that remembers the events each entry cites.
#[derive(Default)]
pub struct MemoryCache {
    entries: Mutex<HashMap<String, (String, Vec<String>)>>,
}

impl MemoryCache {
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    pub fn sources_of(&self, key: &str) -> Option<Vec<String>> {
        self.entries
            .lock()
            .unwrap()
            .get(key)
            .map(|(_, sources)| sources.clone())
    }
}

impl SummaryCache for MemoryCache {
    fn get(&self, key: &str) -> Option<String> {
        self.entries
            .lock()
            .unwrap()
            .get(key)
            .map(|(text, _)| text.clone())
    }

    fn put(&self, key: &str, text: &str, source_event_ids: &[String]) -> io::Result<()> {
        self.entries.lock().unwrap().insert(
            key.to_string(),
            (text.to_string(), source_event_ids.to_vec()),
        );
        Ok(())
    }
}
