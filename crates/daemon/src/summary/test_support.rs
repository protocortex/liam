// SPDX-License-Identifier: Apache-2.0
//! Doubles for the summary tests. `MockLlm` echoes its whole prompt, which would
//! make the grounding and injection tests meaningless, so replies are scripted.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use liam_model::{Llm, ModelError, Result};

use super::generate::SummaryCache;

/// Replies in order, counts its calls and keeps every prompt it was given.
#[derive(Default)]
pub struct ScriptedLlm {
    replies: Mutex<VecDeque<std::result::Result<String, String>>>,
    calls: AtomicUsize,
    prompts: Mutex<Vec<(String, String)>>,
    count_chars: bool,
}

impl ScriptedLlm {
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

    /// Counts tokens as characters, so a test can size a budget exactly.
    pub fn counting_chars(mut self) -> Self {
        self.count_chars = true;
        self
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// The (system, user) pair of every call so far.
    pub fn prompts(&self) -> Vec<(String, String)> {
        self.prompts.lock().unwrap().clone()
    }
}

#[async_trait]
impl Llm for ScriptedLlm {
    async fn complete(&self, system: &str, prompt: &str) -> Result<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.prompts
            .lock()
            .unwrap()
            .push((system.to_string(), prompt.to_string()));
        match self.replies.lock().unwrap().pop_front() {
            Some(Ok(reply)) => Ok(reply),
            Some(Err(error)) => Err(ModelError::Llm(error)),
            None => Err(ModelError::Llm("no scripted reply left".to_string())),
        }
    }

    fn count_tokens(&self, text: &str) -> Option<usize> {
        self.count_chars.then(|| text.chars().count())
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
