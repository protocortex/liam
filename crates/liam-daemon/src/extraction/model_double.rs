// SPDX-License-Identifier: Apache-2.0
//! A model for tests of the extraction hook: it answers every event written
//! as "<Name> works at <Org>." with that fact, counts its calls, and can lose
//! its process on a chosen call.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::bail;
use async_trait::async_trait;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use super::client::{ExtractError, ExtractRequest, JsonExtractor, ModelIdentity};
use super::extractor::{ExtractorProvider, ExtractorSession};
use crate::config::ExtractionConfig;

pub const MODEL_ID: &str = "test-model";
pub const MODEL_SHA: &str = "abc123";

pub fn identity() -> ModelIdentity {
    ModelIdentity {
        model_id: MODEL_ID.into(),
        model_sha256: MODEL_SHA.into(),
    }
}

/// The text of an event the double reads a fact from.
pub fn employment(person: &str, org: &str) -> String {
    format!("{person} works at {org}.")
}

pub struct ModelDouble {
    identity: ModelIdentity,
    calls: AtomicUsize,
    alive: Arc<AtomicBool>,
    /// The call, counted from 1, after which the process is gone.
    dies_on_call: Option<usize>,
    /// The call, counted from 1, that cancels the token and never answers.
    stops_on_call: Option<(usize, CancellationToken)>,
    /// A prompt holding this text gets an answer that is not JSON.
    garbles: Mutex<Option<String>>,
    /// The call, counted from 1, that answers only after the pause.
    slow_on_call: Option<(usize, Duration)>,
}

impl ModelDouble {
    pub fn garble(&self, text: &str) {
        *self.garbles.lock().unwrap() = Some(text.to_owned());
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl JsonExtractor for ModelDouble {
    async fn extract_json(&self, req: &ExtractRequest) -> Result<String, ExtractError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.dies_on_call == Some(call) {
            self.alive.store(false, Ordering::SeqCst);
        }
        if let Some((stop_call, token)) = &self.stops_on_call {
            if *stop_call == call {
                token.cancel();
                std::future::pending::<()>().await;
            }
        }
        if let Some((slow_call, pause)) = self.slow_on_call {
            if slow_call == call {
                tokio::time::sleep(pause).await;
            }
        }
        let garbled = self.garbles.lock().unwrap();
        if garbled
            .as_deref()
            .is_some_and(|text| req.prompt.contains(text))
        {
            return Ok("not json".to_owned());
        }
        Ok(facts_for(&req.prompt))
    }

    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    async fn health(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }
}

/// Opens one shared `ModelDouble`, or refuses to when told the model cannot
/// start.
pub struct ProviderDouble {
    pub model: Arc<ModelDouble>,
    opens: AtomicUsize,
    fail_open: AtomicBool,
}

impl ProviderDouble {
    pub fn new() -> Arc<Self> {
        Self::with(None, None, None)
    }

    /// The model's `call`-th answer takes `pause`.
    pub fn slow_on_call(call: usize, pause: Duration) -> Arc<Self> {
        Self::with(None, None, Some((call, pause)))
    }

    /// The model's process is gone after its `call`-th answer.
    pub fn dying_on_call(call: usize) -> Arc<Self> {
        Self::with(Some(call), None, None)
    }

    /// The daemon begins stopping during the model's `call`-th answer.
    pub fn stopping_on_call(call: usize, token: CancellationToken) -> Arc<Self> {
        Self::with(None, Some((call, token)), None)
    }

    fn with(
        dies_on_call: Option<usize>,
        stops_on_call: Option<(usize, CancellationToken)>,
        slow_on_call: Option<(usize, Duration)>,
    ) -> Arc<Self> {
        Arc::new(Self {
            model: Arc::new(ModelDouble {
                identity: identity(),
                calls: AtomicUsize::new(0),
                alive: Arc::new(AtomicBool::new(true)),
                dies_on_call,
                stops_on_call,
                garbles: Mutex::new(None),
                slow_on_call,
            }),
            opens: AtomicUsize::new(0),
            fail_open: AtomicBool::new(false),
        })
    }

    pub fn fail_open(&self) {
        self.fail_open.store(true, Ordering::SeqCst);
    }

    pub fn opens(&self) -> usize {
        self.opens.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl ExtractorProvider for ProviderDouble {
    async fn open(&self, _config: &ExtractionConfig) -> anyhow::Result<ExtractorSession> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        if self.fail_open.load(Ordering::SeqCst) {
            bail!("the model server cannot start");
        }
        let alive = Arc::clone(&self.model.alive);
        Ok(ExtractorSession {
            extractor: self.model.clone(),
            alive: Box::new(move || alive.load(Ordering::SeqCst)),
            _keepalive: None,
        })
    }
}

/// One fact per event of the prompt that reads "<Name> works at <Org>.".
fn facts_for(prompt: &str) -> String {
    let facts: Vec<_> = event_texts(prompt)
        .iter()
        .enumerate()
        .filter_map(|(number, text)| {
            let (person, org) = text.strip_suffix('.')?.split_once(" works at ")?;
            Some(json!({
                "subject": person.to_lowercase(), "predicate": "works_at", "object": org,
                "polarity": "asserted", "valid_from": null, "valid_until": null,
                "confidence": 0.9, "events": [number],
            }))
        })
        .collect();
    json!({ "facts": facts }).to_string()
}

fn event_texts(prompt: &str) -> Vec<&str> {
    prompt
        .split("<event id=\"")
        .skip(1)
        .filter_map(|chunk| {
            let body = chunk.split_once("\">\n")?.1;
            Some(body.split_once("\n</event>")?.0)
        })
        .collect()
}
