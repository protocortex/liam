// SPDX-License-Identifier: Apache-2.0
//! R10 for the hosted providers, in two tiers. The recorded tier replays
//! committed provider responses through the real adapters and the submit and
//! collect flow, scores the facts the pipeline stores, and runs in normal CI.
//! The live tier asks the real provider, needs a key, is `#[ignore]`d, and is
//! advisory.
//!
//! No provider key was available when this was written, so the recorded
//! responses are replayed from the corpus labels with the deviations in
//! `r10_recorded.json`, shaped the way each provider's batch API returns them.
//! They show the wiring is right and the gate can fail; they do not measure a
//! model.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use super::batch::{AnthropicBatch, GeminiBatch, HostedBatch, OpenAiBatch};
use super::client::HostedClient;
use super::cost::CostRecorder;
use super::extraction_batch::HostedBatchApi;
use super::model::hosted_identity;
use super::secrets::SecretString;
use super::test_support::{metrics, price, KEY, PRICE_VERSION};
use super::transport::{HttpRequest, HttpResponse, HttpTransport, Method, TransportError};
use super::{anthropic, gemini, openai};
use crate::config::{ExtractionConfig, HostedConfig, ModelPin, PricesConfig, Provider, TokenUsage};
use crate::deletion::batch_store::{batch_store_path_for, BatchStore};
use crate::deletion::send_ledger::{send_ledger_path_for, SendLedger};
use crate::extraction::batch_flow::BatchFlow;
use crate::extraction::client::ModelIdentity;
use crate::extraction::extractor::{ExtractorProvider, ExtractorSession};
use crate::extraction::prompt::PROMPT_VERSION;
use crate::extraction::r10::corpus::{generate, ExpectedFact, LabelledEvent, CORPUS_SEED};
use crate::extraction::r10::tiers::{model_labels, run_pipeline_batched, PlentyOfRam};
use crate::extraction::r10::{score, Score};
use crate::extraction::response::Polarity;
use liam_store::{FixedClock, Millis};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// The lowest F1 a recorded fixture may score and still pass. The recorded
/// fixtures score 1, 74/75 and 58/59; the floor leaves room for a fixture that
/// is a little worse and fails one that loses a fifth of the labels.
const RECORDED_FLOOR_F1: f64 = 0.95;
/// Events per request, so the 300 events make 43 requests.
const BATCH_EVENTS: usize = 7;
/// The first run sends, the second collects and writes the facts.
const RUNS: usize = 2;
/// A quality gate for a live run, and how far below its recorded F1 a live
/// median may fall before the advisory says so.
const LIVE_MARGIN: f64 = 0.05;
/// How many times a live run asks, so one unlucky draw does not decide.
const LIVE_REPEATS: usize = 3;
/// The usage a model is priced on when the cheapest passing one is picked: one
/// extraction batch of a million tokens in and a tenth of that out.
const REFERENCE_USAGE: TokenUsage = TokenUsage {
    input_tokens: 1_000_000,
    output_tokens: 100_000,
    cached_input_tokens: 0,
};
const ANSWER_CONFIDENCE: f64 = 0.9;
const RECORDED: &str = include_str!("r10_recorded.json");
const COMMITTED_RESULTS: &str = include_str!("r10_results.toml");
const RECORDED_USAGE: (u64, u64) = (100, 50);

/// What one provider's recorded answers get wrong.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    model_id: String,
    /// Events whose facts the recorded answer leaves out.
    #[serde(default)]
    missed: BTreeSet<usize>,
    /// Events whose fact keeps only the first word of its object.
    #[serde(default)]
    wrong_object: BTreeSet<usize>,
}

fn fixtures() -> BTreeMap<String, Fixture> {
    serde_json::from_str(RECORDED).expect("r10_recorded.json is valid")
}

/// One row of the live results table.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultRow {
    provider: String,
    model_id: String,
    prompt_version: String,
    date: String,
    price_table_version: String,
    f1: f64,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Results {
    rows: Vec<ResultRow>,
}

impl Results {
    fn committed() -> Self {
        toml::from_str(COMMITTED_RESULTS).expect("r10_results.toml is valid")
    }

    /// Fails with what to do when a pin or the prompt differs from a row.
    fn check_bound(&self, pins: &[ModelPin], prompt_version: &str) -> Result<(), String> {
        let mut stale = Vec::new();
        for row in &self.rows {
            if row.prompt_version != prompt_version {
                stale.push(format!(
                    "prompt_version is {prompt_version}, the {} row has {}",
                    row.provider, row.prompt_version
                ));
            }
        }
        for pin in pins {
            let rows: Vec<&ResultRow> = self
                .rows
                .iter()
                .filter(|row| row.provider == pin.provider)
                .collect();
            if !rows.is_empty() && rows.iter().all(|row| row.model_id != pin.model_id) {
                stale.push(format!(
                    "the {} pin is {}, the results have {}",
                    pin.provider, pin.model_id, rows[0].model_id
                ));
            }
        }
        if stale.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "the hosted R10 results are out of date ({}): run the live tier and update \
                 r10_results.toml",
                stale.join("; ")
            ))
        }
    }
}

/// The cheapest recorded model whose F1 reaches `min_f1`, priced on a batch of
/// `REFERENCE_USAGE`; ties go to the lower provider and model id. A model with
/// no price is not a candidate.
fn cheapest_passing(prices: &PricesConfig, results: &Results, min_f1: f64) -> Option<ModelPin> {
    results
        .rows
        .iter()
        .filter(|row| row.f1 >= min_f1)
        .filter_map(|row| {
            let cost = prices
                .models
                .get(&row.model_id)?
                .cost_micro(&REFERENCE_USAGE, true);
            Some((cost, &row.provider, &row.model_id))
        })
        .min()
        .and_then(|(_, provider, model_id)| ModelPin::new(provider, model_id).ok())
}

/// The median of `values`, the mean of the two middle ones for an even count.
fn median(mut values: Vec<f64>) -> Option<f64> {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    match values.len() {
        0 => None,
        n if n % 2 == 1 => Some(values[middle]),
        _ => Some((values[middle - 1] + values[middle]) / 2.0),
    }
}

fn corpus() -> Vec<LabelledEvent> {
    generate(&mut ChaCha8Rng::seed_from_u64(CORPUS_SEED))
}

/// The answer a provider recorded for a prompt: the labels of each event in
/// it, minus what the fixture gets wrong.
fn recorded_answer(prompt: &str, events: &[LabelledEvent], fixture: &Fixture) -> String {
    let by_text: BTreeMap<&str, &LabelledEvent> = events
        .iter()
        .map(|event| (event.text.as_str(), event))
        .collect();
    let facts: Vec<Value> = prompt
        .split("<event id=\"")
        .skip(1)
        .filter_map(|chunk| chunk.split_once("\">\n")?.1.split_once("\n</event>"))
        .enumerate()
        .flat_map(|(position, (text, _))| {
            let event = by_text[text];
            let kept = !fixture.missed.contains(&event.index);
            let wrong = fixture.wrong_object.contains(&event.index);
            event
                .facts
                .iter()
                .filter(move |_| kept)
                .map(move |fact| fact_json(fact, position, wrong))
                .collect::<Vec<_>>()
        })
        .collect();
    json!({ "facts": facts }).to_string()
}

fn fact_json(fact: &ExpectedFact, position: usize, wrong_object: bool) -> Value {
    let object = if wrong_object {
        fact.object.split(' ').next().unwrap_or_default()
    } else {
        &fact.object
    };
    json!({
        "subject": fact.subject.to_lowercase(),
        "predicate": fact.predicate,
        "object": object,
        "polarity": if fact.polarity == Polarity::Ended { "ended" } else { "asserted" },
        "valid_from": null,
        "valid_until": null,
        "confidence": ANSWER_CONFIDENCE,
        "events": [position],
    })
}

/// A provider's batch API replayed from a fixture: it keeps the requests it is
/// sent and answers them all at once.
#[derive(Clone)]
struct RecordedProvider {
    provider: Provider,
    fixture: Fixture,
    events: Arc<Vec<LabelledEvent>>,
    retired: bool,
    state: Arc<Mutex<Recorded>>,
}

#[derive(Default)]
struct Recorded {
    /// Requests by upload, then by batch: (custom id, prompt).
    uploads: Vec<Vec<(String, String)>>,
    batches: Vec<Vec<(String, String)>>,
    urls: Vec<String>,
}

impl RecordedProvider {
    fn new(provider: Provider, fixture: Fixture, events: Arc<Vec<LabelledEvent>>) -> Self {
        Self {
            provider,
            fixture,
            events,
            retired: false,
            state: Arc::default(),
        }
    }

    fn urls(&self) -> Vec<String> {
        self.state.lock().unwrap().urls.clone()
    }

    fn answer(&self, prompt: &str) -> String {
        recorded_answer(prompt, &self.events, &self.fixture)
    }

    fn model(&self) -> &str {
        &self.fixture.model_id
    }

    fn reply(&self, body: Value) -> Result<HttpResponse, TransportError> {
        Ok(HttpResponse {
            status: 200,
            body: serde_json::to_vec(&body).unwrap(),
        })
    }

    fn text(&self, status: u16, body: String) -> Result<HttpResponse, TransportError> {
        Ok(HttpResponse {
            status,
            body: body.into_bytes(),
        })
    }

    /// The provider-shaped reply that carries `answer` from this model.
    fn message(&self, answer: &str) -> Value {
        let (input, output) = RECORDED_USAGE;
        match self.provider {
            Provider::OpenAi => json!({
                "model": self.model(),
                "choices": [{ "message": { "content": answer } }],
                "usage": { "prompt_tokens": input, "completion_tokens": output },
            }),
            Provider::Gemini => json!({
                "modelVersion": self.model(),
                "candidates": [{ "content": { "parts": [{ "text": answer }] } }],
                "usageMetadata": {
                    "promptTokenCount": input, "candidatesTokenCount": output
                },
            }),
            _ => json!({
                "model": self.model(),
                "content": [{ "type": "text", "text": answer }],
                "usage": { "input_tokens": input, "output_tokens": output },
            }),
        }
    }

    fn results(&self, requests: &[(String, String)]) -> Vec<(String, Value)> {
        requests
            .iter()
            .map(|(key, prompt)| (key.clone(), self.message(&self.answer(prompt))))
            .collect()
    }
}

fn parse_batch_index(id: &str, prefix: &str) -> usize {
    id.trim_start_matches(prefix).parse().expect("a batch id")
}

#[async_trait]
impl HttpTransport for RecordedProvider {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
        let path = request
            .url
            .split_once("://")
            .and_then(|(_, rest)| rest.split_once('/'))
            .map_or(String::new(), |(_, path)| format!("/{path}"));
        self.state.lock().unwrap().urls.push(path.clone());
        if request.method == Method::Delete {
            return self.text(200, "{}".to_owned());
        }
        let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
        let submitting = request.method == Method::Post
            && (path.ends_with("/messages/batches")
                || path.ends_with("/v1/batches")
                || path.ends_with(":batchGenerateContent"));
        if self.retired && submitting {
            return self.text(404, "{}".to_owned());
        }
        match self.provider {
            Provider::Anthropic => self.anthropic(&path, &body),
            Provider::OpenAi => self.openai(&path, &body, &request.body),
            _ => self.gemini(&path, &body),
        }
    }
}

impl RecordedProvider {
    fn anthropic(&self, path: &str, body: &Value) -> Result<HttpResponse, TransportError> {
        let mut state = self.state.lock().unwrap();
        if path.ends_with("/messages/batches") {
            let requests = body["requests"]
                .as_array()
                .unwrap()
                .iter()
                .map(|entry| {
                    (
                        entry["custom_id"].as_str().unwrap().to_owned(),
                        entry["params"]["messages"][0]["content"]
                            .as_str()
                            .unwrap()
                            .to_owned(),
                    )
                })
                .collect();
            state.batches.push(requests);
            return self.reply(json!({ "id": format!("msgbatch_{}", state.batches.len() - 1) }));
        }
        let index = parse_batch_index(
            path.rsplit('/')
                .nth(usize::from(path.ends_with("/results")))
                .unwrap(),
            "msgbatch_",
        );
        if path.ends_with("/results") {
            let lines: Vec<String> = self
                .results(&state.batches[index])
                .into_iter()
                .map(|(key, message)| {
                    json!({ "custom_id": key, "result": { "type": "succeeded", "message": message } })
                        .to_string()
                })
                .collect();
            return self.text(200, lines.join("\n"));
        }
        self.reply(json!({ "processing_status": "ended" }))
    }

    fn openai(&self, path: &str, body: &Value, raw: &[u8]) -> Result<HttpResponse, TransportError> {
        let mut state = self.state.lock().unwrap();
        if path == "/v1/files" {
            let requests = String::from_utf8_lossy(raw)
                .lines()
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .filter(|line| line.get("custom_id").is_some())
                .map(|line| {
                    (
                        line["custom_id"].as_str().unwrap().to_owned(),
                        line["body"]["messages"][1]["content"]
                            .as_str()
                            .unwrap()
                            .to_owned(),
                    )
                })
                .collect();
            state.uploads.push(requests);
            return self.reply(json!({ "id": format!("file-{}", state.uploads.len() - 1) }));
        }
        if path == "/v1/batches" {
            let upload = parse_batch_index(body["input_file_id"].as_str().unwrap(), "file-");
            let requests = state.uploads[upload].clone();
            state.batches.push(requests);
            return self.reply(json!({ "id": format!("batch_{}", state.batches.len() - 1) }));
        }
        if let Some(id) = path.strip_prefix("/v1/batches/") {
            let index = parse_batch_index(id, "batch_");
            return self.reply(json!({
                "status": "completed",
                "input_file_id": format!("file-{index}"),
                "output_file_id": format!("out-{index}"),
            }));
        }
        let index = parse_batch_index(
            path.trim_start_matches("/v1/files/")
                .trim_end_matches("/content"),
            "out-",
        );
        let lines: Vec<String> = self
            .results(&state.batches[index])
            .into_iter()
            .map(|(key, message)| {
                json!({ "custom_id": key, "response": { "status_code": 200, "body": message } })
                    .to_string()
            })
            .collect();
        self.text(200, lines.join("\n"))
    }

    fn gemini(&self, path: &str, body: &Value) -> Result<HttpResponse, TransportError> {
        let mut state = self.state.lock().unwrap();
        if path.ends_with(":batchGenerateContent") {
            let requests = body["batch"]["input_config"]["requests"]["requests"]
                .as_array()
                .unwrap()
                .iter()
                .map(|entry| {
                    (
                        entry["metadata"]["key"].as_str().unwrap().to_owned(),
                        entry["request"]["contents"][0]["parts"][0]["text"]
                            .as_str()
                            .unwrap()
                            .to_owned(),
                    )
                })
                .collect();
            state.batches.push(requests);
            return self
                .reply(json!({ "name": format!("batches/r10{}", state.batches.len() - 1) }));
        }
        let index = parse_batch_index(path.rsplit('/').next().unwrap(), "r10");
        let inlined: Vec<Value> = self
            .results(&state.batches[index])
            .into_iter()
            .map(|(key, message)| json!({ "metadata": { "key": key }, "response": message }))
            .collect();
        self.reply(json!({
            "metadata": { "state": "BATCH_STATE_SUCCEEDED" },
            "response": { "inlinedResponses": { "inlinedResponses": inlined } },
        }))
    }
}

/// A provider the loop must never ask directly: hosted extraction goes through
/// the batch flow.
struct NeverAskedDirectly;

#[async_trait]
impl ExtractorProvider for NeverAskedDirectly {
    async fn open(&self, _config: &ExtractionConfig) -> Result<ExtractorSession> {
        anyhow::bail!("the recorded tier must not ask the provider directly")
    }
}

fn pin_of(provider: Provider, fixture: &Fixture) -> ModelPin {
    ModelPin::new(provider.name(), &fixture.model_id).expect("a dated pin")
}

/// The real batch adapter for `provider` over `transport`.
fn adapter(
    provider: Provider,
    fixture: &Fixture,
    transport: RecordedProvider,
) -> Arc<dyn HostedBatch> {
    let host = match provider {
        Provider::Anthropic => anthropic::HOST,
        Provider::OpenAi => openai::HOST,
        _ => gemini::HOST,
    };
    let config = HostedConfig {
        enabled: true,
        allowed_hosts: vec![host.to_owned()],
        ..HostedConfig::default()
    };
    let (metrics, _) = metrics();
    let prices = PricesConfig {
        version: PRICE_VERSION.to_owned(),
        models: [(fixture.model_id.clone(), price())].into(),
    };
    let client = HostedClient::new(transport, &config);
    let (pin, key, costs) = (
        pin_of(provider, fixture),
        SecretString::new(KEY),
        CostRecorder::new(metrics, prices),
    );
    match provider {
        Provider::Anthropic => Arc::new(AnthropicBatch::new(client, pin, key, costs)),
        Provider::OpenAi => Arc::new(OpenAiBatch::new(client, pin, key, costs)),
        _ => Arc::new(GeminiBatch::new(client, pin, key, costs)),
    }
}

/// What a recorded run produced.
struct RecordedRun {
    score: Score,
    requests_submitted: usize,
    batches_failed: usize,
    facts_written: usize,
    urls: Vec<String>,
}

/// Runs the corpus through `provider`'s adapter and the submit and collect
/// flow, as a recorded fixture says the provider answered.
async fn run_recorded(provider: Provider, fixture: Fixture, retired: bool) -> RecordedRun {
    let events = Arc::new(corpus());
    let mut transport = RecordedProvider::new(provider, fixture.clone(), Arc::clone(&events));
    transport.retired = retired;
    let batch = adapter(provider, &fixture, transport.clone());
    let identity: ModelIdentity = hosted_identity(&pin_of(provider, &fixture));
    let config = ExtractionConfig {
        ram_budget_mb: 1,
        batch_events: BATCH_EVENTS,
        ..ExtractionConfig::default()
    };
    let run = run_pipeline_batched(
        &events,
        &NeverAskedDirectly,
        &identity,
        config,
        &PlentyOfRam,
        RUNS,
        |cache, dir| {
            let database = dir.join("liam.db");
            Some(BatchFlow::new(
                Arc::new(HostedBatchApi::new(provider, {
                    let batch = Arc::clone(&batch);
                    move || Ok(Arc::clone(&batch))
                })),
                BatchStore::new(batch_store_path_for(&database)),
                SendLedger::new(send_ledger_path_for(&database)),
                cache,
                Arc::new(FixedClock::new(Millis(0))),
            ))
        },
    )
    .await;
    RecordedRun {
        score: score(&run.extracted(&events).await, &model_labels(&events)),
        requests_submitted: run.report.requests_submitted,
        batches_failed: run.report.batches_failed,
        facts_written: run.report.facts_written,
        urls: transport.urls(),
    }
}

fn provider_of(name: &str) -> Provider {
    Provider::from_name(name).expect("a provider name")
}

// A recorded tier test per provider. Each fixture is derived from the 150
// labels of the corpus (30 cycles of 5 events that state something).

#[tokio::test]
async fn the_clean_anthropic_fixture_scores_every_label() {
    // Arrange: nothing missed or wrong, so tp = 150, fp = 0, fn = 0 and F1 = 1.
    let fixture = fixtures()["anthropic"].clone();

    // Act
    let run = run_recorded(Provider::Anthropic, fixture, false).await;

    // Assert
    assert_eq!((run.score.tp, run.score.fp, run.score.fn_), (150, 0, 0));
    assert_eq!(run.score.f1, 1.0);
    assert_eq!(run.facts_written, 150);
    assert_eq!(run.requests_submitted, 0, "the second run only collects");
}

#[tokio::test]
async fn the_openai_fixture_with_two_wrong_objects_scores_74_over_75() {
    // Arrange: two title events keep the first word of their object. Each is
    // a false positive and a miss: tp = 148, fp = 2, fn = 2, and
    // F1 = 2 * 148 / (2 * 148 + 2 + 2) = 296 / 300 = 74 / 75.
    let fixture = fixtures()["openai"].clone();

    // Act
    let run = run_recorded(Provider::OpenAi, fixture, false).await;

    // Assert
    assert_eq!((run.score.tp, run.score.fp, run.score.fn_), (148, 2, 2));
    assert!((run.score.f1 - 74.0 / 75.0).abs() < 1e-12);
    assert!(run.score.f1 >= RECORDED_FLOOR_F1);
}

#[tokio::test]
async fn the_gemini_fixture_with_five_missed_events_scores_58_over_59() {
    // Arrange: five title events are left out of the answer: tp = 145, fp = 0,
    // fn = 5, and F1 = 2 * 145 / (2 * 145 + 0 + 5) = 290 / 295 = 58 / 59.
    let fixture = fixtures()["gemini"].clone();

    // Act
    let run = run_recorded(Provider::Gemini, fixture, false).await;

    // Assert
    assert_eq!((run.score.tp, run.score.fp, run.score.fn_), (145, 0, 5));
    assert!((run.score.f1 - 58.0 / 59.0).abs() < 1e-12);
    assert!(run.score.f1 >= RECORDED_FLOOR_F1);
}

#[tokio::test]
async fn every_provider_sends_the_corpus_as_one_batch_and_never_asks_directly() {
    for (name, fixture) in fixtures() {
        // Arrange
        let provider = provider_of(&name);

        // Act
        let run = run_recorded(provider, fixture, false).await;

        // Assert: 300 events at 7 per request is 43 requests, in one batch
        let batch_calls = run
            .urls
            .iter()
            .filter(|url| {
                url.ends_with("/messages/batches")
                    || url.as_str() == "/v1/batches"
                    || url.ends_with(":batchGenerateContent")
            })
            .count();
        assert_eq!(batch_calls, 1, "{name}");
        let direct = run.urls.iter().any(|url| {
            url == "/v1/messages"
                || url == "/v1/chat/completions"
                || url.ends_with(":generateContent")
        });
        assert!(!direct, "{name}");
        assert!(run.facts_written > 0, "{name}");
    }
}

#[tokio::test]
async fn each_hosted_adapter_passes_r10_in_the_recorded_tier() {
    // Arrange: the F1 of each committed fixture, as derived above
    let expected = [
        (Provider::Anthropic, 1.0),
        (Provider::OpenAi, 74.0 / 75.0),
        (Provider::Gemini, 58.0 / 59.0),
    ];

    for (provider, f1) in expected {
        // Act
        let run = run_recorded(provider, fixtures()[provider.name()].clone(), false).await;

        // Assert
        assert!((run.score.f1 - f1).abs() < 1e-12, "{}", provider.name());
        assert!(run.score.f1 >= RECORDED_FLOOR_F1, "{}", provider.name());
    }
}

#[tokio::test]
async fn a_degraded_fixture_falls_below_the_floor_by_a_hand_computed_amount() {
    // Arrange: all 30 title events keep only the first word of their object:
    // tp = 120, fp = 30, fn = 30, and F1 = 240 / (240 + 60) = 4 / 5, which is
    // under the 0.95 floor.
    let mut fixture = fixtures()["anthropic"].clone();
    fixture.wrong_object = (0..30).map(|cycle| cycle * 10 + 4).collect();

    // Act
    let run = run_recorded(Provider::Anthropic, fixture, false).await;

    // Assert
    assert_eq!((run.score.tp, run.score.fp, run.score.fn_), (120, 30, 30));
    assert!((run.score.f1 - 0.8).abs() < 1e-12);
    assert!(run.score.f1 < RECORDED_FLOOR_F1);
}

#[tokio::test]
async fn one_wrong_object_lowers_f1_to_149_over_150() {
    // Arrange: tp = 149, fp = 1, fn = 1, and F1 = 298 / 300 = 149 / 150.
    let mut fixture = fixtures()["anthropic"].clone();
    fixture.wrong_object = BTreeSet::from([4]);

    // Act
    let run = run_recorded(Provider::Anthropic, fixture, false).await;

    // Assert
    assert_eq!((run.score.tp, run.score.fp, run.score.fn_), (149, 1, 1));
    assert!((run.score.f1 - 149.0 / 150.0).abs() < 1e-12);
}

#[tokio::test]
async fn a_retired_pin_fails_closed_and_the_tier_reports_it() {
    for (name, fixture) in fixtures() {
        // Arrange and Act
        let run = run_recorded(provider_of(&name), fixture, true).await;

        // Assert: nothing was sent on, nothing was stored, and the run says so
        assert_eq!(run.requests_submitted, 0, "{name}");
        assert_eq!(run.facts_written, 0, "{name}");
        assert!(run.batches_failed > 0, "{name}");
        assert_eq!(run.score.tp, 0, "{name}");
    }
}

#[test]
fn every_fixture_names_the_pin_it_was_recorded_for() {
    // Arrange
    let fixtures = fixtures();

    // Act and Assert
    for provider in [Provider::Anthropic, Provider::OpenAi, Provider::Gemini] {
        let fixture = &fixtures[provider.name()];
        assert!(
            ModelPin::new(provider.name(), &fixture.model_id).is_ok(),
            "{}",
            provider.name()
        );
    }
}

fn row(provider: &str, model_id: &str, f1: f64) -> ResultRow {
    ResultRow {
        provider: provider.to_owned(),
        model_id: model_id.to_owned(),
        prompt_version: PROMPT_VERSION.to_owned(),
        date: "2026-11-04".to_owned(),
        price_table_version: PRICE_VERSION.to_owned(),
        f1,
    }
}

#[test]
fn the_committed_results_are_bound_to_the_recorded_pins_and_this_prompt() {
    // Arrange
    let fixtures = fixtures();
    let pins: Vec<ModelPin> = ["anthropic", "openai", "gemini"]
        .iter()
        .map(|name| pin_of(provider_of(name), &fixtures[*name]))
        .collect();

    // Act
    let bound = Results::committed().check_bound(&pins, PROMPT_VERSION);

    // Assert
    assert_eq!(bound, Ok(()));
}

#[test]
fn a_changed_pin_or_prompt_version_fails_with_a_clear_message() {
    // Arrange
    let results = Results {
        rows: vec![row("openai", "gpt-4o-2024-08-06", 0.9)],
    };
    let moved = ModelPin::new("openai", "gpt-4o-2024-11-20").unwrap();

    // Act
    let pin = results.check_bound(&[moved], PROMPT_VERSION).unwrap_err();
    let prompt = results.check_bound(&[], "next").unwrap_err();

    // Assert
    assert!(pin.contains("the openai pin is gpt-4o-2024-11-20"), "{pin}");
    assert!(pin.contains("update r10_results.toml"), "{pin}");
    assert!(!pin.contains("prompt_version"), "{pin}");
    assert!(prompt.contains("prompt_version is next"), "{prompt}");
}

#[test]
fn the_cheapest_model_that_passes_the_gate_is_picked() {
    // Arrange: the cheapest model fails the gate, so the next cheapest wins
    let rate = |batch_input: u64| crate::config::Price {
        batch_input_micro_per_mtok: batch_input,
        batch_output_micro_per_mtok: batch_input,
        ..crate::config::Price::default()
    };
    let prices = PricesConfig {
        version: PRICE_VERSION.to_owned(),
        models: [
            ("claude-sonnet-4-5-20250929".to_owned(), rate(300)),
            ("gpt-4o-2024-08-06".to_owned(), rate(100)),
            ("gemini-2.0-flash-001".to_owned(), rate(200)),
        ]
        .into(),
    };
    let results = Results {
        rows: vec![
            row("anthropic", "claude-sonnet-4-5-20250929", 0.95),
            row("openai", "gpt-4o-2024-08-06", 0.4),
            row("gemini", "gemini-2.0-flash-001", 0.9),
        ],
    };

    // Act
    let picked = cheapest_passing(&prices, &results, 0.8);
    let none = cheapest_passing(&prices, &results, 0.99);

    // Assert
    assert_eq!(
        picked,
        Some(ModelPin::new("gemini", "gemini-2.0-flash-001").unwrap())
    );
    assert_eq!(none, None);
}

#[test]
fn the_median_of_live_repeats_is_the_middle_value() {
    // Arrange and Act
    let odd = median(vec![0.9, 0.5, 0.7]);
    let even = median(vec![0.5, 0.9, 0.7, 0.6]);
    let empty = median(Vec::new());

    // Assert
    assert_eq!(odd, Some(0.7));
    assert!((even.unwrap() - 0.65).abs() < 1e-12);
    assert_eq!(empty, None);
}

#[test]
fn the_live_tier_repeat_count_and_margin_are_named_constants() {
    // Assert
    assert_eq!(LIVE_REPEATS, 3);
    assert!((LIVE_MARGIN - 0.05).abs() < f64::EPSILON);
}

/// The live tier: asks the real provider, `LIVE_REPEATS` times, and compares
/// the median F1 with the recorded row. Run with the key file named by
/// `LIAM_R10_KEY_FILE` and the pin by `LIAM_R10_MODEL`, for example
/// `LIAM_R10_PROVIDER=anthropic cargo test --features hosted -- --ignored r10_live`.
/// Advisory: it prints what it found and never fails on the score.
#[tokio::test]
#[ignore = "calls a real provider and needs a key"]
async fn r10_live() {
    use crate::config::Config;
    use crate::extraction::r10::tiers::run_pipeline;
    use crate::hosted::summariser::HostedExtractorProvider;

    // Arrange
    let provider = provider_of(&std::env::var("LIAM_R10_PROVIDER").expect("LIAM_R10_PROVIDER"));
    let model_id = std::env::var("LIAM_R10_MODEL").expect("LIAM_R10_MODEL");
    let key_file = std::env::var("LIAM_R10_KEY_FILE").expect("LIAM_R10_KEY_FILE");
    let host = match provider {
        Provider::Anthropic => anthropic::HOST,
        Provider::OpenAi => openai::HOST,
        _ => gemini::HOST,
    };
    let scratch = tempfile::tempdir().unwrap();
    let config = Config {
        database_path: scratch
            .path()
            .join("liam.db")
            .to_string_lossy()
            .into_owned(),
        hosted: HostedConfig {
            enabled: true,
            allowed_hosts: vec![host.to_owned()],
            models: [(provider.name().to_owned(), model_id.clone())].into(),
            key_files: [(provider.name().to_owned(), key_file)].into(),
            ..HostedConfig::default()
        },
        ..Config::default()
    };
    let (metrics, _) = metrics();
    let events = corpus();
    let identity = hosted_identity(&ModelPin::new(provider.name(), &model_id).unwrap());
    let extraction = ExtractionConfig {
        ram_budget_mb: 1,
        batch_events: BATCH_EVENTS,
        ..ExtractionConfig::default()
    };

    // Act
    let mut f1s = Vec::new();
    for _ in 0..LIVE_REPEATS {
        let opener = {
            let (config, metrics) = (config.clone(), metrics.clone());
            move || super::open_model(&config, provider, metrics.clone())
        };
        let asked = run_pipeline(
            &events,
            &HostedExtractorProvider::new(opener),
            &identity,
            extraction.clone(),
            &PlentyOfRam,
        )
        .await;
        f1s.push(score(&asked.extracted(&events).await, &model_labels(&events)).f1);
    }

    // Assert
    let found = median(f1s).expect("at least one repeat");
    let recorded = Results::committed()
        .rows
        .into_iter()
        .find(|row| row.provider == provider.name() && row.model_id == model_id);
    match recorded {
        Some(row) if found + LIVE_MARGIN < row.f1 => {
            eprintln!("R10 live (advisory): median F1 {found:.3} is below the recorded {:.3} by more than {LIVE_MARGIN}", row.f1);
        }
        Some(row) => eprintln!("R10 live: median F1 {found:.3}, recorded {:.3}", row.f1),
        None => eprintln!("R10 live: median F1 {found:.3}, no recorded row for {model_id}"),
    }
}
