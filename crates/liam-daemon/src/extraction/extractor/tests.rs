// SPDX-License-Identifier: Apache-2.0
//! The batch loop against a real file store, a real cache file and a scripted
//! model.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::bail;
use liam_store::{GraphConfig, Millis};
use serde_json::json;
use std::time::Duration;
use tempfile::TempDir;

use super::super::cache::{cache_key, cache_path, KeyParts, NewEntry};
use super::super::client::ExtractError;
use super::*;
use crate::config::ExtractionConfig;

const CLIENT: &str = "claude-code";
const MODEL_SHA: &str = "abc123";
const EMBEDDING_DIMS: usize = 8;
const READ_LIMIT: usize = 10;
const HOUR: i64 = 3_600_000;
const OBSERVED: i64 = 1_685_577_600_000;
const READ_AT: i64 = OBSERVED + 1000 * HOUR;
const WINDOW_END: LogOffset = LogOffset {
    segment: 0,
    index: 100,
};
const WRITE_SOURCE: &str = "consolidation:test-run";
const ABANDON_AFTER: Duration = Duration::from_secs(60);
const RUN_BUDGET: Duration = Duration::from_secs(60);
const FAR_FUTURE: Duration = Duration::from_secs(3600);

enum Reply {
    Text(String),
    Fail(ExtractError),
    /// Answers after a pause, which moves a paused clock.
    After(Duration, String),
    Hang,
}

type Handler = Box<dyn Fn(&ExtractRequest, usize) -> Reply + Send + Sync>;

/// Answers each request from a handler that sees the request and the call number.
struct Script {
    identity: ModelIdentity,
    handler: Handler,
    calls: AtomicUsize,
    prompts: Mutex<Vec<String>>,
    requests: Mutex<Vec<ExtractRequest>>,
}

impl Script {
    fn new(handler: impl Fn(&ExtractRequest, usize) -> Reply + Send + Sync + 'static) -> Arc<Self> {
        Self::with_sha(MODEL_SHA, handler)
    }

    fn with_sha(
        sha: &str,
        handler: impl Fn(&ExtractRequest, usize) -> Reply + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            identity: ModelIdentity {
                model_id: "test-model".into(),
                model_sha256: sha.into(),
            },
            handler: Box::new(handler),
            calls: AtomicUsize::new(0),
            prompts: Mutex::new(Vec::new()),
            requests: Mutex::new(Vec::new()),
        })
    }

    /// Reads "<Name> works at <Org>." from each event of the prompt.
    fn honest() -> Arc<Self> {
        Self::new(|request, _| Reply::Text(facts_for(&request.prompt)))
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn prompt(&self, call: usize) -> String {
        self.prompts.lock().unwrap()[call].clone()
    }
}

#[async_trait]
impl JsonExtractor for Script {
    async fn extract_json(&self, req: &ExtractRequest) -> Result<String, ExtractError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        self.prompts.lock().unwrap().push(req.prompt.clone());
        self.requests.lock().unwrap().push(req.clone());
        match (self.handler)(req, call) {
            Reply::Text(text) => Ok(text),
            Reply::Fail(error) => Err(error),
            Reply::After(pause, text) => {
                tokio::time::sleep(pause).await;
                Ok(text)
            }
            Reply::Hang => std::future::pending().await,
        }
    }

    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    async fn health(&self) -> bool {
        true
    }
}

fn prompt_texts(prompt: &str) -> Vec<&str> {
    prompt
        .split("<event id=\"")
        .skip(1)
        .map(|chunk| {
            let body = chunk.split_once("\">\n").expect("event body").1;
            body.split_once("\n</event>").expect("event end").0
        })
        .collect()
}

fn facts_for(prompt: &str) -> String {
    let facts: Vec<_> = prompt_texts(prompt)
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

struct Provider {
    script: Arc<Script>,
    alive: Arc<AtomicBool>,
    opens: AtomicUsize,
    cannot_start: bool,
}

impl Provider {
    fn new(script: &Arc<Script>) -> Self {
        Self {
            script: Arc::clone(script),
            alive: Arc::new(AtomicBool::new(true)),
            opens: AtomicUsize::new(0),
            cannot_start: false,
        }
    }

    fn opens(&self) -> usize {
        self.opens.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl ExtractorProvider for Provider {
    async fn open(&self, _config: &ExtractionConfig) -> Result<ExtractorSession> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        if self.cannot_start {
            bail!("no model to start");
        }
        let alive = Arc::clone(&self.alive);
        Ok(ExtractorSession {
            extractor: self.script.clone(),
            alive: Box::new(move || alive.load(Ordering::SeqCst)),
            _keepalive: None,
        })
    }
}

struct FakeRam(AtomicU64);

impl FreeRam for FakeRam {
    fn free_mb(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

struct Fixture {
    _dir: TempDir,
    store: DefaultGraph,
    cache: Arc<CacheStore>,
    config: ExtractionConfig,
    permits: Semaphore,
    ram: FakeRam,
}

impl Fixture {
    async fn new() -> Self {
        let dir = TempDir::new().expect("tempdir");
        let database = dir.path().join("liam.db");
        let store =
            DefaultGraph::open(database.to_str().unwrap(), GraphConfig::new(EMBEDDING_DIMS))
                .await
                .expect("open store");
        let cache = Arc::new(CacheStore::open(cache_path(&database)).expect("open cache"));
        let config = ExtractionConfig {
            // Host memory must never decide a test.
            ram_budget_mb: 1,
            batch_events: 2,
            ..ExtractionConfig::default()
        };
        Self {
            _dir: dir,
            store,
            cache,
            config,
            permits: Semaphore::new(1),
            ram: FakeRam(AtomicU64::new(u64::MAX)),
        }
    }

    async fn run(&self, provider: &Provider, inputs: &[ExtractionInput]) -> ExtractionReport {
        self.run_full(
            provider,
            &provider.script.identity,
            inputs,
            Instant::now() + FAR_FUTURE,
        )
        .await
    }

    async fn run_expecting(
        &self,
        provider: &Provider,
        identity: &ModelIdentity,
        inputs: &[ExtractionInput],
    ) -> ExtractionReport {
        self.run_full(provider, identity, inputs, Instant::now() + FAR_FUTURE)
            .await
    }

    async fn run_until(
        &self,
        provider: &Provider,
        inputs: &[ExtractionInput],
        deadline: Instant,
    ) -> ExtractionReport {
        self.run_full(provider, &provider.script.identity, inputs, deadline)
            .await
    }

    async fn run_full(
        &self,
        provider: &Provider,
        identity: &ModelIdentity,
        inputs: &[ExtractionInput],
        deadline: Instant,
    ) -> ExtractionReport {
        let deps = ExtractionDeps {
            provider,
            identity,
            cache: Arc::clone(&self.cache),
            store: &self.store,
            config: &self.config,
            permits: &self.permits,
            free_ram: &self.ram,
            write_source: WRITE_SOURCE,
            deadline,
            window_end: WINDOW_END,
        };
        extract_events(inputs, deps).await.expect("a finished run")
    }

    /// How many live facts the store holds about person `n`.
    async fn facts_about(&self, n: u64) -> usize {
        self.store
            .facts_valid_at(&format!("person{n}"), None, Millis(READ_AT), READ_LIMIT)
            .await
            .expect("read facts")
            .facts
            .len()
    }

    async fn assert_every_fact_has_a_source(&self) {
        let orphans = self.store.facts_without_sources().await.expect("scan");
        assert!(orphans.is_empty(), "facts without a source: {orphans:?}");
    }
}

fn offset(n: u64) -> LogOffset {
    LogOffset {
        segment: 0,
        index: n,
    }
}

fn input(n: u64, scope: Option<&str>) -> ExtractionInput {
    let observed = OBSERVED + i64::try_from(n).unwrap() * HOUR;
    ExtractionInput {
        event_id: format!("e{n}"),
        offset: offset(n),
        scope: scope.map(str::to_owned),
        source: CLIENT.into(),
        observed_at: Millis(observed),
        ingested_at: Millis(observed + 1),
        text: format!("Person{n} works at Org{n}."),
    }
}

fn inputs(count: u64) -> Vec<ExtractionInput> {
    (0..count).map(|n| input(n, None)).collect()
}

#[tokio::test]
async fn each_scope_is_batched_apart_and_its_facts_keep_the_scope() {
    // Arrange
    let fx = Fixture::new().await;
    let script = Script::honest();
    let provider = Provider::new(&script);
    let events = [
        input(0, Some("a")),
        input(1, Some("b")),
        input(2, Some("a")),
        input(3, Some("b")),
    ];

    // Act
    let report = fx.run(&provider, &events).await;

    // Assert
    assert_eq!(script.calls(), 2);
    for call in 0..2 {
        let prompt = script.prompt(call);
        let people: Vec<_> = prompt_texts(&prompt)
            .iter()
            .map(|text| text.chars().nth(6).unwrap())
            .collect();
        assert!(people == ['0', '2'] || people == ['1', '3'], "{people:?}");
    }
    let stored = fx
        .store
        .facts_valid_at("person2", None, Millis(READ_AT), READ_LIMIT)
        .await
        .unwrap();
    assert_eq!(stored.facts[0].scope.as_deref(), Some("a"));
    assert_eq!(report.completed_offset, Some(WINDOW_END));
    fx.assert_every_fact_has_a_source().await;
}

#[tokio::test]
async fn batches_fill_to_the_batch_size_with_no_empty_trailing_batch() {
    for (events, batch_events, calls) in [(4, 2, 2), (2, 2, 1), (3, 1, 3), (5, 2, 3), (1, 8, 1)] {
        // Arrange
        let mut fx = Fixture::new().await;
        fx.config.batch_events = batch_events;
        let script = Script::honest();

        // Act
        fx.run(&Provider::new(&script), &inputs(events)).await;

        // Assert
        assert_eq!(
            script.calls(),
            calls,
            "{events} events, batches of {batch_events}"
        );
    }
}

#[tokio::test]
async fn no_inputs_make_no_model_call_and_finish_at_the_window_end() {
    // Arrange
    let fx = Fixture::new().await;
    let script = Script::honest();
    let provider = Provider::new(&script);

    // Act
    let report = fx.run(&provider, &[]).await;

    // Assert
    assert_eq!((script.calls(), provider.opens()), (0, 0));
    assert_eq!(report.completed_offset, Some(WINDOW_END));
    assert_eq!(report.model, None);
}

#[tokio::test]
async fn a_second_run_is_answered_by_the_cache_and_writes_nothing_new() {
    // Arrange
    let fx = Fixture::new().await;
    let first = Script::honest();
    let first_report = fx.run(&Provider::new(&first), &inputs(4)).await;
    let second = Script::honest();
    let provider = Provider::new(&second);

    // Act
    let report = fx.run(&provider, &inputs(4)).await;

    // Assert
    assert_eq!(first_report.facts_written, 4);
    assert_eq!((second.calls(), provider.opens()), (0, 0));
    assert_eq!(report.facts_written, 0);
    assert_eq!(report.completed_offset, Some(WINDOW_END));
    assert_eq!(report.model, first_report.model);
    for n in 0..4 {
        assert_eq!(fx.facts_about(n).await, 1);
    }
    fx.assert_every_fact_has_a_source().await;
}

#[tokio::test]
async fn cached_batches_finish_even_when_the_model_cannot_start() {
    // Arrange
    let fx = Fixture::new().await;
    fx.run(&Provider::new(&Script::honest()), &inputs(2)).await;
    let mut provider = Provider::new(&Script::honest());
    provider.cannot_start = true;

    // Act
    let report = fx.run(&provider, &inputs(2)).await;

    // Assert
    assert_eq!(provider.opens(), 0);
    assert!(!report.sidecar_lost);
    assert_eq!(report.completed_offset, Some(WINDOW_END));
}

#[tokio::test]
async fn a_model_that_cannot_start_ends_the_run_as_lost() {
    // Arrange
    let fx = Fixture::new().await;
    let mut provider = Provider::new(&Script::honest());
    provider.cannot_start = true;

    // Act
    let report = fx.run(&provider, &inputs(4)).await;

    // Assert
    assert!(report.sidecar_lost);
    assert_eq!(report.completed_offset, None);
    assert_eq!(report.batches_failed, 0);
}

#[tokio::test]
async fn an_event_over_the_size_cap_is_cut_reported_and_still_processed() {
    // Arrange
    let mut fx = Fixture::new().await;
    fx.config.max_event_chars = Some(20);
    let script = Script::new(|_, _| Reply::Text(r#"{"facts": []}"#.into()));
    let mut long = input(0, None);
    long.text = format!("{}{}", "a".repeat(20), "b".repeat(80));

    // Act
    let report = fx.run(&Provider::new(&script), &[long]).await;

    // Assert
    assert_eq!(prompt_texts(&script.prompt(0)), ["a".repeat(20)]);
    assert_eq!(report.events_truncated, 1);
    assert_eq!(report.completed_offset, Some(WINDOW_END));
}

#[tokio::test]
async fn a_failed_batch_stops_its_scope_but_not_the_others() {
    // Arrange: a0, a2 (fails), a4 (never tried), b6.
    let mut fx = Fixture::new().await;
    fx.config.batch_events = 1;
    let script = Script::new(|request, call| match call {
        1 => Reply::Fail(ExtractError::Status(500)),
        _ => Reply::Text(facts_for(&request.prompt)),
    });
    let events = [
        input(0, Some("a")),
        input(2, Some("a")),
        input(4, Some("a")),
        input(6, Some("b")),
    ];

    // Act
    let report = fx.run(&Provider::new(&script), &events).await;

    // Assert
    assert_eq!(script.calls(), 3);
    assert_eq!(report.batches_failed, 1);
    assert_eq!(report.completed_offset, Some(offset(0)));
    let written = [
        fx.facts_about(0).await,
        fx.facts_about(2).await,
        fx.facts_about(4).await,
        fx.facts_about(6).await,
    ];
    assert_eq!(written, [1, 0, 0, 1]);
    fx.assert_every_fact_has_a_source().await;
}

#[tokio::test(start_paused = true)]
async fn a_run_dropped_mid_way_is_finished_by_the_next_without_duplicates() {
    // Arrange: the model hangs on the second batch and the run is abandoned.
    let fx = Fixture::new().await;
    let hangs = Script::new(|request, call| {
        if call == 0 {
            Reply::Text(facts_for(&request.prompt))
        } else {
            Reply::Hang
        }
    });
    let abandoned =
        tokio::time::timeout(ABANDON_AFTER, fx.run(&Provider::new(&hangs), &inputs(4))).await;
    let resumed = Script::honest();

    // Act
    let report = fx.run(&Provider::new(&resumed), &inputs(4)).await;

    // Assert
    assert!(abandoned.is_err());
    assert_eq!(resumed.calls(), 1);
    assert_eq!(report.completed_offset, Some(WINDOW_END));
    for n in 0..4 {
        assert_eq!(fx.facts_about(n).await, 1, "person{n}");
    }
    fx.assert_every_fact_has_a_source().await;
}

fn empty_answer() -> Arc<Script> {
    Script::new(|_, _| Reply::Text(r#"{"facts": []}"#.into()))
}

/// An input whose text is `length` characters long.
fn sized(n: u64, length: usize) -> ExtractionInput {
    ExtractionInput {
        text: "x".repeat(length),
        ..input(n, None)
    }
}

#[tokio::test]
async fn inputs_given_out_of_order_are_batched_in_offset_order() {
    // Arrange
    let fx = Fixture::new().await;
    let script = Script::honest();
    let mut events = inputs(4);
    events.reverse();

    // Act
    fx.run(&Provider::new(&script), &events).await;

    // Assert
    let first = script.prompt(0);
    assert_eq!(
        prompt_texts(&first),
        ["Person0 works at Org0.", "Person1 works at Org1."]
    );
}

#[tokio::test]
async fn a_batch_holds_no_more_text_than_its_share_of_the_context_window() {
    // Arrange: the window allows 200 characters per batch.
    let mut fx = Fixture::new().await;
    fx.config.ctx_tokens = 100;
    fx.config.batch_events = 8;
    let script = empty_answer();
    let events = [sized(0, 90), sized(1, 90), sized(2, 90)];

    // Act
    fx.run(&Provider::new(&script), &events).await;

    // Assert
    assert_eq!(script.calls(), 2);
    assert_eq!(prompt_texts(&script.prompt(0)).len(), 2);
    assert_eq!(prompt_texts(&script.prompt(1)).len(), 1);
}

#[tokio::test]
async fn an_event_over_the_batch_bound_still_gets_a_batch_of_its_own() {
    // Arrange
    let mut fx = Fixture::new().await;
    fx.config.ctx_tokens = 100;
    fx.config.max_event_chars = Some(500);
    let script = empty_answer();

    // Act
    let report = fx.run(&Provider::new(&script), &[sized(0, 300)]).await;

    // Assert
    assert_eq!(script.calls(), 1);
    assert_eq!(report.completed_offset, Some(WINDOW_END));
}

#[tokio::test]
async fn a_model_other_than_the_expected_one_ends_the_run_before_any_call() {
    // Arrange
    let fx = Fixture::new().await;
    let script = Script::with_sha("another-model", |_, _| Reply::Text(String::new()));
    let expected = ModelIdentity {
        model_id: "test-model".into(),
        model_sha256: MODEL_SHA.into(),
    };

    // Act
    let report = fx
        .run_expecting(&Provider::new(&script), &expected, &inputs(2))
        .await;

    // Assert
    assert!(report.sidecar_lost);
    assert_eq!(script.calls(), 0);
    assert_eq!(report.completed_offset, None);
}

#[tokio::test]
async fn a_stored_answer_that_cannot_be_read_is_a_miss_and_the_model_is_asked_again() {
    // Arrange: a first run caches the real answer, then the stored line is replaced by garbage.
    let fx = Fixture::new().await;
    let first = Script::honest();
    fx.run(&Provider::new(&first), &inputs(2)).await;
    let request = first.requests.lock().unwrap()[0].clone();
    let key = cache_key(&KeyParts {
        prompt_version: PROMPT_VERSION,
        identity: &first.identity,
        request: &request,
    });
    fx.cache
        .put(NewEntry {
            key: &key,
            response: "this is not json",
            model_id: &first.identity.model_id,
            model_sha256: &first.identity.model_sha256,
            prompt_version: PROMPT_VERSION,
            source_event_ids: &request.source_event_ids,
        })
        .unwrap();
    let second = Script::honest();

    // Act
    let report = fx.run(&Provider::new(&second), &inputs(2)).await;

    // Assert
    assert_eq!(second.calls(), 1);
    assert_eq!(report.completed_offset, Some(WINDOW_END));
    assert_eq!(report.batches_failed, 0);
}

#[tokio::test]
async fn an_expired_deadline_runs_no_batch() {
    // Arrange
    let fx = Fixture::new().await;
    let script = Script::honest();

    // Act
    let report = fx
        .run_until(&Provider::new(&script), &inputs(4), Instant::now())
        .await;

    // Assert
    assert_eq!(script.calls(), 0);
    assert!(report.deadline_hit);
    assert_eq!(report.completed_offset, None);
}

#[tokio::test(start_paused = true)]
async fn a_batch_waiting_for_a_permit_past_the_deadline_is_skipped() {
    // Arrange
    let mut fx = Fixture::new().await;
    fx.permits = Semaphore::new(0);
    let script = Script::honest();

    // Act
    let report = fx
        .run_until(
            &Provider::new(&script),
            &inputs(2),
            Instant::now() + RUN_BUDGET,
        )
        .await;

    // Assert
    assert_eq!(script.calls(), 0);
    assert!(report.deadline_hit);
    assert!(!report.sidecar_lost);
}

#[tokio::test(start_paused = true)]
async fn a_run_cut_by_the_deadline_leaves_the_cursor_before_the_unattempted_batch() {
    // Arrange: batches in offset order are a[0, 2], b[1, 3], b[5, 7]; the last
    // answers after the deadline.
    let fx = Fixture::new().await;
    let script = Script::new(|request, call| {
        let text = facts_for(&request.prompt);
        if call == 2 {
            Reply::After(FAR_FUTURE, text)
        } else {
            Reply::Text(text)
        }
    });
    let events = [
        input(0, Some("a")),
        input(1, Some("b")),
        input(2, Some("a")),
        input(3, Some("b")),
        input(5, Some("b")),
        input(7, Some("b")),
    ];

    // Act
    let report = fx
        .run_until(
            &Provider::new(&script),
            &events,
            Instant::now() + RUN_BUDGET,
        )
        .await;

    // Assert
    assert!(report.deadline_hit);
    assert_eq!(report.batches_failed, 0);
    assert_eq!(report.completed_offset, Some(offset(3)));
    assert_eq!(fx.facts_about(3).await, 1);
    assert_eq!(fx.facts_about(5).await, 0);
}

#[tokio::test]
async fn a_model_that_dies_after_one_batch_ends_the_run_and_keeps_its_facts() {
    // Arrange
    let fx = Fixture::new().await;
    let alive = Arc::new(AtomicBool::new(true));
    let dies = Arc::clone(&alive);
    let script = Script::new(move |request, _| {
        dies.store(false, Ordering::SeqCst);
        Reply::Text(facts_for(&request.prompt))
    });
    let mut provider = Provider::new(&script);
    provider.alive = alive;

    // Act
    let report = fx.run(&provider, &inputs(4)).await;

    // Assert
    assert_eq!(script.calls(), 1);
    assert!(report.sidecar_lost);
    assert_eq!(report.completed_offset, Some(offset(1)));
    assert_eq!((fx.facts_about(1).await, fx.facts_about(2).await), (1, 0));
}

#[tokio::test]
async fn memory_below_the_budget_ends_the_run_as_lost_before_any_call() {
    // Arrange
    let mut fx = Fixture::new().await;
    fx.config.ram_budget_mb = 4096;
    fx.ram.0.store(4095, Ordering::SeqCst);
    let script = Script::honest();

    // Act
    let report = fx.run(&Provider::new(&script), &inputs(2)).await;

    // Assert
    assert_eq!(script.calls(), 0);
    assert!(report.sidecar_lost);
}

#[tokio::test]
async fn memory_equal_to_the_budget_is_enough() {
    // Arrange
    let mut fx = Fixture::new().await;
    fx.config.ram_budget_mb = 4096;
    fx.ram.0.store(4096, Ordering::SeqCst);
    let script = Script::honest();

    // Act
    let report = fx.run(&Provider::new(&script), &inputs(2)).await;

    // Assert
    assert_eq!(script.calls(), 1);
    assert!(!report.sidecar_lost);
}
