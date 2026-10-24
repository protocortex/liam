// SPDX-License-Identifier: Apache-2.0
//! The two R10 tiers that run in normal CI. The rules tier scores the rules
//! extractor against hand-derived floors. The plumbing tier drives the whole
//! pipeline with a scripted model: it shows the wiring is right, not that a
//! model is good, because the labels are replayed.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use liam_log::LogOffset;
use liam_store::{DefaultGraph, ExtractorName, GraphConfig, Millis, FOREVER};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::sync::Semaphore;
use tokio::time::Instant;

use super::corpus::{
    generate, ExpectedFact, LabelledEvent, Template, CORPUS_SEED, EVENT_COUNT, EVENT_GAP,
    FIRST_OBSERVED, HAS_TITLE, REFERENCES_DATE, REFERENCES_ID, REFERENCES_URL,
};
use super::{normalize_tuple, score, score_by_predicate, Score, Tuple};
use crate::config::ExtractionConfig;
use crate::extraction::cache::{cache_path, CacheStore};
use crate::extraction::client::{ExtractError, JsonExtractor, MockExtractor, ModelIdentity};
use crate::extraction::extractor::{
    extract_events, ExtractionDeps, ExtractionReport, ExtractorProvider, ExtractorSession,
};
use crate::extraction::inputs::ExtractionInput;
use crate::extraction::response::Polarity;
use crate::extraction::rules::extract_rules;
use crate::extraction::sidecar::FreeRam;

const EMBEDDING_DIMS: usize = 8;
const CLIENT: &str = "claude-code";
const SCOPE: &str = "r10";
const WRITE_SOURCE: &str = "consolidation:r10";
const RUN_BUDGET: Duration = Duration::from_secs(3600);
const READ_LIMIT: usize = 10;
const ANSWER_CONFIDENCE: f64 = 0.9;
/// Not a multiple of the template cycle, so some assert/end pairs are split
/// across two batches.
const BATCH_EVENTS: usize = 7;
const HALLUCINATED_OBJECT: &str = "Chief Executive";

/// A floor on one predicate's score, derived by hand from the corpus: it has
/// 30 events of each template, so the counts below are exact. A floor changes
/// only in a commit that updates its derivation.
struct Floor {
    predicate: &'static str,
    precision: f64,
    recall: f64,
}

const FLOORS: [Floor; 3] = [
    // 30 date events, each holding one valid date and the invalid near-miss
    // date being rejected: 30 found, 0 wrong, 0 missed.
    Floor {
        predicate: REFERENCES_DATE,
        precision: 1.0,
        recall: 1.0,
    },
    // 30 URL events found whole, trailing period trimmed; "www.example.com"
    // has no scheme and is not matched.
    Floor {
        predicate: REFERENCES_URL,
        precision: 1.0,
        recall: 1.0,
    },
    // 30 ticket events found. The 30 "SHA-256" events are also matched, since
    // the id rule cannot tell them from a ticket: 30 / (30 + 30) = 0.5. The
    // lowercase "abc-123" is not matched.
    Floor {
        predicate: REFERENCES_ID,
        precision: 0.5,
        recall: 1.0,
    },
];

fn corpus() -> Vec<LabelledEvent> {
    generate(&mut ChaCha8Rng::seed_from_u64(CORPUS_SEED))
}

fn model_labels(events: &[LabelledEvent]) -> Vec<Tuple> {
    events
        .iter()
        .flat_map(|event| event.facts.iter().map(ExpectedFact::tuple))
        .collect()
}

fn rules_labels(events: &[LabelledEvent]) -> Vec<Tuple> {
    events
        .iter()
        .flat_map(|event| {
            event.references.iter().map(|reference| {
                normalize_tuple(
                    &event.rules_subject(),
                    reference.predicate,
                    &reference.object,
                    Polarity::Asserted,
                )
            })
        })
        .collect()
}

fn rules_found(events: &[LabelledEvent]) -> Vec<Tuple> {
    let max_chars = ExtractionConfig::default().max_event_chars();
    events
        .iter()
        .flat_map(|event| extract_rules(&event.id, &event.text, &[], max_chars))
        .map(|found| {
            normalize_tuple(
                &found.subject,
                &found.predicate,
                &found.object,
                Polarity::Asserted,
            )
        })
        .collect()
}

/// The predicates whose score is under its floor, or that were never seen.
fn floors_missed(by_predicate: &BTreeMap<String, Score>) -> Vec<&'static str> {
    FLOORS
        .iter()
        .filter(|floor| {
            by_predicate.get(floor.predicate).is_none_or(|scored| {
                scored.precision < floor.precision || scored.recall < floor.recall
            })
        })
        .map(|floor| floor.predicate)
        .collect()
}

#[test]
fn the_rules_tier_meets_every_floor() {
    // Arrange
    let events = corpus();

    // Act
    let by_predicate = score_by_predicate(&rules_found(&events), &rules_labels(&events));

    // Assert
    assert_eq!(floors_missed(&by_predicate), Vec::<&str>::new());
    let ids = by_predicate[REFERENCES_ID];
    assert_eq!((ids.tp, ids.fp, ids.fn_), (30, 30, 0));
}

#[test]
fn the_rules_tier_is_exact_on_every_template_it_has_no_distractor_for() {
    // Arrange
    let events = corpus();
    let clean: Vec<LabelledEvent> = events
        .into_iter()
        .filter(|event| event.template != Template::StandardName)
        .collect();

    // Act
    let result = score(&rules_found(&clean), &rules_labels(&clean));

    // Assert
    assert_eq!(result.f1, 1.0);
}

#[test]
fn a_degraded_predictor_fails_the_gate() {
    // Arrange: the same output with every URL lost and every id doubled up
    // with a wrong one.
    let events = corpus();
    let mut degraded: Vec<Tuple> = rules_found(&events)
        .into_iter()
        .filter(|tuple| tuple.predicate != REFERENCES_URL)
        .collect();
    degraded.extend((0..EVENT_COUNT).map(|n| {
        normalize_tuple(
            &format!("event:evt-{n:03}"),
            REFERENCES_ID,
            "ZZ-0",
            Polarity::Asserted,
        )
    }));

    // Act
    let missed = floors_missed(&score_by_predicate(&degraded, &rules_labels(&events)));

    // Assert
    assert_eq!(missed, [REFERENCES_URL, REFERENCES_ID]);
}

struct PlentyOfRam;

impl FreeRam for PlentyOfRam {
    fn free_mb(&self) -> u64 {
        u64::MAX
    }
}

struct ScriptedProvider(Arc<MockExtractor>);

#[async_trait]
impl ExtractorProvider for ScriptedProvider {
    async fn open(&self, _config: &ExtractionConfig) -> Result<ExtractorSession> {
        Ok(ExtractorSession {
            extractor: Arc::clone(&self.0) as Arc<dyn JsonExtractor>,
            alive: Box::new(|| true),
            _keepalive: None,
        })
    }
}

#[derive(Clone, Copy)]
enum Fault {
    /// Keeps only the first word of the object, which the text still states.
    WrongObject,
    /// Cites an event number past the end of the batch.
    OutOfBatch,
    /// Answers with an object the text never states.
    Ungrounded,
}

fn identity() -> ModelIdentity {
    ModelIdentity {
        model_id: "scripted".into(),
        model_sha256: "r10-plumbing".into(),
    }
}

fn fact_json(
    fact: &ExpectedFact,
    position: usize,
    batch_len: usize,
    fault: Option<Fault>,
) -> Value {
    let object = match fault {
        Some(Fault::WrongObject) => fact.object.split(' ').next().unwrap_or_default(),
        Some(Fault::Ungrounded) => HALLUCINATED_OBJECT,
        _ => &fact.object,
    };
    let cited = if matches!(fault, Some(Fault::OutOfBatch)) {
        batch_len
    } else {
        position
    };
    json!({
        "subject": fact.subject.to_lowercase(),
        "predicate": fact.predicate,
        "object": object,
        "polarity": if fact.polarity == Polarity::Ended { "ended" } else { "asserted" },
        "valid_from": null,
        "valid_until": null,
        "confidence": ANSWER_CONFIDENCE,
        "events": [cited],
    })
}

/// One answer per batch, in the order the run asks, replaying the labels
/// except where `faults` says otherwise.
fn script(
    events: &[LabelledEvent],
    faults: &BTreeMap<usize, Fault>,
) -> Vec<Result<String, ExtractError>> {
    events
        .chunks(BATCH_EVENTS)
        .map(|batch| {
            let facts: Vec<Value> = batch
                .iter()
                .enumerate()
                .flat_map(|(position, event)| {
                    let fault = faults.get(&event.index).copied();
                    event
                        .facts
                        .iter()
                        .map(move |fact| fact_json(fact, position, batch.len(), fault))
                })
                .collect();
            Ok(json!({ "facts": facts }).to_string())
        })
        .collect()
}

fn input(event: &LabelledEvent) -> ExtractionInput {
    ExtractionInput {
        event_id: event.id.clone(),
        offset: LogOffset {
            segment: 0,
            index: event.index as u64,
        },
        scope: Some(SCOPE.into()),
        source: CLIENT.into(),
        observed_at: event.observed_at,
        ingested_at: Millis(event.observed_at.0 + 1),
        text: event.text.clone(),
    }
}

struct Plumbing {
    _dir: TempDir,
    store: DefaultGraph,
    config: ExtractionConfig,
    report: ExtractionReport,
    read_at: Millis,
}

/// Runs the whole corpus through the pipeline against a real store.
async fn run_plumbing(events: &[LabelledEvent], faults: &BTreeMap<usize, Fault>) -> Plumbing {
    let dir = TempDir::new().expect("tempdir");
    let database = dir.path().join("liam.db");
    let store = DefaultGraph::open(database.to_str().unwrap(), GraphConfig::new(EMBEDDING_DIMS))
        .await
        .expect("open store");
    let config = ExtractionConfig {
        ram_budget_mb: 1,
        batch_events: BATCH_EVENTS,
        ..ExtractionConfig::default()
    };
    let mock = Arc::new(MockExtractor::new(identity(), script(events, faults)));
    let provider = ScriptedProvider(Arc::clone(&mock));
    let model = identity();
    let deps = ExtractionDeps {
        provider: &provider,
        identity: &model,
        cache: Arc::new(CacheStore::open(cache_path(&database)).expect("open cache")),
        store: &store,
        config: &config,
        permits: &Semaphore::new(1),
        free_ram: &PlentyOfRam,
        write_source: WRITE_SOURCE,
        deadline: Instant::now() + RUN_BUDGET,
        window_end: LogOffset {
            segment: 0,
            index: EVENT_COUNT as u64,
        },
    };
    let inputs: Vec<ExtractionInput> = events.iter().map(input).collect();

    let report = extract_events(&inputs, deps).await.expect("a finished run");

    assert_eq!(mock.calls(), events.len().div_ceil(BATCH_EVENTS));
    let read_at = Millis(FIRST_OBSERVED.0 + EVENT_GAP.0 * (EVENT_COUNT as i64 + 1));
    Plumbing {
        _dir: dir,
        store,
        config,
        report,
        read_at,
    }
}

/// The events that state something about each subject, in order.
fn by_subject(events: &[LabelledEvent]) -> BTreeMap<String, Vec<&LabelledEvent>> {
    let mut grouped: BTreeMap<String, Vec<&LabelledEvent>> = BTreeMap::new();
    for event in events {
        for fact in &event.facts {
            grouped
                .entry(fact.subject.to_lowercase())
                .or_default()
                .push(event);
        }
    }
    grouped
}

impl Plumbing {
    async fn valid_at(&self, subject: &str, at: Millis) -> Vec<liam_store::Fact> {
        self.store
            .facts_valid_at(subject, Some(SCOPE), at, READ_LIMIT)
            .await
            .expect("read facts")
            .facts
    }

    /// Rebuilds what the pipeline extracted from what the store holds: a
    /// fact present when its subject was first mentioned was asserted, and
    /// one gone by the end was ended.
    async fn extracted(&self, events: &[LabelledEvent]) -> Vec<Tuple> {
        let mut tuples = Vec::new();
        for (subject, mentions) in by_subject(events) {
            let live = self.valid_at(&subject, self.read_at).await;
            for fact in self.valid_at(&subject, mentions[0].observed_at).await {
                let asserted = |polarity| {
                    normalize_tuple(&fact.subject, &fact.predicate, &fact.object, polarity)
                };
                tuples.push(asserted(Polarity::Asserted));
                if !live.iter().any(|kept| kept.id == fact.id) {
                    tuples.push(asserted(Polarity::Ended));
                }
            }
        }
        tuples
    }

    /// Checks the stored rows against the labels: scope, trust, extractor,
    /// source links and whether the fact is still open.
    async fn assert_store_matches(&self, events: &[LabelledEvent]) {
        let trust = self.config.default_trust * self.config.llm_trust_factor;
        for (subject, mentions) in by_subject(events) {
            let (first, last) = (mentions[0], mentions[mentions.len() - 1]);
            let ends = last.facts[0].polarity == Polarity::Ended;
            let stored = self.valid_at(&subject, first.observed_at).await;
            assert_eq!(stored.len(), 1, "{subject}");
            let fact = &stored[0];
            assert_eq!(fact.scope.as_deref(), Some(SCOPE), "{subject}");
            assert_eq!(fact.trust_score, trust, "{subject}");
            assert_eq!(fact.extractor_id, ExtractorName::Llm, "{subject}");
            assert!(fact.source_event_ids.contains(&first.id), "{subject}");
            let live = self.valid_at(&subject, self.read_at).await;
            if ends {
                assert!(live.is_empty(), "{subject} should be over");
                assert_eq!(fact.valid_until, last.observed_at, "{subject}");
            } else {
                assert_eq!(live.len(), 1, "{subject}");
                assert_eq!(fact.valid_until, FOREVER, "{subject}");
                assert_eq!(fact.source_event_ids, [first.id.as_str()], "{subject}");
            }
        }
    }
}

#[tokio::test]
async fn the_plumbing_tier_stores_exactly_what_the_labels_say() {
    // Arrange
    let events = corpus();

    // Act
    let run = run_plumbing(&events, &BTreeMap::new()).await;

    // Assert
    let result = score(&run.extracted(&events).await, &model_labels(&events));
    assert_eq!((result.tp, result.fp, result.fn_), (150, 0, 0));
    assert_eq!(run.report.facts_written, 150);
    assert_eq!(run.report.facts_dropped, 0);
    assert_eq!(run.report.ended_no_match, 0);
    run.assert_store_matches(&events).await;
}

/// Index of the role-change event of cycle `cycle`.
fn role_event(cycle: usize) -> usize {
    cycle * super::corpus::CYCLE_LEN + 4
}

#[tokio::test]
async fn three_scripted_faults_lower_f1_to_the_hand_computed_value() {
    // Arrange: 150 labels. The wrong object turns one hit into a false
    // positive and a miss; the out-of-batch citation and the ungrounded
    // fact each lose one label. So tp = 147, fp = 1, fn = 3, and
    // F1 = 2 * 147 / (2 * 147 + 1 + 3) = 147 / 149.
    let events = corpus();
    let faults = BTreeMap::from([
        (role_event(2), Fault::WrongObject),
        (role_event(11), Fault::OutOfBatch),
        (role_event(21), Fault::Ungrounded),
    ]);
    assert!(events[role_event(2)].facts[0].predicate == HAS_TITLE);

    // Act
    let run = run_plumbing(&events, &faults).await;

    // Assert
    let result = score(&run.extracted(&events).await, &model_labels(&events));
    assert_eq!((result.tp, result.fp, result.fn_), (147, 1, 3));
    assert!((result.f1 - 147.0 / 149.0).abs() < 1e-12);
    assert_eq!(run.report.facts_dropped, 2);
    assert_eq!(run.report.facts_dropped_ungrounded, 1);
}
