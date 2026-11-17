// SPDX-License-Identifier: Apache-2.0
//! Applying parsed answers to a real file store.

use liam_log::LogOffset;
use liam_store::{Fact, GraphConfig};
use serde_json::{json, Value};
use tempfile::TempDir;

use super::super::response::parse_response;
use super::*;
use crate::config::{declared_predicates, resolve_vocabulary};

const WRITE_SOURCE: &str = "consolidation:test-run";
const CLIENT: &str = "claude-code";
const TRUSTED: &str = "slack";
const TRUSTED_SCORE: f64 = 0.9;
const HOUR: i64 = 3_600_000;
const DAY: i64 = 24 * HOUR;
/// 2023-06-01 00:00 UTC.
const OBSERVED: i64 = 1_685_577_600_000;
const READ_LIMIT: usize = 20;
/// A source whose configured trust the store refuses.
const BAD_SOURCE: &str = "bad-source";
const UNTRUSTWORTHY_SCORE: f64 = 2.0;
const EMBEDDING_DIMS: usize = 8;

struct Fixture {
    _dir: TempDir,
    store: DefaultGraph,
    config: ExtractionConfig,
}

impl Fixture {
    async fn new() -> Self {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("liam.db");
        let store = DefaultGraph::open(path.to_str().unwrap(), GraphConfig::new(EMBEDDING_DIMS))
            .await
            .expect("open store");
        let mut config = ExtractionConfig::default();
        config.trust.insert(TRUSTED.into(), TRUSTED_SCORE);
        Self {
            _dir: dir,
            store,
            config,
        }
    }

    async fn apply(&self, batch: &[ExtractionInput], answer: Value) -> ApplyCounts {
        let counts = self.try_apply(batch, answer).await.expect("applied");
        self.assert_every_fact_has_a_source().await;
        counts
    }

    async fn try_apply(
        &self,
        batch: &[ExtractionInput],
        answer: Value,
    ) -> Result<ApplyCounts, ApplyError> {
        let parsed = parse_response(
            &answer.to_string(),
            batch.len(),
            &declared_predicates(&self.config),
        )
        .expect("a readable answer");
        let vocabulary = resolve_vocabulary(&self.config);
        let ctx = ApplyContext {
            store: &self.store,
            config: &self.config,
            vocabulary: &vocabulary,
            write_source: WRITE_SOURCE,
        };
        let refs: Vec<&ExtractionInput> = batch.iter().collect();
        apply_answer(&ctx, &refs, parsed).await
    }

    async fn live(&self, subject: &str, at: i64) -> Vec<Fact> {
        self.store
            .facts_valid_at(subject, None, Millis(at), READ_LIMIT)
            .await
            .expect("read facts")
            .facts
    }

    async fn assert_every_fact_has_a_source(&self) {
        let orphans = self.store.facts_without_sources().await.expect("scan");
        assert!(orphans.is_empty(), "facts without a source: {orphans:?}");
    }
}

fn input(n: u64, source: &str, text: &str) -> ExtractionInput {
    let observed = OBSERVED + i64::try_from(n).unwrap() * HOUR;
    ExtractionInput {
        event_id: format!("e{n}"),
        offset: LogOffset {
            segment: 0,
            index: n,
        },
        scope: None,
        source: source.into(),
        observed_at: Millis(observed),
        ingested_at: Millis(observed + 1),
        text: text.into(),
    }
}

fn fact(subject: &str, predicate: &str, object: &str, events: &[usize]) -> Value {
    json!({
        "subject": subject, "predicate": predicate, "object": object,
        "polarity": "asserted", "valid_from": null, "valid_until": null,
        "confidence": 0.9, "events": events,
    })
}

fn with(mut fact: Value, field: &str, value: Value) -> Value {
    fact[field] = value;
    fact
}

fn answer(facts: Vec<Value>) -> Value {
    json!({ "facts": facts })
}

#[tokio::test]
async fn facts_land_with_their_sources_trust_and_the_event_time() {
    // Arrange
    let fx = Fixture::new().await;
    let batch = [
        input(0, CLIENT, "Alice works at Acme."),
        input(1, TRUSTED, "Bob lives in Perth."),
    ];
    let facts = answer(vec![
        fact("alice", "works_at", "Acme", &[0]),
        fact("bob", "lives_in", "Perth", &[1]),
    ]);

    // Act
    let counts = fx.apply(&batch, facts).await;

    // Assert
    assert_eq!(counts.written, 2);
    let alice = &fx.live("alice", OBSERVED + DAY).await[0];
    assert_eq!(alice.source_event_ids, ["e0"]);
    assert_eq!(alice.valid_from, batch[0].observed_at);
    assert_eq!(alice.extractor_id, ExtractorName::Llm);
    assert_eq!(
        alice.trust_score,
        fx.config.default_trust * fx.config.llm_trust_factor
    );
    let bob = &fx.live("bob", OBSERVED + DAY).await[0];
    assert_eq!(bob.trust_score, TRUSTED_SCORE * fx.config.llm_trust_factor);
    fx.assert_every_fact_has_a_source().await;
}

#[tokio::test]
async fn a_fact_citing_two_sources_takes_the_strongest_trust() {
    // Arrange
    let fx = Fixture::new().await;
    let batch = [
        input(0, CLIENT, "Alice works at Acme."),
        input(1, TRUSTED, "Alice works at Acme, she said."),
    ];

    // Act
    fx.apply(
        &batch,
        answer(vec![fact("alice", "works_at", "Acme", &[0, 1])]),
    )
    .await;

    // Assert
    let alice = &fx.live("alice", OBSERVED + DAY).await[0];
    assert_eq!(
        alice.trust_score,
        TRUSTED_SCORE * fx.config.llm_trust_factor
    );
    assert_eq!(alice.source_event_ids, ["e0", "e1"]);
}

#[tokio::test]
async fn a_citation_outside_the_batch_is_dropped_and_counted() {
    // Arrange
    let fx = Fixture::new().await;
    let batch = [input(0, CLIENT, "Alice works at Acme.")];

    // Act
    let counts = fx
        .apply(
            &batch,
            answer(vec![fact("alice", "works_at", "Acme", &[7])]),
        )
        .await;

    // Assert
    assert_eq!((counts.written, counts.dropped), (0, 1));
    assert!(fx.live("alice", OBSERVED + DAY).await.is_empty());
}

#[tokio::test]
async fn an_object_the_cited_text_lacks_is_dropped_as_ungrounded() {
    // Arrange
    let fx = Fixture::new().await;
    let batch = [input(0, CLIENT, "Alice works at Acme.")];

    // Act
    let counts = fx
        .apply(
            &batch,
            answer(vec![fact("alice", "works_at", "Globex", &[0])]),
        )
        .await;

    // Assert
    assert_eq!(counts.written, 0);
    assert_eq!((counts.dropped, counts.dropped_ungrounded), (1, 1));
    assert!(fx.live("alice", OBSERVED + DAY).await.is_empty());
}

#[tokio::test]
async fn a_fact_below_the_confidence_floor_is_dropped_and_one_at_it_is_kept() {
    // Arrange
    let fx = Fixture::new().await;
    let batch = [input(
        0,
        CLIENT,
        "Alice works at Acme and Bob works at Initech.",
    )];
    let at_floor = with(
        fact("alice", "works_at", "Acme", &[0]),
        "confidence",
        json!(fx.config.min_confidence),
    );
    let below = with(
        fact("bob", "works_at", "Initech", &[0]),
        "confidence",
        json!(fx.config.min_confidence - 0.01),
    );

    // Act
    let counts = fx.apply(&batch, answer(vec![at_floor, below])).await;

    // Assert
    assert_eq!((counts.written, counts.dropped), (1, 1));
    assert_eq!(fx.live("alice", OBSERVED + DAY).await.len(), 1);
    assert!(fx.live("bob", OBSERVED + DAY).await.is_empty());
}

fn ended(subject: &str, predicate: &str, object: &str, events: &[usize]) -> Value {
    with(
        fact(subject, predicate, object, events),
        "polarity",
        json!("ended"),
    )
}

#[tokio::test]
async fn an_ended_statement_closes_the_live_fact_at_the_event_time() {
    // Arrange
    let fx = Fixture::new().await;
    let start = [input(0, CLIENT, "Alice works at Acme.")];
    let stop = [input(5, CLIENT, "Alice no longer works at Acme.")];
    fx.apply(
        &start,
        answer(vec![fact("alice", "works_at", "Acme", &[0])]),
    )
    .await;

    // Act
    let counts = fx
        .apply(
            &stop,
            answer(vec![ended("alice", "works_at", "Acme", &[0])]),
        )
        .await;

    // Assert
    assert_eq!(counts.written, 1);
    assert_eq!(fx.live("alice", stop[0].observed_at.0 - 1).await.len(), 1);
    assert!(fx.live("alice", stop[0].observed_at.0).await.is_empty());
}

#[tokio::test]
async fn an_ended_statement_with_nothing_to_end_inserts_nothing() {
    // Arrange
    let fx = Fixture::new().await;
    let stop = [input(5, CLIENT, "Alice no longer works at Acme.")];

    // Act
    let counts = fx
        .apply(
            &stop,
            answer(vec![ended("alice", "works_at", "Acme", &[0])]),
        )
        .await;

    // Assert
    assert_eq!((counts.written, counts.ended_no_match), (0, 1));
    assert!(fx.live("alice", OBSERVED + DAY).await.is_empty());
}

#[tokio::test]
async fn an_ended_statement_from_a_weaker_source_than_the_live_fact_is_outranked() {
    // Arrange
    let fx = Fixture::new().await;
    let start = [input(0, TRUSTED, "Alice works at Acme.")];
    let stop = [input(5, CLIENT, "Alice no longer works at Acme.")];
    fx.apply(
        &start,
        answer(vec![fact("alice", "works_at", "Acme", &[0])]),
    )
    .await;

    // Act
    let counts = fx
        .apply(
            &stop,
            answer(vec![ended("alice", "works_at", "Acme", &[0])]),
        )
        .await;

    // Assert
    assert_eq!((counts.written, counts.ended_outranked), (0, 1));
    assert_eq!(fx.live("alice", OBSERVED + DAY).await.len(), 1);
}

#[tokio::test]
async fn an_ended_statement_names_its_own_date_when_it_has_one() {
    // Arrange
    let fx = Fixture::new().await;
    let start = [input(0, CLIENT, "Alice works at Acme.")];
    let stop = [input(5, CLIENT, "Alice left Acme in March 2023.")];
    fx.apply(
        &start,
        answer(vec![fact("alice", "works_at", "Acme", &[0])]),
    )
    .await;
    let dated = with(
        ended("alice", "works_at", "Acme", &[0]),
        "valid_until",
        json!("2023-06-10"),
    );

    // Act
    fx.apply(&stop, answer(vec![dated])).await;

    // Assert
    let until = date_millis("2023-06-10").unwrap().0;
    assert_eq!(fx.live("alice", until - 1).await.len(), 1);
    assert!(fx.live("alice", until).await.is_empty());
}

#[tokio::test]
async fn an_end_that_would_not_fall_after_the_start_is_dropped() {
    // Arrange
    let fx = Fixture::new().await;
    let start = [input(5, CLIENT, "Alice works at Acme.")];
    let stop = [input(6, CLIENT, "Alice left Acme.")];
    fx.apply(
        &start,
        answer(vec![fact("alice", "works_at", "Acme", &[0])]),
    )
    .await;
    let before_start = with(
        ended("alice", "works_at", "Acme", &[0]),
        "valid_until",
        json!("2023-05-01"),
    );

    // Act
    let counts = fx.apply(&stop, answer(vec![before_start])).await;

    // Assert
    assert_eq!((counts.written, counts.dropped), (0, 1));
    assert_eq!(fx.live("alice", OBSERVED + DAY).await.len(), 1);
}

#[tokio::test]
async fn an_asserted_fact_is_historical_only_when_it_ended_by_the_event_time() {
    // Arrange: the event is observed at 2023-06-01 00:00 UTC.
    for (until, historical) in [
        ("2023-05-31", true),
        ("2023-06-01", true),
        ("2023-06-02", false),
    ] {
        let fx = Fixture::new().await;
        let batch = [input(0, CLIENT, "Alice lived in Perth until the move.")];
        let dated = with(
            fact("alice", "lives_in", "Perth", &[0]),
            "valid_until",
            json!(until),
        );

        // Act
        fx.apply(&batch, answer(vec![dated])).await;

        // Assert
        let live_after_event = fx.live("alice", OBSERVED + HOUR).await;
        assert_eq!(live_after_event.is_empty(), historical, "until {until}");
        fx.assert_every_fact_has_a_source().await;
    }
}

#[tokio::test]
async fn a_statement_with_only_an_end_date_starts_at_the_beginning_of_time() {
    // Arrange
    let fx = Fixture::new().await;
    let batch = [input(0, CLIENT, "Alice lived in Perth until 2020.")];
    let dated = with(
        fact("alice", "lives_in", "Perth", &[0]),
        "valid_until",
        json!("2020-01-01"),
    );

    // Act
    fx.apply(&batch, answer(vec![dated])).await;

    // Assert
    let found = fx.live("alice", date_millis("2019-01-01").unwrap().0).await;
    assert_eq!(found[0].valid_from, BEGINNING);
}

#[tokio::test]
async fn an_ended_fact_listed_first_is_still_applied_after_the_one_it_ends() {
    // Arrange
    let fx = Fixture::new().await;
    let batch = [
        input(0, CLIENT, "Alice works at Acme."),
        input(1, CLIENT, "Alice no longer works at Acme."),
    ];
    let listed_backwards = answer(vec![
        ended("alice", "works_at", "Acme", &[1]),
        fact("alice", "works_at", "Acme", &[0]),
    ]);

    // Act
    let counts = fx.apply(&batch, listed_backwards).await;

    // Assert
    assert_eq!((counts.written, counts.ended_no_match), (2, 0));
    assert!(fx.live("alice", OBSERVED + DAY).await.is_empty());
}

#[tokio::test]
async fn contradicting_facts_from_one_event_settle_the_same_whichever_comes_first() {
    // Arrange
    let sydney = fact("alice", "lives_in", "Sydney", &[0]);
    let melbourne = fact("alice", "lives_in", "Melbourne", &[0]);
    let mut settled = Vec::new();

    for order in [
        vec![sydney.clone(), melbourne.clone()],
        vec![melbourne, sydney],
    ] {
        let fx = Fixture::new().await;
        let batch = [input(0, CLIENT, "Alice moved from Sydney to Melbourne.")];

        // Act
        fx.apply(&batch, answer(order)).await;

        // Assert
        let live = fx.live("alice", OBSERVED + DAY).await;
        settled.push(live.iter().map(|f| f.object.clone()).collect::<Vec<_>>());
    }
    assert_eq!(settled[0], settled[1]);
    assert_eq!(settled[0].len(), 1);
}

#[tokio::test]
async fn applying_the_same_answer_again_writes_nothing_new() {
    // Arrange
    let fx = Fixture::new().await;
    let batch = [input(0, CLIENT, "Alice works at Acme.")];
    let facts = answer(vec![fact("alice", "works_at", "Acme", &[0])]);
    fx.apply(&batch, facts.clone()).await;
    let before = fx.live("alice", OBSERVED + DAY).await;

    // Act
    let counts = fx.apply(&batch, facts).await;

    // Assert
    assert_eq!(counts.written, 0);
    assert_eq!(fx.live("alice", OBSERVED + DAY).await, before);
}

#[tokio::test]
async fn a_store_rejection_is_reported_as_one_the_facts_caused_and_writes_none_of_the_run() {
    // Arrange: the second fact comes from a source whose trust the store refuses.
    let mut fx = Fixture::new().await;
    fx.config
        .trust
        .insert(BAD_SOURCE.into(), UNTRUSTWORTHY_SCORE);
    let batch = [
        input(0, CLIENT, "Alice works at Acme."),
        input(1, BAD_SOURCE, "Bob works at Initech."),
    ];
    let facts = answer(vec![
        fact("alice", "works_at", "Acme", &[0]),
        fact("bob", "works_at", "Initech", &[1]),
    ]);

    // Act
    let result = fx.try_apply(&batch, facts).await;

    // Assert
    assert!(matches!(result, Err(ApplyError::Rejected(_))), "{result:?}");
    assert!(fx.live("alice", OBSERVED + DAY).await.is_empty());
    fx.assert_every_fact_has_a_source().await;
}

#[tokio::test]
async fn a_fact_that_starts_after_the_open_end_is_dropped_and_counted_not_a_failure() {
    // Arrange
    let fx = Fixture::new().await;
    let batch = [input(
        0,
        CLIENT,
        "Alice works at Acme and Bob works at Initech.",
    )];
    let after_the_open_end = with(
        fact("alice", "works_at", "Acme", &[0]),
        "valid_from",
        json!("2200-01-01"),
    );
    let fine = fact("bob", "works_at", "Initech", &[0]);

    // Act
    let counts = fx
        .apply(&batch, answer(vec![after_the_open_end, fine]))
        .await;

    // Assert
    assert_eq!((counts.written, counts.dropped), (1, 1));
    assert!(fx.live("alice", OBSERVED + DAY).await.is_empty());
    assert_eq!(fx.live("bob", OBSERVED + DAY).await.len(), 1);
}

#[test]
fn only_a_refusal_the_facts_caused_is_a_rejection() {
    for (error, rejected) in [
        (
            Error::InvalidFactInterval {
                valid_from: 2,
                valid_until: 1,
            },
            true,
        ),
        (Error::InvalidFactTrust(2.0), true),
        (Error::FactWithoutSource, true),
        (Error::InvalidScope("bad".into()), true),
        (
            Error::FactInBatch {
                index: 1,
                source: Box::new(Error::FactWithoutSource),
            },
            true,
        ),
        (Error::Backend("disk".into()), false),
        (Error::LogPoisoned, false),
    ] {
        // Arrange
        let label = error.to_string();

        // Act
        let mapped = ApplyError::from(error);

        // Assert
        assert_eq!(
            matches!(mapped, ApplyError::Rejected(_)),
            rejected,
            "{label}"
        );
    }
}

#[tokio::test]
async fn an_llm_fact_loses_to_a_rules_fact_of_the_same_source_it_contradicts() {
    // Arrange: a rules fact carries the source's full trust.
    let fx = Fixture::new().await;
    let batch = [input(0, CLIENT, "Alice works at Globex.")];
    let rules_fact = NewFact {
        subject: "alice".into(),
        predicate: "works_at".into(),
        object: "Acme".into(),
        scope: None,
        valid_from: batch[0].observed_at,
        valid_until: FOREVER,
        evidence_at: batch[0].observed_at,
        confidence: 1.0,
        trust_score: fx.config.default_trust,
        extractor: ExtractorName::Rules,
        extractor_version: "1".into(),
        source_event_ids: vec!["rules-event".into()],
    };
    fx.store
        .assert_fact(
            rules_fact,
            &resolve_vocabulary(&fx.config).spec("works_at"),
            WRITE_SOURCE,
            ArbitrationCaps::default(),
        )
        .await
        .expect("rules fact");

    // Act
    fx.apply(
        &batch,
        answer(vec![fact("alice", "works_at", "Globex", &[0])]),
    )
    .await;

    // Assert
    let live = fx.live("alice", OBSERVED + DAY).await;
    let objects: Vec<_> = live.iter().map(|f| f.object.as_str()).collect();
    assert_eq!(objects, ["Acme", "Globex"]);
    let llm = live.iter().find(|f| f.object == "Globex").unwrap();
    assert!(llm.trust_score < fx.config.default_trust);
}

#[tokio::test]
async fn the_evidence_time_is_the_latest_of_each_events_earlier_stamp() {
    // Arrange: stamps (observed, ingested) are (1h, 9h) and (7h, 3h), so the
    // earlier stamps are 1h and 3h.
    let fx = Fixture::new().await;
    let at = |hours: i64| Millis(OBSERVED + hours * HOUR);
    let mut first = input(0, CLIENT, "Alice works at Acme.");
    (first.observed_at, first.ingested_at) = (at(1), at(9));
    let mut second = input(1, CLIENT, "Alice works at Acme, she said.");
    (second.observed_at, second.ingested_at) = (at(7), at(3));

    // Act
    fx.apply(
        &[first, second],
        answer(vec![fact("alice", "works_at", "Acme", &[0, 1])]),
    )
    .await;

    // Assert
    let stored = &fx.live("alice", OBSERVED + DAY).await[0];
    assert_eq!(stored.evidence_at, at(3));
}

#[tokio::test]
async fn an_asserted_and_an_ended_fact_from_one_event_apply_in_that_order() {
    // Arrange
    let fx = Fixture::new().await;
    let batch = [input(0, CLIENT, "Alice works at Acme. Alice left Acme.")];
    let ended_first = answer(vec![
        with(
            ended("alice", "works_at", "Acme", &[0]),
            "valid_until",
            json!("2023-06-10"),
        ),
        fact("alice", "works_at", "Acme", &[0]),
    ]);

    // Act
    let counts = fx.apply(&batch, ended_first).await;

    // Assert
    assert_eq!((counts.written, counts.ended_no_match), (2, 0));
    assert!(fx
        .live("alice", date_millis("2023-06-10").unwrap().0)
        .await
        .is_empty());
}

#[test]
fn a_date_is_midnight_utc_and_a_bad_one_is_none() {
    for (text, expected) in [
        ("1970-01-01", Some(0)),
        ("1970-01-02", Some(DAY)),
        ("2000-02-29", Some(951_782_400_000)),
        ("2001-02-29", None),
        ("tomorrow", None),
    ] {
        // Act
        let millis = date_millis(text);

        // Assert
        assert_eq!(millis, expected.map(Millis), "{text}");
    }
}
