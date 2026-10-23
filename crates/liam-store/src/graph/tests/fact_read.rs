// SPDX-License-Identifier: Apache-2.0
//! Bitemporal fact reads against a file store, over one scripted timeline in
//! which facts change, compete and are corrected late.

use super::*;
use crate::graph::fact_read::{CURRENT_FILTER, MAX_FACT_ROWS, MAX_SOURCES_PER_FACT};
use crate::types::{
    ArbitrationCaps, ArbitrationOutcome, Cardinality, EndFact, EndOutcome, ExtractorName, Fact,
    NewFact, PredicateSpec,
};

const DIMS: usize = 8;
const SUBJECT: &str = "alice";
const LIVES_IN: &str = "lives_in";
const WORKS_AT: &str = "works_at";
const T1: i64 = 1000;
const T2: i64 = 2000;
const T3: i64 = 3000;
const T4: i64 = 4000;
const LOW_TRUST: f64 = 0.3;
const MID_TRUST: f64 = 0.5;
const HIGH_TRUST: f64 = 0.9;
const NO_LIMIT: usize = usize::MAX;

struct Store {
    graph: DefaultGraph,
    clock: Arc<FixedClock>,
    _dir: TempDir,
}

async fn store() -> Store {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("facts.db");
    let clock = Arc::new(FixedClock::new(Millis(T1)));
    let graph = DefaultGraph::open_with_clock(
        path.to_str().unwrap(),
        GraphConfig::new(DIMS),
        clock.clone(),
    )
    .await
    .unwrap();
    Store {
        graph,
        clock,
        _dir: dir,
    }
}

fn single() -> PredicateSpec {
    PredicateSpec {
        cardinality: Cardinality::One,
        ..PredicateSpec::default()
    }
}

fn new_fact(subject: &str, predicate: &str, object: &str, trust: f64, at: i64) -> NewFact {
    NewFact {
        subject: subject.into(),
        predicate: predicate.into(),
        object: object.into(),
        scope: None,
        valid_from: Millis(at),
        valid_until: FOREVER,
        evidence_at: Millis(at),
        confidence: 1.0,
        trust_score: trust,
        extractor: ExtractorName::Rules,
        extractor_version: "v1".into(),
        source_event_ids: vec![format!("{subject}-{predicate}-{object}")],
    }
}

impl Store {
    /// Asserts `fact` with the store clock set to `tx`.
    async fn assert_at(&self, tx: i64, fact: NewFact) -> ArbitrationOutcome {
        self.clock.set(Millis(tx));
        self.graph
            .assert_fact(fact, &single(), ArbitrationCaps::default())
            .await
            .unwrap()
    }

    async fn assert_many(&self, fact: NewFact) -> ArbitrationOutcome {
        self.graph
            .assert_fact(fact, &PredicateSpec::default(), ArbitrationCaps::default())
            .await
            .unwrap()
    }

    async fn valid_at(&self, at: i64) -> Vec<Fact> {
        self.graph
            .facts_valid_at(SUBJECT, None, Millis(at), NO_LIMIT)
            .await
            .unwrap()
            .facts
    }

    async fn believed_at(&self, believed: i64, valid: Option<i64>) -> Vec<Fact> {
        self.graph
            .facts_believed_at(SUBJECT, None, Millis(believed), valid.map(Millis), NO_LIMIT)
            .await
            .unwrap()
            .facts
    }
}

fn objects(facts: &[Fact], predicate: &str) -> Vec<String> {
    facts
        .iter()
        .filter(|fact| fact.predicate == predicate)
        .map(|fact| fact.object.clone())
        .collect()
}

/// X at t1, replaced by the higher-trust Y at t2, then a competing Z at t3.
/// A separate key holds X' (since t1), which Y' corrects at t4 back to t2.
async fn timeline() -> Store {
    let s = store().await;
    s.assert_at(T1, new_fact(SUBJECT, LIVES_IN, "Sydney", MID_TRUST, T1))
        .await;
    s.assert_at(T1, new_fact(SUBJECT, WORKS_AT, "Acme", MID_TRUST, T1))
        .await;
    s.assert_at(T2, new_fact(SUBJECT, LIVES_IN, "Melbourne", HIGH_TRUST, T2))
        .await;
    s.assert_at(T3, new_fact(SUBJECT, LIVES_IN, "Perth", LOW_TRUST, T3))
        .await;
    let mut correction = new_fact(SUBJECT, WORKS_AT, "Initech", HIGH_TRUST, T4);
    correction.valid_from = Millis(T2);
    s.assert_at(T4, correction).await;
    s
}

#[tokio::test]
async fn valid_at_before_the_replacement_returns_the_replaced_fact() {
    // Arrange
    let s = timeline().await;

    // Act
    let facts = s.valid_at(T1 + 500).await;

    // Assert
    assert_eq!(objects(&facts, LIVES_IN), ["Sydney"]);
}

#[tokio::test]
async fn valid_at_after_the_replacement_returns_only_the_replacement() {
    // Arrange
    let s = timeline().await;

    // Act
    let facts = s.valid_at(T2 + 500).await;

    // Assert
    assert_eq!(objects(&facts, LIVES_IN), ["Melbourne"]);
}

#[tokio::test]
async fn valid_at_ranks_the_trusted_fact_before_the_competing_one() {
    // Arrange
    let s = timeline().await;

    // Act
    let facts = s.valid_at(T3 + 500).await;

    // Assert
    assert_eq!(objects(&facts, LIVES_IN), ["Melbourne", "Perth"]);
}

#[tokio::test]
async fn believed_at_the_first_period_returns_the_first_fact() {
    // Arrange
    let s = timeline().await;

    // Act
    let facts = s.believed_at(T1 + 500, None).await;

    // Assert
    assert_eq!(objects(&facts, LIVES_IN), ["Sydney"]);
}

#[tokio::test]
async fn believed_at_the_second_period_omits_the_fact_not_yet_recorded() {
    // Arrange
    let s = timeline().await;

    // Act
    let facts = s.believed_at(T2 + 500, None).await;

    // Assert
    assert_eq!(objects(&facts, LIVES_IN), ["Melbourne"]);
}

#[tokio::test]
async fn late_correction_changes_current_belief_but_not_past_belief() {
    // Arrange
    let s = timeline().await;

    // Act
    let now = s.valid_at(T3).await;
    let then = s.believed_at(T3, None).await;

    // Assert
    assert_eq!(objects(&now, WORKS_AT), ["Initech"]);
    assert_eq!(objects(&then, WORKS_AT), ["Acme"]);
}

#[tokio::test]
async fn believed_at_reads_a_past_belief_about_a_different_valid_time() {
    // Arrange
    let s = timeline().await;

    // Act
    let facts = s.believed_at(T3, Some(T1 + 500)).await;

    // Assert
    assert_eq!(objects(&facts, WORKS_AT), ["Acme"]);
    assert_eq!(objects(&facts, LIVES_IN), ["Sydney"]);
}

#[tokio::test]
async fn valid_at_boundaries_include_the_start_and_exclude_the_end() {
    // Arrange
    let s = timeline().await;
    let cases = [
        (T1 - 1, vec![]),
        (T1, vec!["Sydney"]),
        (T2 - 1, vec!["Sydney"]),
        (T2, vec!["Melbourne"]),
    ];

    for (at, expected) in cases {
        // Act
        let facts = s.valid_at(at).await;

        // Assert
        assert_eq!(objects(&facts, LIVES_IN), expected, "valid at {at}");
    }
}

#[tokio::test]
async fn believed_at_boundaries_include_tx_from_and_exclude_tx_to() {
    // Arrange
    let s = timeline().await;
    let cases = [
        (T1 - 1, Some(T1), vec![]),
        (T1, Some(T1), vec!["Sydney"]),
        (T2 - 1, Some(T2 - 1), vec!["Sydney"]),
        (T2, Some(T2 - 1), vec!["Sydney"]),
        (T2, Some(T2), vec!["Melbourne"]),
    ];

    for (believed, valid, expected) in cases {
        // Act
        let facts = s.believed_at(believed, valid).await;

        // Assert
        let case = format!("believed {believed} valid {valid:?}");
        assert_eq!(objects(&facts, LIVES_IN), expected, "{case}");
    }
}

#[tokio::test]
async fn the_original_version_ends_at_its_tx_to_and_the_clone_starts_there() {
    // Arrange
    let s = timeline().await;

    // Act
    let before = s.believed_at(T2 - 1, Some(T1 + 500)).await;
    let at = s.believed_at(T2, Some(T1 + 500)).await;

    // Assert
    let sydney = |facts: &[Fact]| {
        let found: Vec<_> = facts.iter().filter(|f| f.predicate == LIVES_IN).collect();
        assert_eq!(found.len(), 1);
        (found[0].tx_from, found[0].tx_to)
    };
    assert_eq!(sydney(&before), (Millis(T1), Millis(T2)));
    assert_eq!(sydney(&at), (Millis(T2), FOREVER));
}

#[tokio::test]
async fn extreme_and_pre_history_times_return_empty() {
    // Arrange
    let s = timeline().await;

    // Act
    let valid = s.valid_at(i64::MIN).await;
    let believed = s.believed_at(T1 - 500, None).await;

    // Assert
    assert!(valid.is_empty());
    assert!(believed.is_empty());
}

#[tokio::test]
async fn limit_zero_returns_empty() {
    // Arrange
    let s = timeline().await;

    // Act
    let current = s.graph.facts_valid_at(SUBJECT, None, Millis(T3), 0).await;
    let past = s
        .graph
        .facts_believed_at(SUBJECT, None, Millis(T3), None, 0)
        .await;

    // Assert
    for page in [current.unwrap(), past.unwrap()] {
        assert!(page.facts.is_empty());
        assert!(page.truncated);
    }
}

#[tokio::test]
async fn a_huge_limit_is_capped() {
    // Arrange
    let s = store().await;
    for n in 0..=MAX_FACT_ROWS {
        s.assert_many(new_fact(
            SUBJECT,
            "likes",
            &format!("item-{n}"),
            MID_TRUST,
            T1,
        ))
        .await;
    }

    // Act
    let page = s
        .graph
        .facts_valid_at(SUBJECT, None, Millis(T1), NO_LIMIT)
        .await
        .unwrap();

    // Assert
    assert_eq!(page.facts.len(), MAX_FACT_ROWS);
    assert!(page.truncated);
}

#[tokio::test]
async fn a_limit_below_the_match_count_flags_truncation() {
    // Arrange
    let s = timeline().await;
    let cases = [(1, 1, true), (2, 2, true), (3, 3, false), (4, 3, false)];

    for (limit, returned, truncated) in cases {
        // Act
        let page = s
            .graph
            .facts_valid_at(SUBJECT, None, Millis(T3 + 500), limit)
            .await
            .unwrap();

        // Assert
        assert_eq!(page.facts.len(), returned, "limit {limit}");
        assert_eq!(page.truncated, truncated, "limit {limit}");
    }
}

#[tokio::test]
async fn unknown_and_empty_subjects_return_empty() {
    // Arrange
    let s = timeline().await;

    for subject in ["nobody", ""] {
        // Act
        let facts = s
            .graph
            .facts_valid_at(subject, None, Millis(T3), NO_LIMIT)
            .await
            .unwrap()
            .facts;

        // Assert
        assert!(facts.is_empty(), "subject {subject:?}");
    }
}

#[tokio::test]
async fn a_query_never_returns_another_subjects_facts() {
    // Arrange
    let s = timeline().await;
    s.assert_at(T1, new_fact("bob", LIVES_IN, "Oslo", HIGH_TRUST, T1))
        .await;

    // Act
    let bob = s
        .graph
        .facts_valid_at("bob", None, Millis(T3), NO_LIMIT)
        .await
        .unwrap()
        .facts;
    let alice = s.valid_at(T3).await;

    // Assert
    assert_eq!(objects(&bob, LIVES_IN), ["Oslo"]);
    assert!(alice.iter().all(|fact| fact.subject == SUBJECT));
}

#[tokio::test]
async fn equal_rows_order_by_ascending_id() {
    // Arrange
    let s = store().await;
    let mut ids = Vec::new();
    for object in ["a", "b", "c", "d"] {
        let outcome = s
            .assert_many(new_fact(SUBJECT, "likes", object, MID_TRUST, T1))
            .await;
        let ArbitrationOutcome::Inserted { id } = outcome else {
            panic!("expected an insert, got {outcome:?}");
        };
        ids.push(id);
    }
    ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));

    // Act
    let facts = s.valid_at(T1).await;

    // Assert
    let read: Vec<_> = facts.into_iter().map(|fact| fact.id).collect();
    assert_eq!(read, ids);
}

async fn scoped_store() -> Store {
    let s = store().await;
    for (object, scope) in [
        ("none", None),
        ("root", Some("work")),
        ("child", Some("work/liam")),
        ("lookalike", Some("work-archive")),
    ] {
        let mut fact = new_fact(SUBJECT, "likes", object, MID_TRUST, T1);
        fact.scope = scope.map(str::to_owned);
        s.assert_many(fact).await;
    }
    s
}

#[tokio::test]
async fn no_scope_returns_every_scope() {
    // Arrange
    let s = scoped_store().await;

    // Act
    let facts = s.valid_at(T1).await;

    // Assert
    let mut found = objects(&facts, "likes");
    found.sort();
    assert_eq!(found, ["child", "lookalike", "none", "root"]);
}

#[tokio::test]
async fn a_scope_includes_its_descendants_and_not_look_alikes() {
    // Arrange
    let s = scoped_store().await;

    // Act
    let facts = s
        .graph
        .facts_valid_at(SUBJECT, Some("work"), Millis(T1), NO_LIMIT)
        .await
        .unwrap()
        .facts;

    // Assert
    let mut found = objects(&facts, "likes");
    found.sort();
    assert_eq!(found, ["child", "root"]);
}

#[tokio::test]
async fn an_invalid_scope_is_refused() {
    // Arrange
    let s = store().await;

    // Act
    let result = s
        .graph
        .facts_valid_at(SUBJECT, Some("bad scope"), Millis(T1), NO_LIMIT)
        .await;

    // Assert
    assert!(matches!(result, Err(crate::Error::InvalidScope(_))));
}

#[tokio::test]
async fn an_end_on_older_evidence_leaves_one_visible_version() {
    // Arrange
    let s = store().await;
    s.assert_at(T2, new_fact(SUBJECT, LIVES_IN, "Melbourne", HIGH_TRUST, T2))
        .await;
    let end = EndFact {
        subject: SUBJECT.into(),
        predicate: LIVES_IN.into(),
        object: "Melbourne".into(),
        scope: None,
        valid_until: Millis(T3),
        evidence_at: Millis(T1),
        trust_score: HIGH_TRUST,
        source_event_ids: vec!["stale-end".into()],
    };
    let ended = s.graph.end_fact(end, &single()).await.unwrap();

    // Act
    let facts = s.valid_at(T2 + 500).await;

    // Assert
    assert!(matches!(ended, EndOutcome::Ended { .. }), "{ended:?}");
    assert_eq!(objects(&facts, LIVES_IN), ["Melbourne"]);
    assert_eq!(facts[0].valid_until, FOREVER);
}

#[tokio::test]
async fn lineage_history_lists_every_version_with_its_sources() {
    // Arrange
    let s = timeline().await;
    let sydney = s.believed_at(T1, None).await;
    let lineage = sydney
        .iter()
        .find(|fact| fact.object == "Sydney")
        .unwrap()
        .lineage_id
        .clone();

    // Act
    let history = s
        .graph
        .fact_lineage_history(&lineage, NO_LIMIT)
        .await
        .unwrap()
        .facts;

    // Assert
    let spans: Vec<_> = history
        .iter()
        .map(|fact| (fact.tx_from, fact.tx_to, fact.valid_until))
        .collect();
    assert_eq!(
        spans,
        [
            (Millis(T1), Millis(T2), FOREVER),
            (Millis(T2), FOREVER, Millis(T2)),
        ]
    );
    assert!(history
        .iter()
        .all(|fact| fact.source_event_ids == ["alice-lives_in-Sydney"]));
}

#[tokio::test]
async fn sources_are_listed_in_order_and_capped() {
    // Arrange
    let s = store().await;
    let mut fact = new_fact(SUBJECT, LIVES_IN, "Sydney", MID_TRUST, T1);
    fact.source_event_ids = (0..=MAX_SOURCES_PER_FACT)
        .rev()
        .map(|n| format!("event-{n:04}"))
        .collect();
    s.assert_at(T1, fact).await;

    // Act
    let page = s
        .graph
        .facts_valid_at(SUBJECT, None, Millis(T1), NO_LIMIT)
        .await
        .unwrap();

    // Assert
    let sources = &page.facts[0].source_event_ids;
    let expected: Vec<_> = (0..MAX_SOURCES_PER_FACT)
        .map(|n| format!("event-{n:04}"))
        .collect();
    assert_eq!(sources, &expected);
    assert!(page.sources_truncated);
}

#[tokio::test]
async fn sources_exactly_at_the_cap_are_not_flagged() {
    // Arrange
    let s = store().await;
    let mut fact = new_fact(SUBJECT, LIVES_IN, "Sydney", MID_TRUST, T1);
    fact.source_event_ids = (0..MAX_SOURCES_PER_FACT)
        .map(|n| format!("event-{n:04}"))
        .collect();
    s.assert_at(T1, fact).await;

    // Act
    let page = s
        .graph
        .facts_valid_at(SUBJECT, None, Millis(T1), NO_LIMIT)
        .await
        .unwrap();

    // Assert
    assert_eq!(page.facts[0].source_event_ids.len(), MAX_SOURCES_PER_FACT);
    assert!(!page.sources_truncated);
}

impl Store {
    async fn assert_many_at(&self, tx: i64, fact: NewFact) {
        self.clock.set(Millis(tx));
        self.assert_many(fact).await;
    }
}

#[tokio::test]
async fn predicates_order_ascending_whatever_the_insert_order() {
    // Arrange
    let s = store().await;
    s.assert_many(new_fact(SUBJECT, "b_pred", "x", HIGH_TRUST, T1))
        .await;
    s.assert_many(new_fact(SUBJECT, "a_pred", "x", LOW_TRUST, T1))
        .await;

    // Act
    let facts = s.valid_at(T1).await;

    // Assert
    let predicates: Vec<_> = facts.iter().map(|f| f.predicate.as_str()).collect();
    assert_eq!(predicates, ["a_pred", "b_pred"]);
}

#[tokio::test]
async fn higher_confidence_ranks_first_within_equal_trust() {
    // Arrange
    let s = store().await;
    for (predicate, object, confidence) in [
        ("a_pred", "low", 0.2),
        ("a_pred", "high", 0.8),
        ("b_pred", "low", 0.1),
        ("b_pred", "high", 0.7),
    ] {
        let mut fact = new_fact(SUBJECT, predicate, object, MID_TRUST, T1);
        fact.confidence = confidence;
        s.assert_many(fact).await;
    }

    // Act
    let facts = s.valid_at(T1).await;

    // Assert
    let ranked: Vec<_> = facts.iter().map(|f| f.object.as_str()).collect();
    assert_eq!(ranked, ["high", "low", "high", "low"]);
}

#[tokio::test]
async fn the_later_transaction_ranks_first_within_equal_trust_and_confidence() {
    // Arrange
    let s = store().await;
    for predicate in ["a_pred", "b_pred"] {
        s.assert_many_at(T1, new_fact(SUBJECT, predicate, "older", MID_TRUST, T1))
            .await;
        s.assert_many_at(T2, new_fact(SUBJECT, predicate, "newer", MID_TRUST, T1))
            .await;
    }

    // Act
    let facts = s.valid_at(T2).await;

    // Assert
    let ranked: Vec<_> = facts.iter().map(|f| f.object.as_str()).collect();
    assert_eq!(ranked, ["newer", "older", "newer", "older"]);
}

#[tokio::test]
async fn the_current_read_seeks_on_the_facts_current_index() {
    // Arrange
    let s = store().await;
    let sql = format!(
        "EXPLAIN QUERY PLAN SELECT id FROM facts WHERE {CURRENT_FILTER}
         ORDER BY predicate ASC LIMIT 5"
    );
    let params = [SUBJECT.into(), FOREVER.into(), Millis(T1).into()];

    // Act
    let plan = s.graph.backend.query(&sql, &params).await.unwrap();

    // Assert
    let details: Vec<_> = plan.iter().map(|row| row.get_string(3).unwrap()).collect();
    assert!(
        details.iter().any(|d| d.contains("facts_current")
            && d.contains("subject=?")
            && !d.contains("tx_to=")),
        "{details:?}"
    );
}
