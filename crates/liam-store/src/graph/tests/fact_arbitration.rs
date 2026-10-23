// SPDX-License-Identifier: Apache-2.0
//! Fact writes against a real store: what the rows look like after each
//! outcome, that a failed write leaves none, and that racing writers serialize.

use tokio::sync::Barrier;

use super::support::FACT_WRITER;
use super::*;
use crate::ids::FactId;
use crate::types::{
    ArbitrationCaps, ArbitrationOutcome, Cardinality, EndFact, EndOutcome, ExtractorName, NewFact,
    PredicateSpec, RetractOutcome,
};

const DIMS: usize = 8;
pub(super) const START: Millis = Millis(1000);
pub(super) const LATER: Millis = Millis(2000);
const MIDDLE: Millis = Millis(3000);
pub(super) const SMALL_CAP: usize = 2;
const WEAK_TRUST: f64 = 0.3;
const RACE_ROUNDS: usize = 50;
pub(super) const LOW_TRUST: f64 = 0.5;
pub(super) const HIGH_TRUST: f64 = 0.9;

type ErrorCheck = fn(&Error) -> bool;

pub(super) fn one() -> PredicateSpec {
    PredicateSpec {
        cardinality: Cardinality::One,
        ..PredicateSpec::default()
    }
}

pub(super) fn many() -> PredicateSpec {
    PredicateSpec::default()
}

pub(super) fn fact_for(
    predicate: &str,
    object: &str,
    trust: f64,
    event: &str,
    evidence: i64,
) -> NewFact {
    NewFact {
        subject: "alice".into(),
        predicate: predicate.into(),
        object: object.into(),
        scope: None,
        valid_from: Millis(evidence),
        valid_until: FOREVER,
        evidence_at: Millis(evidence),
        confidence: 1.0,
        trust_score: trust,
        extractor: ExtractorName::Rules,
        extractor_version: "v1".into(),
        source_event_ids: vec![event.into()],
    }
}

pub(super) fn fact(object: &str, trust: f64, event: &str, evidence: i64) -> NewFact {
    fact_for("lives_in", object, trust, event, evidence)
}

async fn store_at(t: Millis) -> (DefaultGraph, Arc<FixedClock>) {
    let clock = Arc::new(FixedClock::new(t));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock.clone())
        .await
        .unwrap();
    (g, clock)
}

async fn assert_one(g: &DefaultGraph, fact: NewFact) -> ArbitrationOutcome {
    g.assert_fact(fact, &one(), FACT_WRITER, ArbitrationCaps::default())
        .await
        .unwrap()
}

async fn rows<B: Backend>(g: &Graph<B>, sql: &str) -> Vec<Row> {
    g.backend.query(sql, &[]).await.unwrap()
}

async fn count_of<B: Backend>(g: &Graph<B>, table: &str) -> i64 {
    rows(g, &format!("SELECT COUNT(*) FROM {table}")).await[0]
        .get_i64(0)
        .unwrap()
}

/// Objects of the versions that are current and open, in object order.
async fn live_objects<B: Backend>(g: &Graph<B>) -> Vec<String> {
    let sql = format!(
        "SELECT object FROM facts WHERE tx_to = {} AND valid_until = {} ORDER BY object",
        FOREVER.0, FOREVER.0
    );
    rows(g, &sql)
        .await
        .iter()
        .map(|row| row.get_string(0).unwrap())
        .collect()
}

/// Every version as comparable content, without ids or transaction stamps.
async fn content<B: Backend>(g: &Graph<B>) -> Vec<(String, String, i64, i64, bool)> {
    let sql = format!(
        "SELECT object, trust_score, valid_from, valid_until, tx_to = {} FROM facts",
        FOREVER.0
    );
    let mut content: Vec<_> = rows(g, &sql)
        .await
        .iter()
        .map(|row| {
            (
                row.get_string(0).unwrap(),
                format!("{:.2}", row.get_f64(1).unwrap()),
                row.get_i64(2).unwrap(),
                row.get_i64(3).unwrap(),
                row.get_i64(4).unwrap() == 1,
            )
        })
        .collect();
    content.sort();
    content
}

#[tokio::test]
async fn an_asserted_fact_is_stored_with_its_source() {
    // Arrange
    let (g, _) = store_at(START).await;

    // Act
    let outcome = assert_one(&g, fact("Sydney", LOW_TRUST, "e1", 100)).await;

    // Assert
    assert!(matches!(outcome, ArbitrationOutcome::Inserted { .. }));
    assert_eq!(live_objects(&g).await, ["Sydney"]);
    assert_eq!(count_of(&g, "fact_sources").await, 1);
    assert!(g.facts_without_sources().await.unwrap().is_empty());
}

#[tokio::test]
async fn a_fact_without_a_source_event_is_refused_and_leaves_no_row() {
    // Arrange
    let (g, _) = store_at(START).await;
    let mut unsourced = fact("Sydney", LOW_TRUST, "e1", 100);
    unsourced.source_event_ids.clear();

    // Act
    let error = g
        .assert_fact(unsourced, &one(), FACT_WRITER, ArbitrationCaps::default())
        .await
        .unwrap_err();

    // Assert
    assert!(matches!(error, Error::FactWithoutSource), "{error}");
    assert_eq!(count_of(&g, "facts").await, 0);
}

#[tokio::test]
async fn an_invalid_assertion_is_refused_before_the_store_is_touched() {
    let cases: [(&str, NewFact, ErrorCheck); 5] = [
        ("nan trust", fact("Sydney", f64::NAN, "e1", 100), |e| {
            matches!(e, Error::InvalidFactTrust(_))
        }),
        ("negative trust", fact("Sydney", -0.1, "e1", 100), |e| {
            matches!(e, Error::InvalidFactTrust(_))
        }),
        ("trust above one", fact("Sydney", 1.5, "e1", 100), |e| {
            matches!(e, Error::InvalidFactTrust(_))
        }),
        (
            "empty interval",
            NewFact {
                valid_until: Millis(100),
                ..fact("Sydney", LOW_TRUST, "e1", 100)
            },
            |e| matches!(e, Error::InvalidFactInterval { .. }),
        ),
        (
            "bad scope",
            NewFact {
                scope: Some("has space".into()),
                ..fact("Sydney", LOW_TRUST, "e1", 100)
            },
            |e| matches!(e, Error::InvalidScope(_)),
        ),
    ];
    for (label, invalid, is_expected) in cases {
        // Arrange
        let (g, _) = store_at(START).await;

        // Act
        let result = g
            .assert_fact(invalid, &one(), FACT_WRITER, ArbitrationCaps::default())
            .await;

        // Assert
        assert!(is_expected(&result.unwrap_err()), "{label}");
        assert_eq!(count_of(&g, "facts").await, 0, "{label}");
        assert_eq!(count_of(&g, "fact_sources").await, 0, "{label}");
    }
}

#[tokio::test]
async fn a_many_valued_predicate_keeps_every_object_and_counts_each_source() {
    // Arrange
    let (g, _) = store_at(START).await;

    // Act
    for (object, event) in [("Bob", "e1"), ("Carol", "e2"), ("Dan", "e3"), ("Bob", "e4")] {
        let known = fact_for("knows", object, LOW_TRUST, event, 100);
        g.assert_fact(known, &many(), FACT_WRITER, ArbitrationCaps::default())
            .await
            .unwrap();
    }

    // Assert
    assert_eq!(live_objects(&g).await, ["Bob", "Carol", "Dan"]);
    assert_eq!(count_of(&g, "facts").await, 3);
    assert_eq!(count_of(&g, "fact_sources").await, 4);
}

#[tokio::test]
async fn a_replacement_closes_the_old_version_in_both_timelines() {
    // Arrange
    let (g, clock) = store_at(START).await;
    assert_one(&g, fact("Sydney", LOW_TRUST, "e1", 100)).await;
    clock.set(LATER);

    // Act
    let outcome = assert_one(&g, fact("Melbourne", HIGH_TRUST, "e2", 200)).await;

    // Assert
    assert!(matches!(outcome, ArbitrationOutcome::Replaced { .. }));
    assert_eq!(live_objects(&g).await, ["Melbourne"]);
    let versions = rows(
        &g,
        "SELECT object, tx_from, tx_to, valid_until, closed_by FROM facts ORDER BY rowid",
    )
    .await;
    let shape: Vec<_> = versions
        .iter()
        .map(|row| {
            (
                row.get_string(0).unwrap(),
                row.get_i64(1).unwrap(),
                row.get_i64(2).unwrap(),
                row.get_i64(3).unwrap(),
                !row.get_string(4).unwrap().is_empty(),
            )
        })
        .collect();
    assert_eq!(
        shape,
        [
            ("Sydney".into(), 1000, 2000, FOREVER.0, false),
            ("Melbourne".into(), 2000, FOREVER.0, FOREVER.0, false),
            ("Sydney".into(), 2000, FOREVER.0, 200, true),
        ]
    );
}

#[tokio::test]
async fn a_weaker_fact_is_stored_beside_the_stronger_one() {
    // Arrange
    let (g, _) = store_at(START).await;
    assert_one(&g, fact("Sydney", HIGH_TRUST, "e1", 100)).await;

    // Act
    let outcome = assert_one(&g, fact("Melbourne", LOW_TRUST, "e2", 200)).await;

    // Assert
    assert!(matches!(outcome, ArbitrationOutcome::Competing { .. }));
    assert_eq!(live_objects(&g).await, ["Melbourne", "Sydney"]);
}

#[tokio::test]
async fn repeating_an_assert_writes_nothing() {
    // Arrange
    let (g, _) = store_at(START).await;
    assert_one(&g, fact("Sydney", LOW_TRUST, "e1", 100)).await;
    let before = content(&g).await;

    // Act
    let outcome = assert_one(&g, fact("Sydney", LOW_TRUST, "e1", 100)).await;

    // Assert
    assert!(matches!(
        outcome,
        ArbitrationOutcome::Corroborated { changed: false, .. }
    ));
    assert_eq!(content(&g).await, before);
    assert_eq!(count_of(&g, "fact_sources").await, 1);
}

#[tokio::test]
async fn a_new_event_for_a_live_fact_adds_a_source_and_no_version() {
    // Arrange
    let (g, _) = store_at(START).await;
    assert_one(&g, fact("Sydney", LOW_TRUST, "e1", 100)).await;

    // Act
    let outcome = assert_one(&g, fact("Sydney", LOW_TRUST, "e2", 200)).await;

    // Assert
    assert!(matches!(
        outcome,
        ArbitrationOutcome::Corroborated { changed: true, .. }
    ));
    assert_eq!(count_of(&g, "facts").await, 1);
    assert_eq!(count_of(&g, "fact_sources").await, 2);
}

#[tokio::test]
async fn two_asserts_under_a_frozen_clock_both_keep_their_rows() {
    // Arrange
    let (g, _) = store_at(START).await;

    // Act
    assert_one(&g, fact("Sydney", HIGH_TRUST, "e1", 100)).await;
    assert_one(&g, fact("Melbourne", LOW_TRUST, "e2", 200)).await;

    // Assert
    let stamps = rows(&g, "SELECT DISTINCT tx_from FROM facts").await;
    assert_eq!(stamps.len(), 1);
    assert_eq!(count_of(&g, "facts").await, 2);
}

#[tokio::test]
async fn a_clock_stepped_backwards_never_overlaps_history() {
    // Arrange
    let (g, clock) = store_at(LATER).await;
    assert_one(&g, fact("Sydney", LOW_TRUST, "e1", 100)).await;
    clock.set(START);

    // Act
    assert_one(&g, fact("Melbourne", HIGH_TRUST, "e2", 200)).await;

    // Assert
    let earliest_new = rows(
        &g,
        "SELECT MIN(tx_from) FROM facts WHERE object = 'Melbourne'",
    )
    .await[0]
        .get_i64(0)
        .unwrap();
    let closed_at = rows(
        &g,
        &format!(
            "SELECT tx_to FROM facts WHERE closed_by = '' AND tx_to < {}",
            FOREVER.0
        ),
    )
    .await[0]
        .get_i64(0)
        .unwrap();
    assert_eq!(earliest_new, LATER.0);
    assert_eq!(closed_at, LATER.0);
}

pub(super) fn end(
    object: &str,
    valid_until: i64,
    trust: f64,
    event: &str,
    evidence: i64,
) -> EndFact {
    EndFact {
        subject: "alice".into(),
        predicate: "lives_in".into(),
        object: object.into(),
        scope: None,
        valid_until: Millis(valid_until),
        evidence_at: Millis(evidence),
        trust_score: trust,
        source_event_ids: vec![event.into()],
    }
}

#[tokio::test]
async fn ending_a_fact_closes_it_and_never_inserts_a_live_one() {
    // Arrange
    let (g, _) = store_at(START).await;
    assert_one(&g, fact("Sydney", LOW_TRUST, "e1", 100)).await;

    // Act
    let ended = g
        .end_fact(
            end("sydney", 500, LOW_TRUST, "e2", 200),
            &one(),
            FACT_WRITER,
        )
        .await
        .unwrap();
    let replayed = g
        .end_fact(
            end("sydney", 500, LOW_TRUST, "e2", 200),
            &one(),
            FACT_WRITER,
        )
        .await
        .unwrap();
    let from_new_event = g
        .end_fact(
            end("sydney", 500, LOW_TRUST, "e3", 300),
            &one(),
            FACT_WRITER,
        )
        .await
        .unwrap();

    // Assert
    assert!(matches!(ended, EndOutcome::Ended { .. }));
    assert_eq!(replayed, EndOutcome::NoMatch);
    assert_eq!(from_new_event, EndOutcome::NoMatch);
    assert!(live_objects(&g).await.is_empty());
    let open = rows(
        &g,
        &format!("SELECT valid_until FROM facts WHERE tx_to = {}", FOREVER.0),
    )
    .await;
    assert_eq!(open[0].get_i64(0).unwrap(), 500);
    assert!(g.facts_without_sources().await.unwrap().is_empty());
}

#[tokio::test]
async fn an_end_with_no_match_a_stronger_fact_or_a_bad_interval_changes_nothing() {
    // Arrange
    let (g, _) = store_at(START).await;
    assert_one(&g, fact("Sydney", HIGH_TRUST, "e1", 100)).await;
    let before = content(&g).await;
    let cases = [
        (end("Perth", 500, LOW_TRUST, "e2", 200), EndOutcome::NoMatch),
        (
            end("Sydney", 500, LOW_TRUST, "e2", 200),
            EndOutcome::Outranked,
        ),
        (
            end("Sydney", 100, HIGH_TRUST, "e2", 200),
            EndOutcome::Rejected,
        ),
    ];

    for (statement, expected) in cases {
        // Act
        let outcome = g.end_fact(statement, &one(), FACT_WRITER).await.unwrap();

        // Assert
        assert_eq!(outcome, expected);
        assert_eq!(content(&g).await, before);
    }
}

pub(super) fn lineage_of(outcome: &ArbitrationOutcome) -> FactId {
    match outcome {
        ArbitrationOutcome::Inserted { id } | ArbitrationOutcome::Competing { id } => id.clone(),
        ArbitrationOutcome::Replaced { inserted, .. } => inserted.clone(),
        other => panic!("no new lineage in {other:?}"),
    }
}

#[tokio::test]
async fn a_retracted_fact_leaves_current_belief_but_stays_in_the_history() {
    // Arrange
    let (g, clock) = store_at(START).await;
    let lineage = lineage_of(&assert_one(&g, fact("Sydney", LOW_TRUST, "e1", 100)).await);
    clock.set(LATER);

    // Act
    let outcome = g.retract_fact(&lineage, false, FACT_WRITER).await.unwrap();

    // Assert
    assert_eq!(outcome, RetractOutcome::Retracted { restored: vec![] });
    assert!(live_objects(&g).await.is_empty());
    let closed = rows(&g, "SELECT tx_from, tx_to FROM facts").await;
    assert_eq!(
        (closed[0].get_i64(0).unwrap(), closed[0].get_i64(1).unwrap()),
        (START.0, LATER.0)
    );
}

#[tokio::test]
async fn retracting_an_unknown_lineage_finds_nothing() {
    // Arrange
    let (g, _) = store_at(START).await;

    // Act
    let outcome = g
        .retract_fact(&FactId::new(), false, FACT_WRITER)
        .await
        .unwrap();

    // Assert
    assert_eq!(outcome, RetractOutcome::NotFound);
}

#[tokio::test]
async fn a_retraction_blocks_old_evidence_but_not_newer_or_stronger_evidence() {
    // Arrange
    let (g, clock) = store_at(START).await;
    let lineage = lineage_of(&assert_one(&g, fact("Sydney", LOW_TRUST, "e1", 100)).await);
    clock.set(LATER);
    g.retract_fact(&lineage, false, FACT_WRITER).await.unwrap();

    // Act
    let stale = assert_one(&g, fact("Sydney", LOW_TRUST, "e2", 1500)).await;
    let stronger = assert_one(&g, fact("Sydney", HIGH_TRUST, "e3", 1500)).await;
    g.retract_fact(&lineage_of(&stronger), false, FACT_WRITER)
        .await
        .unwrap();
    let newer = assert_one(&g, fact("Sydney", LOW_TRUST, "e4", 2500)).await;

    // Assert
    assert_eq!(stale, ArbitrationOutcome::Suppressed);
    assert!(matches!(stronger, ArbitrationOutcome::Inserted { .. }));
    assert!(matches!(newer, ArbitrationOutcome::Inserted { .. }));
}

async fn suppression_rows<B: Backend>(g: &Graph<B>) -> Vec<(String, f64, i64)> {
    rows(
        g,
        "SELECT object_key, trust_score, retracted_at FROM fact_suppressions ORDER BY object_key",
    )
    .await
    .iter()
    .map(|row| {
        (
            row.get_string(0).unwrap(),
            row.get_f64(1).unwrap(),
            row.get_i64(2).unwrap(),
        )
    })
    .collect()
}

/// Asserts and retracts Sydney, so the retraction is stamped with the clock.
async fn retract_sydney(
    g: &DefaultGraph,
    clock: &FixedClock,
    at: Millis,
    trust: f64,
    event: &str,
    evidence: i64,
) {
    clock.set(at);
    let lineage = lineage_of(&assert_one(g, fact("Sydney", trust, event, evidence)).await);
    g.retract_fact(&lineage, false, FACT_WRITER).await.unwrap();
}

#[tokio::test]
async fn a_later_retraction_replaces_the_earlier_one_whole() {
    // Arrange
    let (g, clock) = store_at(START).await;
    retract_sydney(&g, &clock, LATER, HIGH_TRUST, "e1", 100).await;
    retract_sydney(&g, &clock, MIDDLE, WEAK_TRUST, "e2", 2500).await;

    // Act
    let between = assert_one(&g, fact("Sydney", 0.8, "e3", 2800)).await;

    // Assert
    assert_eq!(
        suppression_rows(&g).await,
        vec![("sydney".to_string(), WEAK_TRUST, MIDDLE.0)]
    );
    assert!(matches!(between, ArbitrationOutcome::Inserted { .. }));
}

#[tokio::test]
async fn retractions_at_the_same_time_keep_the_higher_trust() {
    // Arrange
    let (g, clock) = store_at(START).await;
    retract_sydney(&g, &clock, MIDDLE, WEAK_TRUST, "e1", 100).await;

    // Act
    retract_sydney(&g, &clock, MIDDLE, LOW_TRUST, "e2", 2800).await;

    // Assert
    assert_eq!(
        suppression_rows(&g).await,
        vec![("sydney".to_string(), LOW_TRUST, MIDDLE.0)]
    );
}

#[tokio::test]
async fn past_the_cap_the_weakest_competing_lineage_is_closed_and_suppressed() {
    // Arrange
    let (g, _) = store_at(START).await;
    let caps = ArbitrationCaps {
        max_competing_per_key: SMALL_CAP,
    };
    let claims = [
        ("Sydney", HIGH_TRUST, "e1", 100),
        ("Melbourne", LOW_TRUST, "e2", 200),
        ("Perth", 0.4, "e3", 300),
    ];
    for (object, trust, event, evidence) in claims {
        g.assert_fact(
            fact(object, trust, event, evidence),
            &one(),
            FACT_WRITER,
            caps,
        )
        .await
        .unwrap();
    }

    // Act
    let outcome = g
        .assert_fact(fact("Hobart", 0.45, "e4", 400), &one(), FACT_WRITER, caps)
        .await
        .unwrap();

    // Assert
    assert!(matches!(outcome, ArbitrationOutcome::Competing { .. }));
    assert_eq!(
        live_objects(&g).await,
        vec!["Hobart", "Melbourne", "Sydney"]
    );
    assert_eq!(
        suppression_rows(&g).await,
        vec![("perth".to_string(), 0.4, START.0)]
    );
}

#[tokio::test]
async fn retracting_with_restore_brings_back_only_what_that_lineage_replaced() {
    // Arrange
    let (g, clock) = store_at(START).await;
    assert_one(&g, fact("Sydney", LOW_TRUST, "e1", 100)).await;
    g.assert_fact(
        fact_for("works_at", "Acme", LOW_TRUST, "e2", 100),
        &one(),
        FACT_WRITER,
        ArbitrationCaps::default(),
    )
    .await
    .unwrap();
    clock.set(LATER);
    let replacing_city = assert_one(&g, fact("Melbourne", HIGH_TRUST, "e3", 200)).await;
    g.assert_fact(
        fact_for("works_at", "Initech", HIGH_TRUST, "e4", 200),
        &one(),
        FACT_WRITER,
        ArbitrationCaps::default(),
    )
    .await
    .unwrap();

    // Act
    let outcome = g
        .retract_fact(&lineage_of(&replacing_city), true, FACT_WRITER)
        .await
        .unwrap();

    // Assert
    assert!(matches!(
        outcome,
        RetractOutcome::Retracted { ref restored } if restored.len() == 1
    ));
    assert_eq!(live_objects(&g).await, ["Initech", "Sydney"]);
    assert!(g.facts_without_sources().await.unwrap().is_empty());
}

#[tokio::test]
async fn restoring_is_refused_while_another_fact_holds_the_key() {
    // Arrange
    let (g, _) = store_at(START).await;
    assert_one(&g, fact("Sydney", HIGH_TRUST, "e1", 100)).await;
    let weaker = lineage_of(&assert_one(&g, fact("Melbourne", LOW_TRUST, "e2", 200)).await);
    let before = content(&g).await;

    // Act
    let outcome = g.retract_fact(&weaker, true, FACT_WRITER).await.unwrap();

    // Assert
    assert_eq!(outcome, RetractOutcome::RestoreBlocked);
    assert_eq!(content(&g).await, before);
    assert_eq!(count_of(&g, "fact_suppressions").await, 0);
}

async fn failing_store() -> Graph<FailingBackend> {
    let clock = Arc::new(FixedClock::new(START));
    Graph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock)
        .await
        .unwrap()
}

#[tokio::test]
async fn a_failure_part_way_through_a_replacement_leaves_the_store_as_it_was() {
    // Arrange
    let g = failing_store().await;
    g.assert_fact(
        fact("Sydney", LOW_TRUST, "e1", 100),
        &one(),
        FACT_WRITER,
        ArbitrationCaps::default(),
    )
    .await
    .unwrap();
    let before = content(&g).await;
    let sources_before = count_of(&g, "fact_sources").await;
    g.backend.set_fail_on_execute(1);

    // Act
    let result = g
        .assert_fact(
            fact("Melbourne", HIGH_TRUST, "e2", 200),
            &one(),
            FACT_WRITER,
            ArbitrationCaps::default(),
        )
        .await;

    // Assert
    assert!(matches!(result, Err(Error::Backend(_))), "{result:?}");
    assert_eq!(content(&g).await, before);
    assert_eq!(count_of(&g, "fact_sources").await, sources_before);
}

async fn serial_content(first: NewFact, second: NewFact) -> Vec<(String, String, i64, i64, bool)> {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("graph.db");
    let g = DefaultGraph::open_with_clock(
        path.to_str().unwrap(),
        GraphConfig::new(DIMS),
        Arc::new(FixedClock::new(START)),
    )
    .await
    .unwrap();
    for fact in [first, second] {
        g.assert_fact(fact, &one(), FACT_WRITER, ArbitrationCaps::default())
            .await
            .unwrap();
    }
    content(&g).await
}

#[tokio::test]
async fn a_second_writer_waits_for_the_first_and_decides_on_what_it_wrote() {
    // Arrange: the first writer is held after reading an empty key, holding the
    // write lock. Without the lock the second would read the same empty key and
    // both facts would end up live.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("graph.db");
    let g: Arc<Graph<FailingBackend>> = Arc::new(
        Graph::open_with_clock(
            path.to_str().unwrap(),
            GraphConfig::new(DIMS),
            Arc::new(FixedClock::new(START)),
        )
        .await
        .unwrap(),
    );
    let gate = Arc::new(QueryGate::new());
    *g.backend.probe.query_gate.lock().unwrap() = Some(gate.clone());
    let weak = fact("Sydney", LOW_TRUST, "e1", 100);
    let strong = fact("Melbourne", HIGH_TRUST, "e2", 200);
    let first = tokio::spawn({
        let (g, weak) = (g.clone(), weak.clone());
        async move {
            g.assert_fact(weak, &one(), FACT_WRITER, ArbitrationCaps::default())
                .await
        }
    });
    gate.wait_until_held().await;
    let mut second = tokio::spawn({
        let (g, strong) = (g.clone(), strong.clone());
        async move {
            g.assert_fact(strong, &one(), FACT_WRITER, ArbitrationCaps::default())
                .await
        }
    });

    // Act
    let blocked = tokio::time::timeout(std::time::Duration::from_millis(100), &mut second).await;
    gate.let_go();
    let first = first.await.unwrap().unwrap();
    let second = second.await.unwrap().unwrap();

    // Assert
    assert!(
        blocked.is_err(),
        "the second writer ran while the first held the lock"
    );
    assert!(matches!(first, ArbitrationOutcome::Inserted { .. }));
    assert!(matches!(second, ArbitrationOutcome::Replaced { .. }));
    assert_eq!(live_objects(&*g).await, ["Melbourne"]);
    assert_eq!(content(&*g).await, serial_content(weak, strong).await);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn racing_asserts_end_in_the_state_of_one_serial_order() {
    // Arrange
    let weak = fact("Sydney", LOW_TRUST, "e1", 100);
    let strong = fact("Melbourne", HIGH_TRUST, "e2", 100);
    let weak_first = serial_content(weak.clone(), strong.clone()).await;
    let strong_first = serial_content(strong.clone(), weak.clone()).await;

    for round in 0..RACE_ROUNDS {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("graph.db");
        let g = Arc::new(
            DefaultGraph::open_with_clock(
                path.to_str().unwrap(),
                GraphConfig::new(DIMS),
                Arc::new(FixedClock::new(START)),
            )
            .await
            .unwrap(),
        );
        let start = Arc::new(Barrier::new(2));
        let racer = |fact: NewFact| {
            let (g, start) = (g.clone(), start.clone());
            tokio::spawn(async move {
                start.wait().await;
                g.assert_fact(fact, &one(), FACT_WRITER, ArbitrationCaps::default())
                    .await
                    .unwrap()
            })
        };

        // Act
        let (a, b) = (racer(weak.clone()), racer(strong.clone()));
        a.await.unwrap();
        b.await.unwrap();

        // Assert
        let settled = content(&*g).await;
        assert!(
            settled == weak_first || settled == strong_first,
            "round {round} ended in neither serial order: {settled:?}"
        );
        assert!(g.facts_without_sources().await.unwrap().is_empty());
    }
}
