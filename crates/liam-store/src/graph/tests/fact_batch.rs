// SPDX-License-Identifier: Apache-2.0
//! `assert_facts`: a batch lands in one transaction and one log record, with
//! the arbitration of `assert_fact`, and a fault anywhere leaves none of it.

use liam_log::event::LogPayload;

use super::fact_arbitration::{fact, fact_for, many, one, HIGH_TRUST, LOW_TRUST, START};
use super::fact_log::{fact_state, fact_writes, replica};
use super::support::{count, snapshot, Env, FACT_WRITER};
use super::*;
use crate::types::{ArbitrationCaps, ArbitrationOutcome, NewFact, PredicateSpec};
use crate::DefaultBackend;

const BATCH_OF_THREE: usize = 3;

fn batch(facts: Vec<NewFact>) -> Vec<(NewFact, PredicateSpec)> {
    facts
        .into_iter()
        .map(|fact| {
            let spec = if fact.predicate == "lives_in" {
                one()
            } else {
                many()
            };
            (fact, spec)
        })
        .collect()
}

fn three_facts() -> Vec<NewFact> {
    vec![
        fact("Sydney", LOW_TRUST, "e1", 100),
        fact_for("works_at", "Acme", LOW_TRUST, "e2", 100),
        fact_for("knows", "Bob", LOW_TRUST, "e3", 100),
    ]
}

/// Every version as comparable content, without ids or transaction stamps.
async fn content<B: Backend>(g: &Graph<B>) -> Vec<String> {
    let sql = format!(
        "SELECT predicate, object, trust_score, valid_from, valid_until, tx_to = {} FROM facts",
        FOREVER.0
    );
    let mut content: Vec<String> = g
        .backend
        .query(&sql, &[])
        .await
        .unwrap()
        .iter()
        .map(|row| format!("{row:?}"))
        .collect();
    content.sort();
    content
}

async fn live_store(env: &Env) -> (DefaultGraph, SharedLog) {
    let clock = Arc::new(FixedClock::new(START));
    env.open::<DefaultBackend>("live.db", clock).await
}

#[tokio::test]
async fn a_batch_of_three_is_one_log_record_and_all_three_facts() {
    // Arrange
    let env = Env::new();
    let (g, _log) = live_store(&env).await;

    // Act
    let outcomes = g
        .assert_facts(
            batch(three_facts()),
            FACT_WRITER,
            ArbitrationCaps::default(),
        )
        .await
        .unwrap();

    // Assert
    assert_eq!(outcomes.len(), BATCH_OF_THREE);
    assert!(outcomes
        .iter()
        .all(|outcome| matches!(outcome, ArbitrationOutcome::Inserted { .. })));
    assert_eq!(count(&g, "facts").await, BATCH_OF_THREE as i64);
    assert_eq!(fact_writes(&env).await.len(), 1);
    assert!(g.facts_without_sources().await.unwrap().is_empty());
}

#[tokio::test]
async fn a_batch_decides_like_the_same_facts_asserted_one_by_one() {
    // Arrange: a mix of corroboration, replacement and a competing claim.
    let facts = vec![
        fact("Sydney", LOW_TRUST, "e1", 100),
        fact("Sydney", LOW_TRUST, "e2", 150),
        fact("Melbourne", HIGH_TRUST, "e3", 200),
        fact("Perth", LOW_TRUST, "e4", 300),
        fact_for("works_at", "Acme", LOW_TRUST, "e5", 100),
        fact_for("works_at", "Acme", HIGH_TRUST, "e6", 120),
    ];
    let one_by_one = DefaultGraph::open_with_clock(
        ":memory:",
        GraphConfig::new(8),
        Arc::new(FixedClock::new(START)),
    )
    .await
    .unwrap();
    let together = DefaultGraph::open_with_clock(
        ":memory:",
        GraphConfig::new(8),
        Arc::new(FixedClock::new(START)),
    )
    .await
    .unwrap();
    let caps = ArbitrationCaps::default();
    let mut single = Vec::new();
    for (fact, spec) in batch(facts.clone()) {
        single.push(
            one_by_one
                .assert_fact(fact, &spec, FACT_WRITER, caps)
                .await
                .unwrap(),
        );
    }

    // Act
    let batched = together
        .assert_facts(batch(facts), FACT_WRITER, caps)
        .await
        .unwrap();

    // Assert
    let kinds = |outcomes: &[ArbitrationOutcome]| -> Vec<String> {
        outcomes
            .iter()
            .map(|outcome| {
                format!("{outcome:?}")
                    .split(['{', ' '])
                    .next()
                    .unwrap()
                    .into()
            })
            .collect()
    };
    assert_eq!(kinds(&batched), kinds(&single));
    assert_eq!(content(&together).await, content(&one_by_one).await);
    assert!(together.facts_without_sources().await.unwrap().is_empty());
}

#[tokio::test]
async fn a_fact_that_closes_an_earlier_fact_of_the_batch_leaves_one_live_and_replays_the_same() {
    // Arrange
    let env = Env::new();
    let (g, log) = live_store(&env).await;
    let facts = vec![
        fact("Sydney", LOW_TRUST, "e1", 100),
        fact("Melbourne", HIGH_TRUST, "e2", 200),
    ];

    // Act
    let outcomes = g
        .assert_facts(batch(facts), FACT_WRITER, ArbitrationCaps::default())
        .await
        .unwrap();
    let replayed = replica(&env, &log, "replica.db").await;
    replayed.catch_up().await.unwrap();

    // Assert
    assert!(matches!(outcomes[1], ArbitrationOutcome::Replaced { .. }));
    let live = g
        .facts_valid_at("alice", None, Millis(500), 10)
        .await
        .unwrap();
    assert_eq!(live.facts.len(), 1);
    assert_eq!(live.facts[0].object, "Melbourne");
    assert_eq!(fact_writes(&env).await.len(), 1);
    assert_eq!(fact_state(&replayed).await, fact_state(&g).await);
    assert_eq!(snapshot(&replayed).await, snapshot(&g).await);
}

#[tokio::test]
async fn a_replayed_batch_equals_the_live_store() {
    // Arrange
    let env = Env::new();
    let (g, log) = live_store(&env).await;
    g.assert_facts(
        batch(three_facts()),
        FACT_WRITER,
        ArbitrationCaps::default(),
    )
    .await
    .unwrap();

    // Act
    let replayed = replica(&env, &log, "replica.db").await;
    replayed.catch_up().await.unwrap();

    // Assert
    assert_eq!(fact_state(&replayed).await, fact_state(&g).await);
}

#[tokio::test]
async fn one_refused_fact_leaves_none_of_the_batch_and_names_its_position() {
    // Arrange
    let env = Env::new();
    let (g, _log) = live_store(&env).await;
    let mut facts = three_facts();
    facts[2].trust_score = 1.5;

    // Act
    let result = g
        .assert_facts(batch(facts), FACT_WRITER, ArbitrationCaps::default())
        .await;

    // Assert
    assert!(
        matches!(&result, Err(Error::FactInBatch { index: 2, source })
            if matches!(**source, Error::InvalidFactTrust(_))),
        "{result:?}"
    );
    assert_eq!(count(&g, "facts").await, 0);
    assert_eq!(count(&g, "fact_sources").await, 0);
    assert!(fact_writes(&env).await.is_empty());
}

#[tokio::test]
async fn a_fault_between_two_writes_of_the_batch_leaves_nothing_visible_and_is_voided() {
    // Arrange: the second statement of the batch's projection fails.
    let env = Env::new();
    let clock = Arc::new(FixedClock::new(START));
    let (g, log) = env.open::<FailingBackend>("live.db", clock).await;
    g.backend.set_fail_on_execute(1);

    // Act
    let result = g
        .assert_facts(
            batch(three_facts()),
            FACT_WRITER,
            ArbitrationCaps::default(),
        )
        .await;
    g.backend.set_fail_on_execute(usize::MAX);
    let replayed = replica(&env, &log, "replica.db").await;
    let report = replayed.catch_up().await.unwrap();

    // Assert
    assert!(matches!(result, Err(Error::Backend(_))), "{result:?}");
    assert_eq!(count(&g, "facts").await, 0);
    assert_eq!(count(&g, "fact_sources").await, 0);
    assert_eq!(report.skipped_voided, 1);
    assert_eq!(count(&replayed, "facts").await, 0);
}

#[tokio::test]
async fn a_batch_over_the_limit_is_refused_before_anything_is_written() {
    // Arrange
    let g = DefaultGraph::open_with_clock(
        ":memory:",
        GraphConfig::new(8),
        Arc::new(FixedClock::new(START)),
    )
    .await
    .unwrap();
    let facts: Vec<NewFact> = (0..=MAX_FACTS_PER_BATCH)
        .map(|n| fact_for("knows", &format!("Person{n}"), LOW_TRUST, "e1", 100))
        .collect();

    // Act
    let result = g
        .assert_facts(batch(facts), FACT_WRITER, ArbitrationCaps::default())
        .await;

    // Assert
    assert!(matches!(result, Err(Error::FactBatchTooLarge { len, max })
            if len == MAX_FACTS_PER_BATCH + 1 && max == MAX_FACTS_PER_BATCH));
    assert_eq!(count(&g, "facts").await, 0);
}

#[tokio::test]
async fn an_empty_batch_writes_and_logs_nothing() {
    // Arrange
    let env = Env::new();
    let (g, _log) = live_store(&env).await;

    // Act
    let outcomes = g
        .assert_facts(Vec::new(), FACT_WRITER, ArbitrationCaps::default())
        .await
        .unwrap();

    // Assert
    assert!(outcomes.is_empty());
    assert!(fact_writes(&env).await.is_empty());
}

#[tokio::test]
async fn a_batch_that_changes_nothing_logs_nothing() {
    // Arrange
    let env = Env::new();
    let (g, _log) = live_store(&env).await;
    let caps = ArbitrationCaps::default();
    g.assert_facts(batch(three_facts()), FACT_WRITER, caps)
        .await
        .unwrap();

    // Act
    let again = g
        .assert_facts(batch(three_facts()), FACT_WRITER, caps)
        .await
        .unwrap();

    // Assert
    assert!(again.iter().all(|outcome| matches!(
        outcome,
        ArbitrationOutcome::Corroborated { changed: false, .. }
    )));
    assert_eq!(fact_writes(&env).await.len(), 1);
    assert!(matches!(
        fact_writes(&env).await[0].payload,
        LogPayload::FactWrite { .. }
    ));
}
