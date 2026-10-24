// SPDX-License-Identifier: Apache-2.0
//! `assert_facts`: a batch lands in one transaction and one log record, with
//! the arbitration of `assert_fact`, and a fault anywhere leaves none of it.

use liam_log::event::LogPayload;

use super::fact_arbitration::{
    fact, fact_for, many, one, HIGH_TRUST, LOW_TRUST, MIDDLE, SMALL_CAP, START,
};
use super::fact_log::{fact_state, fact_writes, replica};
use super::support::{count, event, snapshot, Env, FACT_WRITER, ONE_SEGMENT};
use super::*;
use crate::graph::arbitration::{KeyState, Plan, Suppression};
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

const EMBEDDING_DIMS: usize = 8;
const SCRIPTS: u64 = 60;
const FACTS_PER_SCRIPT: u64 = 7;
const CLIENT_SOURCE: &str = "client:mcp";
const RUN_SOURCE: &str = "consolidation:7";

async fn memory_store(start: Millis) -> (DefaultGraph, Arc<FixedClock>) {
    let clock = Arc::new(FixedClock::new(start));
    let g =
        DefaultGraph::open_with_clock(":memory:", GraphConfig::new(EMBEDDING_DIMS), clock.clone())
            .await
            .unwrap();
    (g, clock)
}

fn kinds(outcomes: &[ArbitrationOutcome]) -> Vec<String> {
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
}

/// Suppression rows and the event linked to each lineage's first object.
async fn bookkeeping<B: Backend>(g: &Graph<B>) -> Vec<String> {
    let mut rows: Vec<String> = Vec::new();
    for sql in [
        "SELECT object_key, trust_score, retracted_at FROM fact_suppressions",
        "SELECT f.object, s.event_id FROM fact_sources s JOIN facts f ON f.id = s.lineage_id",
    ] {
        let found = g.backend.query(sql, &[]).await.unwrap();
        rows.extend(found.iter().map(|row| format!("{row:?}")));
    }
    rows.sort();
    rows
}

/// Asserts `facts` one by one into one store and as a batch into another, and
/// requires the same outcomes, versions, sources and suppressions.
async fn decides_like(facts: Vec<NewFact>, caps: ArbitrationCaps) {
    let (one_by_one, _) = memory_store(START).await;
    let (together, _) = memory_store(START).await;
    let mut single = Vec::new();
    let mut first_refusal = None;
    for (index, (fact, spec)) in batch(facts.clone()).into_iter().enumerate() {
        match one_by_one.assert_fact(fact, &spec, FACT_WRITER, caps).await {
            Ok(outcome) => single.push(outcome),
            Err(_) => {
                first_refusal = Some(index);
                break;
            }
        }
    }

    let batched = together
        .assert_facts(batch(facts.clone()), FACT_WRITER, caps)
        .await;

    // A script one-by-one writes refuse is refused whole, naming the same fact.
    if let Some(index) = first_refusal {
        assert!(
            matches!(&batched, Err(Error::FactInBatch { index: found, .. }) if *found == index),
            "{facts:?}: {batched:?}"
        );
        assert_eq!(count(&together, "facts").await, 0);
        return;
    }
    let batched = batched.unwrap();
    assert_eq!(kinds(&batched), kinds(&single), "{facts:?}");
    assert_eq!(
        content(&together).await,
        content(&one_by_one).await,
        "{facts:?}"
    );
    assert_eq!(
        bookkeeping(&together).await,
        bookkeeping(&one_by_one).await,
        "{facts:?}"
    );
    assert!(together.facts_without_sources().await.unwrap().is_empty());
}

/// A small deterministic generator, so a failing script can be rerun.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, bound: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % bound
    }
}

fn random_script(seed: u64) -> Vec<NewFact> {
    const OBJECTS: [&str; 4] = ["Sydney", "Melbourne", "Perth", "Hobart"];
    const TRUSTS: [f64; 3] = [0.4, LOW_TRUST, HIGH_TRUST];
    let mut random = Lcg(seed);
    (0..FACTS_PER_SCRIPT)
        .map(|n| {
            let object = OBJECTS[random.next(OBJECTS.len() as u64) as usize];
            let trust = TRUSTS[random.next(TRUSTS.len() as u64) as usize];
            let evidence = 100 + 50 * random.next(6) as i64;
            let mut fact = fact(
                object,
                trust,
                &format!("e{}", random.next(4) + n % 2),
                evidence,
            );
            fact.valid_from = Millis(evidence - 50 * random.next(2) as i64);
            fact
        })
        .collect()
}

#[tokio::test]
async fn a_batch_decides_like_the_same_facts_asserted_one_by_one() {
    // Arrange: corroboration, a repeat that adds nothing, replacement and a competing claim.
    let facts = vec![
        fact("Sydney", LOW_TRUST, "e1", 100),
        fact("Sydney", LOW_TRUST, "e2", 150),
        fact("Sydney", LOW_TRUST, "e2", 150),
        fact("Melbourne", HIGH_TRUST, "e3", 200),
        fact("Perth", LOW_TRUST, "e4", 300),
        fact_for("works_at", "Acme", LOW_TRUST, "e5", 100),
        fact_for("works_at", "Acme", HIGH_TRUST, "e6", 120),
    ];

    // Act / Assert
    decides_like(facts, ArbitrationCaps::default()).await;
}

#[tokio::test]
async fn generated_scripts_decide_like_one_by_one_writes_with_and_without_a_small_cap() {
    for seed in 0..SCRIPTS {
        for caps in [
            ArbitrationCaps::default(),
            ArbitrationCaps {
                max_competing_per_key: SMALL_CAP,
            },
        ] {
            decides_like(random_script(seed), caps).await;
        }
    }
}

#[tokio::test]
async fn an_object_evicted_at_the_cap_stays_suppressed_for_a_later_claim_in_the_same_batch() {
    // Arrange: the fourth claim evicts Perth, and the fifth asks for Perth again.
    let caps = ArbitrationCaps {
        max_competing_per_key: SMALL_CAP,
    };
    let facts = vec![
        fact("Sydney", HIGH_TRUST, "e1", 100),
        fact("Melbourne", LOW_TRUST, "e2", 200),
        fact("Perth", 0.4, "e3", 300),
        fact("Hobart", 0.45, "e4", 400),
        fact("Perth", 0.4, "e5", 300),
    ];
    let (together, _) = memory_store(START).await;

    // Act
    let outcomes = together
        .assert_facts(batch(facts.clone()), FACT_WRITER, caps)
        .await
        .unwrap();

    // Assert
    assert!(matches!(outcomes[4], ArbitrationOutcome::Suppressed));
    decides_like(facts, caps).await;
}

#[tokio::test]
async fn two_suppressions_of_one_object_in_a_batch_keep_the_later_one_whole() {
    // Arrange: Perth is evicted twice at different times, the later with the lower trust.
    let (mut state, plans) = (KeyState::default(), |trust: f64, at: i64| Plan {
        suppress: vec![Suppression {
            subject: "alice".into(),
            predicate: "lives_in".into(),
            scope: String::new(),
            object_key: "perth".into(),
            trust_score: trust,
            retracted_at: Millis(at),
        }],
        ..Plan::default()
    });

    // Act
    state.apply(&plans(HIGH_TRUST, 100));
    state.apply(&plans(LOW_TRUST, 200));
    state.apply(&plans(HIGH_TRUST, 150));

    // Assert
    assert_eq!(state.suppressions.len(), 1);
    assert_eq!(state.suppressions[0].retracted_at, Millis(200));
    assert_eq!(state.suppressions[0].trust_score, LOW_TRUST);
}

#[tokio::test]
async fn a_batch_after_a_backwards_clock_step_never_stamps_before_the_version_it_closes() {
    // Arrange
    let (g, clock) = memory_store(MIDDLE).await;
    let caps = ArbitrationCaps::default();
    g.assert_fact(
        fact("Sydney", LOW_TRUST, "e1", 100),
        &one(),
        FACT_WRITER,
        caps,
    )
    .await
    .unwrap();
    clock.set(START);

    // Act
    g.assert_facts(
        batch(vec![fact("Melbourne", HIGH_TRUST, "e2", 200)]),
        FACT_WRITER,
        caps,
    )
    .await
    .unwrap();

    // Assert
    let backwards = g
        .backend
        .query("SELECT COUNT(*) FROM facts WHERE tx_to < tx_from", &[])
        .await
        .unwrap();
    assert_eq!(backwards[0].get_i64(0).unwrap(), 0);
    let newest = g
        .backend
        .query(
            "SELECT MIN(tx_from) FROM facts WHERE object = 'Melbourne'",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(newest[0].get_i64(0).unwrap(), MIDDLE.0);
}

#[tokio::test]
async fn exactly_the_limit_is_accepted_and_one_more_is_refused() {
    // Arrange
    let distinct = |count: usize| -> Vec<NewFact> {
        (0..count)
            .map(|n| fact_for("knows", &format!("Person{n}"), LOW_TRUST, "e1", 100))
            .collect()
    };
    let (g, _) = memory_store(START).await;
    let caps = ArbitrationCaps::default();

    // Act
    let at_limit = g
        .assert_facts(batch(distinct(MAX_FACTS_PER_BATCH)), FACT_WRITER, caps)
        .await;
    let over = g
        .assert_facts(batch(distinct(MAX_FACTS_PER_BATCH + 1)), FACT_WRITER, caps)
        .await;

    // Assert
    assert_eq!(at_limit.unwrap().len(), MAX_FACTS_PER_BATCH);
    assert!(matches!(over, Err(Error::FactBatchTooLarge { .. })));
}

#[tokio::test]
async fn a_fact_refused_at_decide_time_fails_the_batch_with_its_index_and_writes_nothing() {
    // Arrange: Melbourne is live from 200, so a claim starting at 300 on evidence from 100
    // would be a past that never ended before it began.
    let (g, _) = memory_store(START).await;
    let caps = ArbitrationCaps::default();
    g.assert_fact(
        fact("Melbourne", LOW_TRUST, "e1", 200),
        &one(),
        FACT_WRITER,
        caps,
    )
    .await
    .unwrap();
    let mut stale = fact("Sydney", LOW_TRUST, "e2", 100);
    stale.valid_from = Millis(300);
    let facts = vec![fact_for("works_at", "Acme", LOW_TRUST, "e3", 100), stale];

    // Act
    let result = g.assert_facts(batch(facts), FACT_WRITER, caps).await;

    // Assert
    assert!(
        matches!(&result, Err(Error::FactInBatch { index: 1, source })
            if matches!(**source, Error::InvalidFactInterval { .. })),
        "{result:?}"
    );
    assert_eq!(count(&g, "facts").await, 1);
}

/// `(predicate, object)` of every current version, in order.
async fn live_pairs(g: &DefaultGraph) -> Vec<(String, String)> {
    let sql = format!(
        "SELECT predicate, object FROM facts WHERE tx_to = {} ORDER BY predicate, object",
        FOREVER.0
    );
    g.backend
        .query(&sql, &[])
        .await
        .unwrap()
        .iter()
        .map(|row| (row.get_string(0).unwrap(), row.get_string(1).unwrap()))
        .collect()
}

#[tokio::test]
async fn a_rolled_back_batch_over_two_keys_leaves_the_other_key_alone_after_a_later_close() {
    // Arrange: the client states Sydney and Acme, run 7 replaces both in one batch, the
    // client then replaces Melbourne, and run 7 is rolled back.
    let env = Env::new();
    let clock = Arc::new(FixedClock::new(START));
    {
        let (g, _log) = env
            .open::<DefaultBackend>("live.db", Arc::clone(&clock))
            .await;
        let caps = ArbitrationCaps::default();
        let client_facts = [
            fact("Sydney", LOW_TRUST, "e1", 100),
            fact_for("works_at", "Acme", LOW_TRUST, "e2", 100),
        ];
        for (n, fact) in client_facts.into_iter().enumerate() {
            clock.set(Millis(1000 + n as i64));
            g.assert_fact(fact, &one(), CLIENT_SOURCE, caps)
                .await
                .unwrap();
        }
        clock.set(Millis(2000));
        let run = [
            fact("Melbourne", HIGH_TRUST, "e3", 200),
            fact_for("works_at", "Initech", HIGH_TRUST, "e4", 200),
        ];
        let specs = vec![(run[0].clone(), one()), (run[1].clone(), one())];
        g.assert_facts(specs, RUN_SOURCE, caps).await.unwrap();
        clock.set(Millis(3000));
        g.assert_fact(
            fact("Perth", HIGH_TRUST, "e5", 300),
            &one(),
            CLIENT_SOURCE,
            caps,
        )
        .await
        .unwrap();
    }
    env.append(&[event(
        "rollback-7",
        [0; 32],
        LogPayload::RunRolledBack { run_id: "7".into() },
    )]);
    let log = env.log(ONE_SEGMENT);
    let rebuilt = replica(&env, &log, "rebuilt.db").await;
    let expected = vec![
        ("lives_in".to_string(), "Perth".to_string()),
        ("works_at".to_string(), "Acme".to_string()),
    ];

    // Act
    rebuilt.catch_up().await.unwrap();
    let replayed = live_pairs(&rebuilt).await;
    let (again, _) = rebuilt
        .rebuild_from_log(log, RebuildMode::Replace)
        .await
        .unwrap();

    // Assert: Acme was never replaced once the run is gone, and the close of
    // Melbourne reaches only the lives_in key.
    assert_eq!(replayed, expected);
    assert_eq!(live_pairs(&again).await, expected);
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
