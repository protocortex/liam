// SPDX-License-Identifier: Apache-2.0
//! Fact writes through the event log: each decision is logged before it is
//! applied, a replay reproduces the rows without arbitrating again, and a run
//! the log marks as rolled back stays gone. Every test uses a real WAL and
//! database file in its own temp directory.

use liam_log::event::{LogPayload, CURRENT_SCHEMA_VERSION};

use super::fact_arbitration::{
    end, fact, fact_for, lineage_of, many, one, HIGH_TRUST, LATER, LOW_TRUST, SMALL_CAP, START,
};
use super::log_write::{events, RecordingLog};
use super::support::{
    controlled_log, count, event, share, snapshot, Env, FACT_WRITER, ONE_SEGMENT,
};
use super::*;
use crate::types::{ArbitrationCaps, ArbitrationOutcome, EndOutcome};
use crate::DefaultBackend;

const MIDDLE: Millis = Millis(3000);
const OTHER_CLIENT: &str = "client:mcp";
/// The fact tables with the columns that order their rows.
const FACT_TABLES: [(&str, &str); 3] = [
    ("facts", "id"),
    ("fact_sources", "lineage_id, event_id"),
    ("fact_suppressions", "subject, predicate, scope, object_key"),
];
/// Puts the cursor back before the first record, as a store that never replayed.
const REWIND_CURSOR: &str = "UPDATE log_cursor SET last_segment = NULL, last_index = NULL";

/// Every row of the fact tables, ids and stamps included, in a stable order.
async fn fact_state<B: Backend>(g: &Graph<B>) -> Vec<String> {
    let mut state = Vec::new();
    for (table, order) in FACT_TABLES {
        let rows = g
            .backend
            .query(&format!("SELECT * FROM {table} ORDER BY {order}"), &[])
            .await
            .unwrap();
        state.extend(rows.iter().map(|row| format!("{table}: {row:?}")));
    }
    state
}

/// Every outcome class that writes: insert, corroborate, replace, historical,
/// competing, a many-valued insert, an end and a retraction.
async fn script(g: &DefaultGraph, clock: &FixedClock, source: &str) {
    let caps = ArbitrationCaps::default();
    let spec = one();
    let assert = |fact| g.assert_fact(fact, &spec, source, caps);
    assert(fact("Sydney", LOW_TRUST, "e1", 100)).await.unwrap();
    clock.set(LATER);
    let corroborated = assert(fact("Sydney", LOW_TRUST, "e2", 150)).await.unwrap();
    assert!(matches!(
        corroborated,
        ArbitrationOutcome::Corroborated { changed: true, .. }
    ));
    clock.set(MIDDLE);
    let replaced = assert(fact("Melbourne", HIGH_TRUST, "e3", 200))
        .await
        .unwrap();
    assert!(matches!(replaced, ArbitrationOutcome::Replaced { .. }));
    let historical = assert(fact("Perth", LOW_TRUST, "e4", 50)).await.unwrap();
    assert!(matches!(historical, ArbitrationOutcome::Historical { .. }));
    assert(fact_for("works_at", "Acme", HIGH_TRUST, "e5", 100))
        .await
        .unwrap();
    let competing = assert(fact_for("works_at", "Initech", LOW_TRUST, "e6", 150))
        .await
        .unwrap();
    assert!(matches!(competing, ArbitrationOutcome::Competing { .. }));
    for (object, event) in [("tea", "e7"), ("coffee", "e8")] {
        let liked = fact_for("likes", object, LOW_TRUST, event, 100);
        g.assert_fact(liked, &many(), source, caps).await.unwrap();
    }
    let ended = g
        .end_fact(
            end("Melbourne", 5000, HIGH_TRUST, "e9", 300),
            &one(),
            source,
        )
        .await
        .unwrap();
    assert!(matches!(ended, EndOutcome::Ended { .. }), "{ended:?}");
    g.retract_fact(&lineage_of(&competing), false, source)
        .await
        .unwrap();
}

/// A live store that ran `script`, over the log in `env`.
async fn scripted(env: &Env, db: &str, source: &str) -> (DefaultGraph, SharedLog) {
    let clock = Arc::new(FixedClock::new(START));
    let (g, log) = env.open::<DefaultBackend>(db, Arc::clone(&clock)).await;
    script(&g, &clock, source).await;
    (g, log)
}

/// A second store over the log of `env`, which a first store has written.
async fn replica(env: &Env, log: &SharedLog, db: &str) -> DefaultGraph {
    let clock = Arc::new(FixedClock::new(START));
    let (g, _) = env
        .open_over::<DefaultBackend>(Arc::clone(log), db, clock)
        .await;
    g
}

async fn fact_writes(env: &Env) -> Vec<liam_log::event::LogEvent> {
    env.records()
        .await
        .into_iter()
        .map(|record| record.event)
        .filter(|event| matches!(event.payload, LogPayload::FactWrite { .. }))
        .collect()
}

#[tokio::test]
async fn a_fact_write_is_logged_with_its_source_and_schema_version_before_it_is_applied() {
    // Arrange
    let env = Env::new();
    let (g, _log) = scripted(&env, "live.db", OTHER_CLIENT).await;

    // Act
    let logged = fact_writes(&env).await;

    // Assert
    assert!(logged.len() >= 8, "{} fact writes", logged.len());
    for event in &logged {
        assert_eq!(event.source, OTHER_CLIENT);
        assert_eq!(event.schema_version, CURRENT_SCHEMA_VERSION);
    }
    assert!(g.facts_without_sources().await.unwrap().is_empty());
}

#[tokio::test]
async fn a_store_that_replays_the_log_holds_the_facts_the_live_store_holds() {
    // Arrange
    let env = Env::new();
    let (live, log) = scripted(&env, "live.db", FACT_WRITER).await;
    let replica = replica(&env, &log, "replica.db").await;

    // Act
    let report = replica.catch_up().await.unwrap();

    // Assert
    let expected = fact_state(&live).await;
    assert!(report.applied > 0, "{report:?}");
    assert_eq!(report.quarantined, 0);
    for (table, _) in FACT_TABLES {
        assert!(
            expected.iter().any(|row| row.starts_with(table)),
            "{table} is empty"
        );
    }
    assert_eq!(fact_state(&replica).await, expected);
}

#[tokio::test]
async fn a_rebuild_reproduces_the_facts_and_clears_a_stale_extraction_cursor() {
    // Arrange
    let env = Env::new();
    let (live, log) = scripted(&env, "live.db", FACT_WRITER).await;
    let expected = fact_state(&live).await;
    live.backend
        .execute(
            "INSERT INTO extraction_cursor (name, last_segment, last_index, updated_at)
             VALUES ('rules', 0, 0, 1)",
            &[],
        )
        .await
        .unwrap();

    // Act
    let (rebuilt, _) = live
        .rebuild_from_log(log, RebuildMode::Replace)
        .await
        .unwrap();

    // Assert
    assert_eq!(fact_state(&rebuilt).await, expected);
    assert_eq!(count(&rebuilt, "extraction_cursor").await, 0);
}

#[tokio::test]
async fn a_rebuild_over_facts_the_log_does_not_hold_is_refused_in_require_empty_mode() {
    // Arrange
    let env = Env::new();
    let (live, log) = scripted(&env, "live.db", FACT_WRITER).await;

    // Act
    let result = live.rebuild_from_log(log, RebuildMode::RequireEmpty).await;

    // Assert
    assert!(matches!(result, Err(Error::ProjectionNotEmpty)));
}

#[tokio::test]
async fn a_replay_that_runs_twice_changes_nothing_the_second_time() {
    // Arrange
    let env = Env::new();
    let (_live, log) = scripted(&env, "live.db", FACT_WRITER).await;
    let replica = replica(&env, &log, "replica.db").await;
    replica.catch_up().await.unwrap();
    let once = fact_state(&replica).await;
    replica.backend.execute(REWIND_CURSOR, &[]).await.unwrap();

    // Act
    let report = replica.catch_up().await.unwrap();

    // Assert
    assert_eq!(report.quarantined, 0, "{report:?}");
    assert_eq!(fact_state(&replica).await, once);
}

#[tokio::test]
async fn a_no_op_assert_appends_nothing() {
    // Arrange
    let (log, appended) = RecordingLog::new();
    let clock = Arc::new(FixedClock::new(START));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock)
        .await
        .unwrap()
        .with_log(share(log))
        .await
        .unwrap();
    let caps = ArbitrationCaps::default();
    g.assert_fact(
        fact("Sydney", LOW_TRUST, "e1", 100),
        &one(),
        FACT_WRITER,
        caps,
    )
    .await
    .unwrap();
    let logged = events(&appended).len();

    // Act
    let again = g
        .assert_fact(
            fact("Sydney", LOW_TRUST, "e1", 100),
            &one(),
            FACT_WRITER,
            caps,
        )
        .await
        .unwrap();

    // Assert
    assert!(matches!(
        again,
        ArbitrationOutcome::Corroborated { changed: false, .. }
    ));
    assert_eq!(events(&appended).len(), logged);
}

#[tokio::test]
async fn a_fact_write_beside_node_and_edge_writes_leaves_them_as_they_replay_alone() {
    // Arrange
    let env = Env::new();
    let clock = Arc::new(FixedClock::new(START));
    let (live, log) = env
        .open::<DefaultBackend>("live.db", Arc::clone(&clock))
        .await;
    let from = live.insert(super::support::fact_at("alpha")).await.unwrap();
    let to = live.insert(super::support::fact_at("beta")).await.unwrap();
    live.relate(&from, &to, "mentions").await.unwrap();
    script(&live, &clock, FACT_WRITER).await;
    let replica = replica(&env, &log, "replica.db").await;

    // Act
    replica.catch_up().await.unwrap();

    // Assert
    assert_eq!(snapshot(&replica).await, snapshot(&live).await);
    assert_eq!(fact_state(&replica).await, fact_state(&live).await);
}

#[tokio::test]
async fn a_write_whose_apply_fails_is_voided_and_a_replay_skips_it() {
    // Arrange
    let env = Env::new();
    let clock = Arc::new(FixedClock::new(START));
    let (g, log) = env.open::<FailingBackend>("live.db", clock).await;
    g.backend.set_fail_on_execute(0);

    // Act
    let result = g
        .assert_fact(
            fact("Sydney", LOW_TRUST, "e1", 100),
            &one(),
            FACT_WRITER,
            ArbitrationCaps::default(),
        )
        .await;
    let replica = replica(&env, &log, "replica.db").await;
    let report = replica.catch_up().await.unwrap();

    // Assert
    assert!(matches!(result, Err(Error::Backend(_))), "{result:?}");
    let logged: Vec<_> = env.records().await;
    assert!(matches!(
        logged.as_slice(),
        [first, second]
            if matches!(first.event.payload, LogPayload::FactWrite { .. })
                && matches!(&second.event.payload,
                    LogPayload::Voided { target_event_id } if *target_event_id == first.event.event_id)
    ));
    assert_eq!(report.skipped_voided, 1);
    assert_eq!(count(&replica, "facts").await, 0);
}

#[tokio::test]
async fn a_failed_void_append_refuses_the_next_fact_write() {
    // Arrange
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("graph.db");
    let (log, appended) =
        RecordingLog::over(liam_log::test_support::FailingLogWriter::fail_on_nth(2));
    let clock = Arc::new(FixedClock::new(START));
    let g = Graph::<FailingBackend>::open_with_clock(
        path.to_str().unwrap(),
        GraphConfig::new(8),
        clock,
    )
    .await
    .unwrap()
    .with_log(share(log))
    .await
    .unwrap();
    g.backend.set_fail_on_execute(0);
    let caps = ArbitrationCaps::default();
    let first = g
        .assert_fact(
            fact("Sydney", LOW_TRUST, "e1", 100),
            &one(),
            FACT_WRITER,
            caps,
        )
        .await;
    g.backend.set_fail_on_execute(usize::MAX);
    let logged = events(&appended).len();

    // Act
    let second = g
        .assert_fact(
            fact("Sydney", LOW_TRUST, "e2", 100),
            &one(),
            FACT_WRITER,
            caps,
        )
        .await;

    // Assert
    assert!(matches!(first, Err(Error::Backend(_))), "{first:?}");
    assert!(matches!(second, Err(Error::LogPoisoned)), "{second:?}");
    assert_eq!(events(&appended).len(), logged);
}

#[tokio::test]
async fn a_write_logged_but_never_applied_is_applied_by_the_next_catch_up() {
    // Arrange: the commit fails and so does the void, as in a crash between the
    // append and the commit.
    let env = Env::new();
    let (log, control) = controlled_log(&env);
    let clock = Arc::new(FixedClock::new(START));
    let (g, log) = env.open_over::<FailingBackend>(log, "live.db", clock).await;
    control.fail_nth_from_now(2);
    g.backend.set_fail_commit(true);
    let result = g
        .assert_fact(
            fact("Sydney", LOW_TRUST, "e1", 100),
            &one(),
            FACT_WRITER,
            ArbitrationCaps::default(),
        )
        .await;
    assert!(result.is_err(), "{result:?}");
    assert_eq!(count(&g, "facts").await, 0);
    assert_eq!(fact_writes(&env).await.len(), 1);
    drop((g, log));

    // Act
    let reopened = replica(&env, &env.log(ONE_SEGMENT), "live.db").await;
    let report = reopened.catch_up().await.unwrap();

    // Assert
    assert_eq!(report.applied, 1, "{report:?}");
    assert_eq!(count(&reopened, "facts").await, 1);
    assert_eq!(count(&reopened, "fact_sources").await, 1);
}

#[tokio::test]
async fn a_rolled_back_run_stays_gone_after_a_rebuild_and_other_writes_stay() {
    // Arrange: runs "1" and "12" and a client write share the log, then run "1"
    // is marked rolled back.
    let env = Env::new();
    let clock = Arc::new(FixedClock::new(START));
    {
        let (g, _log) = env
            .open::<DefaultBackend>("live.db", Arc::clone(&clock))
            .await;
        let caps = ArbitrationCaps::default();
        for (source, object, event) in [
            ("consolidation:1", "Sydney", "e1"),
            ("consolidation:12", "Melbourne", "e2"),
            (OTHER_CLIENT, "Perth", "e3"),
        ] {
            let liked = fact_for("likes", object, LOW_TRUST, event, 100);
            g.assert_fact(liked, &many(), source, caps).await.unwrap();
        }
    }
    env.append(&[event(
        "rollback-1",
        [0; 32],
        LogPayload::RunRolledBack { run_id: "1".into() },
    )]);
    let (rebuilt, log) = env
        .open::<DefaultBackend>("rebuilt.db", Arc::clone(&clock))
        .await;

    // Act
    rebuilt.catch_up().await.unwrap();

    // Assert
    let objects: Vec<String> = rebuilt
        .backend
        .query("SELECT object FROM facts ORDER BY object", &[])
        .await
        .unwrap()
        .iter()
        .map(|row| row.get_string(0).unwrap())
        .collect();
    assert_eq!(objects, ["Melbourne", "Perth"]);
    let (again, _) = rebuilt
        .rebuild_from_log(log, RebuildMode::Replace)
        .await
        .unwrap();
    assert_eq!(count(&again, "facts").await, 2);
}

#[tokio::test]
async fn evicting_at_the_competing_cap_replays_with_its_suppression() {
    // Arrange
    let env = Env::new();
    let clock = Arc::new(FixedClock::new(START));
    let (live, log) = env
        .open::<DefaultBackend>("live.db", Arc::clone(&clock))
        .await;
    let caps = ArbitrationCaps {
        max_competing_per_key: SMALL_CAP,
    };
    for (object, event) in [
        ("Sydney", "e1"),
        ("Perth", "e2"),
        ("Hobart", "e3"),
        ("Cairns", "e4"),
    ] {
        let trust = if object == "Sydney" {
            HIGH_TRUST
        } else {
            LOW_TRUST
        };
        live.assert_fact(fact(object, trust, event, 100), &one(), FACT_WRITER, caps)
            .await
            .unwrap();
    }
    let replica = replica(&env, &log, "replica.db").await;

    // Act
    replica.catch_up().await.unwrap();

    // Assert
    assert_eq!(fact_state(&replica).await, fact_state(&live).await);
}
