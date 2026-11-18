// SPDX-License-Identifier: Apache-2.0
//! What a rollback puts back: every fact version, the cursor, and the failure
//! counts, compared before and after.

use liam_store::{batch_key, FixedClock};

use super::*;
use crate::extraction::extractor::tracking_id;
use crate::extraction::model_double::MODEL_SHA;
use crate::extraction::prompt::PROMPT_VERSION;

const POISON: &str = "POISON";
const EARLY: Millis = Millis(1500);
const LATE: Millis = Millis(2500);

/// Every version of every fact about `PEOPLE`, closed ones included, as text.
async fn fact_versions(store: &DefaultGraph) -> Vec<String> {
    let mut versions = Vec::new();
    for person in PEOPLE {
        for at in [EARLY, LATE, ALL_LIVE] {
            let page = store
                .facts_valid_at(person, None, at, FACT_LIMIT)
                .await
                .unwrap();
            for fact in page.facts {
                let history = store
                    .fact_lineage_history(&fact.lineage_id, FACT_LIMIT)
                    .await
                    .unwrap();
                versions.extend(history.facts.iter().map(|v| format!("{v:?}")));
            }
        }
    }
    versions.sort();
    versions.dedup();
    versions
}

async fn poison_failures(harness: &Harness, store: &DefaultGraph) -> u32 {
    let (_, event_id) = logged_note(harness, &employment("zed", POISON)).await;
    let key = batch_key(PROMPT_VERSION, MODEL_SHA, &[tracking_id(None, &event_id)]);
    store.batch_failures(&key).await.unwrap()
}

/// Ada's employer changes from Acme to Initech, so a version is closed, and
/// Bob joins.
async fn history(harness: &Harness) -> (MemoryServer, Arc<FixedClock>) {
    let (store, clock) = harness.store_at(Millis(1000)).await;
    let probed = NewNode::now("fact", "kept", "a fact that stays").with_producer(CLIENT);
    store.insert(probed).await.unwrap();
    store.insert(employment_note("ada")).await.unwrap();
    clock.set(Millis(2000));
    let moved = client_note(&employment("ada", "Initech"));
    store.insert(moved).await.unwrap();
    store.insert(employment_note("bob")).await.unwrap();
    (harness.server(store, Arc::new(liam_model::MockLlm)), clock)
}

/// Ada moves again, Cy joins, and one event the model cannot read.
async fn later_events(server: &MemoryServer, clock: &FixedClock) {
    clock.set(Millis(3000));
    let store = server.store().await.unwrap();
    let moved = client_note(&employment("ada", "Globex"));
    store.insert(moved).await.unwrap();
    store.insert(employment_note("cy")).await.unwrap();
    let unreadable = client_note(&employment("zed", POISON));
    store.insert(unreadable).await.unwrap();
}

#[tokio::test]
async fn a_rolled_back_run_restores_every_fact_version_the_cursor_and_the_failure_counts() {
    // Arrange: a committed run with a closed version, then new events
    let harness = Harness::new();
    let (server, clock) = history(&harness).await;
    let provider = ProviderDouble::new();
    let first = harness.job_with_hooks(
        &server,
        honest(),
        None,
        hooks_of(extraction_part(&harness, &provider, config())),
    );
    assert_eq!(first.tick().await, Some(RunOutcome::Committed));
    later_events(&server, &clock).await;
    let store = server.store().await.unwrap();
    let (versions_before, cursor_before) = (fact_versions(&store).await, cursor_of(&store).await);
    drop(store);
    assert!(
        versions_before.len() >= 3,
        "the history holds a closed version: {versions_before:?}"
    );
    provider.model.garble(POISON);
    let second = harness.job_with_hooks(
        &server,
        regresses(),
        None,
        hooks_of(extraction_part(&harness, &provider, config())),
    );

    // Act
    let outcome = second.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::RolledBack));
    let store = server.store().await.unwrap();
    assert_eq!(fact_versions(&store).await, versions_before);
    assert_eq!(cursor_of(&store).await, cursor_before);
    assert_eq!(poison_failures(&harness, &store).await, 0);
}

#[tokio::test]
async fn a_committed_run_counts_the_failure_of_an_event_the_model_cannot_read() {
    // Arrange
    let harness = Harness::new();
    let (server, clock) = history(&harness).await;
    later_events(&server, &clock).await;
    let provider = ProviderDouble::new();
    provider.model.garble(POISON);
    let job = harness.job_with_hooks(
        &server,
        honest(),
        None,
        hooks_of(extraction_part(&harness, &provider, config())),
    );

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    let store = server.store().await.unwrap();
    assert_eq!(poison_failures(&harness, &store).await, 1);
}
