// SPDX-License-Identifier: Apache-2.0
//! The rules pass inside whole runs: it needs no model, and the model pass
//! failing or being absent must not stall it.

use liam_store::relation;

use super::*;

const MENTIONED_IN: &str = "mentioned_in";

/// The facts the rules wrote about the entity page for Ada.
async fn mention_facts(store: &DefaultGraph) -> Vec<Fact> {
    let page = store
        .facts_valid_at("ada", None, ALL_LIVE, FACT_LIMIT)
        .await
        .unwrap();
    page.facts
        .into_iter()
        .filter(|fact| fact.predicate == MENTIONED_IN && fact.extractor_id == ExtractorName::Rules)
        .collect()
}

/// A store with an entity page for Ada that mentions the first note, and
/// `mentions` notes that name her.
async fn with_ada(harness: &Harness, mentions: usize) -> MemoryServer {
    let store = filled(harness, 0).await;
    let ada = store
        .insert(NewNode::entity("person", "Ada"))
        .await
        .unwrap();
    for n in 1..=mentions {
        let note = client_note(&format!("Ada reviewed item {n} today"));
        store.insert(note).await.unwrap();
    }
    let kept = store.query(&liam_store::Query::text("kept")).await.unwrap();
    let kept = kept.first().expect("the kept note").id.clone();
    store
        .link(NewEdge::new(&ada, &kept, relation::MENTIONS))
        .await
        .unwrap();
    harness.server(store, Arc::new(liam_model::MockLlm))
}

#[tokio::test]
async fn rules_only_extraction_writes_mention_facts_and_starts_no_model() {
    // Arrange
    let harness = Harness::new();
    let server = with_ada(&harness, 2).await;
    let store = server.store().await.unwrap();
    let last_event = store.last_applied_offset().await.unwrap();
    let provider = ProviderDouble::new();
    let part = extraction_part(
        &harness,
        &provider,
        ExtractionConfig {
            llm_enabled: false,
            ..config()
        },
    );
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert_eq!(mention_facts(&store).await.len(), 2);
    assert_eq!(rules_cursor_of(&store).await, last_event);
    assert_eq!(cursor_of(&store).await, None);
    assert_eq!(provider.opens(), 0);
}

#[tokio::test]
async fn a_model_that_never_starts_does_not_stall_the_rules_over_three_runs() {
    // Arrange: two events a run, so the later notes are only in later windows
    let harness = Harness::new();
    let server = with_ada(&harness, 3).await;
    let provider = ProviderDouble::new();
    provider.fail_open();
    let part = extraction_part(
        &harness,
        &provider,
        ExtractionConfig {
            max_events_per_run: 2,
            ..config()
        },
    );
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));
    let store = server.store().await.unwrap();

    // Act
    let mut cursors = Vec::new();
    for _ in 0..3 {
        job.tick().await;
        cursors.push(rules_cursor_of(&store).await);
    }

    // Assert
    assert!(cursors.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(mention_facts(&store).await.len(), 3);
    let facts = facts_of(&store).await;
    assert!(facts
        .iter()
        .all(|fact| fact.extractor_id == ExtractorName::Rules));
    let manifest = RunManifest::load(&harness.manifest_path())
        .unwrap()
        .unwrap();
    assert!(done(&manifest, "extract:partial"));
}

#[tokio::test]
async fn the_rules_facts_of_a_rolled_back_run_are_gone_and_the_next_run_writes_them_again() {
    // Arrange
    let harness = Harness::new();
    let server = with_ada(&harness, 2).await;
    let provider = ProviderDouble::new();
    let part = extraction_part(&harness, &provider, config());
    let regressing = harness.job_with_hooks(&server, regresses(), None, hooks_of(part));

    // Act
    let outcome = regressing.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::RolledBack));
    let store = server.store().await.unwrap();
    assert!(mention_facts(&store).await.is_empty());
    assert_eq!(rules_cursor_of(&store).await, None);
    drop(store);
    let part = extraction_part(&harness, &provider, config());
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));
    job.tick().await;
    let store = server.store().await.unwrap();
    assert_eq!(mention_facts(&store).await.len(), 2);
}
