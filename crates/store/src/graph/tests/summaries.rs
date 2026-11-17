// SPDX-License-Identifier: Apache-2.0
//! The summary store: what a summary keeps beside its text, that one child set
//! has one live summary, that a write lands whole or not at all, and which nodes
//! the tree builder may read.

use std::collections::BTreeSet;

use super::support::{count, fact, injected};
use super::*;
use crate::graph::summaries::child_hash;
use crate::types::{
    ChildKind, EventRef, EventRole, NewSummary, PutOutcome, SummaryChild, TreeState,
};

const DIMS: usize = 8;
const LEVEL: u8 = 1;
const MODEL: &str = "model-a";
const PROMPT: &str = "prompt-1";
const WRITER: &str = "consolidation:test";

fn clock() -> Arc<FixedClock> {
    Arc::new(FixedClock::new(Millis(1000)))
}

async fn open() -> DefaultGraph {
    DefaultGraph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock())
        .await
        .unwrap()
}

fn vector() -> Vec<f32> {
    vec![0.5; DIMS]
}

fn event(id: &str) -> EventRef {
    EventRef {
        event_id: id.to_string(),
        synthetic: false,
    }
}

fn new_summary(scope: &str) -> NewSummary {
    NewSummary {
        level: LEVEL,
        scope: scope.to_string(),
        text: "a summary".to_string(),
        events_fed: 2,
        valid_min: Millis(100),
        valid_max: Millis(900),
        model_id: MODEL.to_string(),
        prompt_version: PROMPT.to_string(),
    }
}

fn nodes(ids: &[&str]) -> Vec<SummaryChild> {
    ids.iter()
        .map(|id| SummaryChild {
            id: (*id).to_string(),
            kind: ChildKind::Node,
        })
        .collect()
}

async fn put<B: Backend>(g: &Graph<B>, children: &[&str], closes: &[String]) -> Result<PutOutcome> {
    g.put_summary(
        new_summary(""),
        &nodes(children),
        &[event("e1"), event("e2")],
        &vector(),
        closes,
        WRITER,
    )
    .await
}

fn id_of(outcome: PutOutcome) -> String {
    match outcome {
        PutOutcome::Inserted(id) | PutOutcome::Unchanged(id) => id,
    }
}

#[tokio::test]
async fn a_new_summary_is_stored_with_its_children_sources_and_vector() {
    // Arrange
    let g = open().await;

    // Act
    let outcome = put(&g, &["a", "b"], &[]).await.unwrap();

    // Assert
    let PutOutcome::Inserted(id) = outcome else {
        panic!("expected an insert, got {outcome:?}");
    };
    let live = g.live_summaries(LEVEL, "").await.unwrap();
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].id, id);
    assert_eq!(live[0].text.as_str(), "a summary");
    assert_eq!(live[0].derivation, Derivation::Inferred);
    assert_eq!(
        live[0].child_hash,
        child_hash(LEVEL, "", &["a".into(), "b".into()])
    );
    assert_eq!(
        (live[0].valid_min, live[0].valid_max),
        (Millis(100), Millis(900))
    );
    assert_eq!((live[0].tx_from, live[0].tx_to), (Millis(1000), FOREVER));
    assert_eq!(count(&g, "summary_children").await, 2);
    assert_eq!(
        g.summary_sources(&id).await.unwrap(),
        [event("e1"), event("e2")]
    );
    assert_eq!(
        g.summary_embeddings("", LEVEL).await.unwrap(),
        [(id, vector())]
    );
}

#[tokio::test]
async fn the_children_of_a_summary_are_listed_with_their_kind() {
    // Arrange
    let g = open().await;
    let upper = NewSummary {
        level: 2,
        ..new_summary("")
    };
    let children: Vec<SummaryChild> = ["b", "a"]
        .iter()
        .map(|id| SummaryChild {
            id: (*id).to_string(),
            kind: ChildKind::Summary,
        })
        .collect();
    let outcome = g
        .put_summary(upper, &children, &[event("e1")], &vector(), &[], WRITER)
        .await
        .unwrap();

    // Act
    let listed = g.summary_children(&id_of(outcome)).await.unwrap();

    // Assert
    let ids: Vec<(&str, ChildKind)> = listed.iter().map(|c| (c.id.as_str(), c.kind)).collect();
    assert_eq!(ids, [("a", ChildKind::Summary), ("b", ChildKind::Summary)]);
    assert!(ChildKind::parse("edge").is_err());
}

#[tokio::test]
async fn a_child_of_the_wrong_kind_for_the_level_is_refused_and_leaves_no_row() {
    // Arrange
    let g = open().await;
    let summary_child = vec![SummaryChild {
        id: "a".to_string(),
        kind: ChildKind::Summary,
    }];
    let upper = NewSummary {
        level: 2,
        ..new_summary("")
    };

    // Act
    let at_level_one = g
        .put_summary(
            new_summary(""),
            &summary_child,
            &[event("e1")],
            &vector(),
            &[],
            WRITER,
        )
        .await;
    let at_level_two = g
        .put_summary(
            upper,
            &nodes(&["a"]),
            &[event("e1")],
            &vector(),
            &[],
            WRITER,
        )
        .await;

    // Assert
    assert!(matches!(at_level_one, Err(Error::InvalidReference(_))));
    assert!(matches!(at_level_two, Err(Error::InvalidReference(_))));
    assert_eq!(count(&g, "summaries").await, 0);
}

#[tokio::test]
async fn the_same_child_set_is_unchanged_and_a_changed_one_replaces_it_in_one_step() {
    // Arrange
    let g = open().await;
    let first = id_of(put(&g, &["a", "b"], &[]).await.unwrap());

    // Act
    let again = put(&g, &["b", "a"], &[]).await.unwrap();
    let second = put(&g, &["a", "c"], std::slice::from_ref(&first))
        .await
        .unwrap();

    // Assert
    assert_eq!(again, PutOutcome::Unchanged(first.clone()));
    let PutOutcome::Inserted(second) = second else {
        panic!("expected an insert, got {second:?}");
    };
    let live = g.live_summaries(LEVEL, "").await.unwrap();
    assert_eq!(live.iter().map(|s| &s.id).collect::<Vec<_>>(), [&second]);
    assert_eq!(count(&g, "summaries").await, 2);
}

#[tokio::test]
async fn a_child_set_that_returns_to_an_earlier_value_gets_a_new_live_row() {
    // Arrange: {a, b} replaced by {a, c}, which {a, b} then replaces
    let g = open().await;
    let first = id_of(put(&g, &["a", "b"], &[]).await.unwrap());
    let second = id_of(
        put(&g, &["a", "c"], std::slice::from_ref(&first))
            .await
            .unwrap(),
    );

    // Act
    let third = put(&g, &["a", "b"], &[second]).await.unwrap();

    // Assert
    let PutOutcome::Inserted(third) = third else {
        panic!("expected an insert, got {third:?}");
    };
    assert_ne!(third, first);
    assert_eq!(count(&g, "summaries").await, 3);
    let live = g.live_summaries(LEVEL, "").await.unwrap();
    assert_eq!(live.iter().map(|s| &s.id).collect::<Vec<_>>(), [&third]);
}

#[tokio::test]
async fn a_summary_with_no_sources_or_no_children_is_refused_and_leaves_no_row() {
    // Arrange
    let g = open().await;

    // Act
    let no_sources = g
        .put_summary(new_summary(""), &nodes(&["a"]), &[], &vector(), &[], WRITER)
        .await;
    let no_children = g
        .put_summary(new_summary(""), &[], &[event("e1")], &vector(), &[], WRITER)
        .await;

    // Assert
    assert!(matches!(no_sources, Err(Error::SummaryWithoutSources(_))));
    assert!(matches!(no_children, Err(Error::InvalidReference(_))));
    assert_eq!(count(&g, "summaries").await, 0);
}

#[tokio::test]
async fn a_vector_of_the_wrong_width_is_refused() {
    // Arrange
    let g = open().await;

    // Act
    let result = g
        .put_summary(
            new_summary(""),
            &nodes(&["a"]),
            &[event("e1")],
            &[0.5; DIMS + 1],
            &[],
            WRITER,
        )
        .await;

    // Assert
    assert!(matches!(result, Err(Error::Dimension { .. })));
    assert_eq!(count(&g, "summaries").await, 0);
}

#[tokio::test]
async fn closing_a_summary_that_is_not_open_is_an_error() {
    // Arrange
    let g = open().await;
    let id = id_of(put(&g, &["a"], &[]).await.unwrap());
    g.close_summary(&id, Millis(2000), WRITER).await.unwrap();

    // Act
    let again = g.close_summary(&id, Millis(3000), WRITER).await;
    let replace = put(&g, &["b"], std::slice::from_ref(&id)).await;

    // Assert
    assert!(matches!(again, Err(Error::SummaryNotOpen(_))));
    assert!(matches!(replace, Err(Error::SummaryNotOpen(_))));
    assert_eq!(
        count(&g, "summaries").await,
        1,
        "the replacement rolled back"
    );
}

#[tokio::test]
async fn the_closure_reports_a_child_whose_events_the_summary_lacks() {
    // Arrange: two nodes written with no log, so their origins are synthetic
    let g = open().await;
    let covered = g.insert(fact("one")).await.unwrap();
    let missed = g.insert(fact("two")).await.unwrap();
    let covered_events = g.node_event_ids(&covered, EventRole::Origin).await.unwrap();
    let children = [
        SummaryChild {
            id: covered.as_str().to_string(),
            kind: ChildKind::Node,
        },
        SummaryChild {
            id: missed.as_str().to_string(),
            kind: ChildKind::Node,
        },
    ];
    let id = id_of(
        g.put_summary(
            new_summary(""),
            &children,
            &covered_events,
            &vector(),
            &[],
            WRITER,
        )
        .await
        .unwrap(),
    );

    // Act
    let report = g.check_closure(&id).await.unwrap();

    // Assert
    assert_eq!(report.uncovered_children, [missed.as_str().to_string()]);
    assert!(!report.is_closed());
    assert_eq!((report.real_events, report.synthetic_events), (0, 1));
}

#[tokio::test]
async fn the_closure_of_a_parent_covers_the_events_of_its_child_summary() {
    // Arrange
    let g = open().await;
    let child = id_of(put(&g, &["a", "b"], &[]).await.unwrap());
    let parent_children = [SummaryChild {
        id: child,
        kind: ChildKind::Summary,
    }];

    // Act
    let parent = id_of(
        g.put_summary(
            NewSummary {
                level: LEVEL + 1,
                ..new_summary("")
            },
            &parent_children,
            &[event("e1"), event("e2"), event("e3")],
            &vector(),
            &[],
            WRITER,
        )
        .await
        .unwrap(),
    );

    // Assert
    let report = g.check_closure(&parent).await.unwrap();
    assert!(report.is_closed(), "{report:?}");
    assert_eq!((report.real_events, report.synthetic_events), (3, 0));
}

#[test]
fn a_child_hash_ignores_order_and_changes_with_the_level_scope_and_children() {
    // Arrange
    let ids = |names: &[&str]| -> Vec<String> { names.iter().map(|n| (*n).to_string()).collect() };

    // Act
    let base = child_hash(1, "", &ids(&["a", "b"]));

    // Assert
    assert_eq!(base, child_hash(1, "", &ids(&["b", "a"])));
    for other in [
        child_hash(2, "", &ids(&["a", "b"])),
        child_hash(1, "work", &ids(&["a", "b"])),
        child_hash(1, "", &ids(&["a", "c"])),
        child_hash(1, "", &ids(&["ab"])),
    ] {
        assert_ne!(base, other);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_tasks_putting_one_child_set_leave_one_live_row_and_one_unchanged() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("race.db");
    let g = Arc::new(
        DefaultGraph::open_with_clock(path.to_str().unwrap(), GraphConfig::new(DIMS), clock())
            .await
            .unwrap(),
    );
    let barrier = Arc::new(tokio::sync::Barrier::new(2));

    // Act
    let tasks: Vec<_> = (0..2)
        .map(|_| {
            let (g, barrier) = (Arc::clone(&g), Arc::clone(&barrier));
            tokio::spawn(async move {
                barrier.wait().await;
                put(&g, &["a", "b"], &[]).await.unwrap()
            })
        })
        .collect();
    let mut outcomes = Vec::new();
    for task in tasks {
        outcomes.push(task.await.unwrap());
    }

    // Assert
    let inserted = outcomes
        .iter()
        .filter(|o| matches!(o, PutOutcome::Inserted(_)))
        .count();
    assert_eq!(inserted, 1, "{outcomes:?}");
    assert_eq!(g.live_summaries(LEVEL, "").await.unwrap().len(), 1);
}

#[tokio::test]
async fn the_database_refuses_two_live_summaries_of_one_child_set() {
    // Arrange
    let g = open().await;
    put(&g, &["a"], &[]).await.unwrap();
    let hash = child_hash(LEVEL, "", &["a".to_string()]);

    // Act
    let duplicate = g
        .backend
        .execute(
            "INSERT INTO summaries
             (id, level, child_hash, text, events_fed, valid_min, valid_max, model_id,
              prompt_version, tx_from)
             VALUES ('other', 1, ?1, 't', 1, 1, 1, 'm', 'p', 1)",
            &[hash.into()],
        )
        .await;

    // Assert
    assert!(
        matches!(duplicate, Err(Error::Constraint(_))),
        "{duplicate:?}"
    );
}

/// The table a statement writes: the one after `INTO`, or the one `UPDATE` names.
fn table_of(sql: &str) -> String {
    let words: Vec<&str> = sql.split_whitespace().collect();
    let at = match words[0] {
        "UPDATE" => 1,
        _ => words.iter().position(|w| *w == "INTO").unwrap() + 1,
    };
    words[at].to_string()
}

#[tokio::test]
async fn a_write_that_fails_at_any_statement_leaves_no_summary_row_and_keeps_the_old_one() {
    // Arrange: a live summary that the new one replaces
    let g: Graph<FailingBackend> =
        Graph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock())
            .await
            .unwrap();
    let old = id_of(put(&g, &["a", "b"], &[]).await.unwrap());
    let mut baseline = Vec::new();
    for table in ["summary_children", "summary_sources", "summary_embeddings"] {
        baseline.push((table, count(&g, table).await));
    }
    let mut failed_tables = BTreeSet::new();
    let mut succeeded_at = None;

    for statement in 0..64 {
        g.backend.probe.executed.lock().unwrap().clear();
        g.backend.set_fail_on_execute(statement);

        // Act
        let result = put(&g, &["a", "c"], std::slice::from_ref(&old)).await;

        // Assert
        match result {
            Ok(_) => {
                succeeded_at = Some(statement);
                break;
            }
            Err(error) => assert!(injected(&error), "{statement}: {error}"),
        }
        let failing = g.backend.probe.executed.lock().unwrap().last().cloned();
        failed_tables.insert(table_of(&failing.expect("a statement ran")));
        for (table, kept) in &baseline {
            let rows = count(&g, table).await;
            assert_eq!(
                rows, *kept,
                "{statement}: {table} kept rows of the failed write"
            );
        }
        assert_eq!(count(&g, "summaries").await, 1, "{statement}");
        let live = g.live_summaries(LEVEL, "").await.unwrap();
        assert_eq!(live.iter().map(|s| &s.id).collect::<Vec<_>>(), [&old]);
    }

    // Assert: every table the write touches failed once, and the cap held
    assert!(succeeded_at.is_some(), "the write never went through");
    let written: BTreeSet<String> = [
        "summaries",
        "summary_children",
        "summary_sources",
        "summary_embeddings",
    ]
    .map(String::from)
    .into();
    assert_eq!(failed_tables, written);
}

#[tokio::test]
async fn the_inputs_are_the_recorded_nodes_of_one_scope_that_have_a_vector() {
    // Arrange
    let g = open().await;
    let kept = g
        .insert(fact("kept").with_embedding(vector()))
        .await
        .unwrap();
    let mut page = NewNode::entity("person", "Ada").with_embedding(vector());
    page.content = "composed".into();
    g.insert(page).await.unwrap();
    g.insert(
        fact("elsewhere")
            .with_scope("work")
            .with_embedding(vector()),
    )
    .await
    .unwrap();
    g.insert(fact("no vector")).await.unwrap();

    // Act
    let inputs = g.summary_inputs("").await.unwrap();
    let work = g.summary_inputs("work").await.unwrap();

    // Assert
    assert_eq!(inputs.items.len(), 1);
    assert_eq!(inputs.items[0].id, kept);
    assert_eq!(inputs.items[0].content.as_str(), "kept");
    assert_eq!(inputs.items[0].vector, vector());
    assert_eq!(inputs.items[0].origin_events.len(), 1);
    assert_eq!(inputs.skipped_no_vector, 1);
    assert_eq!(work.items.len(), 1);
    assert_eq!(work.items[0].scope.as_deref(), Some("work"));
}

#[tokio::test]
async fn an_input_carries_the_producer_its_node_was_sealed_for_and_none_without_one() {
    // Arrange
    let g = open().await;
    let sealed = g
        .insert(fact("sealed").with_embedding(vector()))
        .await
        .unwrap();
    g.insert(fact("plain").with_embedding(vector()))
        .await
        .unwrap();
    g.backend
        .execute(
            "INSERT INTO node_keys (node_id, key_id, source) VALUES (?1, 'key-1', 'client:mcp')",
            &[sealed.as_str().into()],
        )
        .await
        .unwrap();

    // Act
    let inputs = g.summary_inputs("").await.unwrap();

    // Assert
    let sources = |id: &NodeId| {
        inputs
            .items
            .iter()
            .find(|item| &item.id == id)
            .map(|item| item.sources.clone())
    };
    assert_eq!(sources(&sealed), Some(vec!["client:mcp".to_owned()]));
    let plain_sources: Vec<_> = inputs
        .items
        .iter()
        .filter(|item| item.id != sealed)
        .map(|item| item.sources.clone())
        .collect();
    assert_eq!(plain_sources, [Vec::<String>::new()]);
}

#[tokio::test]
async fn the_scopes_are_those_of_recorded_nodes_and_open_summaries_and_not_of_entity_pages() {
    // Arrange
    let g = open().await;
    g.insert(fact("unscoped")).await.unwrap();
    g.insert(fact("in work").with_scope("work")).await.unwrap();
    g.insert(fact("in work too").with_scope("work"))
        .await
        .unwrap();
    let mut page = NewNode::entity("person", "Ada").with_scope("pages");
    page.content = "composed".into();
    g.insert(page).await.unwrap();

    // Act
    let scopes = g.summary_scopes().await.unwrap();

    // Assert
    assert_eq!(scopes, ["", "work"]);
}

#[tokio::test]
async fn a_scope_with_only_an_open_summary_is_still_listed() {
    // Arrange
    let g = open().await;
    g.put_summary(
        new_summary("retired"),
        &nodes(&["child"]),
        &[event("e")],
        &vector(),
        &[],
        WRITER,
    )
    .await
    .unwrap();

    // Act
    let scopes = g.summary_scopes().await.unwrap();

    // Assert
    assert_eq!(scopes, ["retired"]);
}

#[tokio::test]
async fn a_summary_embedding_survives_the_orphan_sweep() {
    // Arrange
    let g = open().await;
    let id = id_of(put(&g, &["a"], &[]).await.unwrap());

    // Act
    g.backend.vector_sweep_orphans().await.unwrap();

    // Assert
    assert_eq!(
        g.summary_embeddings("", LEVEL).await.unwrap(),
        [(id, vector())]
    );
}

#[tokio::test]
async fn failures_count_up_and_start_again_under_a_new_prompt_or_model() {
    // Arrange
    let g = open().await;
    let hash = "child-set";

    // Act
    let first = g
        .record_summary_failure(hash, "timeout", PROMPT, MODEL)
        .await
        .unwrap();
    let second = g
        .record_summary_failure(hash, "ungrounded", PROMPT, MODEL)
        .await
        .unwrap();
    let new_prompt = g
        .record_summary_failure(hash, "timeout", "prompt-2", MODEL)
        .await
        .unwrap();
    let new_model = g
        .record_summary_failure(hash, "timeout", "prompt-2", "model-b")
        .await
        .unwrap();

    // Assert
    assert_eq!((first, second, new_prompt, new_model), (1, 2, 1, 1));
    let stored = g.summary_failures(hash).await.unwrap().unwrap();
    assert_eq!(
        (
            stored.failures,
            stored.model_id.as_str(),
            stored.last_error.as_str()
        ),
        (1, "model-b", "timeout")
    );
    assert!(g.summary_failures("other").await.unwrap().is_none());
}

#[tokio::test]
async fn a_dissolving_summary_counts_its_runs_until_it_is_forgotten() {
    // Arrange
    let g = open().await;

    // Act
    let first = g.note_dissolving("s1").await.unwrap();
    let second = g.note_dissolving("s1").await.unwrap();
    let other = g.note_dissolving("s2").await.unwrap();
    g.forget_dissolving("s1").await.unwrap();
    let again = g.note_dissolving("s1").await.unwrap();

    // Assert
    assert_eq!((first, second, other, again), (1, 2, 1, 1));
}

#[tokio::test]
async fn tree_state_is_stored_per_scope_and_level_and_overwritten() {
    // Arrange
    let g = open().await;
    let state = |level, built, items| TreeState {
        scope: "work".to_string(),
        level,
        last_full_build_at: Millis(built),
        baseline_item_count: items,
        unattached_count: 0,
    };
    g.put_tree_state(&state(1, 10, 5)).await.unwrap();
    g.put_tree_state(&state(2, 20, 2)).await.unwrap();

    // Act
    g.put_tree_state(&state(1, 30, 7)).await.unwrap();

    // Assert
    assert_eq!(
        g.tree_state("work", 1).await.unwrap(),
        Some(state(1, 30, 7))
    );
    assert_eq!(
        g.tree_state("work", 2).await.unwrap(),
        Some(state(2, 20, 2))
    );
    assert_eq!(g.tree_state("", 1).await.unwrap(), None);
}

async fn put_in(g: &DefaultGraph, scope: &str, level: u8, children: &[&str]) -> String {
    let summary = NewSummary {
        level,
        ..new_summary(scope)
    };
    let kind = if level == 1 {
        ChildKind::Node
    } else {
        ChildKind::Summary
    };
    let children: Vec<SummaryChild> = children
        .iter()
        .map(|id| SummaryChild {
            id: (*id).to_string(),
            kind,
        })
        .collect();
    let outcome = g
        .put_summary(summary, &children, &[event("e1")], &vector(), &[], WRITER)
        .await
        .unwrap();
    id_of(outcome)
}

#[tokio::test]
async fn summaries_within_a_scope_include_its_descendants_and_not_a_lookalike() {
    // Arrange
    let g = open().await;
    let work = put_in(&g, "work", 1, &["a"]).await;
    let child = put_in(&g, "work/liam", 1, &["b"]).await;
    put_in(&g, "work-archive", 1, &["c"]).await;
    let none = put_in(&g, "", 1, &["d"]).await;

    // Act
    let within = g.summaries_within(Some("work"), None, None).await.unwrap();
    let every = g.summaries_within(None, None, None).await.unwrap();

    // Assert
    let ids = |entries: &[crate::types::SummaryEntry]| -> BTreeSet<String> {
        entries.iter().map(|e| e.summary.id.clone()).collect()
    };
    assert_eq!(ids(&within), BTreeSet::from([work, child]));
    assert_eq!(every.len(), 4);
    assert!(ids(&every).contains(&none));
}

#[tokio::test]
async fn summaries_come_back_by_level_then_id_with_closed_ones_left_out() {
    // Arrange
    let g = open().await;
    let low = put_in(&g, "", 1, &["a"]).await;
    let other = put_in(&g, "", 1, &["b"]).await;
    let upper = put_in(&g, "", 2, &[&low, &other]).await;
    let gone = put_in(&g, "", 1, &["c"]).await;
    g.close_summary(&gone, Millis(1500), WRITER).await.unwrap();

    // Act
    let all = g.summaries_within(None, None, None).await.unwrap();
    let level_two = g.summaries_within(None, Some(2), None).await.unwrap();

    // Assert
    let mut level_one = [low, other];
    level_one.sort();
    let [first, second] = level_one;
    let order: Vec<&str> = all.iter().map(|e| e.summary.id.as_str()).collect();
    assert_eq!(order, [first.as_str(), second.as_str(), upper.as_str()]);
    assert_eq!(level_two.len(), 1);
    assert_eq!(level_two[0].summary.id, upper);
    assert_eq!(level_two[0].vector.as_deref(), Some(vector().as_slice()));
}

#[tokio::test]
async fn only_summaries_known_and_finished_by_the_instant_are_returned_for_it() {
    // Arrange
    let clock = clock();
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock.clone())
        .await
        .unwrap();
    let early = put_in(&g, "", 1, &["a"]).await;
    clock.set(Millis(2000));
    let late = NewSummary {
        valid_max: Millis(3000),
        ..new_summary("")
    };
    g.put_summary(late, &nodes(&["b"]), &[event("e1")], &vector(), &[], WRITER)
        .await
        .unwrap();

    // Act
    let before_either = g
        .summaries_within(None, None, Some(Millis(500)))
        .await
        .unwrap();
    let after_the_first = g
        .summaries_within(None, None, Some(Millis(1000)))
        .await
        .unwrap();
    let stored_but_still_running = g
        .summaries_within(None, None, Some(Millis(2500)))
        .await
        .unwrap();

    // Assert
    assert!(before_either.is_empty());
    assert_eq!(after_the_first.len(), 1);
    assert_eq!(after_the_first[0].summary.id, early);
    assert_eq!(stored_but_still_running.len(), 1);
}

#[tokio::test]
async fn a_summary_without_a_vector_of_the_stores_width_has_none() {
    // Arrange
    let g = open().await;
    let wide = put_in(&g, "", 1, &["a"]).await;
    g.backend
        .execute(
            "UPDATE summary_embeddings SET dims = 2, embedding = x'0000803f00000040'
             WHERE summary_id = ?1",
            &[wide.as_str().into()],
        )
        .await
        .unwrap();
    let bare = put_in(&g, "", 1, &["b"]).await;
    g.backend
        .execute(
            "DELETE FROM summary_embeddings WHERE summary_id = ?1",
            &[bare.as_str().into()],
        )
        .await
        .unwrap();

    // Act
    let entries = g.summaries_within(None, None, None).await.unwrap();

    // Assert
    assert_eq!(entries.len(), 2);
    assert!(entries.iter().all(|entry| entry.vector.is_none()));
}

#[tokio::test]
async fn a_summary_is_found_by_id_open_or_closed() {
    // Arrange
    let g = open().await;
    let id = put_in(&g, "", 1, &["a"]).await;
    g.close_summary(&id, Millis(1500), WRITER).await.unwrap();

    // Act
    let found = g.summary(&id).await.unwrap();
    let missing = g.summary("nope").await.unwrap();

    // Assert
    assert_eq!(found.unwrap().tx_to, Millis(1500));
    assert!(missing.is_none());
}
