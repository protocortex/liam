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
    let mut children = nodes(&["b"]);
    children.push(SummaryChild {
        id: "a".to_string(),
        kind: ChildKind::Summary,
    });
    let outcome = g
        .put_summary(new_summary(""), &children, &[event("e1")], &vector(), &[])
        .await
        .unwrap();

    // Act
    let listed = g.summary_children(&id_of(outcome)).await.unwrap();

    // Assert
    assert_eq!(
        listed,
        [
            SummaryChild {
                id: "a".to_string(),
                kind: ChildKind::Summary
            },
            SummaryChild {
                id: "b".to_string(),
                kind: ChildKind::Node
            },
        ]
    );
    assert!(ChildKind::parse("edge").is_err());
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
        .put_summary(new_summary(""), &nodes(&["a"]), &[], &vector(), &[])
        .await;
    let no_children = g
        .put_summary(new_summary(""), &[], &[event("e1")], &vector(), &[])
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
    g.close_summary(&id, Millis(2000)).await.unwrap();

    // Act
    let again = g.close_summary(&id, Millis(3000)).await;
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
        g.put_summary(new_summary(""), &children, &covered_events, &vector(), &[])
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
