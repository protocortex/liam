// SPDX-License-Identifier: Apache-2.0
//! The exit criteria of the summary tree: each check can fail on a hand-built
//! input, and both hold on the output of every tool over a store a real pass
//! has summarized, written through a real log.

use std::sync::Arc;

use async_trait::async_trait;
use liam_daemon::config::SummaryConfig;
use liam_daemon::consolidation::hooks::{SummaryCtx, SummaryHook};
use liam_daemon::consolidation::maintenance::refresh_clusters;
use liam_model::{Llm, MockEmbedder};
use liam_store::{
    relation, ContentEmbedder, DefaultGraph, EmbedError, EventRole, GraphConfig, Millis, NewNode,
    NodeId,
};
use rmcp::handler::server::wrapper::Parameters;
use tokio::sync::Semaphore;

use super::*;
use crate::extraction::client::ModelIdentity;
use crate::log_startup::test_support::{database_in, log_dir_in, standalone_log, DIMS};
use crate::mcp::summaries::SummariesArgs;
use crate::mcp::{AskArgs, ClustersArgs, MemoryServer, RecallArgs, TimelineArgs};
use crate::summary::hooks::SummaryHooks;
use crate::summary::summariser::LocalSummariser;
use crate::summary::test_support::{MemoryCache, ScriptedLlm};

const FACT_SENTINEL: &str = "fxsentinel7";
const PAGE_SENTINEL: &str = "pgsentinel7";
const SUMMARY_SENTINEL: &str = "smsentinel7";
const SUMMARY_MARKER: &str = "summary/level";
const GROUP_NAMES: [&str; 3] = ["alpha", "beta", "gamma"];
const GROUP_SIZE: usize = 4;
const SOURCE: &str = "consolidation:test";
const ASK_REPLY: &str = "The ledger notes cover quarterly forecasts [1].";
const ASK_TIMEOUT_SECS: u64 = 30;
const CONTEXT_TOKENS: usize = 8192;
/// Any summary is close enough, so the question need not resemble its text.
const MATCH_ANY_SUMMARY: f32 = -1.0;
const LEVEL_ONE_REPLY: &str =
    "These ledger notes cover quarterly forecasts for the team smsentinel7.";
const LEVEL_TWO_REPLY: &str =
    "Together the ledger notes cover quarterly forecasts for the team smsentinel7.";

fn event(id: &str, synthetic: bool) -> EventRef {
    EventRef {
        event_id: id.to_string(),
        synthetic,
    }
}

fn real(id: &str) -> EventRef {
    event(id, false)
}

fn node_child(id: &str) -> SummaryChild {
    SummaryChild {
        id: id.to_string(),
        kind: ChildKind::Node,
    }
}

fn summary_child(id: &str) -> SummaryChild {
    SummaryChild {
        id: id.to_string(),
        kind: ChildKind::Summary,
    }
}

fn row(id: &str, children: Vec<SummaryChild>, sources: Vec<EventRef>) -> SummaryRow {
    SummaryRow {
        id: id.to_string(),
        children,
        sources,
    }
}

fn log_of(ids: &[&str]) -> BTreeSet<String> {
    ids.iter().map(|id| id.to_string()).collect()
}

/// Two nodes with one real event each, a summary of both, and one above it.
fn sound_snapshot() -> StoreSnapshot {
    StoreSnapshot {
        summaries: vec![
            row(
                "s1",
                vec![node_child("n1"), node_child("n2")],
                vec![real("e1"), real("e2")],
            ),
            row(
                "s2",
                vec![summary_child("s1")],
                vec![real("e1"), real("e2")],
            ),
        ],
        node_events: HashMap::from([
            ("n1".to_string(), vec![real("e1")]),
            ("n2".to_string(), vec![real("e2")]),
        ]),
    }
}

#[test]
fn a_sound_tree_passes_the_trace() {
    // Arrange
    let snapshot = sound_snapshot();

    // Act
    let result = check_trace(&snapshot, &log_of(&["e1", "e2"]));

    // Assert
    assert_eq!(result, Ok(()));
}

#[test]
fn a_store_of_synthetic_events_only_reports_synthetic_only() {
    // Arrange
    let snapshot = StoreSnapshot {
        summaries: vec![row("s1", vec![node_child("n1")], vec![event("e1", true)])],
        node_events: HashMap::from([("n1".to_string(), vec![event("e1", true)])]),
    };

    // Act
    let result = check_trace(&snapshot, &log_of(&["e1"]));

    // Assert
    let error = result.unwrap_err();
    assert_eq!(error, TraceError::SyntheticOnly);
    assert!(error.to_string().contains("synthetic only"), "{error}");
}

#[test]
fn a_closure_of_synthetic_events_fails_in_a_store_with_real_events_elsewhere() {
    // Arrange
    let snapshot = StoreSnapshot {
        summaries: vec![
            row("s1", vec![node_child("n1")], vec![event("e1", true)]),
            row("s2", vec![node_child("n2")], vec![real("e2")]),
        ],
        node_events: HashMap::from([
            ("n1".to_string(), vec![event("e1", true)]),
            ("n2".to_string(), vec![real("e2")]),
        ]),
    };

    // Act
    let result = check_trace(&snapshot, &log_of(&["e1", "e2"]));

    // Assert
    assert_eq!(
        result,
        Err(TraceError::SyntheticClosure {
            summary: "s1".to_string()
        })
    );
}

#[test]
fn a_child_event_missing_from_the_parent_closure_fails() {
    // Arrange
    let mut snapshot = sound_snapshot();
    snapshot.summaries[0].sources = vec![real("e1")];

    // Act
    let result = check_trace(&snapshot, &log_of(&["e1", "e2"]));

    // Assert
    assert_eq!(
        result,
        Err(TraceError::MissingFromParent {
            summary: "s1".to_string(),
            event: "e2".to_string()
        })
    );
}

#[test]
fn an_event_no_child_carries_fails() {
    // Arrange
    let mut snapshot = sound_snapshot();
    snapshot.summaries[0].sources.push(real("e9"));

    // Act
    let result = check_trace(&snapshot, &log_of(&["e1", "e2", "e9"]));

    // Assert
    assert_eq!(
        result,
        Err(TraceError::NotFromChildren {
            summary: "s1".to_string(),
            event: "e9".to_string()
        })
    );
}

#[test]
fn an_event_the_log_does_not_hold_fails() {
    // Arrange
    let snapshot = sound_snapshot();

    // Act
    let result = check_trace(&snapshot, &log_of(&["e1"]));

    // Assert
    assert_eq!(
        result,
        Err(TraceError::UnknownEvent {
            summary: "s1".to_string(),
            event: "e2".to_string()
        })
    );
}

#[test]
fn a_summary_with_no_sources_or_an_unknown_child_fails() {
    // Arrange
    let mut bare = sound_snapshot();
    bare.summaries[0].sources.clear();
    let mut orphaned = sound_snapshot();
    orphaned.summaries[0].children.push(node_child("gone"));
    let log = log_of(&["e1", "e2"]);

    // Act
    let (bare, orphaned) = (check_trace(&bare, &log), check_trace(&orphaned, &log));

    // Assert
    assert_eq!(
        bare,
        Err(TraceError::NoSources {
            summary: "s1".to_string()
        })
    );
    assert_eq!(
        orphaned,
        Err(TraceError::UnknownChild {
            summary: "s1".to_string(),
            child: "gone".to_string()
        })
    );
}

fn truth() -> Vec<Entry> {
    vec![
        Entry {
            tokens: vec!["recorded-text".to_string()],
            ids: vec!["FACTHANDLE".to_string()],
            derivation: Derivation::Recorded,
        },
        Entry {
            tokens: vec!["inferred-text".to_string()],
            ids: vec!["PAGEHANDLE".to_string()],
            derivation: Derivation::Inferred,
        },
    ]
}

const LABELLED_PAGE: &str =
    "[person PAGEHANDLE] inferred-text\ninferred-text body\nprovenance: derivation: inferred; basis: unknown";

#[test]
fn a_page_in_strict_output_fails_by_its_text_or_its_handle() {
    // Arrange
    let by_text = "[1] person/x\ninferred-text\nprovenance: derivation: inferred; basis: unknown";
    let by_handle = "[person PAGEHANDLE]";

    // Act
    let (text, handle) = (
        check_separation(by_text, Mode::Strict, &truth(), &[]),
        check_separation(by_handle, Mode::Strict, &truth(), &[]),
    );

    // Assert
    assert_eq!(
        text,
        Err(SeparationError::InferredLeaked {
            found: "inferred-text".to_string()
        })
    );
    assert_eq!(
        handle,
        Err(SeparationError::InferredLeaked {
            found: "PAGEHANDLE".to_string()
        })
    );
}

#[test]
fn an_inferred_item_shown_with_the_recorded_label_fails_in_infer_output() {
    // Arrange
    let mutated = LABELLED_PAGE.replace("derivation: inferred", "derivation: recorded");

    // Act
    let result = check_separation(&mutated, Mode::Infer, &truth(), &[]);

    // Assert
    assert_eq!(
        result,
        Err(SeparationError::WrongLabel {
            token: "inferred-text".to_string(),
            expected: "inferred",
            found: Some("recorded".to_string())
        })
    );
}

#[test]
fn an_inferred_item_with_no_label_fails_but_a_recorded_head_alone_passes() {
    // Arrange
    let unlabelled = "[1] person/inferred-text\n[2] fact/recorded-text";
    let heads_only = "[fact FACTHANDLE] recorded-text\n[fact OTHER] recorded-text";

    // Act
    let (unlabelled, heads_only) = (
        check_separation(unlabelled, Mode::Infer, &truth(), &[]),
        check_separation(heads_only, Mode::Strict, &truth(), &[]),
    );

    // Assert
    assert_eq!(
        unlabelled,
        Err(SeparationError::WrongLabel {
            token: "inferred-text".to_string(),
            expected: "inferred",
            found: None
        })
    );
    assert_eq!(heads_only, Ok(()));
}

#[test]
fn the_exempted_page_may_appear_in_strict_output_only_labelled_inferred() {
    // Arrange
    let exempt = ["PAGEHANDLE".to_string()];
    let unlabelled = LABELLED_PAGE.replace("derivation: inferred", "derivation: recorded");

    // Act
    let labelled = check_separation(LABELLED_PAGE, Mode::Strict, &truth(), &exempt);
    let mutated = check_separation(&unlabelled, Mode::Strict, &truth(), &exempt);

    // Assert
    assert_eq!(labelled, Ok(()));
    assert_eq!(
        mutated,
        Err(SeparationError::ExemptUnlabelled {
            id: "PAGEHANDLE".to_string()
        })
    );
}

#[test]
fn a_label_belongs_to_its_own_item_and_not_to_the_next_one() {
    // Arrange
    let output = "[1] fact/recorded-text\n[2] page/inferred-text\nprovenance: derivation: inferred";

    // Act
    let result = check_separation(output, Mode::Infer, &truth(), &[]);

    // Assert
    assert_eq!(result, Ok(()));
}

/// Answers every prompt with the same text.
struct FixedLlm;

#[async_trait]
impl Llm for FixedLlm {
    async fn complete(&self, _system: &str, _prompt: &str) -> liam_model::Result<String> {
        Ok(ASK_REPLY.to_string())
    }
}

struct FirstAxisEmbedder;

#[async_trait]
impl ContentEmbedder for FirstAxisEmbedder {
    async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(axis(0))
    }
}

fn axis(index: usize) -> Vec<f32> {
    let mut vector = vec![0.0; DIMS];
    vector[index] = 1.0;
    vector
}

/// A store written through a log, with notes, a page that mentions one of them,
/// and the summaries one real pass wrote over the notes.
struct World {
    _dir: tempfile::TempDir,
    server: MemoryServer,
    page: NodeId,
    fact: NodeId,
    notes: Vec<NodeId>,
}

impl World {
    async fn build() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let log = standalone_log(&log_dir_in(dir.path()));
        let store = DefaultGraph::open(
            database_in(dir.path()).to_str().unwrap(),
            GraphConfig::new(DIMS),
        )
        .await
        .unwrap()
        .with_log(log)
        .await
        .unwrap();
        let mut notes = Vec::new();
        for (group, name) in GROUP_NAMES.iter().enumerate() {
            for n in 0..GROUP_SIZE {
                let marked = group == 0 && n == 0;
                let sentinel = if marked { FACT_SENTINEL } else { "" };
                let node = NewNode::now(
                    "fact",
                    format!("Note {n} {sentinel}").trim_end().to_string(),
                    format!("ledger notes number {n} cover quarterly forecasts for the {name} team {sentinel}")
                    .trim_end()
                    .to_string(),
                )
                .with_embedding(axis(group))
                .with_valid_from(Millis(1000 + (group * 100 + n) as i64));
                notes.push(store.insert(node).await.unwrap());
            }
        }
        let fact = notes[0].clone();
        let mut page = NewNode::entity("person", PAGE_SENTINEL);
        page.content = format!("{PAGE_SENTINEL} profile: ledger notes cover quarterly forecasts");
        let page = store.upsert_by(page).await.unwrap();
        store
            .relate(&page, &fact, relation::MENTIONS)
            .await
            .unwrap();
        summarize(&store).await;
        refresh_clusters(&store).await;
        let server = MemoryServer::new(
            Arc::new(store),
            Arc::new(MockEmbedder::new(DIMS)),
            Arc::new(liam_model::IdentityReranker),
            Arc::new(FixedLlm),
            ASK_TIMEOUT_SECS,
            false,
            CONTEXT_TOKENS,
            1,
        )
        .with_min_summary_similarity(MATCH_ANY_SUMMARY);
        Self {
            _dir: dir,
            server,
            page,
            fact,
            notes,
        }
    }

    async fn store(&self) -> impl std::ops::Deref<Target = DefaultGraph> + '_ {
        self.server.store().await.unwrap()
    }

    /// The live summaries of every level, and the events of the notes under them.
    async fn snapshot(&self) -> StoreSnapshot {
        let store = self.store().await;
        let mut summaries = Vec::new();
        for level in 1..=crate::config::MAX_SUMMARY_DEPTH {
            for summary in store.live_summaries(level, "").await.unwrap() {
                summaries.push(SummaryRow {
                    children: store.summary_children(&summary.id).await.unwrap(),
                    sources: store.summary_sources(&summary.id).await.unwrap(),
                    id: summary.id,
                });
            }
        }
        let mut node_events = HashMap::new();
        for note in &self.notes {
            let events = store.node_event_ids(note, EventRole::Origin).await.unwrap();
            node_events.insert(note.as_str().to_string(), events);
        }
        StoreSnapshot {
            summaries,
            node_events,
        }
    }

    /// The ids the notes' writes minted in the log.
    async fn minted(&self) -> BTreeSet<String> {
        self.snapshot()
            .await
            .node_events
            .values()
            .flatten()
            .map(|event| event.event_id.clone())
            .collect()
    }

    async fn truth(&self) -> Vec<Entry> {
        let summary_ids = self
            .snapshot()
            .await
            .summaries
            .into_iter()
            .map(|summary| summary.id)
            .collect();
        vec![
            Entry {
                tokens: vec![FACT_SENTINEL.to_string()],
                ids: vec![self.fact.handle().to_string()],
                derivation: Derivation::Recorded,
            },
            Entry {
                tokens: vec![PAGE_SENTINEL.to_string()],
                ids: vec![self.page.handle().to_string()],
                derivation: Derivation::Inferred,
            },
            Entry {
                tokens: vec![SUMMARY_SENTINEL.to_string(), SUMMARY_MARKER.to_string()],
                ids: summary_ids,
                derivation: Derivation::Inferred,
            },
        ]
    }

    async fn ask(&self, mode: Mode) -> String {
        let question = format!("{FACT_SENTINEL} {PAGE_SENTINEL}");
        self.server
            .ask(Parameters(AskArgs {
                question,
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: Some(mode),
            }))
            .await
    }
}

/// Runs one real summarize pass: three groups of notes, then the level above.
async fn summarize(store: &DefaultGraph) {
    let hooks = SummaryHooks {
        config: SummaryConfig {
            enabled: true,
            ..SummaryConfig::default()
        },
        summariser: Arc::new(LocalSummariser::new(
            Arc::new(ScriptedLlm::replying(&[
                LEVEL_ONE_REPLY,
                LEVEL_ONE_REPLY,
                LEVEL_ONE_REPLY,
                LEVEL_TWO_REPLY,
            ])),
            ModelIdentity {
                model_id: "model".to_string(),
                model_sha256: String::new(),
            },
        )),
        gate: None,
        permits: Arc::new(Semaphore::new(1)),
        cache: Arc::new(MemoryCache::default()),
        embedder: Arc::new(FirstAxisEmbedder),
        context_tokens: CONTEXT_TOKENS,
    };
    let report = hooks
        .summarize(&SummaryCtx {
            store,
            source: SOURCE.to_string(),
        })
        .await
        .unwrap();
    assert_eq!(report.written, 4, "the fixture needs its summaries");
}

#[tokio::test]
async fn every_summary_of_a_pipeline_run_traces_to_real_events() {
    // Arrange
    let world = World::build().await;

    // Act
    let snapshot = world.snapshot().await;
    let result = check_trace(&snapshot, &world.minted().await);

    // Assert
    assert_eq!(result, Ok(()));
    assert_eq!(snapshot.summaries.len(), 4);
    assert!(snapshot
        .summaries
        .iter()
        .flat_map(|summary| &summary.sources)
        .all(|event| !event.synthetic));
}

#[tokio::test]
async fn strict_output_of_every_tool_holds_no_inferred_content_and_shows_the_recorded() {
    // Arrange
    let world = World::build().await;
    let truth = world.truth().await;
    let exempt = [world.page.handle().to_string()];
    let recall = world
        .server
        .recall(Parameters(RecallArgs {
            query: format!("{FACT_SENTINEL} {PAGE_SENTINEL} ledger notes"),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: Some(Mode::Strict),
        }))
        .await;
    let timeline = world
        .server
        .timeline(Parameters(TimelineArgs {
            entity: world.page.as_str().to_string(),
            limit: None,
            mode: Some(Mode::Strict),
        }))
        .await;
    let clusters = world
        .server
        .clusters(Parameters(ClustersArgs {
            k: None,
            members: None,
            mode: Some(Mode::Strict),
        }))
        .await;
    let ask = world.ask(Mode::Strict).await;

    // Act
    let outputs = [
        ("recall", &recall),
        ("timeline", &timeline),
        ("clusters", &clusters),
        ("ask", &ask),
    ];
    let results: Vec<_> = outputs
        .iter()
        .map(|(tool, out)| (*tool, check_separation(out, Mode::Strict, &truth, &exempt)))
        .collect();

    // Assert
    for (tool, result) in results {
        assert_eq!(result, Ok(()), "{tool}");
    }
    for (tool, out) in [("recall", &recall), ("timeline", &timeline), ("ask", &ask)] {
        assert!(out.contains(FACT_SENTINEL), "{tool} lost the fact: {out}");
    }
    assert!(clusters.contains("[fact "), "{clusters}");
    assert!(timeline.contains(world.page.handle()), "{timeline}");
    assert!(!recall.contains(world.page.handle()), "{recall}");
    assert!(!ask.contains(SUMMARY_MARKER), "{ask}");
}

#[tokio::test]
async fn infer_output_labels_the_page_and_the_summaries_inferred() {
    // Arrange
    let world = World::build().await;
    let truth = world.truth().await;

    // Act
    let ask = world.ask(Mode::Infer).await;
    let listing = world
        .server
        .summaries(Parameters(SummariesArgs {
            query: None,
            level: None,
            scope: None,
            id: None,
            limit: None,
        }))
        .await;

    // Assert
    for (tool, out) in [("ask", &ask), ("summaries", &listing)] {
        assert_eq!(
            check_separation(out, Mode::Infer, &truth, &[]),
            Ok(()),
            "{tool}: {out}"
        );
    }
    assert!(ask.starts_with("answer (inferred):"), "{ask}");
    assert!(ask.contains(PAGE_SENTINEL), "{ask}");
    assert!(ask.contains(SUMMARY_MARKER), "{ask}");
    assert!(ask.contains(FACT_SENTINEL), "{ask}");
    assert!(listing.contains(SUMMARY_SENTINEL), "{listing}");
}
