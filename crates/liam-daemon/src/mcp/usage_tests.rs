// SPDX-License-Identifier: Apache-2.0
//! What `recall` and `ask` count as use of a memory.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use liam_model::{Llm, MockEmbedder};
use liam_store::{DefaultGraph, GraphConfig, NewEdge, NewNode, NodeId, Query, HANDLE_LEN};
use rmcp::handler::server::wrapper::Parameters;

use super::{AskArgs, MemoryServer, Mode, RecallArgs};
use crate::tiering::usage::UsageRecorder;

const DIMS: usize = 8;
const FENCE_OPEN: &str = "<<<EVIDENCE";
const TRACKED: usize = 100;
const WIDE_POOL_MATCHES: usize = 20;
const CUT_K: usize = 3;

/// Answers the sufficiency pre-pass with `verdict` and the answer prompt with
/// `answer`, and keeps the prompts it saw.
struct ScriptedLlm {
    verdict: &'static str,
    answer: Option<&'static str>,
    prompts: Mutex<Vec<String>>,
}

impl ScriptedLlm {
    fn new(verdict: &'static str, answer: Option<&'static str>) -> Arc<Self> {
        Arc::new(Self {
            verdict,
            answer,
            prompts: Mutex::new(Vec::new()),
        })
    }

    fn answer_prompt(&self) -> String {
        self.prompts
            .lock()
            .unwrap()
            .iter()
            .find(|prompt| !prompt.contains("Reply YES or NO"))
            .cloned()
            .expect("the answer prompt was built")
    }
}

#[async_trait::async_trait]
impl Llm for ScriptedLlm {
    async fn complete(&self, _system: &str, prompt: &str) -> liam_model::Result<String> {
        self.prompts.lock().unwrap().push(prompt.to_string());
        if prompt.contains("Reply YES or NO") {
            return Ok(self.verdict.to_string());
        }
        self.answer
            .map(str::to_string)
            .ok_or_else(|| liam_model::ModelError::Llm("the model failed".into()))
    }
}

struct Fixture {
    server: MemoryServer,
    store: Arc<DefaultGraph>,
    usage: Arc<UsageRecorder>,
}

async fn fixture(llm: Arc<dyn Llm>, sufficiency: bool, context_tokens: usize) -> Fixture {
    let store = Arc::new(
        DefaultGraph::open(":memory:", GraphConfig::new(DIMS))
            .await
            .unwrap(),
    );
    let usage = Arc::new(UsageRecorder::new(TRACKED));
    let server = MemoryServer::new(
        store.clone(),
        Arc::new(MockEmbedder::new(DIMS)),
        Arc::new(liam_model::IdentityReranker),
        llm,
        1,
        sufficiency,
        context_tokens,
        1,
    )
    .with_usage_recorder(usage.clone());
    Fixture {
        server,
        store,
        usage,
    }
}

async fn plain() -> Fixture {
    fixture(ScriptedLlm::new("YES", Some("the answer")), false, 8192).await
}

async fn note(store: &DefaultGraph, label: &str, content: &str) -> NodeId {
    store
        .insert(NewNode::now("fact", label, content))
        .await
        .unwrap()
}

fn recall_args(query: &str, k: Option<usize>) -> RecallArgs {
    RecallArgs {
        query: query.to_string(),
        kind: None,
        scope: None,
        k,
        as_of: None,
        mode: None,
    }
}

fn ask_args(question: &str, k: Option<usize>, mode: Option<Mode>) -> AskArgs {
    AskArgs {
        question: question.to_string(),
        kind: None,
        scope: None,
        k,
        as_of: None,
        mode,
    }
}

fn counted(fixture: &Fixture) -> Vec<String> {
    fixture
        .usage
        .hits()
        .into_iter()
        .map(|(node, _)| node.as_str().to_string())
        .collect()
}

#[tokio::test]
async fn recall_counts_the_ranked_results_it_returns_and_no_more() {
    // Arrange: ten matches, three asked for
    let fixture = plain().await;
    for index in 0..10 {
        note(
            &fixture.store,
            &format!("note {index}"),
            "the zorbnax gadget",
        )
        .await;
    }

    // Act
    let out = fixture
        .server
        .recall(Parameters(recall_args("zorbnax gadget", Some(3))))
        .await;

    // Assert
    assert_eq!(out.matches("[fact ").count(), 3, "{out}");
    let hits = fixture.usage.hits();
    assert_eq!(hits.len(), 3);
    assert!(hits.iter().all(|(_, hits)| *hits == 1));
}

#[tokio::test]
async fn recall_counts_only_the_cut_hits_when_it_reranks_a_wider_pool() {
    // Arrange: more matches than the rerank pool holds, three asked for
    let fixture = plain().await;
    for index in 0..WIDE_POOL_MATCHES {
        note(
            &fixture.store,
            &format!("note {index}"),
            "the zorbnax gadget",
        )
        .await;
    }

    // Act
    let out = fixture
        .server
        .recall(Parameters(recall_args("zorbnax gadget", Some(CUT_K))))
        .await;

    // Assert: the counted ids are exactly the returned ones
    let counted = counted(&fixture);
    assert_eq!(counted.len(), CUT_K);
    assert_eq!(out.matches("[fact ").count(), CUT_K, "{out}");
    for id in &counted {
        let handle = &id[..HANDLE_LEN];
        assert!(out.contains(handle), "{handle} was counted but not shown");
    }
}

#[tokio::test]
async fn recall_does_not_count_a_neighbour_pulled_in_by_an_edge() {
    // Arrange
    let fixture = plain().await;
    let seed = note(&fixture.store, "seed", "quartzmarker").await;
    let neighbour = note(&fixture.store, "neighbour", "unrelated words").await;
    fixture
        .store
        .link(NewEdge::new(&seed, &neighbour, "relates_to"))
        .await
        .unwrap();

    // Act
    let out = fixture
        .server
        .recall(Parameters(recall_args("quartzmarker", Some(2))))
        .await;

    // Assert: the neighbour is shown and not counted
    assert!(out.contains("unrelated words"), "{out}");
    assert_eq!(counted(&fixture), [seed.as_str()]);
}

#[tokio::test]
async fn a_recall_with_no_match_counts_nothing() {
    // Arrange
    let fixture = plain().await;

    // Act
    let out = fixture
        .server
        .recall(Parameters(recall_args("nothing stored", None)))
        .await;

    // Assert
    assert_eq!(out, "no relevant memory");
    assert!(counted(&fixture).is_empty());
}

#[tokio::test]
async fn a_query_made_straight_on_the_store_counts_nothing() {
    // Arrange
    let fixture = plain().await;
    note(&fixture.store, "note", "the zorbnax gadget").await;

    // Act
    let hits = fixture
        .store
        .query_explained(&Query::text("zorbnax"))
        .await
        .unwrap();

    // Assert
    assert_eq!(hits.len(), 1);
    assert!(counted(&fixture).is_empty());
}

#[tokio::test]
async fn ask_counts_the_evidence_that_reached_the_model_after_trimming() {
    // Arrange: five items and a budget that holds fewer
    let llm = ScriptedLlm::new("YES", Some("the answer"));
    let fixture = fixture(llm.clone(), false, 1000).await;
    for index in 1..=5 {
        note(
            &fixture.store,
            &format!("Detail {index}"),
            &format!(
                "The gizmoflux widget detail number {index} explains a long history of \
                 engineering decisions and requirements gathered over many quarters of \
                 careful planning and review that led to this specific numbered outcome."
            ),
        )
        .await;
    }

    // Act
    fixture
        .server
        .ask(Parameters(ask_args(
            "What does the gizmoflux widget explain?",
            Some(5),
            None,
        )))
        .await;

    // Assert
    let in_prompt = llm.answer_prompt().matches(FENCE_OPEN).count();
    assert!(in_prompt < 5, "the budget should have trimmed the evidence");
    assert_eq!(fixture.usage.hits().len(), in_prompt);
}

#[tokio::test]
async fn ask_counts_nothing_when_the_sufficiency_check_refuses() {
    // Arrange
    let fixture = fixture(ScriptedLlm::new("NO", Some("the answer")), true, 8192).await;
    note(&fixture.store, "note", "the zorbnax gadget uses libSQL").await;

    // Act
    fixture
        .server
        .ask(Parameters(ask_args("zorbnax gadget storage", None, None)))
        .await;

    // Assert
    assert!(counted(&fixture).is_empty());
}

#[tokio::test]
async fn ask_counts_nothing_when_there_is_no_relevant_memory() {
    // Arrange
    let fixture = plain().await;

    // Act
    let out = fixture
        .server
        .ask(Parameters(ask_args("nothing stored", None, None)))
        .await;

    // Assert
    assert_eq!(out, "no relevant memory");
    assert!(counted(&fixture).is_empty());
}

#[tokio::test]
async fn ask_counts_the_evidence_when_the_model_fails_after_the_sufficiency_pass() {
    // Arrange
    let fixture = fixture(ScriptedLlm::new("YES", None), true, 8192).await;
    let id = note(&fixture.store, "note", "the zorbnax gadget uses libSQL").await;

    // Act
    fixture
        .server
        .ask(Parameters(ask_args("zorbnax gadget storage", None, None)))
        .await;

    // Assert
    assert_eq!(counted(&fixture), [id.as_str()]);
}

#[tokio::test]
async fn strict_ask_counts_the_evidence_it_returns() {
    // Arrange
    let fixture = plain().await;
    let id = note(&fixture.store, "note", "the zorbnax gadget uses libSQL").await;

    // Act
    fixture
        .server
        .ask(Parameters(ask_args(
            "zorbnax gadget storage",
            None,
            Some(Mode::Strict),
        )))
        .await;

    // Assert
    assert_eq!(counted(&fixture), [id.as_str()]);
}

#[tokio::test]
async fn ask_counts_nothing_when_no_generation_permit_is_granted_in_time() {
    // Arrange: the only permit is held, so the request gives up after its
    // one second deadline and returns the evidence without the model seeing it
    let fixture = plain().await;
    note(&fixture.store, "note", "the zorbnax gadget uses libSQL").await;
    let _held = fixture
        .server
        .generation_permits_handle()
        .acquire_owned()
        .await
        .unwrap();

    // Act
    let out = tokio::time::timeout(
        Duration::from_secs(10),
        fixture
            .server
            .ask(Parameters(ask_args("zorbnax gadget storage", None, None))),
    )
    .await
    .expect("the request gives up at its own deadline");

    // Assert
    assert!(out.contains("zorbnax"), "{out}");
    assert!(counted(&fixture).is_empty());
}
