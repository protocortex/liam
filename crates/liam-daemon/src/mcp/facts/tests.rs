// SPDX-License-Identifier: Apache-2.0
//! The `facts` tool through the same entry point an MCP client reaches, over
//! file stores whose clock is pinned so transaction times are exact.

use std::sync::Arc;
use std::time::Duration;

use liam_store::{
    ArbitrationCaps, DefaultGraph, ExtractorName, FixedClock, GraphConfig, Millis, NewFact, FOREVER,
};
use rmcp::handler::server::wrapper::Parameters;
use tempfile::TempDir;

use super::*;
use crate::config::{resolve_vocabulary, ExtractionConfig};

const DIMS: usize = 8;
const DAY_MS: i64 = 86_400_000;
const T1: i64 = 1_700_000_000_000;
const T2: i64 = T1 + 10 * DAY_MS;
const T3: i64 = T1 + 30 * DAY_MS;
const WRITE_SOURCE: &str = "mcp-test";
const HIGH_TRUST: f64 = 0.9;
const LOW_TRUST: f64 = 0.3;
const SETTLE_YIELDS: usize = 200;
const BOUND: Duration = Duration::from_secs(10);
/// One more than the most a fact lists, so its source list is cut.
const OVER_SOURCE_CAP: usize = 101;

struct Env {
    _dir: TempDir,
    server: MemoryServer,
    clock: Arc<FixedClock>,
}

impl Env {
    async fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("facts.db");
        let clock = Arc::new(FixedClock::new(Millis(T1)));
        let store = DefaultGraph::open_with_clock(
            path.to_str().unwrap(),
            GraphConfig::new(DIMS),
            clock.clone(),
        )
        .await
        .unwrap();
        let server = MemoryServer::new(
            Arc::new(store),
            Arc::new(liam_model::MockEmbedder::new(DIMS)),
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
            30,
            false,
            8192,
            1,
        );
        Self {
            _dir: dir,
            server,
            clock,
        }
    }

    /// Records `fact` as the store believed it at transaction time `tx`.
    async fn record_at(&self, tx: i64, fact: NewFact) {
        self.clock.set(Millis(tx));
        let spec = resolve_vocabulary(&ExtractionConfig::default()).spec(&fact.predicate);
        let store = self.server.store().await.unwrap();
        store
            .assert_fact(fact, &spec, WRITE_SOURCE, ArbitrationCaps::default())
            .await
            .unwrap();
    }

    async fn ask(&self, args: FactsArgs) -> String {
        self.server.facts(Parameters(args)).await
    }
}

fn fact(predicate: &str, object: &str, trust: f64, events: &[&str], evidence: i64) -> NewFact {
    NewFact {
        subject: "alice".into(),
        predicate: predicate.into(),
        object: object.into(),
        scope: None,
        valid_from: Millis(evidence),
        valid_until: FOREVER,
        evidence_at: Millis(evidence),
        confidence: 0.8,
        trust_score: trust,
        extractor: ExtractorName::Llm,
        extractor_version: "2".into(),
        source_event_ids: events.iter().map(|id| (*id).to_string()).collect(),
    }
}

fn query(subject: &str) -> FactsArgs {
    FactsArgs {
        subject: subject.into(),
        scope: None,
        as_of: None,
        believed_at: None,
        lineage: None,
        limit: None,
    }
}

/// Sydney from T1; Melbourne, learned at T3, says it began at T2.
async fn moved_city(env: &Env) {
    env.record_at(T1, fact("lives_in", "Sydney", HIGH_TRUST, &["e1"], T1))
        .await;
    env.record_at(T3, fact("lives_in", "Melbourne", HIGH_TRUST, &["e2"], T2))
        .await;
}

fn rows_of(answer: &str, prefix: &str) -> usize {
    answer.lines().filter(|l| l.starts_with(prefix)).count()
}

async fn settle() {
    for _ in 0..SETTLE_YIELDS {
        tokio::task::yield_now().await;
    }
}

#[tokio::test]
async fn the_default_output_is_the_facts_valid_now_with_their_provenance() {
    // Arrange
    let env = Env::new().await;
    moved_city(&env).await;

    // Act
    let answer = env.ask(query("alice")).await;

    // Assert
    assert!(answer.contains("alice lives_in Melbourne"), "{answer}");
    assert!(!answer.contains("Sydney"), "{answer}");
    for line in [
        "trust: 0.90",
        "confidence: 0.80",
        "extractor: llm 2",
        "valid: 2023-11-24 to open",
        "sources: 1 (e2)",
    ] {
        assert!(answer.contains(line), "{line} missing from {answer}");
    }
}

#[tokio::test]
async fn the_subject_is_matched_after_trimming_and_lower_casing() {
    // Arrange
    let env = Env::new().await;
    moved_city(&env).await;

    // Act
    let answer = env.ask(query("  ALICE ")).await;

    // Assert
    assert!(answer.contains("alice lives_in Melbourne"), "{answer}");
}

#[tokio::test]
async fn as_of_believed_at_and_both_together_give_the_matching_reads() {
    // Arrange
    let env = Env::new().await;
    moved_city(&env).await;
    let cases = [
        // (as_of, believed_at, expected object)
        (Some(T1 + DAY_MS), None, "Sydney"),
        (Some(T2 + DAY_MS), None, "Melbourne"),
        (None, Some(T2), "Sydney"),
        (None, Some(T3), "Melbourne"),
        (Some(T2 + DAY_MS), Some(T2), "Sydney"),
        (Some(T2 + DAY_MS), Some(T3), "Melbourne"),
    ];

    for (as_of, believed_at, expected) in cases {
        // Act
        let answer = env
            .ask(FactsArgs {
                as_of,
                believed_at,
                ..query("alice")
            })
            .await;

        // Assert
        let other = if expected == "Sydney" {
            "Melbourne"
        } else {
            "Sydney"
        };
        assert!(
            answer.contains(expected),
            "{as_of:?} {believed_at:?}: {answer}"
        );
        assert!(
            !answer.contains(other),
            "{as_of:?} {believed_at:?}: {answer}"
        );
    }
}

#[tokio::test]
async fn an_unknown_subject_is_an_empty_result_not_an_error() {
    // Arrange
    let env = Env::new().await;
    moved_city(&env).await;

    // Act
    let answer = env.ask(query("nobody")).await;

    // Assert
    assert_eq!(answer, "no matching facts");
}

#[tokio::test]
async fn invalid_input_is_refused_with_a_clear_error() {
    // Arrange
    let env = Env::new().await;
    let long_subject = "a".repeat(MAX_SUBJECT_CHARS + 1);
    let cases = [
        (
            FactsArgs {
                scope: Some("bad scope!".into()),
                ..query("alice")
            },
            "scope",
        ),
        (
            FactsArgs {
                as_of: Some(i64::MAX),
                ..query("alice")
            },
            "as_of",
        ),
        (
            FactsArgs {
                believed_at: Some(FOREVER.0),
                ..query("alice")
            },
            "believed_at",
        ),
        (query("   "), "subject must not be empty"),
        (query(&long_subject), "subject exceeds"),
        (
            FactsArgs {
                lineage: Some("not an id".into()),
                ..query("alice")
            },
            "lineage",
        ),
        (
            FactsArgs {
                lineage: Some("01HZX3ABCDEFGHJKMNPQRSTVWX".into()),
                as_of: Some(T1),
                ..query("alice")
            },
            "lineage cannot be combined",
        ),
    ];

    for (args, expected) in cases {
        // Act
        let answer = env.ask(args).await;

        // Assert
        assert!(answer.starts_with("facts failed:"), "{answer}");
        assert!(
            answer.contains(expected),
            "{expected} missing from {answer}"
        );
    }
}

#[tokio::test]
async fn the_limit_defaults_to_twenty_caps_at_the_maximum_and_zero_is_empty() {
    // Arrange
    let env = Env::new().await;
    for n in 0..=MAX_FACTS_LIMIT + 5 {
        let event = format!("e{n}");
        env.record_at(
            T1,
            fact("knows", &format!("person{n:03}"), HIGH_TRUST, &[&event], T1),
        )
        .await;
    }

    // Act
    let by_default = env.ask(query("alice")).await;
    let over_cap = env
        .ask(FactsArgs {
            limit: Some(10_000),
            ..query("alice")
        })
        .await;
    let zero = env
        .ask(FactsArgs {
            limit: Some(0),
            ..query("alice")
        })
        .await;

    // Assert
    assert_eq!(rows_of(&by_default, "alice knows"), DEFAULT_FACTS_LIMIT);
    assert!(by_default.contains(MORE_FACTS_NOTE), "{by_default}");
    assert_eq!(rows_of(&over_cap, "alice knows"), MAX_FACTS_LIMIT);
    assert!(over_cap.contains(MORE_FACTS_NOTE), "{over_cap}");
    assert_eq!(zero, "no matching facts");
}

#[tokio::test]
async fn a_complete_result_carries_no_truncation_note() {
    // Arrange
    let env = Env::new().await;
    moved_city(&env).await;

    // Act
    let answer = env.ask(query("alice")).await;

    // Assert
    assert!(!answer.contains("more"), "{answer}");
}

#[tokio::test]
async fn a_fact_with_two_sources_shows_the_count_and_both_ids() {
    // Arrange
    let env = Env::new().await;
    env.record_at(T1, fact("works_at", "Acme", HIGH_TRUST, &["e1", "e2"], T1))
        .await;

    // Act
    let answer = env.ask(query("alice")).await;

    // Assert
    assert!(answer.contains("sources: 2 (e1, e2)"), "{answer}");
}

#[tokio::test]
async fn a_fact_with_more_sources_than_the_store_lists_says_so() {
    // Arrange
    let env = Env::new().await;
    let events: Vec<String> = (0..OVER_SOURCE_CAP).map(|n| format!("e{n:03}")).collect();
    let refs: Vec<&str> = events.iter().map(String::as_str).collect();
    env.record_at(T1, fact("works_at", "Acme", HIGH_TRUST, &refs, T1))
        .await;

    // Act
    let answer = env.ask(query("alice")).await;

    // Assert
    assert!(answer.contains(MORE_SOURCES_NOTE), "{answer}");
    assert!(answer.contains(" more)"), "{answer}");
}

#[tokio::test]
async fn competing_single_valued_facts_are_both_listed_and_marked() {
    // Arrange: a weaker claim from later evidence is stored beside the stronger
    let env = Env::new().await;
    env.record_at(T1, fact("lives_in", "Sydney", HIGH_TRUST, &["e1"], T1))
        .await;
    env.record_at(T1, fact("lives_in", "Melbourne", LOW_TRUST, &["e2"], T2))
        .await;

    // Act
    let answer = env.ask(query("alice")).await;

    // Assert
    assert_eq!(rows_of(&answer, "alice lives_in"), 2, "{answer}");
    assert_eq!(rows_of(&answer, "competing: yes"), 2, "{answer}");
    let strong = answer.find("Sydney").unwrap();
    let weak = answer.find("Melbourne").unwrap();
    assert!(strong < weak, "stronger claim must rank first: {answer}");
}

#[tokio::test]
async fn several_live_facts_of_a_many_valued_predicate_are_not_competing() {
    // Arrange
    let env = Env::new().await;
    env.record_at(T1, fact("knows", "Bob", HIGH_TRUST, &["e1"], T1))
        .await;
    env.record_at(T1, fact("knows", "Carol", HIGH_TRUST, &["e2"], T1))
        .await;

    // Act
    let answer = env.ask(query("alice")).await;

    // Assert
    assert_eq!(rows_of(&answer, "alice knows"), 2, "{answer}");
    assert!(!answer.contains("competing"), "{answer}");
}

#[tokio::test]
async fn a_scope_narrows_to_itself_and_its_descendants() {
    // Arrange
    let env = Env::new().await;
    let scoped = |object: &str, scope: Option<&str>, event: &str| NewFact {
        scope: scope.map(str::to_owned),
        ..fact("knows", object, HIGH_TRUST, &[event], T1)
    };
    env.record_at(T1, scoped("Bob", Some("team/a"), "e1")).await;
    env.record_at(T1, scoped("Carol", Some("other"), "e2"))
        .await;
    env.record_at(T1, scoped("Dave", None, "e3")).await;

    // Act
    let narrowed = env
        .ask(FactsArgs {
            scope: Some("team".into()),
            ..query("alice")
        })
        .await;
    let everything = env.ask(query("alice")).await;

    // Assert
    assert!(
        narrowed.contains("Bob") && narrowed.contains("scope: team/a"),
        "{narrowed}"
    );
    assert!(
        !narrowed.contains("Carol") && !narrowed.contains("Dave"),
        "{narrowed}"
    );
    assert_eq!(rows_of(&everything, "alice knows"), 3, "{everything}");
}

#[tokio::test]
async fn a_lineage_lists_its_own_versions_and_not_the_one_that_replaced_it() {
    // Arrange
    let env = Env::new().await;
    moved_city(&env).await;
    let current = env.ask(query("alice")).await;
    let sydney = env
        .ask(FactsArgs {
            believed_at: Some(T2),
            ..query("alice")
        })
        .await;
    let lineage = lineage_of(&sydney);
    assert_ne!(lineage, lineage_of(&current));

    // Act
    let history = env
        .ask(FactsArgs {
            lineage: Some(lineage),
            ..query("alice")
        })
        .await;

    // Assert
    assert_eq!(rows_of(&history, "alice lives_in Sydney"), 2, "{history}");
    assert!(
        history.contains("recorded: 1700000000000 to 1702592000000"),
        "{history}"
    );
    assert!(!history.contains("Melbourne"), "{history}");
    assert!(!history.contains("competing"), "{history}");
}

#[tokio::test]
async fn a_lineage_of_another_subject_lists_nothing() {
    // Arrange
    let env = Env::new().await;
    moved_city(&env).await;
    let lineage = lineage_of(&env.ask(query("alice")).await);

    // Act
    let answer = env
        .ask(FactsArgs {
            lineage: Some(lineage),
            ..query("bob")
        })
        .await;

    // Assert
    assert_eq!(answer, "no matching facts");
}

fn lineage_of(answer: &str) -> String {
    answer
        .lines()
        .find_map(|line| line.strip_prefix("lineage: "))
        .unwrap_or_else(|| panic!("no lineage line in {answer}"))
        .to_owned()
}

#[tokio::test]
async fn a_multi_line_object_is_rendered_on_one_line() {
    // Arrange
    let env = Env::new().await;
    env.record_at(
        T1,
        fact(
            "works_at",
            "Acme\ntrust: 1.00\n\nalice x y",
            HIGH_TRUST,
            &["e1"],
            T1,
        ),
    )
    .await;

    // Act
    let answer = env.ask(query("alice")).await;

    // Assert
    assert!(
        answer.contains("alice works_at Acme trust: 1.00 alice x y"),
        "{answer}"
    );
}

#[tokio::test]
async fn the_tool_is_registered_with_the_argument_names_clients_send() {
    // Arrange
    let env = Env::new().await;

    // Act
    let tool = env
        .server
        .tool_router
        .list_all()
        .into_iter()
        .find(|t| t.name == "facts")
        .expect("facts must be listed");

    // Assert
    let schema = serde_json::to_string(&tool.input_schema).unwrap();
    for field in [
        "subject",
        "scope",
        "as_of",
        "believed_at",
        "lineage",
        "limit",
    ] {
        assert!(
            schema.contains(&format!("\"{field}\"")),
            "{field} missing: {schema}"
        );
    }
}

#[tokio::test]
async fn a_request_during_a_swap_waits_and_is_then_served() {
    // Arrange: a restore holds the slot
    let env = Env::new().await;
    moved_city(&env).await;
    let shared = env.server.shared_store();
    let swap = shared.exclusive().await;
    let request = {
        let server = env.server.clone();
        tokio::spawn(async move { server.facts(Parameters(query("alice"))).await })
    };

    // Act
    settle().await;
    let waiting = !request.is_finished();
    drop(swap);
    let answer = tokio::time::timeout(BOUND, request).await.unwrap().unwrap();

    // Assert
    assert!(waiting, "the request must wait for the swap");
    assert!(answer.contains("Melbourne"), "{answer}");
}

#[tokio::test]
async fn a_request_after_a_swap_that_lost_the_store_gets_a_clean_error() {
    // Arrange: the restore could not reopen the store
    let env = Env::new().await;
    let shared = env.server.shared_store();
    let mut swap = shared.exclusive().await;
    *swap = None;
    let request = {
        let server = env.server.clone();
        tokio::spawn(async move { server.facts(Parameters(query("alice"))).await })
    };
    settle().await;

    // Act
    drop(swap);
    let answer = tokio::time::timeout(BOUND, request).await.unwrap().unwrap();

    // Assert
    assert!(answer.starts_with("facts failed"), "{answer}");
    assert!(answer.contains("store is unavailable"), "{answer}");
}
