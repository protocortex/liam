// SPDX-License-Identifier: Apache-2.0
//! Does an entity page in `recall` help when an entity has many mentions?
//!
//! Two arms run over the same stores, questions and embedder, through the
//! same `recall` the daemon serves: A has no page, C has the page the daemon
//! builds. The sets are hand written, in English, Portuguese and Spanish: per
//! language 8 entities with 20 or 30 memories over many dates, with updates
//! that supersede an earlier value and with near-duplicate memories, and 40
//! questions (single fact, aggregate over several memories, latest state).
//! They are synthetic and small, and say nothing about real traffic.
//!
//! The default test runs a small part of the set with the mock embedder. The
//! ignored test runs all of it with arctic and applies the rule that a page may
//! not lose to no page by more than `TOLERANCE` in any language, and must win on
//! the latest-state questions.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use liam_daemon::retrieval_eval::ndcg_at_10;
use liam_model::{Embedder, MockEmbedder, MockLlm, TextKind};
use liam_store::{relation, DefaultGraph, FixedClock, GraphConfig, Millis, NewNode, NodeId, Query};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::mcp::{MemoryServer, RecallArgs};

const LANGUAGES: [(&str, &str); 3] = [
    ("en", include_str!("data/en.json")),
    ("pt", include_str!("data/pt.json")),
    ("es", include_str!("data/es.json")),
];

/// Moment the questions are asked: after the last memory of every set.
const ASK_DATE: &str = "2026-11-19";

/// Results a question reads, and the `k` of the recall that serves it.
const RECALL_K: usize = 10;

/// Results an agent is assumed to read.
const READ_TOP: usize = 3;

/// Most a page may lose to no page, in nDCG@10, in any language.
const TOLERANCE: f64 = 0.01;

const MOCK_DIMS: usize = 8;
const BUILD_DEADLINE: Duration = Duration::from_secs(60);
const LATEST: &str = "latest";

#[derive(Deserialize)]
struct DataSet {
    entities: Vec<DataEntity>,
    questions: Vec<DataQuestion>,
}

#[derive(Deserialize)]
struct DataEntity {
    id: String,
    kind: String,
    name: String,
    memories: Vec<DataMemory>,
}

#[derive(Deserialize)]
struct DataMemory {
    id: String,
    date: String,
    text: String,
}

#[derive(Deserialize)]
struct DataQuestion {
    kind: String,
    entity: String,
    text: String,
    gold: Vec<String>,
    /// Every term must appear for a text to answer the question; `a|b`
    /// accepts either spelling.
    answer: Vec<String>,
    /// Terms of the superseded values, for the latest-state questions.
    #[serde(default)]
    stale: Vec<String>,
}

fn load(json: &str) -> DataSet {
    serde_json::from_str(json).expect("the data file parses")
}

/// Epoch milliseconds of a `YYYY-MM-DD` date at 00:00 UTC.
fn date_ms(date: &str) -> i64 {
    let mut parts = date.split('-');
    let mut next = || parts.next().expect("a date part");
    let year: i32 = next().parse().expect("a year");
    let month: u8 = next().parse().expect("a month");
    let day: u8 = next().parse().expect("a day");
    let date = time::Date::from_calendar_date(year, time::Month::try_from(month).unwrap(), day)
        .expect("a calendar date");
    date.midnight().assume_utc().unix_timestamp() * 1000
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric()
}

/// Whether `haystack` holds `term` (or any `a|b` alternative of it) as a whole
/// word run, ignoring case.
fn has_term(haystack: &str, term: &str) -> bool {
    let haystack = haystack.to_lowercase();
    term.split('|').any(|alternative| {
        let needle = alternative.to_lowercase();
        haystack.match_indices(&needle).any(|(start, _)| {
            let before = haystack[..start].chars().next_back();
            let after = haystack[start + needle.len()..].chars().next();
            !before.is_some_and(is_word_char) && !after.is_some_and(is_word_char)
        })
    })
}

fn has_all(haystack: &str, terms: &[String]) -> bool {
    terms.iter().all(|term| has_term(haystack, term))
}

fn has_any(haystack: &str, terms: &[String]) -> bool {
    terms.iter().any(|term| has_term(haystack, term))
}

/// One result of a recall: its node, and its text without the head and the
/// lines that describe it.
struct Shown {
    id: NodeId,
    text: String,
}

/// What one question scored in one arm.
struct Outcome {
    kind: String,
    ndcg: f64,
    /// Latest-state only: the newest value is in the first results and no
    /// result holding only a stale value ranks above it.
    latest_ok: Option<bool>,
}

struct Arm {
    store: Arc<DefaultGraph>,
    server: MemoryServer,
    memory_ids: HashMap<String, NodeId>,
    page_ids: HashMap<String, NodeId>,
}

/// Opens a store holding the set's entities and memories, each memory
/// backdated to its date, stored under its vector and linked from its entity.
/// With `pages`, builds the page of every entity the way the daemon does.
async fn seed(
    set: &DataSet,
    embedder: Arc<dyn Embedder>,
    only: Option<&[&str]>,
    pages: bool,
) -> Arm {
    let ask = Millis(date_ms(ASK_DATE));
    let clock = Arc::new(FixedClock::new(ask));
    let store =
        DefaultGraph::open_with_clock(":memory:", GraphConfig::new(embedder.dims()), clock.clone())
            .await
            .expect("open the store");
    let store = Arc::new(store);
    let server = MemoryServer::new(
        store.clone(),
        embedder.clone(),
        Arc::new(MockLlm),
        BUILD_DEADLINE.as_secs(),
        8192,
        1,
    );
    let mut memory_ids = HashMap::new();
    let mut entity_ids = HashMap::new();
    for entity in set
        .entities
        .iter()
        .filter(|e| only.is_none_or(|ids| ids.contains(&e.id.as_str())))
    {
        clock.set(Millis(date_ms(&entity.memories[0].date)));
        let entity_id = store
            .upsert_by(NewNode::entity(&entity.kind, &entity.name))
            .await
            .expect("entity");
        for memory in &entity.memories {
            let at = Millis(date_ms(&memory.date));
            clock.set(at);
            let vector = embedder
                .embed(&memory.text, TextKind::Document)
                .await
                .expect("embed a memory");
            let node = NewNode::now("fact", "note", &memory.text)
                .with_valid_from(at)
                .with_embedding(vector);
            let id = store.insert(node).await.expect("memory");
            store
                .relate(&entity_id, &id, relation::MENTIONS)
                .await
                .expect("mentions edge");
            memory_ids.insert(memory.id.clone(), id);
        }
        entity_ids.insert(entity.id.clone(), entity_id);
    }
    clock.set(ask);
    let mut page_ids = HashMap::new();
    if pages {
        for entity in set
            .entities
            .iter()
            .filter(|e| entity_ids.contains_key(&e.id))
        {
            let deadline = tokio::time::Instant::now() + BUILD_DEADLINE;
            server
                .resynthesize_entity_as(&store, entity_ids[&entity.id].clone(), deadline, None)
                .await
                .expect("build the page");
            let hits = store
                .query_explained(&Query::text(entity.name.as_str()).with_inferred(true))
                .await
                .expect("find the page");
            let page = hits
                .iter()
                .find(|h| h.hit.label == entity.name && !h.hit.retrieved.as_text().is_empty())
                .expect("the page was written");
            page_ids.insert(entity.id.clone(), page.hit.id.clone());
        }
    }
    Arm {
        store,
        server,
        memory_ids,
        page_ids,
    }
}

/// The results of `recall`, in order.
async fn recall(arm: &Arm, question: &str) -> Vec<Shown> {
    let reply = arm
        .server
        .recall_with(
            RecallArgs {
                query: question.to_string(),
                kind: None,
                scope: None,
                k: Some(RECALL_K),
                as_of: None,
                mode: None,
            },
            CancellationToken::new(),
        )
        .await;
    let result = reply.into_call_result();
    let text = result.content[0].as_text().expect("text").text.clone();
    let mut shown = Vec::new();
    for block in text.split("\n\n").filter(|b| b.starts_with('[')) {
        let mut lines = block.lines();
        let head = lines
            .next()
            .expect("a block that starts with a bracket has a line");
        let handle = head[1..]
            .split(']')
            .next()
            .unwrap()
            .split(' ')
            .nth(1)
            .unwrap();
        let id = arm
            .store
            .resolve_handle(handle)
            .await
            .expect("a known handle");
        let body: Vec<&str> = lines
            .filter(|l| {
                !["provenance:", "kind:", "confidence:", "attributes:"]
                    .iter()
                    .any(|prefix| l.starts_with(prefix))
            })
            .collect();
        shown.push(Shown {
            id,
            text: body.join("\n"),
        });
    }
    shown
}

/// Whether the newest value is among the first results and no result with
/// only a stale value ranks above it.
fn latest_ok(texts: &[&str], question: &DataQuestion) -> bool {
    let newest = texts.iter().position(|t| has_all(t, &question.answer));
    let stale_only = texts
        .iter()
        .position(|t| has_any(t, &question.stale) && !has_any(t, &question.answer));
    newest.is_some_and(|n| n < READ_TOP && stale_only.is_none_or(|s| s > n))
}

async fn run_arm(set: &DataSet, arm: &Arm, only: Option<&[&str]>) -> Vec<Outcome> {
    let mut outcomes = Vec::new();
    for question in set
        .questions
        .iter()
        .filter(|q| only.is_none_or(|ids| ids.contains(&q.entity.as_str())))
    {
        let shown = recall(arm, &question.text).await;
        let page = arm.page_ids.get(&question.entity);
        let page_text = page.and_then(|id| shown.iter().find(|s| &s.id == id));
        let mut relevant: HashSet<&str> = question
            .gold
            .iter()
            .map(|g| arm.memory_ids[g].as_str())
            .collect();
        // A page answers when it holds every answer term.
        if let (Some(id), Some(page)) = (page, page_text) {
            if has_all(&page.text, &question.answer) {
                relevant.insert(id.as_str());
            }
        }
        let retrieved: Vec<String> = shown.iter().map(|s| s.id.as_str().to_string()).collect();
        let texts: Vec<&str> = shown.iter().map(|s| s.text.as_str()).collect();
        outcomes.push(Outcome {
            kind: question.kind.clone(),
            ndcg: ndcg_at_10(&retrieved, &relevant),
            latest_ok: (question.kind == LATEST).then(|| latest_ok(&texts, question)),
        });
    }
    outcomes
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values.fold((0.0, 0), |(sum, count), v| (sum + v, count + 1));
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn ndcg_of(outcomes: &[Outcome], kind: Option<&str>) -> f64 {
    mean(
        outcomes
            .iter()
            .filter(|o| kind.is_none_or(|k| o.kind == k))
            .map(|o| o.ndcg),
    )
}

fn latest_ok_rate(outcomes: &[Outcome]) -> f64 {
    mean(
        outcomes
            .iter()
            .filter_map(|o| o.latest_ok)
            .map(|ok| f64::from(u8::from(ok))),
    )
}

/// The text of the live page of entity `name`.
async fn page_text(arm: &Arm, name: &str) -> String {
    let hits = arm
        .store
        .query_explained(&Query::text(name).with_inferred(true))
        .await
        .unwrap();
    hits.iter()
        .find(|h| h.hit.label == name && h.hit.retrieved.is_inferred())
        .map(|h| h.hit.retrieved.as_text().to_string())
        .unwrap_or_default()
}

#[tokio::test]
async fn with_the_mock_embedder_every_page_holds_the_newest_memory_of_its_entity() {
    // Arrange: two entities of the English set, the mock embedder
    let set = load(LANGUAGES[0].1);
    let only = ["e01", "e04"];
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(MOCK_DIMS));

    // Act
    let comparisons = compare(&embedder, Some(&only)).await;
    let with = seed(&set, embedder, Some(&only), true).await;

    // Assert: every question was scored in both arms, and the page of each
    // entity holds its newest memory whatever the vectors say
    assert_eq!(comparisons.len(), 1);
    assert_eq!(comparisons[0].without.len(), 10);
    assert_eq!(comparisons[0].with.len(), 10);
    for outcome in comparisons[0].without.iter().chain(&comparisons[0].with) {
        assert!(outcome.ndcg.is_finite());
    }
    for entity in set
        .entities
        .iter()
        .filter(|e| only.contains(&e.id.as_str()))
    {
        let page = page_text(&with, &entity.name).await;
        let newest = entity.memories.last().unwrap();
        assert!(page.contains(&newest.text), "{page}");
        assert!(page.starts_with(&format!("{} ({})\n", entity.name, entity.kind)));
    }
}

#[test]
fn the_sets_have_what_the_evaluation_needs() {
    // Arrange / Act / Assert: per language 8 entities with 20 to 60 memories
    // over distinct dates, and at least 40 questions of each kind
    for (code, json) in LANGUAGES {
        let set = load(json);
        assert!(set.entities.len() >= 8, "{code}");
        assert!(set.questions.len() >= 40, "{code}");
        for entity in &set.entities {
            assert!(
                (20..=60).contains(&entity.memories.len()),
                "{code} {}",
                entity.id
            );
            let dates: HashSet<&str> = entity.memories.iter().map(|m| m.date.as_str()).collect();
            assert_eq!(dates.len(), entity.memories.len(), "{code} {}", entity.id);
        }
        for kind in ["single", "aggregate", LATEST] {
            assert!(
                set.questions.iter().any(|q| q.kind == kind),
                "{code} {kind}"
            );
        }
        for question in &set.questions {
            let entity = set
                .entities
                .iter()
                .find(|e| e.id == question.entity)
                .unwrap();
            let gold: Vec<&str> = question
                .gold
                .iter()
                .map(|id| {
                    entity
                        .memories
                        .iter()
                        .find(|m| &m.id == id)
                        .unwrap()
                        .text
                        .as_str()
                })
                .collect();
            let gold = gold.join(" ");
            assert!(has_all(&gold, &question.answer), "{code} {}", question.text);
            assert!(!has_any(&gold, &question.stale), "{code} {}", question.text);
        }
    }
}

#[test]
fn a_term_matches_whole_words_and_alternatives_only() {
    // Arrange / Act / Assert
    assert!(has_term("Moved to Berlin in July", "Berlim|Berlin"));
    assert!(!has_term("version 13.7 shipped", "3.7"));
    assert!(has_term("conversion rose to 6% in May", "6%"));
    assert!(!has_term("we go home", "Elixir"));
}

/// How a language scored without a page and with one.
struct Comparison {
    code: &'static str,
    without: Vec<Outcome>,
    with: Vec<Outcome>,
}

impl Comparison {
    /// The scores on one line, as the run prints them.
    fn summary(&self) -> String {
        format!(
            "{}: nDCG@10 A {:.3} C {:.3}; latest A {:.3} C {:.3}; latest ok A {:.3} C {:.3}",
            self.code,
            ndcg_of(&self.without, None),
            ndcg_of(&self.with, None),
            ndcg_of(&self.without, Some(LATEST)),
            ndcg_of(&self.with, Some(LATEST)),
            latest_ok_rate(&self.without),
            latest_ok_rate(&self.with),
        )
    }
}

/// Scores every language of the sets, or the entities `only` names of the first
/// one, with and without pages.
async fn compare(embedder: &Arc<dyn Embedder>, only: Option<&[&str]>) -> Vec<Comparison> {
    let mut comparisons = Vec::new();
    for (code, json) in LANGUAGES
        .into_iter()
        .take(only.map_or(LANGUAGES.len(), |_| 1))
    {
        let set = load(json);
        let without = seed(&set, embedder.clone(), only, false).await;
        let with = seed(&set, embedder.clone(), only, true).await;
        comparisons.push(Comparison {
            code,
            without: run_arm(&set, &without, only).await,
            with: run_arm(&set, &with, only).await,
        });
    }
    comparisons
}

/// Prints each language's numbers, then applies the gate.
fn report(comparisons: &[Comparison]) -> Result<(), String> {
    comparisons.iter().for_each(|c| println!("{}", c.summary()));
    check_gate(comparisons)
}

/// The rule a page must meet: it may not lose to no page by more than the
/// tolerance in any language, and it must win on the latest-state questions
/// overall.
fn check_gate(comparisons: &[Comparison]) -> Result<(), String> {
    for comparison in comparisons {
        let lost = ndcg_of(&comparison.without, None) - ndcg_of(&comparison.with, None);
        if lost > TOLERANCE {
            return Err(format!(
                "{}: pages lose to no pages by {lost:.3}",
                comparison.code
            ));
        }
    }
    let latest = |pick: fn(&Comparison) -> &Vec<Outcome>| {
        let all: Vec<Outcome> = comparisons
            .iter()
            .flat_map(|c| pick(c).iter())
            .map(|o| Outcome {
                kind: o.kind.clone(),
                ndcg: o.ndcg,
                latest_ok: o.latest_ok,
            })
            .collect();
        ndcg_of(&all, Some(LATEST))
    };
    let (without, with) = (latest(|c| &c.without), latest(|c| &c.with));
    if with > without {
        Ok(())
    } else {
        Err(format!(
            "pages do not win on the latest-state questions: {with:.3} against {without:.3}"
        ))
    }
}

fn outcome(kind: &str, ndcg: f64) -> Outcome {
    Outcome {
        kind: kind.to_string(),
        ndcg,
        latest_ok: None,
    }
}

fn comparison(code: &'static str, without: f64, with: f64, latest: (f64, f64)) -> Comparison {
    Comparison {
        code,
        without: vec![outcome("single", without), outcome(LATEST, latest.0)],
        with: vec![outcome("single", with), outcome(LATEST, latest.1)],
    }
}

#[test]
fn the_gate_passes_a_page_within_the_tolerance_that_wins_on_latest_state() {
    // Arrange: one language where the page ties on single facts and wins on latest state
    let comparisons = [comparison("en", 0.80, 0.80, (0.60, 0.70))];

    // Act / Assert
    assert_eq!(report(&comparisons), Ok(()));
}

#[test]
fn the_gate_fails_a_language_where_the_page_loses_by_more_than_the_tolerance() {
    // Arrange
    let comparisons = [comparison("pt", 0.90, 0.60, (0.50, 0.50))];

    // Act
    let verdict = check_gate(&comparisons);

    // Assert
    assert!(verdict.unwrap_err().starts_with("pt: pages lose"));
}

#[test]
fn the_gate_fails_a_page_that_does_not_win_on_latest_state() {
    // Arrange
    let comparisons = [comparison("es", 0.80, 0.80, (0.60, 0.60))];

    // Act
    let verdict = check_gate(&comparisons);

    // Assert
    assert!(verdict.unwrap_err().starts_with("pages do not win"));
}

#[tokio::test]
async fn the_summary_names_the_language_and_both_arms() {
    // Arrange
    let comparison = comparison("en", 0.5, 0.6, (0.4, 0.5));

    // Act
    let line = comparison.summary();

    // Assert
    assert!(line.starts_with("en: nDCG@10 A 0.450 C 0.550"), "{line}");
}

/// Runs every language with arctic and checks the rule a page must meet.
///
/// Needs the arctic files, about 311 MB, in a cache directory named by
/// `LIAM_ARCTIC_CACHE_DIR`; they are downloaded there on first use. Run it
/// with:
///
/// ```text
/// LIAM_ARCTIC_CACHE_DIR=<dir> cargo test --release -j 2 -p liam-daemon \
///   --bin liamd -- --ignored --nocapture entity_page::evaluation::
/// ```
#[tokio::test]
#[ignore = "loads the arctic embedder, which needs a model download"]
async fn with_arctic_a_page_does_not_lose_to_no_page_and_wins_on_latest_state() {
    let mut config = crate::config::Config::default();
    config.embedder.provider = liam_daemon::models::LOCAL_PROVIDER.to_string();
    config.embedder.cache_dir =
        std::env::var("LIAM_ARCTIC_CACHE_DIR").expect("LIAM_ARCTIC_CACHE_DIR names the cache");
    let embedder = liam_daemon::models::build_embedder(&config).expect("load arctic");

    let comparisons = compare(&embedder, None).await;

    report(&comparisons).unwrap();
}
