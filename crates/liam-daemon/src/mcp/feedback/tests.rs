// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::{Duration, Instant};

use liam_model::{IdentityReranker, MockEmbedder, MockLlm};
use liam_store::{CountedFeedback, DefaultGraph, FixedClock, Millis, NewNode, NodeId};
use rmcp::handler::server::wrapper::Parameters;

use super::*;
use crate::mcp::{AskArgs, Mode, RecallArgs};
use crate::tiering::test_support::logged_store;

const DIMS: usize = 8;
const START: Millis = Millis(10 * 86_400_000);
const PER_HOUR: u32 = 100;
const MAX_ISSUED: usize = 100;
const WINDOW_DAYS: i64 = 30;

struct Fixture {
    server: MemoryServer,
    store: Arc<DefaultGraph>,
    clock: Arc<FixedClock>,
    _dir: tempfile::TempDir,
}

async fn fixture(per_hour: Option<u32>) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(FixedClock::new(START));
    let store = Arc::new(logged_store(dir.path(), clock.clone(), DIMS).await);
    let mut server = MemoryServer::new(
        store.clone(),
        Arc::new(MockEmbedder::new(DIMS)),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
        30,
        false,
        8192,
        1,
    );
    if let Some(per_hour) = per_hour {
        server = server.with_feedback(per_hour, MAX_ISSUED);
    }
    Fixture {
        server,
        store,
        clock,
        _dir: dir,
    }
}

fn verdict(handle: &str, correct: bool) -> FeedbackArgs {
    FeedbackArgs {
        result_id: handle.to_string(),
        correct,
    }
}

async fn recall(server: &MemoryServer, query: &str) -> String {
    server
        .recall(Parameters(RecallArgs {
            query: query.to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await
}

/// The handle `recall` printed for the first result.
fn handle_in(output: &str) -> String {
    let head = output.lines().next().expect("a result");
    head.trim_start_matches('[')
        .split(']')
        .next()
        .and_then(|bracket| bracket.split(' ').nth(1))
        .expect("a handle in the bracket")
        .to_string()
}

async fn verdicts(store: &DefaultGraph) -> Vec<CountedFeedback> {
    let snapshot = store
        .usage_snapshot(Millis(START.0 + 1), WINDOW_DAYS)
        .await
        .unwrap();
    snapshot
        .items
        .into_iter()
        .flat_map(|item| item.feedback)
        .collect()
}

fn node(text: &str) -> NodeId {
    NodeId::from_raw(text)
}

#[tokio::test]
async fn a_verdict_on_a_recalled_result_is_logged_with_the_producer() {
    // Arrange
    let fixture = fixture(Some(PER_HOUR)).await;
    fixture.server.set_producer("agent-a");
    fixture
        .store
        .insert(NewNode::now("fact", "Mascot", "a wombat named Pixel"))
        .await
        .unwrap();
    let handle = handle_in(&recall(&fixture.server, "wombat Pixel").await);

    // Act
    let out = give(&fixture.server, verdict(&handle, true)).await;

    // Assert
    assert_eq!(out, "feedback recorded");
    let given = verdicts(&fixture.store).await;
    assert_eq!(given.len(), 1);
    assert_eq!(given[0].producer, "agent-a");
    assert!(given[0].correct);
}

#[tokio::test]
async fn a_handle_in_lower_case_matches() {
    // Arrange
    let fixture = fixture(Some(PER_HOUR)).await;
    fixture.server.set_producer("agent-a");
    fixture
        .store
        .insert(NewNode::now("fact", "Mascot", "a wombat named Pixel"))
        .await
        .unwrap();
    let handle = handle_in(&recall(&fixture.server, "wombat Pixel").await).to_lowercase();

    // Act
    let out = give(&fixture.server, verdict(&handle, false)).await;

    // Assert
    assert_eq!(out, "feedback recorded");
    assert!(!verdicts(&fixture.store).await[0].correct);
}

#[tokio::test]
async fn a_handle_never_issued_is_refused_and_nothing_is_logged() {
    // Arrange
    let fixture = fixture(Some(PER_HOUR)).await;
    let other = fixture
        .store
        .insert(NewNode::now("fact", "Unseen", "never recalled"))
        .await
        .unwrap();

    // Act
    let out = give(&fixture.server, verdict(other.handle(), true)).await;

    // Assert
    assert!(
        out.starts_with("feedback failed: no result with handle"),
        "{out}"
    );
    assert!(verdicts(&fixture.store).await.is_empty());
}

#[tokio::test]
async fn a_verdict_after_the_result_was_superseded_lands_on_the_same_lineage() {
    // Arrange
    let fixture = fixture(Some(PER_HOUR)).await;
    fixture
        .store
        .insert(NewNode::now("fact", "Role", "Ada is an engineer").with_subject("ada"))
        .await
        .unwrap();
    let handle = handle_in(&recall(&fixture.server, "Ada engineer").await);
    fixture
        .store
        .upsert_by(NewNode::now("fact", "Role", "Ada is a manager").with_subject("ada"))
        .await
        .unwrap();

    // Act
    let out = give(&fixture.server, verdict(&handle, true)).await;

    // Assert
    assert_eq!(out, "feedback recorded");
    assert_eq!(verdicts(&fixture.store).await.len(), 1);
}

#[tokio::test]
async fn a_verdict_on_a_result_that_has_since_been_removed_says_so() {
    // Arrange: a retention rule removes the node once it is old enough
    let fixture = fixture(Some(PER_HOUR)).await;
    fixture
        .store
        .insert(NewNode::now("episode", "Chat", "an old chat about wombats"))
        .await
        .unwrap();
    let handle = handle_in(&recall(&fixture.server, "wombats chat").await);
    fixture.clock.set(Millis(START.0 + Millis::days(10).0));
    let policy = liam_store::RetentionPolicy::keep("episode", Millis::days(1));
    fixture.store.gc(&policy).await.unwrap();

    // Act
    let out = give(&fixture.server, verdict(&handle, true)).await;

    // Assert
    assert!(out.contains("no longer exists"), "{out}");
}

#[tokio::test]
async fn a_producer_over_its_hourly_budget_is_refused() {
    // Arrange
    let fixture = fixture(Some(2)).await;
    fixture.server.set_producer("agent-a");
    fixture
        .store
        .insert(NewNode::now("fact", "Mascot", "a wombat named Pixel"))
        .await
        .unwrap();
    let handle = handle_in(&recall(&fixture.server, "wombat Pixel").await);

    // Act
    let outcomes = [
        give(&fixture.server, verdict(&handle, true)).await,
        give(&fixture.server, verdict(&handle, true)).await,
        give(&fixture.server, verdict(&handle, true)).await,
    ];

    // Assert
    assert_eq!(outcomes[1], "feedback recorded");
    // Both accepted verdicts are logged; the second of the day does not count.
    assert!(
        outcomes[2].contains("agent-a has used its verdicts"),
        "{}",
        outcomes[2]
    );
    assert_eq!(verdicts(&fixture.store).await.len(), 1);
}

#[tokio::test]
async fn two_producers_can_disagree_and_one_producer_counts_once_a_day() {
    // Arrange: a clone is another connection, with its own handles
    let fixture = fixture(Some(PER_HOUR)).await;
    let other = fixture.server.clone();
    fixture.server.set_producer("agent-a");
    other.set_producer("agent-b");
    fixture
        .store
        .insert(NewNode::now("fact", "Mascot", "a wombat named Pixel"))
        .await
        .unwrap();
    let mine = handle_in(&recall(&fixture.server, "wombat Pixel").await);
    let theirs = handle_in(&recall(&other, "wombat Pixel").await);

    // Act
    give(&fixture.server, verdict(&mine, true)).await;
    give(&other, verdict(&theirs, false)).await;
    give(&fixture.server, verdict(&mine, false)).await;

    // Assert: a's second verdict of the day is kept and does not count
    let mut given: Vec<_> = verdicts(&fixture.store)
        .await
        .into_iter()
        .map(|v| (v.producer, v.correct))
        .collect();
    given.sort();
    assert_eq!(
        given,
        [
            ("agent-a".to_string(), true),
            ("agent-b".to_string(), false)
        ]
    );
}

#[tokio::test]
async fn a_connection_cannot_use_a_handle_issued_to_another() {
    // Arrange
    let fixture = fixture(Some(PER_HOUR)).await;
    let other = fixture.server.clone();
    fixture
        .store
        .insert(NewNode::now("fact", "Mascot", "a wombat named Pixel"))
        .await
        .unwrap();
    let handle = handle_in(&recall(&fixture.server, "wombat Pixel").await);

    // Act
    let out = give(&other, verdict(&handle, true)).await;

    // Assert
    assert!(out.contains("was returned on this connection"), "{out}");
}

#[tokio::test]
async fn every_handle_ask_shows_can_be_given_a_verdict() {
    // Arrange
    let fixture = fixture(Some(PER_HOUR)).await;
    fixture
        .store
        .insert(NewNode::now("fact", "Mascot", "a wombat named Pixel"))
        .await
        .unwrap();

    // Act
    let strict = fixture
        .server
        .ask(Parameters(AskArgs {
            question: "wombat Pixel".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: Some(Mode::Strict),
        }))
        .await;
    let shown = strict
        .lines()
        .find_map(|line| line.strip_prefix("handle: "))
        .expect("a handle on the evidence block")
        .to_string();
    let out = give(&fixture.server, verdict(&shown, true)).await;

    // Assert
    assert_eq!(out, "feedback recorded");
}

#[tokio::test]
async fn without_tiering_the_tool_says_verdicts_are_not_kept() {
    // Arrange
    let fixture = fixture(None).await;

    // Act
    let out = give(&fixture.server, verdict("ABCDEF", true)).await;

    // Assert
    assert_eq!(
        out,
        "feedback failed: tiering is not enabled, so verdicts are not kept"
    );
}

#[test]
fn the_oldest_issued_handle_is_evicted_at_the_bound() {
    // Arrange
    let issued = IssuedHandles::new(2);
    let (a, b, c) = (node("AAAA1"), node("BBBB1"), node("CCCC1"));

    // Act
    issued.issue([&a, &b, &c]);

    // Assert
    assert_eq!(
        issued.resolve("AAAA"),
        Err(FeedbackError::NotIssued("AAAA".into()))
    );
    assert_eq!(issued.resolve("bbbb"), Ok(b));
    assert_eq!(issued.resolve("CCCC1"), Ok(c));
}

#[test]
fn two_issued_ids_sharing_a_handle_are_ambiguous_and_both_named() {
    // Arrange
    let issued = IssuedHandles::new(10);
    let (a, b) = (node("01M6FHXD6P4B6A"), node("01M6FHXD6P4B6B"));
    issued.issue([&b, &a]);

    // Act
    let result = issued.resolve("01M6FHXD6P4B6");

    // Assert
    assert_eq!(
        result,
        Err(FeedbackError::Ambiguous {
            handle: "01M6FHXD6P4B6".into(),
            candidates: vec![a.as_str().into(), b.as_str().into()],
        })
    );
}

#[test]
fn an_empty_or_non_alphanumeric_handle_is_invalid() {
    // Arrange
    let issued = IssuedHandles::new(10);

    for handle in ["", "  ", "ab-c", "ab*", "a b"] {
        // Act
        let result = issued.resolve(handle);

        // Assert
        assert_eq!(result, Err(FeedbackError::InvalidHandle), "{handle:?}");
    }
}

#[test]
fn a_clone_of_the_handles_starts_empty() {
    // Arrange
    let issued = IssuedHandles::new(10);
    issued.issue([&node("AAAA1")]);

    // Act
    let cloned = issued.clone();

    // Assert
    assert!(issued.resolve("AAAA").is_ok());
    assert!(cloned.resolve("AAAA").is_err());
}

#[test]
fn the_throttle_refuses_past_the_budget_and_allows_again_when_the_window_passes() {
    // Arrange
    let mut throttle = Throttle::new(2);
    let start = Instant::now();
    let later = |secs: u64| start + Duration::from_secs(secs);

    // Act
    let first = throttle.admit("a", start);
    let second = throttle.admit("a", later(10));
    let third = throttle.admit("a", later(20));
    let other_producer = throttle.admit("b", later(20));
    let just_inside = throttle.admit("a", later(3599));
    let after_first_expires = throttle.admit("a", later(3600));

    // Assert
    assert!(first.is_ok() && second.is_ok());
    assert_eq!(third, Err(Duration::from_secs(3580)));
    assert!(other_producer.is_ok());
    assert_eq!(just_inside, Err(Duration::from_secs(1)));
    assert!(after_first_expires.is_ok());
}
