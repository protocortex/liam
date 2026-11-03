// SPDX-License-Identifier: Apache-2.0
//! The gate's decisions, the counts, and that no item text reaches a log line.

use std::collections::BTreeMap;
use std::io::Write;
use std::sync::{Arc, Mutex};

use liam_log::LogOffset;
use liam_store::{Millis, NodeId, RecordedText};

use super::*;
use crate::extraction::cache::{CacheStore, NewEntry};

const PROVIDER: &str = "anthropic";
const REGION: &str = "us";
const SECRET_TEXT: &str = "the-private-sentence";

fn policy() -> PrivacyConfig {
    PrivacyConfig {
        allowed_scopes: vec!["work".into()],
        local_only_sources: vec!["client:diary".into()],
        provider_regions: BTreeMap::from([(PROVIDER.into(), REGION.into())]),
        attested_regions: BTreeMap::from([("work".into(), vec![REGION.into()])]),
    }
}

fn item<'a>(scope: Option<&'a str>, sources: &'a [String]) -> GateItem<'a> {
    GateItem { scope, sources }
}

fn input(scope: Option<&str>, source: &str, text: &str) -> ExtractionInput {
    ExtractionInput {
        event_id: format!("event-{text}"),
        offset: LogOffset {
            segment: 0,
            index: 0,
        },
        scope: scope.map(str::to_owned),
        source: source.to_owned(),
        observed_at: Millis(0),
        ingested_at: Millis(0),
        text: text.to_owned(),
    }
}

fn summary_input(scope: Option<&str>, sources: &[&str]) -> SummaryInput {
    SummaryInput {
        id: NodeId::from_raw("node"),
        kind: "note".into(),
        label: "label".into(),
        content: RecordedText::for_test(SECRET_TEXT),
        valid_from: Millis(0),
        scope: scope.map(str::to_owned),
        origin_events: Vec::new(),
        sources: sources.iter().map(|s| (*s).to_owned()).collect(),
        vector: Vec::new(),
    }
}

#[test]
fn with_no_policy_every_item_is_denied() {
    // Arrange
    let gate = PrivacyGate::new(PrivacyConfig::default());

    // Act
    let decision = gate.decide(&item(Some("work"), &[]), PROVIDER);

    // Assert
    assert_eq!(
        decision,
        Decision::Deny {
            reason: DenyReason::ScopeNotAllowed
        }
    );
}

#[test]
fn an_allowed_scope_with_an_attested_region_is_allowed() {
    // Arrange
    let gate = PrivacyGate::new(policy());
    let sources = ["client:mcp".to_owned()];

    // Act
    let decision = gate.decide(&item(Some("work"), &sources), PROVIDER);

    // Assert
    assert_eq!(decision, Decision::Allow);
}

#[test]
fn a_local_only_source_denies_an_item_in_an_allowed_scope() {
    // Arrange
    let gate = PrivacyGate::new(policy());
    let sources = ["client:diary".to_owned()];

    // Act
    let decision = gate.decide(&item(Some("work"), &sources), PROVIDER);

    // Assert
    assert_eq!(
        decision,
        Decision::Deny {
            reason: DenyReason::LocalOnlySource
        }
    );
}

#[test]
fn an_unattested_region_denies_and_so_does_an_unknown_provider() {
    // Arrange
    let mut other_region = policy();
    other_region
        .provider_regions
        .insert(PROVIDER.into(), "eu".into());
    let gate = PrivacyGate::new(other_region);
    let unknown = PrivacyGate::new(policy());

    // Act
    let wrong_region = gate.decide(&item(Some("work"), &[]), PROVIDER);
    let no_region = unknown.decide(&item(Some("work"), &[]), "openai");

    // Assert
    let denied = Decision::Deny {
        reason: DenyReason::RegionNotAttested,
    };
    assert_eq!(wrong_region, denied);
    assert_eq!(no_region, denied);
}

#[test]
fn an_item_with_no_sources_is_judged_on_scope_and_region_alone() {
    // Arrange
    let gate = PrivacyGate::new(policy());

    // Act
    let decision = gate.decide(&item(Some("work"), &[]), PROVIDER);

    // Assert
    assert_eq!(decision, Decision::Allow);
}

#[test]
fn a_child_scope_is_allowed_by_its_ancestor_and_a_parent_is_not_by_its_child() {
    // Arrange
    let child_only = PrivacyConfig {
        allowed_scopes: vec!["work/liam".into()],
        attested_regions: BTreeMap::from([("work/liam".into(), vec![REGION.into()])]),
        ..policy()
    };
    let by_ancestor = PrivacyGate::new(policy());
    let by_child = PrivacyGate::new(child_only);

    // Act
    let child = by_ancestor.decide(&item(Some("work/liam"), &[]), PROVIDER);
    let exact = by_ancestor.decide(&item(Some("work"), &[]), PROVIDER);
    let parent = by_child.decide(&item(Some("work"), &[]), PROVIDER);

    // Assert
    assert_eq!(child, Decision::Allow);
    assert_eq!(exact, Decision::Allow);
    assert_eq!(
        parent,
        Decision::Deny {
            reason: DenyReason::ScopeNotAllowed
        }
    );
}

#[test]
fn a_region_attested_for_one_scope_does_not_cover_another() {
    // Arrange
    let both_allowed = PrivacyConfig {
        allowed_scopes: vec!["a".into(), "b".into()],
        attested_regions: BTreeMap::from([("a".into(), vec![REGION.into()])]),
        ..policy()
    };
    let gate = PrivacyGate::new(both_allowed);

    // Act
    let attested = gate.decide(&item(Some("a"), &[]), PROVIDER);
    let not = gate.decide(&item(Some("b"), &[]), PROVIDER);

    // Assert
    assert_eq!(attested, Decision::Allow);
    assert_eq!(
        not,
        Decision::Deny {
            reason: DenyReason::RegionNotAttested
        }
    );
}

#[test]
fn three_of_ten_denied_leaves_seven_in_the_prompt_and_counts_the_three_by_reason() {
    // Arrange
    let gate = PrivacyGate::new(policy());
    let mut inputs: Vec<ExtractionInput> = (0..7)
        .map(|n| input(Some("work"), "client:mcp", &format!("kept {n}")))
        .collect();
    inputs.push(input(Some("home"), "client:mcp", "elsewhere"));
    inputs.push(input(Some("work"), "client:diary", "private"));
    inputs.push(input(None, "client:mcp", "unscoped"));

    // Act
    let (allowed, counts) = gate.filter_inputs(PROVIDER, &inputs);

    // Assert
    assert_eq!(allowed.len(), 7);
    assert_eq!(counts.total(), 3);
    assert_eq!(counts.of(DenyReason::ScopeNotAllowed), 2);
    assert_eq!(counts.of(DenyReason::LocalOnlySource), 1);
    assert_eq!(inputs.len(), 10, "the caller's list is untouched");
}

#[test]
fn when_all_ten_are_denied_nothing_is_left_to_send() {
    // Arrange
    let gate = PrivacyGate::new(PrivacyConfig::default());
    let inputs: Vec<ExtractionInput> = (0..10)
        .map(|n| input(Some("work"), "client:mcp", &format!("item {n}")))
        .collect();

    // Act
    let (allowed, counts) = gate.filter_inputs(PROVIDER, &inputs);

    // Assert
    assert!(allowed.is_empty());
    assert_eq!(counts.total(), 10);
}

#[test]
fn one_local_only_source_keeps_the_whole_summary_at_home() {
    // Arrange
    let gate = PrivacyGate::new(policy());
    let clean = [
        summary_input(Some("work"), &["client:mcp"]),
        summary_input(Some("work"), &["client:mcp"]),
    ];
    let mixed = [
        summary_input(Some("work"), &["client:mcp"]),
        summary_input(Some("work"), &["client:diary"]),
    ];

    // Act
    let sent = gate.decide_summary(PROVIDER, &clean);
    let kept = gate.decide_summary(PROVIDER, &mixed);

    // Assert
    assert_eq!(sent, Decision::Allow);
    assert_eq!(
        kept,
        Decision::Deny {
            reason: DenyReason::LocalOnlySource
        }
    );
}

#[test]
fn a_cached_answer_is_not_served_once_its_scope_leaves_the_allow_list() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let cache = CacheStore::open(dir.path().join("cache.jsonl")).unwrap();
    cache
        .put(NewEntry {
            key: "k",
            response: "{}",
            model_id: "m",
            model_sha256: "s",
            prompt_version: "p",
            source_event_ids: &[],
        })
        .unwrap();
    let before = PrivacyGate::new(policy());
    let after = PrivacyGate::new(PrivacyConfig {
        allowed_scopes: Vec::new(),
        ..policy()
    });
    let work = item(Some("work"), &[]);

    // Act
    let served = before.cached(PROVIDER, &work, &cache, "k");
    let withheld = after.cached(PROVIDER, &work, &cache, "k");

    // Assert
    assert_eq!(served.as_deref(), Some("{}"));
    assert_eq!(withheld, None);
}

#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn a_denial_is_logged_with_its_reason_and_never_with_the_item_text() {
    // Arrange
    let captured = Captured::default();
    let writer = captured.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(move || writer.clone())
        .with_ansi(false)
        .finish();
    let gate = PrivacyGate::new(policy());
    let inputs = [input(Some("home"), "client:mcp", SECRET_TEXT)];

    // Act
    tracing::subscriber::with_default(subscriber, || {
        gate.filter_inputs(PROVIDER, &inputs);
    });

    // Assert
    let log = String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();
    assert!(log.contains("scope_not_allowed=1"), "{log}");
    assert!(!log.contains(SECRET_TEXT), "{log}");
}
