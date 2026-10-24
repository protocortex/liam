// SPDX-License-Identifier: Apache-2.0
//! The text extraction reads from log events. Only events a client wrote about
//! the world are eligible: the daemon's own writes and cancelled ones are not,
//! so extraction never feeds on its own output.

use std::collections::{BTreeMap, HashSet};

use liam_log::event::{LogEvent, LogPayload, NodeRow, RowEffect};
use liam_log::LogOffset;
use liam_store::Millis;

use crate::config::RESERVED_PRODUCER_PREFIX;

/// One event's text in one scope, with what the fact rules need to know about
/// where it came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtractionInput {
    pub event_id: String,
    pub offset: LogOffset,
    pub scope: Option<String>,
    /// The writing producer, which the trust table is keyed on.
    pub source: String,
    pub observed_at: Millis,
    pub ingested_at: Millis,
    pub text: String,
}

/// The inputs of `events`, in the order given. An event with several scopes
/// gives one input per scope. Text under `min_event_chars` characters is left
/// out, and the caller still counts such an event as processed.
pub fn events_to_inputs(
    events: &[(LogOffset, LogEvent)],
    min_event_chars: usize,
) -> Vec<ExtractionInput> {
    let voided: HashSet<&str> = events
        .iter()
        .filter_map(|(_, event)| match &event.payload {
            LogPayload::Voided { target_event_id } => Some(target_event_id.as_str()),
            _ => None,
        })
        .collect();

    events
        .iter()
        .filter(|(_, event)| {
            !event.source.starts_with(RESERVED_PRODUCER_PREFIX)
                && !voided.contains(event.event_id.as_str())
        })
        .flat_map(|(offset, event)| {
            texts_by_scope(&event.payload)
                .into_iter()
                .filter(|(_, text)| text.chars().count() >= min_event_chars)
                .map(move |(scope, text)| ExtractionInput {
                    event_id: event.event_id.clone(),
                    offset: *offset,
                    scope,
                    source: event.source.clone(),
                    observed_at: Millis(event.observed_at),
                    ingested_at: Millis(event.ingested_at),
                    text,
                })
        })
        .collect()
}

/// Each fact row's label and content, joined per scope. Payloads that do not
/// state anything about the world give nothing.
fn texts_by_scope(payload: &LogPayload) -> BTreeMap<Option<String>, String> {
    let rows: Vec<&NodeRow> = match payload {
        LogPayload::NodeWrite(row) => vec![row],
        LogPayload::EpisodeBatch(effects) => effects
            .iter()
            .filter_map(|effect| match effect {
                RowEffect::Node(row) => Some(row),
                RowEffect::Edge(_) => None,
            })
            .collect(),
        LogPayload::EdgeWrite(_)
        | LogPayload::Tombstone(_)
        | LogPayload::DuplicateOf { .. }
        | LogPayload::Voided { .. }
        | LogPayload::FactWrite { .. }
        | LogPayload::RunRolledBack { .. } => Vec::new(),
    };
    let mut by_scope: BTreeMap<Option<String>, String> = BTreeMap::new();
    for row in rows.into_iter().filter(|row| !is_entity_page(row)) {
        let text = by_scope.entry(row.scope.clone()).or_default();
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&row.label);
        if !row.content.is_empty() {
            text.push('\n');
            text.push_str(&row.content);
        }
    }
    by_scope
}

/// An entity page keys its subject on its own name and has no content until
/// the daemon summarizes it. A client fact that merely shares its subject with
/// its label has content, so it is kept.
fn is_entity_page(row: &NodeRow) -> bool {
    row.content.is_empty()
        && row.subject.as_deref() == Some(row.label.trim().to_lowercase().as_str())
}

#[cfg(test)]
mod tests {
    use liam_log::event::{EdgeRow, FactRowEffect};

    use super::*;

    const CLIENT: &str = "claude-code";
    const MIN_CHARS: usize = 10;

    fn offset(index: u64) -> LogOffset {
        LogOffset { segment: 0, index }
    }

    fn event(id: &str, source: &str, payload: LogPayload) -> LogEvent {
        LogEvent {
            event_id: id.into(),
            content_hash: [0; 32],
            source: source.into(),
            trust_score: 1.0,
            observed_at: 1_000,
            ingested_at: 2_000,
            encryption_key_id: None,
            schema_version: 2,
            payload,
        }
    }

    fn node(label: &str, content: &str, scope: Option<&str>) -> NodeRow {
        NodeRow {
            id: format!("node-{label}"),
            kind: "note".into(),
            label: label.into(),
            content: content.into(),
            producer: CLIENT.into(),
            attributes: "{}".into(),
            scope: scope.map(str::to_owned),
            subject: None,
            confidence: 1.0,
            valid_from: 0,
            valid_from_supplied: false,
            valid_until: 0,
            tx_from: 0,
            tx_to: 0,
        }
    }

    fn entity(name: &str) -> NodeRow {
        NodeRow {
            subject: Some(name.trim().to_lowercase()),
            ..node(name, "", None)
        }
    }

    fn edge() -> EdgeRow {
        EdgeRow {
            id: "e".into(),
            src: "a".into(),
            dst: "b".into(),
            edge_type: "mentions".into(),
            attributes: "{}".into(),
            tx_from: 0,
            tx_to: 0,
        }
    }

    fn fact_row() -> FactRowEffect {
        FactRowEffect {
            id: "f".into(),
            lineage_id: "f".into(),
            subject: "alice".into(),
            predicate: "works_at".into(),
            object: "Acme".into(),
            object_key: "acme".into(),
            scope: String::new(),
            evidence_at: 0,
            valid_from: 0,
            valid_until: 0,
            tx_from: 0,
            tx_to: 0,
            closed_by: String::new(),
            confidence: 1.0,
            trust_score: 1.0,
            extractor_id: "llm".into(),
            extractor_version: "1".into(),
        }
    }

    fn numbered(events: Vec<LogEvent>) -> Vec<(LogOffset, LogEvent)> {
        events
            .into_iter()
            .enumerate()
            .map(|(index, event)| (offset(index as u64), event))
            .collect()
    }

    fn ids(inputs: &[ExtractionInput]) -> Vec<&str> {
        inputs.iter().map(|input| input.event_id.as_str()).collect()
    }

    #[test]
    fn only_node_and_episode_payloads_with_text_become_inputs() {
        // Arrange
        let ineligible = [
            LogPayload::EdgeWrite(edge()),
            LogPayload::Tombstone(Vec::new()),
            LogPayload::DuplicateOf {
                first_event_id: "n".into(),
            },
            LogPayload::Voided {
                target_event_id: "elsewhere".into(),
            },
            LogPayload::FactWrite {
                at: 0,
                closed_version_ids: Vec::new(),
                inserted: vec![fact_row()],
                source_links: Vec::new(),
                suppressions: Vec::new(),
            },
            LogPayload::RunRolledBack {
                run_id: "run".into(),
            },
        ];
        let mut events = vec![
            event(
                "node",
                CLIENT,
                LogPayload::NodeWrite(node("Alice", "works at Acme", None)),
            ),
            event(
                "episode",
                CLIENT,
                LogPayload::EpisodeBatch(vec![
                    RowEffect::Node(node("Bob", "lives in Perth", None)),
                    RowEffect::Edge(edge()),
                ]),
            ),
        ];
        events.extend(
            ineligible
                .into_iter()
                .enumerate()
                .map(|(n, payload)| event(&format!("skip-{n}"), CLIENT, payload)),
        );

        // Act
        let inputs = events_to_inputs(&numbered(events), MIN_CHARS);

        // Assert
        assert_eq!(ids(&inputs), ["node", "episode"]);
        assert_eq!(inputs[0].text, "Alice\nworks at Acme");
        assert_eq!(inputs[1].text, "Bob\nlives in Perth");
    }

    #[test]
    fn an_input_keeps_where_the_event_sits_and_who_wrote_it() {
        // Arrange
        let events = numbered(vec![
            event(
                "skipped",
                "consolidation:7",
                LogPayload::RunRolledBack { run_id: "7".into() },
            ),
            event(
                "kept",
                CLIENT,
                LogPayload::NodeWrite(node("Alice", "works at Acme", Some("team/a"))),
            ),
        ]);

        // Act
        let inputs = events_to_inputs(&events, MIN_CHARS);

        // Assert
        assert_eq!(
            inputs,
            vec![ExtractionInput {
                event_id: "kept".into(),
                offset: offset(1),
                scope: Some("team/a".into()),
                source: CLIENT.into(),
                observed_at: Millis(1_000),
                ingested_at: Millis(2_000),
                text: "Alice\nworks at Acme".into(),
            }]
        );
    }

    #[test]
    fn the_daemons_own_writes_never_become_inputs() {
        // Arrange
        let body = || LogPayload::NodeWrite(node("Alice", "works at Acme", None));
        let events = numbered(vec![
            event("run", "consolidation:01RUN", body()),
            event("bare", "consolidation", body()),
            event("client", CLIENT, body()),
        ]);

        // Act
        let inputs = events_to_inputs(&events, MIN_CHARS);

        // Assert
        assert_eq!(ids(&inputs), ["client"]);
    }

    #[test]
    fn an_event_voided_in_the_window_gives_no_input() {
        // Arrange
        let body = || LogPayload::NodeWrite(node("Alice", "works at Acme", None));
        let events = numbered(vec![
            event("cancelled", CLIENT, body()),
            event("kept", CLIENT, body()),
            event(
                "void",
                CLIENT,
                LogPayload::Voided {
                    target_event_id: "cancelled".into(),
                },
            ),
        ]);

        // Act
        let inputs = events_to_inputs(&events, MIN_CHARS);

        // Assert
        assert_eq!(ids(&inputs), ["kept"]);
    }

    #[test]
    fn an_entity_page_row_is_skipped_and_the_facts_beside_it_are_kept() {
        // Arrange
        let events = numbered(vec![event(
            "episode",
            CLIENT,
            LogPayload::EpisodeBatch(vec![
                RowEffect::Node(entity("  Alice ")),
                RowEffect::Node(node("Fact", "Alice works at Acme", None)),
            ]),
        )]);

        // Act
        let inputs = events_to_inputs(&events, MIN_CHARS);

        // Assert
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].text, "Fact\nAlice works at Acme");
    }

    #[test]
    fn a_fact_whose_subject_equals_its_label_but_has_content_is_kept() {
        // Arrange
        let row = NodeRow {
            subject: Some("alice".into()),
            ..node("Alice", "works at Acme", None)
        };
        let events = numbered(vec![event("fact", CLIENT, LogPayload::NodeWrite(row))]);

        // Act
        let inputs = events_to_inputs(&events, MIN_CHARS);

        // Assert
        assert_eq!(ids(&inputs), ["fact"]);
    }

    #[test]
    fn an_episode_in_two_scopes_gives_one_input_per_scope() {
        // Arrange
        let events = numbered(vec![event(
            "episode",
            CLIENT,
            LogPayload::EpisodeBatch(vec![
                RowEffect::Node(node("One", "first note", Some("b"))),
                RowEffect::Node(node("Two", "second note", Some("a"))),
                RowEffect::Node(node("Three", "third note", Some("b"))),
            ]),
        )]);

        // Act
        let inputs = events_to_inputs(&events, MIN_CHARS);

        // Assert
        let seen: Vec<_> = inputs
            .iter()
            .map(|input| (input.scope.as_deref(), input.text.as_str()))
            .collect();
        assert_eq!(
            seen,
            [
                (Some("a"), "Two\nsecond note"),
                (Some("b"), "One\nfirst note\nThree\nthird note"),
            ]
        );
    }

    #[test]
    fn text_is_kept_at_the_minimum_and_dropped_one_character_below() {
        // Arrange
        let at_minimum = "x".repeat(MIN_CHARS);
        let below = "x".repeat(MIN_CHARS - 1);
        let events = numbered(vec![
            event(
                "at",
                CLIENT,
                LogPayload::NodeWrite(node(&at_minimum, "", None)),
            ),
            event(
                "below",
                CLIENT,
                LogPayload::NodeWrite(node(&below, "", None)),
            ),
        ]);

        // Act
        let inputs = events_to_inputs(&events, MIN_CHARS);

        // Assert
        assert_eq!(ids(&inputs), ["at"]);
    }
}
