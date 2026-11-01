// SPDX-License-Identifier: Apache-2.0
//! The one place a retrieved item is formatted. Every label is read off the
//! `Retrieved` variant, so a tool cannot show inferred text under a recorded
//! label, and every item can carry the provenance line that says where it came
//! from.

use liam_store::{Basis, EventRef, Retrieved};

use crate::ask::fmt_millis;

/// Event ids shown on one provenance line before the rest are counted.
pub const MAX_PROVENANCE_IDS: usize = 8;

/// The derivation a variant stands for.
pub fn derivation_label(retrieved: &Retrieved) -> &'static str {
    match retrieved {
        Retrieved::Recorded(hit) => hit.derivation.as_str(),
        Retrieved::Inferred(hit) => hit.derivation.as_str(),
    }
}

/// `[{kind} {handle}] {label}`, with `— known since` when `known_since_ms` is
/// given. The handle sits inside the bracket and nothing else does, because
/// anything more breaks `resolve_handle`'s alphanumeric gate (ADR-0004).
pub fn bracket_head(kind: &str, handle: &str, label: &str, known_since_ms: Option<i64>) -> String {
    let head = format!("[{kind} {handle}] {label}");
    with_known_since(head, known_since_ms)
}

/// `[{n}] ({kind}) {label} — known since {date}`, the head of an evidence block.
pub fn evidence_head(n: usize, kind: &str, label: &str, known_since_ms: i64) -> String {
    with_known_since(format!("[{n}] ({kind}) {label}"), Some(known_since_ms))
}

fn with_known_since(head: String, known_since_ms: Option<i64>) -> String {
    match known_since_ms {
        Some(ms) => format!("{head} — known since {}", fmt_millis(ms)),
        None => head,
    }
}

/// `provenance: derivation: recorded; events: a, b` for a recorded row, and
/// `provenance: derivation: inferred; basis: a, b` (or `basis: unknown`, plus
/// `summary: <level>`) for an inferred one.
pub fn provenance_line(retrieved: &Retrieved) -> String {
    let mut parts = vec![format!("derivation: {}", derivation_label(retrieved))];
    match retrieved {
        Retrieved::Recorded(hit) => parts.push(format!("events: {}", event_list(&hit.events))),
        Retrieved::Inferred(hit) => {
            parts.push(match &hit.basis {
                Basis::Known(events) => format!("basis: {}", event_list(events)),
                Basis::Unknown => "basis: unknown".to_string(),
            });
            if let Some(level) = hit.summary_level {
                parts.push(format!("summary: {level}"));
            }
        }
    }
    format!("provenance: {}", parts.join("; "))
}

/// The ids, `MAX_PROVENANCE_IDS` at most, with the number left out stated.
fn event_list(events: &[EventRef]) -> String {
    if events.is_empty() {
        return "none".to_string();
    }
    let shown: Vec<String> = events
        .iter()
        .take(MAX_PROVENANCE_IDS)
        .map(|event| match event.synthetic {
            true => format!("{} (synthetic)", event.event_id),
            false => event.event_id.clone(),
        })
        .collect();
    let left_out = events.len().saturating_sub(MAX_PROVENANCE_IDS);
    match left_out {
        0 => shown.join(", "),
        n => format!("{} (+{n} more)", shown.join(", ")),
    }
}

/// An item as a tool prints it: its head, its text, any trailing lines, then
/// its provenance line.
pub fn render_item(head: &str, retrieved: &Retrieved, trailing: &[String]) -> String {
    let mut lines = vec![head.to_string(), retrieved.as_text().to_string()];
    lines.extend(trailing.iter().cloned());
    lines.push(provenance_line(retrieved));
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use liam_store::{Derivation, InferredHit, InferredText, RecordedHit, RecordedText};

    use super::*;

    fn event(id: &str, synthetic: bool) -> EventRef {
        EventRef {
            event_id: id.to_string(),
            synthetic,
        }
    }

    fn recorded(derivation: Derivation, events: Vec<EventRef>) -> Retrieved {
        Retrieved::Recorded(RecordedHit {
            content: RecordedText::for_test("said so"),
            events,
            derivation,
        })
    }

    fn inferred(basis: Basis, summary_level: Option<u8>) -> Retrieved {
        Retrieved::Inferred(InferredHit {
            text: InferredText::for_test("composed"),
            basis,
            derivation: Derivation::Inferred,
            summary_level,
        })
    }

    #[test]
    fn each_variant_gets_the_label_its_type_implies() {
        // Arrange
        let cases = [
            (
                recorded(Derivation::Recorded, vec![event("e1", false)]),
                "provenance: derivation: recorded; events: e1",
            ),
            (
                recorded(Derivation::Extracted, vec![event("e1", false)]),
                "provenance: derivation: extracted; events: e1",
            ),
            (
                inferred(Basis::Known(vec![event("e2", false)]), None),
                "provenance: derivation: inferred; basis: e2",
            ),
            (
                inferred(Basis::Unknown, Some(2)),
                "provenance: derivation: inferred; basis: unknown; summary: 2",
            ),
        ];

        for (retrieved, expected) in cases {
            // Act
            let line = provenance_line(&retrieved);

            // Assert
            assert_eq!(line, expected);
        }
    }

    #[test]
    fn a_long_event_list_is_capped_with_the_count_left_out() {
        // Arrange
        let events: Vec<EventRef> = (0..40).map(|i| event(&format!("e{i:02}"), false)).collect();
        let retrieved = inferred(Basis::Known(events), None);

        // Act
        let line = provenance_line(&retrieved);

        // Assert
        assert!(line.contains("e07 (+32 more)"), "{line}");
        assert!(!line.contains("e08"), "{line}");
    }

    #[test]
    fn a_synthetic_event_is_marked_and_a_row_with_no_events_says_none() {
        // Arrange
        let marked = recorded(Derivation::Recorded, vec![event("e1", true)]);
        let bare = recorded(Derivation::Recorded, Vec::new());

        // Act
        let (marked, bare) = (provenance_line(&marked), provenance_line(&bare));

        // Assert
        assert!(marked.ends_with("events: e1 (synthetic)"), "{marked}");
        assert!(bare.ends_with("events: none"), "{bare}");
    }

    #[test]
    fn the_bracket_head_keeps_the_handle_alone_in_the_bracket() {
        // Arrange
        let (kind, handle, label) = ("fact", "01ABCDEFGHJKM", "Role");

        // Act
        let plain = bracket_head(kind, handle, label, None);
        let dated = bracket_head(kind, handle, label, Some(0));

        // Assert
        assert_eq!(plain, "[fact 01ABCDEFGHJKM] Role");
        assert_eq!(dated, "[fact 01ABCDEFGHJKM] Role — known since 1970-01-01");
    }

    #[test]
    fn an_item_ends_with_its_provenance_line() {
        // Arrange
        let retrieved = recorded(Derivation::Recorded, vec![event("e1", false)]);

        // Act
        let item = render_item(
            "[fact h] Role",
            &retrieved,
            &["confidence: 0.50".to_string()],
        );

        // Assert
        assert_eq!(
            item,
            "[fact h] Role\nsaid so\nconfidence: 0.50\nprovenance: derivation: recorded; events: e1"
        );
    }
}
