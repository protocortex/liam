// SPDX-License-Identifier: Apache-2.0
//! What a delete job was asked to forget.

use liam_log::event::{LogEvent, LogPayload, NodeRow, RowEffect};
use serde::{Deserialize, Serialize};

/// A subject (with its scope), or one logged event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeleteTarget {
    Subject {
        subject: String,
        scope: Option<String>,
        /// Also the subject's groups in every scope below `scope`.
        include_children: bool,
    },
    Event {
        event_id: String,
    },
}

impl DeleteTarget {
    /// Whether a plain (unsealed) `event` holds rows this target covers, which
    /// destroying keys cannot reach.
    pub fn covers_plain(&self, event: &LogEvent) -> bool {
        match self {
            Self::Event { event_id } => &event.event_id == event_id && event.payload.holds_rows(),
            Self::Subject { .. } => node_rows(&event.payload)
                .iter()
                .any(|node| self.covers_node(node)),
        }
    }

    fn covers_node(&self, node: &NodeRow) -> bool {
        let Self::Subject {
            subject,
            scope,
            include_children,
        } = self
        else {
            return false;
        };
        let same_subject = node
            .subject
            .as_deref()
            .is_some_and(|other| normalised(other) == normalised(subject));
        same_subject && scope_covers(scope.as_deref(), *include_children, node.scope.as_deref())
    }
}

fn normalised(subject: &str) -> String {
    subject.trim().to_lowercase()
}

/// An empty scope is none, as the store keeps it.
fn scope_covers(wanted: Option<&str>, include_children: bool, row: Option<&str>) -> bool {
    let wanted = wanted.unwrap_or_default();
    let row = row.unwrap_or_default();
    if include_children && wanted.is_empty() {
        return true;
    }
    row == wanted
        || (include_children
            && row
                .strip_prefix(wanted)
                .is_some_and(|rest| rest.starts_with('/')))
}

/// The node rows a payload carries.
pub fn node_rows(payload: &LogPayload) -> Vec<&NodeRow> {
    match payload {
        LogPayload::NodeWrite(node) => vec![node],
        LogPayload::EpisodeBatch(effects) => effects
            .iter()
            .filter_map(|effect| match effect {
                RowEffect::Node(node) => Some(node),
                RowEffect::Edge(_) => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use liam_log::event::CURRENT_SCHEMA_VERSION;

    use super::*;

    fn node(subject: Option<&str>, scope: Option<&str>) -> NodeRow {
        NodeRow {
            id: "n".into(),
            kind: "fact".into(),
            label: "l".into(),
            content: "c".into(),
            producer: "p".into(),
            attributes: "{}".into(),
            scope: scope.map(Into::into),
            subject: subject.map(Into::into),
            confidence: 1.0,
            valid_from: 0,
            valid_from_supplied: true,
            valid_until: i64::MAX,
            tx_from: 0,
            tx_to: i64::MAX,
            derivation: None,
            basis: Vec::new(),
        }
    }

    fn write(id: &str, row: NodeRow) -> LogEvent {
        LogEvent {
            event_id: id.into(),
            content_hash: [0; 32],
            source: "s".into(),
            trust_score: 1.0,
            observed_at: 0,
            ingested_at: 0,
            encryption_key_id: None,
            schema_version: CURRENT_SCHEMA_VERSION,
            payload: LogPayload::NodeWrite(row),
        }
    }

    fn subject(name: &str, scope: Option<&str>, include_children: bool) -> DeleteTarget {
        DeleteTarget::Subject {
            subject: name.into(),
            scope: scope.map(Into::into),
            include_children,
        }
    }

    #[test]
    fn a_plain_row_of_the_subject_is_covered_whatever_its_case() {
        // Arrange
        let event = write("e", node(Some(" Alice "), None));

        // Act
        let covered = subject("alice", None, false).covers_plain(&event);

        // Assert
        assert!(covered);
    }

    #[test]
    fn a_plain_row_of_another_subject_or_without_one_is_not_covered() {
        // Arrange
        let target = subject("alice", None, false);

        // Act
        let other = target.covers_plain(&write("e", node(Some("bob"), None)));
        let none = target.covers_plain(&write("e", node(None, None)));

        // Assert
        assert!(!other);
        assert!(!none);
    }

    #[test]
    fn scope_decides_which_plain_rows_are_covered() {
        // Arrange
        let in_scope = |target: &DeleteTarget, scope: Option<&str>| {
            target.covers_plain(&write("e", node(Some("alice"), scope)))
        };
        let exact = subject("alice", Some("work"), false);
        let tree = subject("alice", Some("work"), true);
        let everywhere = subject("alice", None, true);

        // Act and Assert
        assert!(in_scope(&exact, Some("work")));
        assert!(!in_scope(&exact, Some("work/liam")));
        assert!(in_scope(&tree, Some("work/liam")));
        assert!(!in_scope(&tree, Some("work-archive")));
        assert!(!in_scope(&tree, None));
        assert!(in_scope(&everywhere, Some("anything")));
        assert!(in_scope(&everywhere, None));
        assert!(!in_scope(&subject("alice", None, false), Some("work")));
    }

    #[test]
    fn a_plain_event_target_is_covered_only_when_it_holds_rows() {
        // Arrange
        let target = DeleteTarget::Event {
            event_id: "e".into(),
        };
        let mut marker = write("e", node(None, None));
        marker.payload = LogPayload::Voided {
            target_event_id: "x".into(),
        };

        // Act and Assert
        assert!(target.covers_plain(&write("e", node(None, None))));
        assert!(!target.covers_plain(&write("other", node(None, None))));
        assert!(!target.covers_plain(&marker));
    }

    #[test]
    fn an_episode_batch_row_of_the_subject_is_covered() {
        // Arrange
        let mut event = write("e", node(None, None));
        event.payload = LogPayload::EpisodeBatch(vec![
            RowEffect::Node(node(None, None)),
            RowEffect::Node(node(Some("alice"), None)),
        ]);

        // Act
        let covered = subject("alice", None, false).covers_plain(&event);

        // Assert
        assert!(covered);
    }
}
