// SPDX-License-Identifier: Apache-2.0
//! Finding what a delete destroys. One pass over the whole log, rolled-back
//! runs included, collects the events whose rows sit under the target's key
//! groups, then the fact writes and derived artifacts that cite them: those
//! are sealed under keys of their own, so only an opened record shows what it
//! cites.

use std::collections::BTreeSet;
use std::fmt;

use futures_util::StreamExt;
use liam_log::envelope::RowKind;
use liam_log::event::{DerivedArtifact, LogPayload};
use liam_log::reader::{LogReader, LogRecord, ReaderError};

use super::target::{node_rows, DeleteTarget};

/// Why the scan could not name what to destroy. No key has been touched.
#[derive(Debug)]
pub enum ScanError {
    /// A row of the target is plain in the log, where destroying a key cannot
    /// reach it. `liamd encrypt-log` seals it first.
    PlaintextLogPresent {
        event_id: String,
    },
    /// The event shares its key group with a subject's rows, so deleting it
    /// alone would leave the group alive. Delete the subject instead.
    EventInSubjectGroup {
        event_id: String,
    },
    NothingToDelete,
    /// Archived segments could not be reached, so the scan could not see every
    /// row.
    PartitionUnreachable {
        first_segment: u64,
        last_segment: u64,
    },
    Reader(ReaderError),
}

impl fmt::Display for ScanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlaintextLogPresent { event_id } => write!(
                f,
                "event {event_id} is plain in the log and cannot be deleted by destroying a \
                 key; run liamd encrypt-log first"
            ),
            Self::EventInSubjectGroup { event_id } => write!(
                f,
                "event {event_id} holds rows of a subject, so it cannot be deleted alone; \
                 delete the subject"
            ),
            Self::NothingToDelete => write!(f, "nothing in the log matches the target"),
            Self::PartitionUnreachable {
                first_segment,
                last_segment,
            } => write!(
                f,
                "segments {first_segment} to {last_segment} are archived and not reachable, \
                 so the log cannot be scanned in full"
            ),
            Self::Reader(error) => write!(f, "the log could not be scanned: {error}"),
        }
    }
}

impl std::error::Error for ScanError {}

impl From<ReaderError> for ScanError {
    fn from(error: ReaderError) -> Self {
        match error {
            ReaderError::PartitionUnreachable {
                first_segment,
                last_segment,
            } => Self::PartitionUnreachable {
                first_segment,
                last_segment,
            },
            other => Self::Reader(other),
        }
    }
}

/// What a delete destroys, as the log shows it.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Found {
    pub event_ids: Vec<String>,
    pub node_ids: Vec<String>,
    /// The key groups of the target, then the row keys of every citing write.
    pub key_ids: Vec<String>,
}

/// Scans the log for `target`, whose subject key groups are `groups` (empty
/// for an event target).
pub async fn scan(
    reader: &dyn LogReader,
    target: &DeleteTarget,
    groups: &[String],
) -> Result<Found, ScanError> {
    let mut state = State::new(groups);
    let mut records = reader.scan(None);
    while let Some(record) = records.next().await {
        state.visit(target, &record?)?;
    }
    state.finish()
}

struct State {
    groups: BTreeSet<String>,
    events: BTreeSet<String>,
    nodes: BTreeSet<String>,
    /// Ids a later record may cite: events, nodes, and artifacts already doomed.
    doomed: BTreeSet<String>,
    keys: BTreeSet<String>,
    target_seen: bool,
}

impl State {
    fn new(groups: &[String]) -> Self {
        Self {
            groups: groups.iter().cloned().collect(),
            events: BTreeSet::new(),
            nodes: BTreeSet::new(),
            doomed: BTreeSet::new(),
            keys: groups.iter().cloned().collect(),
            target_seen: false,
        }
    }

    fn visit(&mut self, target: &DeleteTarget, record: &LogRecord) -> Result<(), ScanError> {
        let event = &record.event;
        if record.row_keys.is_empty() {
            if target.covers_plain(event) {
                return Err(ScanError::PlaintextLogPresent {
                    event_id: event.event_id.clone(),
                });
            }
            return Ok(());
        }
        if self.is_target(target, record) {
            self.take_target(target, record)?;
        } else if self.cites_doomed(&event.payload) {
            self.take_citing(record);
        }
        Ok(())
    }

    fn is_target(&self, target: &DeleteTarget, record: &LogRecord) -> bool {
        match target {
            DeleteTarget::Event { event_id } => &record.event.event_id == event_id,
            DeleteTarget::Subject { .. } => record
                .row_keys
                .iter()
                .any(|row| self.groups.contains(&row.key_id)),
        }
    }

    fn take_target(&mut self, target: &DeleteTarget, record: &LogRecord) -> Result<(), ScanError> {
        self.target_seen = true;
        let event_id = &record.event.event_id;
        if matches!(target, DeleteTarget::Event { .. }) {
            if node_rows(&record.event.payload)
                .iter()
                .any(|node| node.subject.is_some())
            {
                return Err(ScanError::EventInSubjectGroup {
                    event_id: event_id.clone(),
                });
            }
            self.keys
                .extend(record.row_keys.iter().map(|row| row.key_id.clone()));
        }
        for row in &record.row_keys {
            let ours = self.keys.contains(&row.key_id);
            if ours && row.kind == RowKind::Node {
                self.nodes.insert(row.row_id.clone());
                self.doomed.insert(row.row_id.clone());
            }
        }
        self.events.insert(event_id.clone());
        self.doomed.insert(event_id.clone());
        Ok(())
    }

    fn cites_doomed(&self, payload: &LogPayload) -> bool {
        match payload {
            LogPayload::FactWrite { source_links, .. } => source_links
                .iter()
                .any(|(_, event_id)| self.doomed.contains(event_id)),
            LogPayload::DerivedArtifact(artifact) => self.artifact_cites_doomed(artifact),
            _ => false,
        }
    }

    fn artifact_cites_doomed(&self, artifact: &DerivedArtifact) -> bool {
        artifact
            .child_ids
            .iter()
            .any(|child| self.doomed.contains(child))
            || artifact
                .source_event_ids
                .iter()
                .any(|basis| self.doomed.contains(&basis.event_id))
    }

    /// A write that cites doomed rows is destroyed with its own row keys, and
    /// what cites it goes next.
    fn take_citing(&mut self, record: &LogRecord) {
        self.events.insert(record.event.event_id.clone());
        self.doomed.insert(record.event.event_id.clone());
        for row in &record.row_keys {
            self.keys.insert(row.key_id.clone());
            if row.kind == RowKind::Artifact {
                self.doomed.insert(row.row_id.clone());
            }
        }
    }

    fn finish(self) -> Result<Found, ScanError> {
        if !self.target_seen {
            return Err(ScanError::NothingToDelete);
        }
        Ok(Found {
            event_ids: self.events.into_iter().collect(),
            node_ids: self.nodes.into_iter().collect(),
            key_ids: self.keys.into_iter().collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use futures_util::stream;
    use liam_log::envelope::RowKeyRef;
    use liam_log::event::{BasisRef, LogEvent, NodeRow, CURRENT_SCHEMA_VERSION};
    use liam_log::reader::LogStream;
    use liam_log::LogOffset;

    use super::*;

    const ALICE: &str = "key-alice";
    const BOB: &str = "key-bob";

    /// Serves fixed records, then an optional failure.
    struct FixedReader {
        records: Vec<LogRecord>,
        failure: Option<(u64, u64)>,
    }

    impl LogReader for FixedReader {
        fn scan(&self, _from: Option<LogOffset>) -> LogStream {
            let mut items: Vec<Result<LogRecord, ReaderError>> =
                self.records.iter().cloned().map(Ok).collect();
            if let Some((first_segment, last_segment)) = self.failure {
                items.push(Err(ReaderError::PartitionUnreachable {
                    first_segment,
                    last_segment,
                }));
            }
            Box::pin(stream::iter(items))
        }

        fn scan_through(&self, from: Option<LogOffset>, _through: LogOffset) -> LogStream {
            self.scan(from)
        }
    }

    fn reader(records: Vec<LogRecord>) -> FixedReader {
        FixedReader {
            records,
            failure: None,
        }
    }

    fn node(id: &str, subject: Option<&str>) -> NodeRow {
        NodeRow {
            id: id.into(),
            kind: "fact".into(),
            label: id.into(),
            content: "secret".into(),
            producer: "p".into(),
            attributes: "{}".into(),
            scope: None,
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

    fn record(
        index: u64,
        event_id: &str,
        payload: LogPayload,
        keys: &[(&str, RowKind, &str)],
    ) -> LogRecord {
        let event = LogEvent {
            event_id: event_id.into(),
            content_hash: [0; 32],
            source: "s".into(),
            trust_score: 1.0,
            observed_at: 0,
            ingested_at: 0,
            encryption_key_id: None,
            schema_version: CURRENT_SCHEMA_VERSION,
            payload,
        };
        let mut record = LogRecord::new(LogOffset { segment: 1, index }, event);
        record.row_keys = keys
            .iter()
            .map(|(row_id, kind, key_id)| RowKeyRef {
                row_id: (*row_id).into(),
                kind: *kind,
                key_id: (*key_id).into(),
            })
            .collect();
        record
    }

    fn node_event(
        index: u64,
        event_id: &str,
        node_id: &str,
        subject: Option<&str>,
        key: &str,
    ) -> LogRecord {
        record(
            index,
            event_id,
            LogPayload::NodeWrite(node(node_id, subject)),
            &[(node_id, RowKind::Node, key)],
        )
    }

    fn fact_write(index: u64, event_id: &str, cites: &str, fact_key: &str) -> LogRecord {
        record(
            index,
            event_id,
            LogPayload::FactWrite {
                at: 0,
                closed_version_ids: Vec::new(),
                inserted: Vec::new(),
                source_links: vec![("lineage".into(), cites.into())],
                suppressions: Vec::new(),
            },
            &[(event_id, RowKind::Fact, fact_key)],
        )
    }

    fn artifact(
        index: u64,
        event_id: &str,
        id: &str,
        children: &[&str],
        events: &[&str],
        key: &str,
    ) -> LogRecord {
        record(
            index,
            event_id,
            LogPayload::DerivedArtifact(DerivedArtifact {
                artifact_id: id.into(),
                kind: "summary".into(),
                level: 1,
                scope: String::new(),
                text: "composed".into(),
                child_ids: children.iter().map(|c| (*c).into()).collect(),
                source_event_ids: events
                    .iter()
                    .map(|event_id| BasisRef {
                        event_id: (*event_id).into(),
                        synthetic: false,
                    })
                    .collect(),
                events_fed: 1,
                derivation: "inferred".into(),
                model_id: "m".into(),
                prompt_version: "p".into(),
                closes: Vec::new(),
                valid_min: 0,
                valid_max: 0,
                at: 0,
            }),
            &[(id, RowKind::Artifact, key)],
        )
    }

    fn alice() -> DeleteTarget {
        DeleteTarget::Subject {
            subject: "alice".into(),
            scope: None,
            include_children: false,
        }
    }

    fn groups() -> Vec<String> {
        vec![ALICE.into()]
    }

    #[tokio::test]
    async fn the_events_nodes_and_group_of_a_subject_are_found() {
        // Arrange
        let log = reader(vec![
            node_event(0, "e1", "n1", Some("alice"), ALICE),
            node_event(1, "e2", "n2", Some("bob"), BOB),
        ]);

        // Act
        let found = scan(&log, &alice(), &groups()).await.unwrap();

        // Assert
        assert_eq!(found.event_ids, ["e1"]);
        assert_eq!(found.node_ids, ["n1"]);
        assert_eq!(found.key_ids, [ALICE]);
    }

    #[tokio::test]
    async fn a_fact_write_of_another_subject_that_cites_the_target_is_found_by_its_row_key() {
        // Arrange
        let log = reader(vec![
            node_event(0, "e1", "n1", Some("alice"), ALICE),
            node_event(1, "e2", "n2", Some("bob"), BOB),
            fact_write(2, "e3", "e1", "key-fact-1"),
            fact_write(3, "e4", "e2", "key-fact-2"),
        ]);

        // Act
        let found = scan(&log, &alice(), &groups()).await.unwrap();

        // Assert
        assert_eq!(found.event_ids, ["e1", "e3"]);
        assert_eq!(found.key_ids, [ALICE, "key-fact-1"]);
    }

    #[tokio::test]
    async fn an_artifact_citing_a_node_or_an_event_or_another_doomed_artifact_is_found() {
        // Arrange
        let log = reader(vec![
            node_event(0, "e1", "n1", Some("alice"), ALICE),
            node_event(1, "e2", "n2", Some("bob"), BOB),
            artifact(2, "a1", "art-1", &["n1"], &[], "key-art-1"),
            artifact(3, "a2", "art-2", &[], &["e1"], "key-art-2"),
            artifact(4, "a3", "art-3", &["art-1"], &[], "key-art-3"),
            artifact(5, "a4", "art-4", &["n2"], &["e2"], "key-art-4"),
        ]);

        // Act
        let found = scan(&log, &alice(), &groups()).await.unwrap();

        // Assert
        assert_eq!(found.event_ids, ["a1", "a2", "a3", "e1"]);
        assert_eq!(
            found.key_ids,
            [ALICE, "key-art-1", "key-art-2", "key-art-3"]
        );
    }

    #[tokio::test]
    async fn a_plain_row_of_the_subject_refuses_the_scan() {
        // Arrange
        let mut plain = node_event(1, "e2", "n2", Some("alice"), ALICE);
        plain.row_keys.clear();
        let log = reader(vec![node_event(0, "e1", "n1", Some("alice"), ALICE), plain]);

        // Act
        let result = scan(&log, &alice(), &groups()).await;

        // Assert
        assert!(
            matches!(result, Err(ScanError::PlaintextLogPresent { event_id }) if event_id == "e2")
        );
    }

    #[tokio::test]
    async fn a_plain_row_of_someone_else_does_not_refuse_the_scan() {
        // Arrange
        let mut plain = node_event(1, "e2", "n2", Some("bob"), BOB);
        plain.row_keys.clear();
        let log = reader(vec![node_event(0, "e1", "n1", Some("alice"), ALICE), plain]);

        // Act
        let found = scan(&log, &alice(), &groups()).await.unwrap();

        // Assert
        assert_eq!(found.event_ids, ["e1"]);
    }

    #[tokio::test]
    async fn an_event_without_a_subject_is_deleted_by_its_own_keys() {
        // Arrange
        let log = reader(vec![
            node_event(0, "e1", "n1", None, "key-event-1"),
            node_event(1, "e2", "n2", None, "key-event-2"),
        ]);
        let target = DeleteTarget::Event {
            event_id: "e1".into(),
        };

        // Act
        let found = scan(&log, &target, &[]).await.unwrap();

        // Assert
        assert_eq!(found.event_ids, ["e1"]);
        assert_eq!(found.node_ids, ["n1"]);
        assert_eq!(found.key_ids, ["key-event-1"]);
    }

    #[tokio::test]
    async fn an_event_under_a_subject_key_is_refused() {
        // Arrange
        let log = reader(vec![node_event(0, "e1", "n1", Some("alice"), ALICE)]);
        let target = DeleteTarget::Event {
            event_id: "e1".into(),
        };

        // Act
        let result = scan(&log, &target, &[]).await;

        // Assert
        assert!(
            matches!(result, Err(ScanError::EventInSubjectGroup { event_id }) if event_id == "e1")
        );
    }

    #[tokio::test]
    async fn a_plain_event_target_is_refused() {
        // Arrange
        let mut plain = node_event(0, "e1", "n1", None, "key");
        plain.row_keys.clear();
        let target = DeleteTarget::Event {
            event_id: "e1".into(),
        };

        // Act
        let result = scan(&reader(vec![plain]), &target, &[]).await;

        // Assert
        assert!(matches!(result, Err(ScanError::PlaintextLogPresent { .. })));
    }

    #[tokio::test]
    async fn a_target_the_log_does_not_hold_is_nothing_to_delete() {
        // Arrange
        let log = reader(vec![node_event(0, "e2", "n2", Some("bob"), BOB)]);
        let event = DeleteTarget::Event {
            event_id: "missing".into(),
        };

        // Act
        let subject = scan(&log, &alice(), &groups()).await;
        let by_event = scan(&log, &event, &[]).await;

        // Assert
        assert!(matches!(subject, Err(ScanError::NothingToDelete)));
        assert!(matches!(by_event, Err(ScanError::NothingToDelete)));
    }

    #[tokio::test]
    async fn an_unreachable_archive_fails_the_scan_with_the_segments() {
        // Arrange
        let log = FixedReader {
            records: vec![node_event(0, "e1", "n1", Some("alice"), ALICE)],
            failure: Some((2, 5)),
        };

        // Act
        let result = scan(&log, &alice(), &groups()).await;

        // Assert
        let error = result.expect_err("must fail");
        assert!(matches!(
            error,
            ScanError::PartitionUnreachable {
                first_segment: 2,
                last_segment: 5
            }
        ));
        assert!(error.to_string().contains("2 to 5"));
    }

    #[tokio::test]
    async fn any_other_reader_failure_is_passed_on() {
        // Arrange
        struct Broken;
        impl LogReader for Broken {
            fn scan(&self, _from: Option<LogOffset>) -> LogStream {
                Box::pin(stream::iter([Err(ReaderError::MissingSegment {
                    sequence: 3,
                })]))
            }
            fn scan_through(&self, from: Option<LogOffset>, _through: LogOffset) -> LogStream {
                self.scan(from)
            }
        }

        // Act
        let result = scan(&Broken, &alice(), &groups()).await;

        // Assert
        assert!(matches!(result, Err(ScanError::Reader(_))));
    }

    #[test]
    fn every_error_says_what_failed() {
        // Arrange
        let errors = [
            ScanError::PlaintextLogPresent {
                event_id: "e1".into(),
            },
            ScanError::EventInSubjectGroup {
                event_id: "e1".into(),
            },
            ScanError::NothingToDelete,
        ];

        // Act
        let texts: Vec<String> = errors.iter().map(ToString::to_string).collect();

        // Assert
        assert!(texts[0].contains("encrypt-log"));
        assert!(texts[1].contains("delete the subject"));
        assert!(texts[2].contains("nothing"));
    }
}
