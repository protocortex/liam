// SPDX-License-Identifier: Apache-2.0

use futures_util::StreamExt;

use super::*;
use crate::crypto::TestMaster;
use crate::event::{
    DerivedArtifact, EdgeRow, FactRowEffect, NodeRow, TierName, TombstoneTable, TombstoneTarget,
    CURRENT_SCHEMA_VERSION,
};
use crate::reader::SequentialScanReader;
use crate::wal::{WalConfig, WalWriter};
use crate::LogWriter;

const SCOPE: &str = "work";

fn registry(dir: &std::path::Path) -> Arc<KeyRegistry> {
    Arc::new(KeyRegistry::open(dir, Arc::new(TestMaster::new(3))).expect("registry"))
}

struct Fixture {
    _dir: tempfile::TempDir,
    registry: Arc<KeyRegistry>,
    sealer: Sealer,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = registry(dir.path());
    let sealer = Sealer::new(Arc::clone(&registry));
    Fixture {
        _dir: dir,
        registry,
        sealer,
    }
}

fn node(id: &str, subject: Option<&str>) -> NodeRow {
    NodeRow {
        id: id.into(),
        kind: "fact".into(),
        label: format!("label {id}"),
        content: format!("secret content of {id}"),
        producer: "agent-a".into(),
        attributes: "{}".into(),
        scope: Some(SCOPE.into()),
        subject: subject.map(Into::into),
        confidence: 0.5,
        valid_from: 1,
        valid_from_supplied: true,
        valid_until: 9,
        tx_from: 2,
        tx_to: 9,
        derivation: None,
        basis: Vec::new(),
    }
}

fn edge(id: &str, src: &str, dst: &str) -> EdgeRow {
    EdgeRow {
        id: id.into(),
        src: src.into(),
        dst: dst.into(),
        edge_type: "relates_to".into(),
        attributes: "{}".into(),
        tx_from: 2,
        tx_to: 9,
    }
}

fn event(id: &str, payload: LogPayload) -> LogEvent {
    LogEvent {
        event_id: id.into(),
        content_hash: [7; 32],
        source: "agent-a".into(),
        trust_score: 0.9,
        observed_at: 1,
        ingested_at: 2,
        encryption_key_id: None,
        schema_version: CURRENT_SCHEMA_VERSION,
        payload,
    }
}

fn episode(id: &str) -> LogEvent {
    event(
        id,
        LogPayload::EpisodeBatch(vec![
            RowEffect::Node(node("a1", Some("Alice"))),
            RowEffect::Node(node("a2", Some("alice"))),
            RowEffect::Node(node("b1", Some("Bob"))),
            RowEffect::Node(node("p1", None)),
        ]),
    )
}

fn sealed(event: &LogEvent) -> &SealedRecord {
    match &event.payload {
        LogPayload::Sealed(record) => record,
        other => panic!("expected a sealed payload, got {other:?}"),
    }
}

fn key_of<'a>(record: &'a SealedRecord, row_id: &str) -> &'a str {
    &record
        .rows
        .iter()
        .find(|row| row.row_id == row_id)
        .expect("row")
        .key_id
}

fn fact_write() -> LogPayload {
    LogPayload::FactWrite {
        at: 5,
        closed_version_ids: Vec::new(),
        inserted: vec![FactRowEffect {
            id: "fact-1".into(),
            lineage_id: "lineage-1".into(),
            subject: "alice".into(),
            predicate: "likes".into(),
            object: "tea".into(),
            object_key: "tea".into(),
            scope: SCOPE.into(),
            evidence_at: 1,
            valid_from: 1,
            valid_until: 9,
            tx_from: 2,
            tx_to: 9,
            closed_by: String::new(),
            confidence: 0.8,
            trust_score: 0.9,
            extractor_id: "x".into(),
            extractor_version: "1".into(),
        }],
        source_links: vec![("lineage-1".into(), "event-1".into())],
        suppressions: Vec::new(),
    }
}

fn artifact() -> DerivedArtifact {
    DerivedArtifact {
        artifact_id: "art-1".into(),
        kind: "summary".into(),
        level: 1,
        scope: SCOPE.into(),
        text: "secret summary".into(),
        child_ids: vec!["a1".into()],
        source_event_ids: Vec::new(),
        events_fed: 1,
        derivation: "inferred".into(),
        model_id: "m".into(),
        prompt_version: "1".into(),
        closes: Vec::new(),
        valid_min: 1,
        valid_max: 9,
        at: 3,
    }
}

fn opened(fixture: &Fixture, event: &LogEvent) -> OpenedRecord {
    open_record(&fixture.registry, event).expect("open")
}

#[test]
fn an_episode_seals_into_one_record_with_a_row_per_effect_under_its_own_group() {
    // Arrange
    let fixture = fixture();
    let plain = episode("e1");

    // Act
    let event = fixture
        .sealer
        .seal(&plain, &KeyHints::default())
        .expect("seal");
    let record = sealed(&event);

    // Assert
    assert_eq!(record.rows.len(), 4);
    assert_eq!(
        key_of(record, "a1"),
        key_of(record, "a2"),
        "Alice and alice share a group"
    );
    assert_ne!(key_of(record, "a1"), key_of(record, "b1"));
    assert_ne!(key_of(record, "b1"), key_of(record, "p1"));
    assert_eq!(
        event.encryption_key_id.as_deref(),
        Some(key_of(record, "a1"))
    );
    assert!(record
        .rows
        .iter()
        .all(|row| !row.body.windows(6).any(|w| w == b"secret")));
    assert_eq!(opened(&fixture, &event).event.payload, plain.payload);
}

#[test]
fn a_destroyed_subject_shreds_only_its_rows() {
    // Arrange
    let fixture = fixture();
    let event = fixture
        .sealer
        .seal(&episode("e1"), &KeyHints::default())
        .unwrap();
    let bob = key_of(sealed(&event), "b1").to_owned();
    fixture.registry.destroy(&bob).unwrap();

    // Act
    let opened = opened(&fixture, &event);

    // Assert
    let LogPayload::EpisodeBatch(effects) = &opened.event.payload else {
        panic!("expected an episode");
    };
    assert_eq!(effects.len(), 3);
    assert_eq!(opened.shredded, vec![bob]);
}

#[test]
fn a_record_whose_rows_are_all_shredded_is_an_empty_episode_with_the_keys_named() {
    // Arrange
    let fixture = fixture();
    let plain = event("e1", LogPayload::NodeWrite(node("a1", Some("alice"))));
    let event = fixture.sealer.seal(&plain, &KeyHints::default()).unwrap();
    let key = key_of(sealed(&event), "a1").to_owned();
    fixture.registry.destroy(&key).unwrap();

    // Act
    let opened = opened(&fixture, &event);

    // Assert
    assert_eq!(opened.event.payload, LogPayload::EpisodeBatch(Vec::new()));
    assert_eq!(opened.shredded, vec![key]);
}

#[test]
fn a_row_under_a_destroying_key_reads_as_shredded() {
    // Arrange
    let fixture = fixture();
    let event = fixture
        .sealer
        .seal(&episode("e1"), &KeyHints::default())
        .unwrap();
    let alice = key_of(sealed(&event), "a1").to_owned();
    fixture.registry.begin_destroy(&alice).unwrap();

    // Act
    let opened = opened(&fixture, &event);

    // Assert
    assert_eq!(opened.shredded, vec![alice]);
}

#[test]
fn sealing_into_a_destroying_group_fails_and_appends_nothing() {
    // Arrange
    let fixture = fixture();
    let first = fixture
        .sealer
        .seal(
            &event("e1", LogPayload::NodeWrite(node("a1", Some("alice")))),
            &KeyHints::default(),
        )
        .unwrap();
    fixture
        .registry
        .begin_destroy(key_of(sealed(&first), "a1"))
        .unwrap();
    let mut appended = false;

    // Act
    let result = fixture.sealer.seal_then(
        &event("e2", LogPayload::NodeWrite(node("a2", Some("Alice")))),
        &KeyHints::default(),
        |_| appended = true,
    );

    // Assert
    assert!(matches!(
        result,
        Err(SealError::Crypto(CryptoError::KeyGroupLocked { .. }))
    ));
    assert!(!appended);
}

#[test]
fn header_hashes_follow_the_key_group() {
    // Arrange
    let fixture = fixture();
    let seal = |id: &str, subject: Option<&str>| {
        fixture
            .sealer
            .seal(
                &event(id, LogPayload::NodeWrite(node("same", subject))),
                &KeyHints::default(),
            )
            .unwrap()
    };

    // Act
    let first = seal("e1", Some("alice"));
    let again = seal("e2", Some("alice"));
    let other = seal("e3", Some("bob"));

    // Assert
    assert_eq!(first.content_hash, again.content_hash);
    assert_ne!(first.content_hash, other.content_hash);
    assert_ne!(first.content_hash, [7; 32]);
}

#[test]
fn an_event_with_nothing_to_hash_has_a_zero_hash_and_a_duplicate_pointer_is_zeroed() {
    // Arrange
    let fixture = fixture();
    let feedback = event(
        "e1",
        LogPayload::Feedback {
            lineage_key: "n:1".into(),
            correct: true,
            producer: "p".into(),
            at: 1,
        },
    );
    let duplicate = event(
        "e2",
        LogPayload::DuplicateOf {
            first_event_id: "e1".into(),
        },
    );

    // Act
    let sealed_feedback = fixture
        .sealer
        .seal(&feedback, &KeyHints::default())
        .unwrap();
    let plain_duplicate = fixture
        .sealer
        .seal(&duplicate, &KeyHints::default())
        .unwrap();

    // Assert
    assert_eq!(sealed_feedback.content_hash, [0; 32]);
    assert_eq!(plain_duplicate.content_hash, [0; 32]);
    assert_eq!(plain_duplicate.payload, duplicate.payload);
}

#[test]
fn a_changed_header_fails_authentication() {
    // Arrange
    let fixture = fixture();
    let mut event = fixture
        .sealer
        .seal(&episode("e1"), &KeyHints::default())
        .unwrap();
    event.source = "agent-b".into();

    // Act
    let result = open_record(&fixture.registry, &event);

    // Assert
    assert!(matches!(result, Err(OpenError::Authentication { .. })));
}

#[test]
fn a_row_moved_to_another_record_or_swapped_within_one_fails_authentication() {
    // Arrange
    let fixture = fixture();
    let first = fixture
        .sealer
        .seal(&episode("e1"), &KeyHints::default())
        .unwrap();
    let mut second = fixture
        .sealer
        .seal(&episode("e2"), &KeyHints::default())
        .unwrap();
    let mut moved = first.clone();
    let mut swapped = first.clone();
    let LogPayload::Sealed(from_second) = &mut second.payload else {
        unreachable!()
    };
    let LogPayload::Sealed(target) = &mut moved.payload else {
        unreachable!()
    };
    target.rows[0].nonce = from_second.rows[0].nonce;
    target.rows[0].body = from_second.rows[0].body.clone();
    let LogPayload::Sealed(inside) = &mut swapped.payload else {
        unreachable!()
    };
    let (a, b) = (inside.rows[0].clone(), inside.rows[1].clone());
    inside.rows[0].nonce = b.nonce;
    inside.rows[0].body = b.body;
    inside.rows[1].nonce = a.nonce;
    inside.rows[1].body = a.body;

    // Act
    let moved_result = open_record(&fixture.registry, &moved);
    let swapped_result = open_record(&fixture.registry, &swapped);

    // Assert
    assert!(matches!(
        moved_result,
        Err(OpenError::Authentication { .. })
    ));
    assert!(matches!(
        swapped_result,
        Err(OpenError::Authentication { .. })
    ));
}

#[test]
fn plain_events_open_unchanged() {
    // Arrange
    let fixture = fixture();
    let events = [
        event(
            "e1",
            LogPayload::Tombstone(vec![TombstoneTarget {
                table: TombstoneTable::Nodes,
                id: "n".into(),
            }]),
        ),
        event(
            "e2",
            LogPayload::Voided {
                target_event_id: "e1".into(),
            },
        ),
        event("e3", LogPayload::RunRolledBack { run_id: "r".into() }),
        event(
            "e4",
            LogPayload::DerivedArtifactClosed {
                artifact_id: "a".into(),
                at: 1,
            },
        ),
    ];

    for plain in events {
        // Act
        let sealed = fixture.sealer.seal(&plain, &KeyHints::default()).unwrap();
        let opened = opened(&fixture, &sealed);

        // Assert
        assert_eq!(sealed, plain);
        assert_eq!(opened.event, plain);
        assert!(opened.shredded.is_empty());
    }
}

#[test]
fn an_edge_is_sealed_under_its_source_group_or_a_hint_or_the_event_key() {
    // Arrange
    let fixture = fixture();
    let batch = event(
        "e1",
        LogPayload::EpisodeBatch(vec![
            RowEffect::Node(node("a1", Some("alice"))),
            RowEffect::Node(node("b1", Some("bob"))),
            RowEffect::Edge(edge("same-event", "a1", "b1")),
            RowEffect::Edge(edge("hinted", "elsewhere", "b1")),
            RowEffect::Edge(edge("unknown", "nowhere", "b1")),
        ]),
    );
    let hinted = fixture
        .registry
        .key_for_subject(Some(SCOPE), "carol")
        .unwrap();
    let hints = KeyHints {
        edge_sources: [("elsewhere".to_owned(), hinted.clone())].into(),
        ..KeyHints::default()
    };

    // Act
    let event = fixture.sealer.seal(&batch, &hints).unwrap();
    let record = sealed(&event);

    // Assert
    assert_eq!(key_of(record, "same-event"), key_of(record, "a1"));
    assert_eq!(key_of(record, "hinted"), hinted);
    assert_ne!(key_of(record, "unknown"), key_of(record, "a1"));
    assert_ne!(key_of(record, "unknown"), key_of(record, "b1"));
    assert_eq!(opened(&fixture, &event).event.payload, batch.payload);
}

#[test]
fn nonces_and_ciphertexts_never_repeat() {
    // Arrange
    let fixture = fixture();
    let plain = event(
        "e1",
        LogPayload::EpisodeBatch(vec![
            RowEffect::Node(node("a1", Some("alice"))),
            RowEffect::Node(node("a1", Some("alice"))),
            RowEffect::Node(node("a1", Some("alice"))),
        ]),
    );

    // Act
    let first = fixture.sealer.seal(&plain, &KeyHints::default()).unwrap();
    let second = fixture.sealer.seal(&plain, &KeyHints::default()).unwrap();

    // Assert
    let nonces: std::collections::HashSet<_> = sealed(&first)
        .rows
        .iter()
        .chain(&sealed(&second).rows)
        .map(|row| row.nonce)
        .collect();
    assert_eq!(nonces.len(), 6);
    assert_ne!(sealed(&first).rows[0].body, sealed(&second).rows[0].body);
}

#[test]
fn a_record_with_no_rows_stays_plain() {
    // Arrange
    let fixture = fixture();
    let empty_episode = event("e1", LogPayload::EpisodeBatch(Vec::new()));
    let empty_usage = event(
        "e2",
        LogPayload::AccessBatch {
            day: 4,
            entries: Vec::new(),
        },
    );

    // Act
    let first = fixture
        .sealer
        .seal(&empty_episode, &KeyHints::default())
        .unwrap();
    let second = fixture
        .sealer
        .seal(&empty_usage, &KeyHints::default())
        .unwrap();

    // Assert
    assert_eq!(first, empty_episode);
    assert_eq!(second, empty_usage);
    assert!(opened(&fixture, &first).shredded.is_empty());
}

#[test]
fn facts_and_artifacts_have_a_key_of_their_own() {
    // Arrange
    let fixture = fixture();
    let subject = fixture
        .registry
        .key_for_subject(Some(SCOPE), "alice")
        .unwrap();
    let fact = fixture
        .sealer
        .seal(&event("e1", fact_write()), &KeyHints::default())
        .unwrap();
    let derived = fixture
        .sealer
        .seal(
            &event("e2", LogPayload::DerivedArtifact(artifact())),
            &KeyHints::default(),
        )
        .unwrap();
    fixture.registry.destroy(&subject).unwrap();

    // Act
    let still_fact = opened(&fixture, &fact);
    let still_artifact = opened(&fixture, &derived);
    fixture
        .registry
        .destroy(&sealed(&fact).rows[0].key_id)
        .unwrap();
    let gone = opened(&fixture, &fact);

    // Assert
    assert_eq!(still_fact.event.payload, fact_write());
    assert_eq!(
        still_artifact.event.payload,
        LogPayload::DerivedArtifact(artifact())
    );
    assert_eq!(gone.event.payload, LogPayload::EpisodeBatch(Vec::new()));
    assert_eq!(gone.shredded.len(), 1);
    assert_eq!(sealed(&fact).rows[0].row_id, "fact-1");
}

#[test]
fn usage_keeps_its_day_when_every_entry_is_shredded() {
    // Arrange
    let fixture = fixture();
    let owner = fixture
        .registry
        .key_for_subject(Some(SCOPE), "alice")
        .unwrap();
    let hints = KeyHints {
        lineages: [("e:work:alice".to_owned(), owner.clone())].into(),
        ..KeyHints::default()
    };
    let usage = event(
        "e1",
        LogPayload::AccessBatch {
            day: 19_000,
            entries: vec![
                UsageRow {
                    lineage_key: "e:work:alice".into(),
                    hits: 2,
                    last_access_at: 5,
                },
                UsageRow {
                    lineage_key: "n:free".into(),
                    hits: 1,
                    last_access_at: 6,
                },
            ],
        },
    );
    let event = fixture.sealer.seal(&usage, &hints).unwrap();
    let before = opened(&fixture, &event);
    fixture.registry.destroy(&owner).unwrap();

    // Act
    let after = opened(&fixture, &event);

    // Assert
    assert_eq!(before.event.payload, usage.payload);
    assert_eq!(
        after.event.payload,
        LogPayload::AccessBatch {
            day: 19_000,
            entries: vec![UsageRow {
                lineage_key: "n:free".into(),
                hits: 1,
                last_access_at: 6
            }],
        }
    );
    assert!(sealed(&event).rows.iter().all(|row| row.row_id.is_empty()));
}

#[test]
fn feedback_and_tier_moves_follow_their_lineage_hint() {
    // Arrange
    let fixture = fixture();
    let owner = fixture
        .registry
        .key_for_subject(Some(SCOPE), "alice")
        .unwrap();
    let hints = KeyHints {
        lineages: [("e:work:alice".to_owned(), owner.clone())].into(),
        ..KeyHints::default()
    };
    let feedback = event(
        "e1",
        LogPayload::Feedback {
            lineage_key: "e:work:alice".into(),
            correct: true,
            producer: "p".into(),
            at: 1,
        },
    );
    let tier = event(
        "e2",
        LogPayload::TierMove {
            lineage_key: "e:work:alice".into(),
            to: TierName::Cold,
            run_id: "r".into(),
            reason: "idle".into(),
            changed_at: 3,
            residency_until: None,
        },
    );

    // Act
    let sealed_feedback = fixture.sealer.seal(&feedback, &hints).unwrap();
    let sealed_tier = fixture.sealer.seal(&tier, &hints).unwrap();

    // Assert
    assert_eq!(sealed(&sealed_feedback).rows[0].key_id, owner);
    assert_eq!(sealed(&sealed_tier).rows[0].key_id, owner);
    assert_eq!(
        opened(&fixture, &sealed_feedback).event.payload,
        feedback.payload
    );
    assert_eq!(opened(&fixture, &sealed_tier).event.payload, tier.payload);
}

#[test]
fn a_key_the_registry_does_not_know_is_an_error_not_a_shred() {
    // Arrange
    let fixture = fixture();
    let event = fixture
        .sealer
        .seal(&episode("e1"), &KeyHints::default())
        .unwrap();
    let other_dir = tempfile::tempdir().unwrap();
    let lost = registry(other_dir.path());

    // Act
    let result = open_record(&lost, &event);

    // Assert
    assert!(matches!(result, Err(OpenError::KeyMissing { .. })));
}

#[test]
fn a_rows_shape_that_does_not_match_is_refused() {
    // Arrange
    let bodies = vec![RowBody::Usage(UsageRow {
        lineage_key: "n:1".into(),
        hits: 1,
        last_access_at: 1,
    })];

    // Act
    let payload = assemble(&SealedShape::Episode, bodies);

    // Assert
    assert_eq!(payload, None);
}

fn writer(dir: &std::path::Path) -> WalWriter<crate::wal::SystemClock> {
    WalWriter::open_with_system_clock(
        dir,
        WalConfig {
            segment_max_bytes: 1 << 20,
            rotate_interval_secs: 3_600,
        },
    )
    .expect("writer")
}

#[tokio::test]
async fn the_opening_reader_returns_plain_rows_and_the_shredded_keys() {
    // Arrange
    let keys_dir = tempfile::tempdir().unwrap();
    let registry = registry(keys_dir.path());
    let sealer = Sealer::new(Arc::clone(&registry));
    let log_dir = tempfile::tempdir().unwrap();
    let mut log = writer(log_dir.path());
    let plain = episode("e1");
    let sealed_event = sealer.seal(&plain, &KeyHints::default()).unwrap();
    let bob = key_of(sealed(&sealed_event), "b1").to_owned();
    log.append(&sealed_event).unwrap();
    log.append(&event(
        "e2",
        LogPayload::Voided {
            target_event_id: "e1".into(),
        },
    ))
    .unwrap();
    registry.destroy(&bob).unwrap();
    let reader = OpeningReader::new(
        SequentialScanReader::local(log_dir.path()).unwrap(),
        registry,
    );

    // Act
    let records: Vec<_> = reader.scan(None).collect().await;
    let through: Vec<_> = reader
        .scan_through(
            None,
            LogOffset {
                segment: 0,
                index: 0,
            },
        )
        .collect()
        .await;

    // Assert
    let first = records[0].as_ref().unwrap();
    let LogPayload::EpisodeBatch(effects) = &first.event.payload else {
        panic!("expected an episode")
    };
    assert_eq!(effects.len(), 3);
    assert_eq!(first.shredded, vec![bob]);
    assert!(records[1].as_ref().unwrap().shredded.is_empty());
    assert_eq!(through.len(), 1);
}

#[tokio::test]
async fn the_opening_reader_stops_at_a_missing_key() {
    // Arrange
    let keys_dir = tempfile::tempdir().unwrap();
    let sealer = Sealer::new(registry(keys_dir.path()));
    let log_dir = tempfile::tempdir().unwrap();
    let mut log = writer(log_dir.path());
    log.append(&sealer.seal(&episode("e1"), &KeyHints::default()).unwrap())
        .unwrap();
    log.append(&event(
        "e2",
        LogPayload::Voided {
            target_event_id: "e1".into(),
        },
    ))
    .unwrap();
    let lost_dir = tempfile::tempdir().unwrap();
    let reader = OpeningReader::new(
        SequentialScanReader::local(log_dir.path()).unwrap(),
        registry(lost_dir.path()),
    );

    // Act
    let items: Vec<_> = reader.scan(None).collect().await;

    // Assert
    assert_eq!(items.len(), 1, "nothing follows the error");
    assert!(matches!(items[0], Err(ReaderError::KeyMissing { .. })));
}

#[tokio::test]
async fn the_opening_reader_reports_a_tampered_record_with_its_event_id() {
    // Arrange
    let keys_dir = tempfile::tempdir().unwrap();
    let registry = registry(keys_dir.path());
    let sealer = Sealer::new(Arc::clone(&registry));
    let log_dir = tempfile::tempdir().unwrap();
    let mut log = writer(log_dir.path());
    let mut tampered = sealer.seal(&episode("e1"), &KeyHints::default()).unwrap();
    tampered.source = "agent-b".into();
    log.append(&tampered).unwrap();
    let reader = OpeningReader::new(
        SequentialScanReader::local(log_dir.path()).unwrap(),
        registry,
    );

    // Act
    let items: Vec<_> = reader.scan(None).collect().await;

    // Assert
    assert!(matches!(&items[0], Err(ReaderError::Sealed { event_id, .. }) if event_id == "e1"));
}
