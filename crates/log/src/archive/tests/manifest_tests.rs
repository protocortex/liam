// SPDX-License-Identifier: Apache-2.0
//! A manifest that cannot be trusted, and an archived copy that disagrees with
//! the original it was made from.

use super::*;

const UNSUPPORTED_VERSION: u32 = MANIFEST_VERSION + 1;
const SHIFTED_EVENTS: usize = 100;
const LISTED_EVENTS: u64 = 2;

fn write_manifest_bytes(env: &Env, bytes: &[u8]) {
    fs::write(env.log().join(MANIFEST_FILE), bytes).unwrap();
}

fn unreadable_reason(result: Result<Vec<ArchivedPartition>, ArchiveError>) -> String {
    match result {
        Err(ArchiveError::ManifestUnreadable { reason, .. }) => reason,
        other => panic!("expected ManifestUnreadable, got {other:?}"),
    }
}

#[test]
fn a_log_without_a_manifest_has_archived_nothing() {
    // Arrange
    let env = Env::new();

    // Act
    let manifest = read_manifest(env.log()).unwrap();

    // Assert
    assert!(manifest.is_empty());
}

#[test]
fn a_manifest_of_garbage_bytes_is_unreadable() {
    // Arrange
    let env = Env::new();
    write_manifest_bytes(&env, &[0xFF; 3]);

    // Act
    let result = read_manifest(env.log());

    // Assert
    assert!(!unreadable_reason(result).is_empty());
}

#[test]
fn a_manifest_of_another_version_says_which_versions_it_found_and_reads() {
    // Arrange
    let env = Env::new();
    let other_version = postcard::to_stdvec(&ManifestFile {
        version: UNSUPPORTED_VERSION,
        partitions: Vec::new(),
    })
    .unwrap();
    write_manifest_bytes(&env, &other_version);

    // Act
    let reason = unreadable_reason(read_manifest(env.log()));

    // Assert
    assert_eq!(
        reason,
        format!(
            "it is version {UNSUPPORTED_VERSION}, but this build reads version {MANIFEST_VERSION}"
        )
    );
}

#[test]
fn a_manifest_that_cannot_be_opened_as_a_file_is_an_io_error() {
    // Arrange
    let env = Env::new();
    fs::create_dir(env.log().join(MANIFEST_FILE)).unwrap();

    // Act
    let result = read_manifest(env.log());

    // Assert
    assert!(matches!(result, Err(ArchiveError::Io(_))), "{result:?}");
}

#[test]
fn a_partition_without_events_has_a_first_offset_and_no_last() {
    // Arrange
    let empty = ArchivedPartition {
        sequence: 4,
        events: 0,
        path: PathBuf::from("archive/4.parquet"),
    };
    let two = ArchivedPartition {
        events: LISTED_EVENTS,
        ..empty.clone()
    };

    // Act / Assert
    assert_eq!(empty.first(), at(4, 0));
    assert_eq!(empty.last(), None);
    assert_eq!(two.last(), Some(at(4, 1)));
}

#[tokio::test]
async fn an_archived_copy_with_other_events_than_the_kept_original_stops_the_finish() {
    // Arrange
    let env = Env::new();
    env.write(&events_by_age()).await;
    env.run_with(&StopAt(ArchiveStep::Recorded))
        .expect_err("the fault stops it after recording partition 0");
    let other = Env::new();
    let shifted: Vec<LogEvent> = events_by_age()
        .iter()
        .enumerate()
        .map(|(index, aged_event)| LogEvent {
            ingested_at: aged_event.ingested_at,
            ..event(index + SHIFTED_EVENTS)
        })
        .collect();
    other.write(&shifted).await;
    other.run().unwrap();
    fs::copy(
        parquet_path(other.archive(), 0),
        parquet_path(env.archive(), 0),
    )
    .unwrap();

    // Act
    let result = env.run();

    // Assert
    match result {
        Err(ArchiveError::CopyMismatch { sequence, detail }) => {
            assert_eq!(sequence, 0);
            assert_eq!(detail, "the events read back differ");
        }
        other => panic!("expected CopyMismatch, got {other:?}"),
    }
    assert!(has_parquet(env.log(), 0), "the original stays");
}
