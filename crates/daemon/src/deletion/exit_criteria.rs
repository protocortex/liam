// SPDX-License-Identifier: Apache-2.0
//! The deletion exit criterion over a real store and a real log: a canary
//! planted in every store and file class is gone after a delete, present in a
//! control before it, and any residual left on purpose is caught by the one
//! detector that watches its class.

mod detect;
mod fixture;

use std::collections::BTreeSet;
use std::io::Write;

use liam_log::crypto::{KeyLookup, KeyState};
use liam_store::{LineageTarget, NewNode, Residual};

use self::detect::{lineage_scan, log_view, Deleted, FileClass};
use self::fixture::{
    alice, files_under, Fixture, ALICE_REQUEST, BOB_CANARY, BOB_REQUEST, CANARY, EDGE_CANARY,
    HOSTED_BATCH, HOSTED_PROVIDER, ROLLED_BACK_CANARY,
};
use super::batch_lineage::{BatchCanceller, ProviderCopies};
use super::batch_store::{batch_store_path_for, BatchState, BatchStore};
use super::job::{DeleteJob, JobStore};
use super::registry::LineageRegistry;
use super::scan::ScanError;
use super::send_ledger::{send_ledger_path_for, SendLedger};
use super::service::{CrashPoint, DeleteError, RunOutcome};
use super::target::DeleteTarget;
use super::JobState;

/// Text in a node nobody owns, deleted by its event.
const FREE_CANARY: &str = "pwhe6120freecanary";
const LEFTOVER_DIR: &str = "restore-leftover";
const LEFTOVER_FILE: &str = "snapshot.db";

fn job_named(fixture: &Fixture, id: &str) -> DeleteJob {
    JobStore::new(super::job_file_path_for(&fixture.database()))
        .load()
        .unwrap()
        .into_iter()
        .find(|job| job.id == id)
        .expect("the job is saved")
}

/// Requests and runs a delete of `target`, returning the finished job.
async fn delete(fixture: &Fixture, target: DeleteTarget) -> DeleteJob {
    let service = fixture.service();
    let preview = service.dry_run(&target).await.expect("the dry run");
    assert!(!preview.found.key_ids.is_empty(), "{preview}");
    let id = service.request(target).unwrap();
    assert_eq!(service.run(&id).await.unwrap(), RunOutcome::Done);
    job_named(fixture, &id)
}

fn deleted_by(fixture: &Fixture, job: &DeleteJob) -> Deleted {
    Deleted {
        events: job.event_ids.iter().cloned().collect(),
        nodes: job.node_ids.iter().cloned().collect(),
        lineages: BTreeSet::from([fixture.alice_lineage.as_str().to_owned()]),
    }
}

async fn insert_free(fixture: &Fixture, token: &str) -> liam_store::NodeId {
    let note = NewNode::now("fact", "free", format!("free {token}"));
    fixture.subjects.store.insert(note).await.unwrap()
}

#[tokio::test]
async fn the_canary_is_in_every_class_before_a_subject_delete_and_in_none_after() {
    // Arrange
    let fixture = Fixture::new().await;
    let before = log_view(&fixture).await;
    let naturally_holding = [
        FileClass::Database,
        FileClass::Snapshots,
        FileClass::Cache,
        FileClass::Manifest,
        FileClass::Cooldown,
        FileClass::HostedBatches,
    ];
    for class in naturally_holding {
        assert!(class.holds_needle(&fixture), "{class:?} control");
    }
    for class in FileClass::ALL {
        assert!(
            class.finds_the_needle_in_a_control_copy(&fixture),
            "{class:?} scan can fail"
        );
    }
    for token in [CANARY, EDGE_CANARY, ROLLED_BACK_CANARY, BOB_CANARY] {
        assert!(before.shows(token), "{token} readable in the log before");
    }

    // Act
    let job = delete(&fixture, alice()).await;

    // Assert
    let after = log_view(&fixture).await;
    assert_eq!(job.state, JobState::Done);
    for class in FileClass::ALL {
        assert!(!class.holds_needle(&fixture), "{class:?} still holds it");
    }
    for token in [CANARY, EDGE_CANARY, ROLLED_BACK_CANARY] {
        assert!(!after.shows(token), "{token} still opens in the log");
    }
    assert!(after.shredded > 0, "the log reports shredded rows");
    assert!(after.shows(BOB_CANARY), "bob's row opens");
    assert!(
        FileClass::Database.holds_token(&fixture, BOB_CANARY),
        "bob's note stays in the database"
    );
    for key in &job.key_ids {
        assert_eq!(fixture.subjects.keys.state(key), Some(KeyState::Destroyed));
        assert!(matches!(
            fixture.subjects.keys.open_key(key).unwrap(),
            KeyLookup::Destroyed
        ));
    }
}

#[tokio::test]
async fn nothing_in_the_database_refers_to_what_a_subject_delete_removed() {
    // Arrange
    let fixture = Fixture::new().await;

    // Act
    let job = delete(&fixture, alice()).await;
    let scan = lineage_scan(&fixture, &deleted_by(&fixture, &job)).await;

    // Assert
    assert!(scan.referencing.is_empty(), "{:?}", scan.referencing);
    assert!(scan.unclassified.is_empty(), "{:?}", scan.unclassified);
}

#[tokio::test]
async fn the_lineage_scan_finds_a_row_left_naming_a_deleted_id() {
    // Arrange
    let fixture = Fixture::new().await;
    let job = delete(&fixture, alice()).await;
    let deleted = deleted_by(&fixture, &job);
    let event = deleted.events.iter().next().unwrap().clone();
    let database = libsql::Builder::new_local(fixture.database().to_str().unwrap())
        .build()
        .await
        .unwrap();
    database
        .connect()
        .unwrap()
        .execute(
            "INSERT INTO log_quarantine (event_id, segment, seg_index, reason, at)
             VALUES (?1, 0, 0, 'left', 0)",
            [event],
        )
        .await
        .unwrap();

    // Act
    let scan = lineage_scan(&fixture, &deleted).await;

    // Assert
    assert_eq!(scan.referencing, ["log_quarantine.event_id: 1"]);
}

#[tokio::test]
async fn a_subjectless_event_is_deleted_alone_and_one_under_a_subject_is_refused() {
    // Arrange
    let fixture = Fixture::new().await;
    let free = insert_free(&fixture, FREE_CANARY).await;
    let free_event = fixture.origin_events(&free).await.remove(0);
    let alice_event = fixture.origin_events(&fixture.alice_note).await.remove(0);
    let refused_target = DeleteTarget::Event {
        event_id: alice_event,
    };
    let service = fixture.service();
    let refused_id = service.request(refused_target).unwrap();

    // Act
    let refused = service.run(&refused_id).await;
    let job = delete(
        &fixture,
        DeleteTarget::Event {
            event_id: free_event,
        },
    )
    .await;

    // Assert
    assert!(matches!(
        refused,
        Err(DeleteError::Refused(ScanError::EventInSubjectGroup { .. }))
    ));
    assert_eq!(job.state, JobState::Done);
    let view = log_view(&fixture).await;
    assert!(!view.shows(FREE_CANARY));
    for kept in [CANARY, BOB_CANARY] {
        assert!(view.shows(kept), "{kept} was not part of the target");
    }
    assert!(!FileClass::Database.holds_token(&fixture, FREE_CANARY));
}

#[tokio::test]
async fn deleting_the_source_of_an_edge_removes_its_body_and_deleting_the_destination_does_not() {
    // Arrange
    let source_deleted = Fixture::new().await;
    let destination_deleted = Fixture::new().await;
    let bob = DeleteTarget::Subject {
        subject: "bob".into(),
        scope: None,
        include_children: false,
    };

    // Act
    delete(&source_deleted, alice()).await;
    delete(&destination_deleted, bob).await;

    // Assert
    assert!(!log_view(&source_deleted).await.shows(EDGE_CANARY));
    assert!(
        log_view(&destination_deleted).await.shows(EDGE_CANARY),
        "the documented residual: the edge body is sealed under its source"
    );
}

#[tokio::test]
async fn a_residual_left_in_a_file_class_is_caught_by_that_classs_scan() {
    for class in FileClass::ALL {
        // Arrange
        let fixture = Fixture::new().await;
        delete(&fixture, alice()).await;
        assert!(!class.holds_needle(&fixture), "{class:?} starts clean");

        // Act
        plant_residual(&fixture, class).await;

        // Assert
        assert!(class.holds_needle(&fixture), "{class:?} residual missed");
    }
}

/// Puts the class's needle back where its scan looks.
async fn plant_residual(fixture: &Fixture, class: FileClass) {
    match class {
        FileClass::Database => {
            insert_free(fixture, CANARY).await;
        }
        FileClass::Snapshots => {
            insert_free(fixture, CANARY).await;
            fixture
                .subjects
                .store
                .snapshot(&fixture.snapshot_dir())
                .await
                .unwrap();
        }
        FileClass::Cache => {
            let cache = crate::extraction::cache::CacheStore::open(fixture.cache_file()).unwrap();
            cache
                .put(crate::extraction::cache::NewEntry {
                    key: "left",
                    response: CANARY,
                    model_id: "model",
                    model_sha256: "sha",
                    prompt_version: "prompt-1",
                    source_event_ids: &["left".to_owned()],
                })
                .unwrap();
        }
        other => {
            let needle = other.needle(fixture);
            let file = other.files(fixture).remove(0);
            let mut handle = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(file)
                .unwrap();
            handle.write_all(needle.as_bytes()).unwrap();
        }
    }
}

#[tokio::test]
async fn a_readable_row_left_in_the_log_is_caught_by_the_log_view() {
    // Arrange
    let fixture = Fixture::new().await;
    delete(&fixture, alice()).await;
    assert!(!log_view(&fixture).await.shows(CANARY));

    // Act
    insert_free(&fixture, CANARY).await;

    // Assert
    assert!(log_view(&fixture).await.shows(CANARY));
}

#[tokio::test]
async fn a_store_left_unpurged_is_reported_by_verify_all_under_its_own_name() {
    // Arrange
    let probe = Fixture::new().await;
    let stores = registry_of(&probe)
        .find_all(&target_of(&probe).await)
        .await
        .unwrap();
    let names: Vec<String> = stores.into_iter().map(|(name, _)| name).collect();
    assert!(names.len() >= 5, "{names:?}");
    drop(probe);

    for (skipped, name) in names.iter().enumerate() {
        let fixture = Fixture::new().await;
        let registry = registry_of(&fixture);
        let target = target_of(&fixture).await;

        // Act
        for index in (0..registry.store_count()).filter(|index| *index != skipped) {
            registry.purge_at(index, &target).await.unwrap();
        }
        let left = stores_of(&registry.verify_all(&target).await.unwrap());

        // Assert
        assert!(left.contains(name), "{name} not reported, got {left:?}");
        if name != "graph" {
            assert_eq!(left, BTreeSet::from([name.clone()]), "{name}");
        }
    }
}

fn stores_of(residuals: &[Residual]) -> BTreeSet<String> {
    residuals
        .iter()
        .map(|residual| residual.store.clone())
        .collect()
}

fn registry_of(fixture: &Fixture) -> LineageRegistry {
    let states: Vec<std::sync::Arc<dyn super::memory::NodeState>> = vec![fixture.usage.clone()];
    LineageRegistry::for_database(
        &fixture.database(),
        &fixture.snapshot_dir(),
        std::sync::Arc::clone(&fixture.subjects.store),
        states,
        std::sync::Arc::new(super::batch_lineage::NoProviderDelete),
    )
    .unwrap()
}

/// What a job for alice resolves to, saved just before it destroys anything.
async fn target_of(fixture: &Fixture) -> LineageTarget {
    let service = fixture
        .service()
        .stopping_at(CrashPoint::AfterDeletionRecorded);
    let id = service.request(alice()).unwrap();
    service.run(&id).await.unwrap_err();
    let job = job_named(fixture, &id);
    fixture
        .subjects
        .store
        .resolve_target(&LineageTarget {
            job_id: job.id,
            event_ids: job.event_ids,
            key_ids: job.key_ids,
            ..LineageTarget::default()
        })
        .await
        .unwrap()
}

#[tokio::test]
async fn a_delete_after_a_rolled_back_run_finishes_and_no_canary_comes_back() {
    // Arrange
    let fixture = Fixture::new().await;
    let before = log_view(&fixture).await;
    assert!(before.shows(ROLLED_BACK_CANARY));

    // Act
    delete(&fixture, alice()).await;
    let after = log_view(&fixture).await;

    // Assert
    assert!(!after.shows(ROLLED_BACK_CANARY));
    assert!(!after.shows(CANARY));
}

#[tokio::test]
async fn a_snapshot_left_by_an_interrupted_restore_is_zeroed_and_gone() {
    // Arrange
    let fixture = Fixture::new().await;
    let leftover = fixture.snapshot_dir().join(LEFTOVER_DIR);
    std::fs::create_dir_all(&leftover).unwrap();
    std::fs::copy(fixture.database(), leftover.join(LEFTOVER_FILE)).unwrap();
    assert!(!files_under(&fixture.snapshot_dir()).is_empty());

    // Act
    delete(&fixture, alice()).await;

    // Assert
    assert!(files_under(&fixture.snapshot_dir()).is_empty());
    assert!(!leftover.join(LEFTOVER_FILE).exists());
}

/// What is left of a fixture once the store and its locks are released, so
/// another store can open the same log.
struct Released {
    dir: tempfile::TempDir,
    keys: std::sync::Arc<liam_log::crypto::KeyRegistry>,
    config: crate::config::Config,
    surviving: Vec<[u8; 32]>,
}

async fn release(fixture: Fixture) -> Released {
    let surviving = fixture
        .subjects
        .store
        .projection_hash_multiset()
        .await
        .unwrap();
    let Fixture { dir, subjects, .. } = fixture;
    Released {
        dir,
        keys: subjects.keys,
        config: subjects.config,
        surviving,
    }
}

/// Opens a new database over the released log, which replays it from the start.
async fn rebuilt(released: &Released) -> (std::path::PathBuf, Vec<[u8; 32]>) {
    let database = released.dir.path().join("rebuilt.db");
    let started = crate::log_startup::open_store_with_log(
        &released.config,
        &database,
        &crate::log_startup::test_support::log_dir_in(released.dir.path()),
        crate::log_startup::test_support::embedder(),
        &crate::shutdown::Shutdown::default(),
    )
    .await
    .expect("the log replays");
    let hashes = started.store.projection_hash_multiset().await.unwrap();
    (database, hashes)
}

fn holds_in(database: &std::path::Path, token: &str) -> bool {
    let mut wal = database.as_os_str().to_owned();
    wal.push("-wal");
    [database.to_path_buf(), std::path::PathBuf::from(wal)]
        .iter()
        .any(|file| std::fs::read(file).is_ok_and(|b| detect::contains(&b, token.as_bytes())))
}

#[tokio::test]
async fn a_rebuild_after_the_delete_matches_the_surviving_store_and_brings_nothing_back() {
    // Arrange
    let fixture = Fixture::new().await;
    delete(&fixture, alice()).await;
    let released = release(fixture).await;

    // Act
    let (database, hashes) = rebuilt(&released).await;

    // Assert
    assert_eq!(hashes, released.surviving);
    assert!(!holds_in(&database, CANARY));
    assert!(holds_in(&database, BOB_CANARY));
}

#[tokio::test]
async fn a_rebuild_after_the_delete_and_an_archive_pass_reads_the_archive_and_matches() {
    // Arrange
    let fixture = Fixture::with_closed_segments().await;
    delete(&fixture, alice()).await;
    let archive = fixture.archive_dir();
    std::fs::create_dir_all(&archive).unwrap();
    let log_dir = fixture.log_dir();
    let released = release(fixture).await;
    wait_for_compaction(&log_dir).await;

    // Act
    let report = liam_log::archive::archive_partitions(
        &log_dir,
        &archive,
        liam_log::archive::ArchivePolicy {
            after_days: 0,
            zstd_level: 3,
        },
        i64::MAX,
        &liam_log::archive::NoFault,
    )
    .unwrap();
    let archived = detect::log_view_of(&log_dir, &released.keys).await;
    let (database, hashes) = rebuilt(&released).await;

    // Assert
    assert!(!report.archived.is_empty(), "{report:?}");
    assert!(!files_under(&archive).is_empty());
    assert!(!archived.shows(CANARY));
    assert!(
        archived.shows(BOB_CANARY),
        "the archive is read, not skipped"
    );
    assert_eq!(hashes, released.surviving);
    assert!(!holds_in(&database, CANARY));
}

/// Waits for the background compactor to turn every closed segment into a
/// Parquet file; only the open segment keeps its write-ahead log.
async fn wait_for_compaction(log_dir: &std::path::Path) {
    const POLL: std::time::Duration = std::time::Duration::from_millis(25);
    const POLLS: usize = 400;
    let wals = || {
        files_under(log_dir)
            .iter()
            .filter(|file| file.extension().is_some_and(|ext| ext == "wal"))
            .count()
    };
    for _ in 0..POLLS {
        if wals() <= 1 {
            return;
        }
        tokio::time::sleep(POLL).await;
    }
    panic!("closed segments were not compacted");
}

/// Cancels every batch it is asked about and reports whether the provider
/// still holds a copy, remembering the calls.
struct Cancelling {
    copies: ProviderCopies,
    calls: std::sync::Mutex<Vec<(String, String)>>,
}

#[async_trait::async_trait]
impl BatchCanceller for Cancelling {
    async fn cancel(&self, provider: &str, batch_id: &str) -> anyhow::Result<ProviderCopies> {
        self.calls
            .lock()
            .unwrap()
            .push((provider.to_owned(), batch_id.to_owned()));
        Ok(self.copies)
    }
}

#[tokio::test]
async fn a_pending_hosted_batch_is_cancelled_and_nothing_of_alice_is_left_in_its_files() {
    // Arrange
    let fixture = Fixture::new().await;
    let canceller = std::sync::Arc::new(Cancelling {
        copies: ProviderCopies::Deleted,
        calls: std::sync::Mutex::default(),
    });
    let service = fixture.service_with(Some(canceller.clone()));
    let target = alice();
    let held_before = FileClass::HostedBatches.holds_needle(&fixture);

    // Act
    let id = service.request(target).unwrap();
    let outcome = service.run(&id).await.unwrap();

    // Assert
    let job = job_named(&fixture, &id);
    let database = fixture.database();
    let table = BatchStore::new(batch_store_path_for(&database))
        .load()
        .unwrap();
    let ledger = SendLedger::new(send_ledger_path_for(&database))
        .load()
        .unwrap();
    assert!(held_before, "the batch files named alice before");
    assert_eq!(outcome, RunOutcome::Done);
    assert!(!FileClass::HostedBatches.holds_needle(&fixture));
    assert!(!FileClass::Cache.holds_needle(&fixture));
    assert_eq!(
        *canceller.calls.lock().unwrap(),
        [(HOSTED_PROVIDER.to_owned(), HOSTED_BATCH.to_owned())]
    );
    assert!(job.notices.is_empty(), "{:?}", job.notices);
    assert_eq!(table.len(), 1);
    assert_eq!(table[0].cache_key, BOB_REQUEST);
    assert_eq!(
        table[0].state,
        BatchState::Cancelled,
        "bob's request is sent again"
    );
    assert_eq!(ledger.len(), 1);
    assert_eq!(ledger[0].cache_key, BOB_REQUEST);
    assert!(table.iter().all(|entry| entry.cache_key != ALICE_REQUEST));
}

#[tokio::test]
async fn a_provider_with_no_delete_endpoint_leaves_one_notice_in_the_job_result() {
    // Arrange
    let fixture = Fixture::new().await;
    let canceller = std::sync::Arc::new(Cancelling {
        copies: ProviderCopies::Kept,
        calls: std::sync::Mutex::default(),
    });
    let service = fixture.service_with(Some(canceller));

    // Act
    let id = service.request(alice()).unwrap();
    service.run(&id).await.unwrap();

    // Assert
    let job = job_named(&fixture, &id);
    assert_eq!(job.notices.len(), 1, "{:?}", job.notices);
    assert!(
        job.notices[0].starts_with(HOSTED_PROVIDER),
        "{:?}",
        job.notices
    );
    assert!(!FileClass::HostedBatches.holds_needle(&fixture));
}
