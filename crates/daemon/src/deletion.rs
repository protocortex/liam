// SPDX-License-Identifier: Apache-2.0
//! Deleting what a client asked to forget. The log holds sealed rows, so a
//! delete destroys keys; this module holds the commands that act on them.

pub(crate) mod batch_lineage;
pub(crate) mod batch_store;
mod command;
#[cfg(test)]
mod exit_criteria;
mod files;
pub(crate) mod job;
mod memory;
mod migrate;
mod preview;
mod registry;
mod scan;
mod scrub;
pub(crate) mod send_ledger;
mod service;
mod target;
#[cfg(test)]
pub(crate) mod test_support;

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use liam_daemon::event_log;
use liam_log::crypto::KeyRegistry;
use liam_store::{DefaultGraph, SystemClock};

use crate::config::Config;
use crate::{resolve_config_path, resolve_log_dir, storelock};
pub(crate) use batch_lineage::{BatchCanceller, ProviderCopies};
pub use command::{run as delete_command, DeleteAction};
pub(crate) use job::job_file_path_for;
pub(crate) use job::DeleteJob;
#[cfg(test)]
pub(crate) use job::JobState;
use job::{DeadlineAlert, JobStore};
pub use memory::NodeState;
pub use preview::DeletePreview;
pub use scan::ScanError;
pub use service::DeleteError;
use service::{DeleteJobPending, DeleteService, StoreParts};
pub use target::DeleteTarget;

/// `liamd encrypt-log`: seals the plaintext events of the log in place. Takes
/// the store lock, so it refuses to run beside a daemon.
pub async fn encrypt_log(config: &Config) -> anyhow::Result<()> {
    let database_path = resolve_config_path("database_path", &config.database_path)?;
    let log_dir = resolve_log_dir(config, &database_path)?;
    let _lock = storelock::StoreLock::acquire(&database_path)
        .context("encrypt-log needs the store to itself: stop liamd, then rerun")?;
    let report = migrate::encrypt_log(config, &database_path, &log_dir).await?;
    tracing::info!(
        partitions = report.partitions_rewritten,
        events = report.events_sealed,
        "the log is sealed"
    );
    println!(
        "sealed {} events in {} partitions",
        report.events_sealed, report.partitions_rewritten
    );
    Ok(())
}

/// Every delete job in the job file beside `database_path`.
pub(crate) fn open_jobs(database_path: &Path) -> anyhow::Result<Vec<DeleteJob>> {
    JobStore::new(job_file_path_for(database_path)).load()
}

/// Finishes the delete jobs a stopped daemon left, before it serves anyone. A
/// job that cannot finish (a failure, or a scrub a reader still holds) refuses
/// the start with [`DeleteJobPending`], because rows of a half-deleted subject
/// would otherwise be served. The caller holds the store lock.
pub async fn resume_pending(
    config: &Config,
    database_path: &Path,
    log_dir: &Path,
    store: Arc<DefaultGraph>,
    keys: Option<Arc<KeyRegistry>>,
    states: Vec<Arc<dyn NodeState>>,
    lock: &storelock::StoreLock,
) -> anyhow::Result<()> {
    let open = JobStore::new(job_file_path_for(database_path))
        .load()?
        .iter()
        .filter(|job| !job.state.is_final())
        .count();
    if open == 0 {
        return Ok(());
    }
    let keys =
        keys.context("a delete job is open, but the log has no key registry to resume it")?;
    let parts = StoreParts {
        store,
        reader: event_log::log_reader(log_dir, Some(&keys))?,
        keys,
        clock: Arc::new(SystemClock),
    };
    let service = DeleteService::open(config, database_path, parts, states, lock)?;
    tracing::info!(open, "resuming delete jobs");
    let report = service.resume_all(false).await?;
    for (job_id, alert) in service.alerts()? {
        match alert {
            DeadlineAlert::Overdue => {
                tracing::error!(job = %job_id, "delete job is past its deadline");
            }
            _ => tracing::warn!(job = %job_id, "delete job is close to its deadline"),
        }
    }
    if report.pending.is_empty() {
        return Ok(());
    }
    Err(DeleteJobPending {
        job_ids: report.pending,
    }
    .into())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use liam_log::event::LogPayload;
    use liam_log::reader::{LogReader, SequentialScanReader};
    use liam_store::{Millis, NewNode, NodeId};

    use super::*;
    use crate::config::{EncryptionConfig, MasterKeySource};
    use crate::log_startup::test_support::{config_in, database_in, embedder, log_dir_in};
    use crate::log_startup::{open_store_with_log, StartedStore};
    use crate::shutdown::Shutdown;

    fn encrypted(dir: &Path) -> Config {
        let mut config = config_in(dir);
        config.encryption = EncryptionConfig {
            master_key: Some(MasterKeySource::File),
            key_file: Some(dir.join("master.key").to_str().unwrap().to_owned()),
        };
        config
    }

    async fn start(config: &Config, dir: &Path) -> StartedStore {
        open_store_with_log(
            config,
            &database_in(dir),
            &log_dir_in(dir),
            embedder(),
            &Shutdown::default(),
        )
        .await
        .expect("start")
    }

    /// A store written without encryption: two subjects, one note with none,
    /// and a feedback verdict that names the first subject's lineage.
    async fn plaintext_store(dir: &Path) -> Vec<NodeId> {
        let started = start(&config_in(dir), dir).await;
        let mut ids = Vec::new();
        for (label, subject) in [
            ("one", Some("alice")),
            ("two", Some("bob")),
            ("three", None),
        ] {
            let mut note = NewNode::now("fact", label, format!("secret {label}"));
            if let Some(subject) = subject {
                note = note.with_subject(subject);
            }
            ids.push(started.store.insert(note).await.expect("insert"));
        }
        let lineage = started
            .store
            .lineage_of(&ids[0])
            .await
            .unwrap()
            .expect("a lineage");
        started
            .store
            .put_feedback_logged(&lineage, "client", true)
            .await
            .expect("feedback");
        ids
    }

    async fn raw_events(dir: &Path) -> Vec<liam_log::event::LogEvent> {
        use futures_util::StreamExt;
        let reader = SequentialScanReader::local(&log_dir_in(dir)).unwrap();
        reader
            .scan(None)
            .map(|record| record.expect("record").event)
            .collect()
            .await
    }

    #[tokio::test]
    async fn a_plaintext_log_is_sealed_and_the_store_reads_it_back_the_same() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let ids = plaintext_store(dir.path()).await;
        let before = raw_events(dir.path()).await;
        let config = encrypted(dir.path());

        // Act
        encrypt_log(&config).await.expect("encrypt");

        // Assert
        let after = raw_events(dir.path()).await;
        assert_eq!(after.len(), before.len());
        for (sealed, plain) in after.iter().zip(&before) {
            assert_eq!(sealed.event_id, plain.event_id);
            assert!(
                matches!(sealed.payload, LogPayload::Sealed(_)),
                "{:?}",
                sealed.payload
            );
        }
        let store = liam_daemon::store_open::open_database(&config, &database_in(dir.path()))
            .await
            .unwrap();
        for id in &ids {
            let key = store.key_of_node(id).await.unwrap();
            assert!(
                key.is_some(),
                "{id:?} must name the key group that seals it"
            );
        }
        drop(store);
        let restarted = start(&config, dir.path()).await;
        for id in &ids {
            let node = restarted.store.get(id, Millis::now()).await.unwrap();
            assert!(node.is_some(), "{id:?} must still be there");
        }
    }

    #[tokio::test]
    async fn a_failed_delete_job_refuses_the_start_before_anything_is_served() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let config = {
            let subjects = test_support::sealed_subjects(dir.path()).await;
            subjects.config.clone()
        };
        let database = database_in(dir.path());
        crate::embedder_fingerprint::record(
            &database,
            &crate::embedder_fingerprint::EmbedderFingerprint::from_config(&config),
        )
        .unwrap();
        let mut job = job::DeleteJob::new(
            "job-failed".into(),
            target::DeleteTarget::Event {
                event_id: "e".into(),
            },
            0,
            1,
        );
        job.state = job::JobState::Failed;
        job::JobStore::new(job::job_file_path_for(&database))
            .save(&job)
            .unwrap();

        // Act
        let result = crate::serve_until(crate::cli::Mode::Stdio, config, Shutdown::default()).await;

        // Assert
        let error = result.expect_err("the start must be refused");
        let pending = error.downcast_ref::<DeleteJobPending>().expect("typed");
        assert_eq!(pending.job_ids, ["job-failed"]);
        assert!(
            pending.to_string().contains("liamd delete --resume"),
            "{pending}"
        );
    }

    #[tokio::test]
    async fn a_second_run_finds_nothing_to_seal() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        plaintext_store(dir.path()).await;
        let config = encrypted(dir.path());
        encrypt_log(&config).await.expect("first run");
        let sealed = raw_events(dir.path()).await;

        // Act
        encrypt_log(&config).await.expect("second run");

        // Assert
        assert_eq!(raw_events(dir.path()).await, sealed);
    }

    #[tokio::test]
    async fn the_command_refuses_without_encryption_configured() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        plaintext_store(dir.path()).await;

        // Act
        let result = encrypt_log(&config_in(dir.path())).await;

        // Assert
        let error = format!("{:#}", result.expect_err("must refuse"));
        assert!(error.contains("master_key"), "{error}");
    }

    #[tokio::test]
    async fn the_command_refuses_while_another_process_holds_the_store() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        plaintext_store(dir.path()).await;
        let config = encrypted(dir.path());
        let _held = storelock::StoreLock::acquire(&database_in(dir.path())).unwrap();

        // Act
        let result = encrypt_log(&config).await;

        // Assert
        let error = format!("{:#}", result.expect_err("must refuse"));
        assert!(error.contains("store to itself"), "{error}");
    }
}
