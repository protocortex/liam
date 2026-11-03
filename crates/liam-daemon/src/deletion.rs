// SPDX-License-Identifier: Apache-2.0
//! Deleting what a client asked to forget. The log holds sealed rows, so a
//! delete destroys keys; this module holds the commands that act on them.

mod migrate;

use anyhow::Context;

use crate::config::Config;
use crate::{resolve_config_path, resolve_log_dir, storelock};

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
        let restarted = start(&config, dir.path()).await;
        for id in &ids {
            let node = restarted.store.get(id, Millis::now()).await.unwrap();
            assert!(node.is_some(), "{id:?} must still be there");
        }
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
