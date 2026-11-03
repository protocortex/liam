// SPDX-License-Identifier: Apache-2.0
//! Builds the extraction part the daemon runs, from its configuration and the
//! files beside the database.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use liam_daemon::event_log;
use liam_log::crypto::KeyRegistry;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use super::cache::{cache_path, CacheStore};
use super::hooks::ExtractionHooks;
use super::identity::{GgufFile, IdentitySource};
use super::provider::{pidfile_path, SidecarProvider, SystemStarter};
use super::sidecar::{SidecarGuard, SystemRam};
use crate::config::ExtractionConfig;

/// What the daemon knows that the configuration does not say.
pub struct DaemonFiles<'a> {
    pub database_path: &'a Path,
    pub log_dir: &'a Path,
    /// The key registry of a sealed log.
    pub keys: Option<Arc<KeyRegistry>>,
    /// The daemon's generation permits, shared with the other model users.
    pub permits: Arc<Semaphore>,
    pub stopping: CancellationToken,
}

/// Clears a model server an earlier daemon left behind. A killed daemon cannot
/// stop its child, so this runs once at startup, whether or not extraction is
/// on. Without `llama_server_path` there is no executable to check a process
/// against, so a pidfile is left alone.
pub fn sweep_stale_sidecar(config: &ExtractionConfig, database_path: &Path) {
    let pidfile = pidfile_path(database_path);
    if config.llama_server_path.is_none() && pidfile.exists() {
        warn!("extraction: a model server may be left running, but extraction.llama_server_path is not set to find it");
    }
    let outcome = SidecarGuard::sweep_stale(config, &pidfile);
    debug!(
        ?outcome,
        "extraction: swept for a model server left by an earlier daemon"
    );
}

/// The extraction part for `config`. The model file is not read here: the
/// first model pass hashes it, so a start with extraction on stays quick.
///
/// # Errors
///
/// Fails when the response cache or the event log cannot be opened.
pub fn build(config: &ExtractionConfig, files: DaemonFiles<'_>) -> anyhow::Result<ExtractionHooks> {
    let identity: Arc<dyn IdentitySource> = Arc::new(GgufFile::new(
        config.model_gguf.as_deref().map(PathBuf::from),
    ));
    let starter = SystemStarter {
        pidfile: pidfile_path(files.database_path),
    };
    let cache = CacheStore::open(cache_path(files.database_path))
        .context("could not open the extraction response cache")?;
    let reader = event_log::log_reader(files.log_dir, files.keys.as_ref())
        .context("could not open the event log for extraction")?;
    Ok(ExtractionHooks {
        config: config.clone(),
        reader,
        provider: Arc::new(SidecarProvider::new(Arc::new(starter), identity.clone())),
        identity,
        cache: Arc::new(cache),
        permits: files.permits,
        free_ram: Arc::new(SystemRam),
        stopping: files.stopping,
    })
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::extraction::identity::hash_file_for;

    fn built(config: &ExtractionConfig, dir: &TempDir) -> ExtractionHooks {
        let files = DaemonFiles {
            database_path: &dir.path().join("liam.db"),
            log_dir: dir.path(),
            keys: None,
            permits: Arc::new(Semaphore::new(1)),
            stopping: CancellationToken::new(),
        };
        build(config, files).unwrap()
    }

    fn config_with_model(path: &Path) -> ExtractionConfig {
        ExtractionConfig {
            model_gguf: Some(path.to_string_lossy().into_owned()),
            ..ExtractionConfig::default()
        }
    }

    #[tokio::test]
    async fn the_hooks_name_the_configured_model_file_when_asked() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let model = dir.path().join("model.gguf");
        std::fs::write(&model, b"weights").unwrap();
        let hooks = built(&config_with_model(&model), &dir);

        // Act
        let identity = hooks.identity.identity().await.unwrap();

        // Assert
        assert_eq!(identity.model_id, "model.gguf");
    }

    #[tokio::test]
    async fn an_unreadable_model_fails_when_asked_not_when_built() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let config = config_with_model(&dir.path().join("absent.gguf"));

        // Act
        let hooks = built(&config, &dir);

        // Assert
        assert!(hooks.identity.identity().await.is_err());
    }

    #[test]
    fn building_the_hooks_does_not_read_the_model() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let model = dir.path().join("model.gguf");
        std::fs::write(&model, b"weights").unwrap();

        // Act
        built(&config_with_model(&model), &dir);

        // Assert
        assert!(!hash_file_for(&model).exists());
    }

    #[test]
    fn a_leftover_pidfile_is_cleared_only_when_the_server_path_is_known() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let database = dir.path().join("liam.db");
        let pidfile = pidfile_path(&database);
        let unset = ExtractionConfig::default();
        let set = ExtractionConfig {
            llama_server_path: Some("/nonexistent/llama-server".into()),
            ..ExtractionConfig::default()
        };
        std::fs::write(&pidfile, "not a record").unwrap();

        // Act
        sweep_stale_sidecar(&unset, &database);
        let kept = pidfile.exists();
        sweep_stale_sidecar(&set, &database);

        // Assert
        assert!(kept);
        assert!(!pidfile.exists());
    }

    /// Writes the two sealed events and hands back the still-open writer, so a
    /// test that needs more appends never reopens the log. Reopening races any
    /// concurrently spawned child process: until it execs, the child holds a
    /// copy of the released lock descriptor, so the lock reads as held.
    fn sealed_log(
        dir: &TempDir,
        shred_second: bool,
        wal: liam_log::wal::WalConfig,
    ) -> (
        Arc<KeyRegistry>,
        Vec<liam_log::event::LogEvent>,
        liam_log::wal::WalWriter<liam_log::wal::SystemClock>,
    ) {
        use liam_log::envelope::{KeyHints, Sealer};
        use liam_log::event::{LogEvent, LogPayload, NodeRow, CURRENT_SCHEMA_VERSION};
        use liam_log::wal::WalWriter;
        use liam_log::LogWriter;

        let encryption = crate::config::EncryptionConfig {
            master_key: Some(crate::config::MasterKeySource::File),
            key_file: Some(dir.path().join("master.key").to_str().unwrap().to_owned()),
        };
        let registry = event_log::open_key_registry(dir.path(), &encryption)
            .unwrap()
            .unwrap();
        let sealer = Sealer::new(Arc::clone(&registry));
        let write = |id: &str, subject: &str| LogEvent {
            event_id: format!("event-{id}"),
            content_hash: [1; 32],
            source: "client".into(),
            trust_score: 1.0,
            observed_at: 1,
            ingested_at: 1,
            encryption_key_id: None,
            schema_version: CURRENT_SCHEMA_VERSION,
            payload: LogPayload::NodeWrite(NodeRow {
                id: id.into(),
                kind: "fact".into(),
                label: format!("label {id}"),
                content: format!("content {id}"),
                producer: "p".into(),
                attributes: "{}".into(),
                scope: None,
                subject: Some(subject.into()),
                confidence: 1.0,
                valid_from: 1,
                valid_from_supplied: true,
                valid_until: 2,
                tx_from: 1,
                tx_to: 2,
                derivation: None,
                basis: Vec::new(),
            }),
        };
        let events: Vec<_> = [write("a", "alice"), write("b", "bob")]
            .iter()
            .map(|event| sealer.seal(event, &KeyHints::default()).unwrap())
            .collect();
        let mut writer = WalWriter::open_with_system_clock(dir.path(), wal).unwrap();
        for event in &events {
            writer.append(event).unwrap();
        }
        if shred_second {
            let LogPayload::Sealed(record) = &events[1].payload else {
                unreachable!()
            };
            registry.destroy(&record.rows[0].key_id).unwrap();
        }
        (registry, events, writer)
    }

    #[tokio::test]
    async fn the_extraction_reader_opens_sealed_records_and_passes_a_shredded_one() {
        use futures_util::StreamExt;
        use liam_log::event::LogPayload;

        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (registry, _events, _writer) =
            sealed_log(&dir, true, crate::config::LogConfig::default().wal());
        let files = DaemonFiles {
            database_path: &dir.path().join("liam.db"),
            log_dir: dir.path(),
            keys: Some(registry),
            permits: Arc::new(Semaphore::new(1)),
            stopping: CancellationToken::new(),
        };
        let hooks = build(&ExtractionConfig::default(), files).unwrap();

        // Act
        let records: Vec<_> = hooks.reader.scan(None).collect().await;

        // Assert
        let first = records[0].as_ref().unwrap();
        assert!(matches!(first.event.payload, LogPayload::NodeWrite(_)));
        let second = records[1].as_ref().unwrap();
        assert_eq!(second.event.payload, LogPayload::EpisodeBatch(Vec::new()));
        assert_eq!(second.shredded.len(), 1);
    }

    #[tokio::test]
    async fn the_extraction_reader_stops_at_a_key_the_registry_lost() {
        use futures_util::StreamExt;
        use liam_log::reader::ReaderError;

        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let _log = sealed_log(&dir, false, crate::config::LogConfig::default().wal());
        let elsewhere = tempfile::tempdir().unwrap();
        let lost = event_log::open_key_registry(
            elsewhere.path(),
            &crate::config::EncryptionConfig {
                master_key: Some(crate::config::MasterKeySource::File),
                key_file: Some(
                    elsewhere
                        .path()
                        .join("other.key")
                        .to_str()
                        .unwrap()
                        .to_owned(),
                ),
            },
        )
        .unwrap()
        .unwrap();
        let files = DaemonFiles {
            database_path: &dir.path().join("liam.db"),
            log_dir: dir.path(),
            keys: Some(lost),
            permits: Arc::new(Semaphore::new(1)),
            stopping: CancellationToken::new(),
        };
        let hooks = build(&ExtractionConfig::default(), files).unwrap();

        // Act
        let records: Vec<_> = hooks.reader.scan(None).collect().await;

        // Assert
        assert_eq!(records.len(), 1);
        assert!(matches!(records[0], Err(ReaderError::KeyMissing { .. })));
    }

    #[tokio::test]
    async fn the_extraction_reader_stops_at_an_archived_partition_that_went_missing() {
        use futures_util::StreamExt;
        use liam_log::archive::{archive_partitions, ArchivePolicy, NoFault};
        use liam_log::compactor::Compactor;
        use liam_log::reader::ReaderError;
        use liam_log::wal::WalConfig;
        use liam_log::LogWriter;

        // Arrange: segment 0 is compacted, archived, and then lost
        let dir = tempfile::tempdir().unwrap();
        let (registry, events, mut writer) = sealed_log(
            &dir,
            false,
            WalConfig {
                segment_max_bytes: 1,
                rotate_interval_secs: u64::MAX,
            },
        );
        let filler = liam_log::event::LogEvent {
            event_id: "filler".into(),
            encryption_key_id: None,
            payload: liam_log::event::LogPayload::Tombstone(Vec::new()),
            ..events[0].clone()
        };
        writer.append(&filler).unwrap();
        drop(writer);
        let first_wal = dir.path().join(format!("{:020}.wal", 0));
        Compactor::local(dir.path())
            .unwrap()
            .compact_segment(&first_wal)
            .await
            .unwrap();
        let archive = tempfile::tempdir().unwrap();
        let policy = ArchivePolicy {
            after_days: 0,
            zstd_level: 3,
        };
        archive_partitions(dir.path(), archive.path(), policy, i64::MAX, &NoFault).unwrap();
        for entry in std::fs::read_dir(archive.path()).unwrap() {
            std::fs::remove_file(entry.unwrap().path()).unwrap();
        }
        let files = DaemonFiles {
            database_path: &dir.path().join("liam.db"),
            log_dir: dir.path(),
            keys: Some(registry),
            permits: Arc::new(Semaphore::new(1)),
            stopping: CancellationToken::new(),
        };
        let hooks = build(&ExtractionConfig::default(), files).unwrap();

        // Act
        let records: Vec<_> = hooks.reader.scan(None).collect().await;

        // Assert
        assert_eq!(records.len(), 1);
        assert!(matches!(
            records[0],
            Err(ReaderError::PartitionUnreachable { .. })
        ));
    }
}
