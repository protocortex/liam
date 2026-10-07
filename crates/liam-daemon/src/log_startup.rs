// SPDX-License-Identifier: Apache-2.0
//! Opening the store on its event log at daemon startup: attach the log, copy
//! in any rows that predate it, and replay what the log holds that the store
//! has not applied, all before the first request is served.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use liam_model::Embedder;
use liam_store::{DefaultGraph, GraphConfig};

use liam_daemon::config::Config;
use liam_daemon::event_log;
use liam_daemon::models::{resolve_log_dir, StoreEmbedder};

/// The store at `database_path`, ready to serve through its log. Every failure
/// is fatal: serving a store whose log is missing, foreign or unreadable would
/// accept writes the log never records.
pub async fn open_store_with_log(
    config: &Config,
    database_path: &Path,
    embedder: Arc<dyn Embedder>,
) -> anyhow::Result<DefaultGraph> {
    let log_dir = resolve_log_dir(config, database_path)?;
    tracing::info!(log_dir = %log_dir.display(), "opening the event log");
    let writer = event_log::open_writer(&log_dir, config.log.wal())?;
    let log = event_log::shared_log(writer, &log_dir, config.log.bloom()?)?;

    let store = DefaultGraph::open(
        database_path.to_str().unwrap_or(&config.database_path),
        GraphConfig::new(config.embedding_dims).with_read_pool_size(config.read_pool_size),
    )
    .await?;
    // The embedder goes in before the replay so restored nodes get vectors.
    let store = store
        .with_log(log)
        .await
        .with_context(|| {
            format!(
                "the event log in {} cannot be used with the database {}; the cause is below",
                log_dir.display(),
                database_path.display()
            )
        })?
        .with_embedder(Arc::new(StoreEmbedder(embedder)));

    if store.needs_backfill().await? {
        tracing::info!("logging the rows that predate the event log, writes wait until it is done");
        let report = store.backfill_log_from_projection().await.context(
            "copying the existing rows into the event log failed; fix the cause and \
                 restart, the copy resumes where it stopped",
        )?;
        tracing::info!(?report, "backfill completed");
    }
    let report = store
        .catch_up()
        .await
        .context("replaying the event log into the database failed")?;
    tracing::info!(?report, "event log caught up");
    Ok(store)
}

/// Fixtures shared with the maintenance tick tests in `main.rs`.
#[cfg(test)]
pub(crate) mod test_support {
    use std::path::{Path, PathBuf};

    use futures_util::StreamExt;
    use liam_log::dedup::{BloomConfig, HashBloom};
    use liam_log::event::{LogEvent, LogPayload, NodeRow, CURRENT_SCHEMA_VERSION};
    use liam_log::hash::node_row_hash;
    use liam_log::reader::{LogReader, SequentialScanReader};
    use liam_log::wal::{WalConfig, WalWriter};
    use liam_log::LogWriter;
    use liam_store::{EventLog, SharedLog, FOREVER};

    use super::*;

    pub(crate) const DIMS: usize = 8;

    const WAL: WalConfig = WalConfig {
        segment_max_bytes: 1 << 20,
        rotate_interval_secs: 3600,
    };

    /// A config whose database, socket and log all live under `dir`, with the
    /// mock models, so a test starts the real startup path against a temp dir.
    pub(crate) fn config_in(dir: &Path) -> Config {
        let mut config = Config {
            database_path: dir.join("liam.db").to_str().unwrap().to_string(),
            socket_path: dir.join("liamd.sock").to_str().unwrap().to_string(),
            embedding_dims: DIMS,
            ..Config::default()
        };
        config.llm.warmup = false;
        config.llm.max_concurrent_generations = 1;
        config.log.dir = Some(log_dir_in(dir).to_str().unwrap().to_string());
        config
    }

    pub(crate) fn log_dir_in(dir: &Path) -> PathBuf {
        dir.join("events")
    }

    pub(crate) fn database_in(dir: &Path) -> PathBuf {
        dir.join("liam.db")
    }

    pub(crate) fn embedder() -> Arc<dyn Embedder> {
        Arc::new(liam_model::MockEmbedder::new(DIMS))
    }

    /// A log over the directory that the test, not the daemon, writes through:
    /// the way a store written by an older run of the daemon was logged.
    pub(crate) fn standalone_log(log_dir: &Path) -> SharedLog {
        let writer = WalWriter::open_with_system_clock(log_dir, WAL).expect("open the wal");
        let reader = SequentialScanReader::local(log_dir).expect("open the reader");
        Arc::new(tokio::sync::Mutex::new(EventLog::new(
            Box::new(writer),
            Arc::new(reader),
            HashBloom::new(BloomConfig::default()),
        )))
    }

    /// Rows the store holds with no log at all, as before the log existed.
    pub(crate) async fn seed_unlogged(database: &Path, contents: &[&str]) {
        let store = DefaultGraph::open(database.to_str().unwrap(), GraphConfig::new(DIMS))
            .await
            .expect("open the store");
        for content in contents {
            store
                .insert(liam_store::NewNode::now("fact", "label", *content))
                .await
                .expect("insert a row");
        }
    }

    /// A database written through a log that is then deleted, so the next open
    /// finds a log with a different id than the one the database belongs to.
    pub(crate) async fn database_whose_log_was_deleted(dir: &Path) {
        let log_dir = log_dir_in(dir);
        {
            let store =
                DefaultGraph::open(database_in(dir).to_str().unwrap(), GraphConfig::new(DIMS))
                    .await
                    .expect("open the store")
                    .with_log(standalone_log(&log_dir))
                    .await
                    .expect("attach the log");
            store
                .insert(liam_store::NewNode::now("fact", "label", "kept"))
                .await
                .expect("insert a row");
        }
        std::fs::remove_dir_all(&log_dir).expect("delete the log");
    }

    /// A node write appended behind the store's back, the way a crash between
    /// the log append and the projection leaves one.
    pub(crate) fn append_unapplied_node(log_dir: &Path, id: &str, content: &str) {
        let row = NodeRow {
            id: id.into(),
            kind: "fact".into(),
            label: "label".into(),
            content: content.into(),
            producer: "agent-a".into(),
            attributes: "{}".into(),
            scope: None,
            subject: None,
            confidence: 0.75,
            valid_from: 500,
            valid_from_supplied: true,
            valid_until: FOREVER.0,
            tx_from: 1000,
            tx_to: FOREVER.0,
        };
        let event = LogEvent {
            event_id: format!("event-{id}"),
            content_hash: node_row_hash(&row),
            source: "agent-a".into(),
            trust_score: 0.75,
            observed_at: 500,
            ingested_at: 1000,
            encryption_key_id: None,
            schema_version: CURRENT_SCHEMA_VERSION,
            payload: LogPayload::NodeWrite(row),
        };
        let mut writer = WalWriter::open_with_system_clock(log_dir, WAL).expect("open the wal");
        writer.append(&event).expect("append the event");
    }

    /// Every event in the log directory, in write order.
    pub(crate) async fn logged_events(log_dir: &Path) -> Vec<LogEvent> {
        if !log_dir.exists() {
            return Vec::new();
        }
        let reader = SequentialScanReader::local(log_dir).expect("open the reader");
        let mut stream = reader.scan(None);
        let mut events = Vec::new();
        while let Some(record) = stream.next().await {
            events.push(record.expect("read a record").event);
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use liam_log::event::LogPayload;
    use liam_log::wal::MANIFEST_NAME;
    use liam_store::{Error as StoreError, Millis, NewNode, NodeId};

    use super::test_support::*;
    use super::*;

    async fn start(dir: &Path) -> anyhow::Result<DefaultGraph> {
        let config = config_in(dir);
        open_store_with_log(&config, &database_in(dir), embedder()).await
    }

    fn node_writes(events: &[liam_log::event::LogEvent]) -> usize {
        events
            .iter()
            .filter(|event| matches!(event.payload, LogPayload::NodeWrite(_)))
            .count()
    }

    async fn has_node(graph: &DefaultGraph, id: &str) -> bool {
        graph
            .get(&NodeId::from_raw(id), Millis::now())
            .await
            .expect("read the node")
            .is_some()
    }

    #[tokio::test]
    async fn a_fresh_database_starts_with_a_log_directory_and_the_log_attached() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();

        // Act
        let graph = start(dir.path()).await.expect("a fresh store must start");

        // Assert: the log exists on disk, is attached, and owes no backfill
        assert!(
            log_dir_in(dir.path()).join(MANIFEST_NAME).is_file(),
            "startup must create the log directory"
        );
        assert_eq!(
            graph.needs_backfill().await.ok(),
            Some(false),
            "the log must be attached and the backfill gate clear"
        );
    }

    #[tokio::test]
    async fn without_a_configured_dir_the_log_is_created_beside_the_database() {
        // Arrange: no [log] dir, so the default derives from database_path
        let dir = tempfile::tempdir().unwrap();
        let mut config = config_in(dir.path());
        config.log.dir = None;

        // Act
        open_store_with_log(&config, &database_in(dir.path()), embedder())
            .await
            .expect("a fresh store must start");

        // Assert
        assert!(dir.path().join("liam.log").join(MANIFEST_NAME).is_file());
    }

    #[tokio::test]
    async fn a_write_after_startup_goes_through_the_log() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let graph = start(dir.path()).await.expect("start");

        // Act
        graph
            .insert(NewNode::now("fact", "label", "after startup"))
            .await
            .expect("a write on a started store must be accepted");

        // Assert
        let events = logged_events(&log_dir_in(dir.path())).await;
        assert_eq!(node_writes(&events), 1);
    }

    #[tokio::test]
    async fn a_database_from_before_the_log_is_backfilled_at_startup() {
        // Arrange: two rows and no log
        let dir = tempfile::tempdir().unwrap();
        seed_unlogged(&database_in(dir.path()), &["first", "second"]).await;

        // Act
        let graph = start(dir.path()).await.expect("an upgrade must start");

        // Assert: the log holds both rows, the gate is clear, writes are allowed
        let events = logged_events(&log_dir_in(dir.path())).await;
        assert_eq!(node_writes(&events), 2, "both rows must be in the log");
        assert_eq!(graph.needs_backfill().await.ok(), Some(false));
        graph
            .insert(NewNode::now("fact", "label", "third"))
            .await
            .expect("writes must be allowed once the backfill completed");
    }

    #[tokio::test]
    async fn a_log_first_store_reopened_has_its_backfill_gate_cleared_without_new_events() {
        // Arrange: a store written through its log from the start, so the
        // backfill was never run for it, then closed
        let dir = tempfile::tempdir().unwrap();
        let log_dir = log_dir_in(dir.path());
        {
            let store = DefaultGraph::open(
                database_in(dir.path()).to_str().unwrap(),
                GraphConfig::new(DIMS),
            )
            .await
            .unwrap()
            .with_log(standalone_log(&log_dir))
            .await
            .unwrap();
            store
                .insert(NewNode::now("fact", "label", "logged first"))
                .await
                .unwrap();
        }

        // Act
        let graph = start(dir.path()).await.expect("reopen");

        // Assert: nothing was logged twice, and writes are allowed
        assert_eq!(node_writes(&logged_events(&log_dir).await), 1);
        assert_eq!(graph.needs_backfill().await.ok(), Some(false));
        graph
            .insert(NewNode::now("fact", "label", "next"))
            .await
            .expect("writes must be allowed after startup cleared the gate");
        assert_eq!(node_writes(&logged_events(&log_dir).await), 2);
    }

    #[tokio::test]
    async fn an_event_the_log_holds_but_the_store_never_applied_is_applied_at_startup() {
        // Arrange: a started store, closed, then an event appended behind it
        let dir = tempfile::tempdir().unwrap();
        drop(start(dir.path()).await.expect("first start"));
        append_unapplied_node(&log_dir_in(dir.path()), "late-node", "from a crash");

        // Act
        let graph = start(dir.path()).await.expect("second start");

        // Assert
        assert!(
            has_node(&graph, "late-node").await,
            "startup must replay the unapplied event"
        );
    }

    #[tokio::test]
    async fn startup_is_idempotent() {
        // Arrange: an upgrade, so the first start does real work
        let dir = tempfile::tempdir().unwrap();
        let log_dir = log_dir_in(dir.path());
        seed_unlogged(&database_in(dir.path()), &["first", "second"]).await;
        drop(start(dir.path()).await.expect("first start"));
        let after_first = logged_events(&log_dir).await;

        // Act
        let graph = start(dir.path()).await.expect("second start");

        // Assert: nothing new was logged and nothing is left to apply
        assert_eq!(after_first.len(), 2);
        assert_eq!(logged_events(&log_dir).await, after_first);
        let report = graph.catch_up().await.unwrap();
        assert_eq!(report.applied, 0);
    }

    #[tokio::test]
    async fn a_log_that_does_not_belong_to_the_database_stops_startup() {
        // Arrange: a database whose log directory was deleted, so the next
        // start opens a new log with a different id
        let dir = tempfile::tempdir().unwrap();
        database_whose_log_was_deleted(dir.path()).await;

        // Act
        let error = start(dir.path())
            .await
            .map(|_| ())
            .expect_err("a database with a foreign log must not start");

        // Assert: the store's own error survives for the operator
        let store_error = error
            .chain()
            .find_map(|cause| cause.downcast_ref::<StoreError>());
        assert!(
            matches!(store_error, Some(StoreError::LogIdMismatch { .. })),
            "{error:#}"
        );
    }

    #[tokio::test]
    async fn a_log_directory_that_cannot_be_created_stops_startup_naming_the_path() {
        // Arrange: the configured directory sits under a regular file
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("blocker");
        std::fs::write(&blocker, b"not a directory").unwrap();
        let mut config = config_in(dir.path());
        config.log.dir = Some(blocker.join("events").to_str().unwrap().to_string());

        // Act
        let error = open_store_with_log(&config, &database_in(dir.path()), embedder())
            .await
            .map(|_| ())
            .expect_err("an unusable log directory must stop startup");

        // Assert
        let message = format!("{error:#}");
        assert!(message.contains("blocker"), "{message}");
    }

    #[tokio::test]
    async fn the_daemon_never_starts_serving_when_the_log_check_fails() {
        // Arrange: the same foreign-log database, through the real serve path
        let dir = tempfile::tempdir().unwrap();
        database_whose_log_was_deleted(dir.path()).await;
        let config = config_in(dir.path());

        // Act: a daemon that went on to serve would never return
        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            crate::serve_with_store(crate::cli::Mode::Serve, config),
        )
        .await
        .expect("serve_with_store must fail at startup instead of serving");

        // Assert
        let error = outcome.expect_err("startup must fail closed");
        let store_error = error
            .chain()
            .find_map(|cause| cause.downcast_ref::<StoreError>());
        assert!(
            matches!(store_error, Some(StoreError::LogIdMismatch { .. })),
            "{error:#}"
        );
        assert!(
            !dir.path().join("liamd.sock").exists(),
            "no listener may have been bound"
        );
    }
}
