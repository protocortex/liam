// SPDX-License-Identifier: Apache-2.0
//! Opening the store on its event log at daemon startup: attach the log, copy
//! in any rows that predate it, and replay what the log holds that the store
//! has not applied, all before the first request is served.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use liam_log::compactor::{ChannelSink, Compactor};
use liam_model::Embedder;
use liam_store::{DefaultGraph, GraphConfig, SharedLog};
use tokio::task::JoinHandle;

use liam_daemon::config::Config;
use liam_daemon::event_log;
use liam_daemon::models::StoreEmbedder;

use crate::shutdown::Shutdown;

/// The store ready to serve, and the task that turns the log's closed segments
/// into Parquet for as long as the store lives.
pub struct StartedStore {
    pub store: DefaultGraph,
    pub compactor: JoinHandle<()>,
}

/// The store at `database_path` over the log in `log_dir`. Every failure is
/// fatal: serving a store whose log is missing, foreign or unreadable would
/// accept writes the log never records.
pub async fn open_store_with_log(
    config: &Config,
    database_path: &Path,
    log_dir: &Path,
    embedder: Arc<dyn Embedder>,
    shutdown: &Shutdown,
) -> anyhow::Result<StartedStore> {
    tracing::info!(log_dir = %log_dir.display(), "opening the event log");
    // Both checks come before the writer creates anything, so a mistyped
    // directory is refused instead of becoming a log.
    event_log::ensure_usable_as_log(log_dir)?;
    let store = open_database(config, database_path).await?;
    refuse_a_new_log_for_a_logged_database(&store, database_path, log_dir).await?;

    let (sink, closed_segments) = ChannelSink::channel();
    let writer = event_log::open_writer(log_dir, config.log.wal())?.with_sink(Box::new(sink));
    // Segments closed before a crash are not announced again, so they are
    // found here.
    let recovered = compactor_for(log_dir)?
        .recover(Some(writer.open_sequence()))
        .await
        .with_context(|| {
            format!(
                "compacting the closed segments left in {} failed; fix the cause and restart",
                log_dir.display()
            )
        })?;
    tracing::info!(report = ?recovered, "closed log segments recovered");

    let log = event_log::shared_log(
        writer,
        log_dir,
        config.log.bloom()?,
        Some(shutdown.poison_hook()),
    )?;
    let store = open_store_on_log(store, database_path, log_dir, log, embedder).await?;
    let compactor = compactor_for(log_dir)?.spawn(closed_segments);
    Ok(StartedStore { store, compactor })
}

pub(crate) async fn open_database(
    config: &Config,
    database_path: &Path,
) -> anyhow::Result<DefaultGraph> {
    let path = database_path
        .to_str()
        .context("the database path is not UTF-8")?;
    let graph = GraphConfig::new(config.embedding_dims).with_read_pool_size(config.read_pool_size);
    Ok(DefaultGraph::open(path, graph).await?)
}

/// A database already tied to a log belongs to that log, so a directory with
/// no log in it is a wrong path, not a place for a new one.
async fn refuse_a_new_log_for_a_logged_database(
    store: &DefaultGraph,
    database_path: &Path,
    log_dir: &Path,
) -> anyhow::Result<()> {
    let Some(log_id) = store.stored_log_id().await? else {
        return Ok(());
    };
    anyhow::ensure!(
        event_log::holds_log(log_dir),
        "the database {} belongs to the log {log_id}, but {} holds no log; fix [log] dir, or \
         restore the log there",
        database_path.display(),
        log_dir.display()
    );
    Ok(())
}

fn compactor_for(log_dir: &Path) -> anyhow::Result<Compactor> {
    Compactor::local(log_dir).with_context(|| {
        format!(
            "could not open the log in {} to compact it",
            log_dir.display()
        )
    })
}

/// Attaches `log` to `store`, copies in the rows that predate it and replays
/// what the store has not applied.
pub(crate) async fn open_store_on_log(
    store: DefaultGraph,
    database_path: &Path,
    log_dir: &Path,
    log: SharedLog,
    embedder: Arc<dyn Embedder>,
) -> anyhow::Result<DefaultGraph> {
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
    use std::sync::atomic::{AtomicBool, Ordering};

    use futures_util::{stream, StreamExt};
    use liam_log::dedup::{BloomConfig, HashBloom};
    use liam_log::event::{LogEvent, LogPayload, NodeRow, CURRENT_SCHEMA_VERSION};
    use liam_log::hash::node_row_hash;
    use liam_log::reader::{LogReader, LogStream, ReaderError, SequentialScanReader};
    use liam_log::wal::{SystemClock, WalConfig, WalError, WalWriter};
    use liam_log::{LogOffset, LogWriter};
    use liam_store::{EventLog, PoisonHook, SharedLog, FOREVER};

    use super::*;

    pub(crate) const DIMS: usize = 8;

    const WAL: WalConfig = WalConfig {
        segment_max_bytes: 1 << 20,
        rotate_interval_secs: 3600,
    };

    /// Closes the segment after every append.
    pub(crate) const SEGMENT_PER_EVENT: WalConfig = WalConfig {
        segment_max_bytes: 1,
        rotate_interval_secs: 3600,
    };

    /// A config whose database, socket and log all live under `dir`, with the
    /// mock models, so a test starts the real startup path against a temp dir.
    pub(crate) fn config_in(dir: &Path) -> Config {
        let mut config = Config {
            database_path: database_in(dir).to_str().unwrap().to_string(),
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
        let writer = event_log::open_writer(log_dir, WAL).expect("open the wal");
        event_log::shared_log(writer, log_dir, BloomConfig::default(), None).expect("open the log")
    }

    /// A log over `writer` and `reader`, which can be made to fail.
    pub(crate) fn log_over(
        writer: impl LogWriter + 'static,
        reader: Arc<dyn LogReader>,
        on_poisoned: Option<PoisonHook>,
    ) -> SharedLog {
        let mut log = EventLog::new(
            Box::new(writer),
            reader,
            HashBloom::new(BloomConfig::default()),
        );
        if let Some(hook) = on_poisoned {
            log = log.with_poison_hook(hook);
        }
        Arc::new(tokio::sync::Mutex::new(log))
    }

    /// A real writer that refuses every append, as a `WalWriter` does after a
    /// failed fsync, once its flag is set.
    pub(crate) struct ToggledWriter {
        inner: WalWriter<SystemClock>,
        poisoned: Arc<AtomicBool>,
    }

    impl ToggledWriter {
        pub(crate) fn open(log_dir: &Path, poisoned: bool) -> (Self, Arc<AtomicBool>) {
            let flag = Arc::new(AtomicBool::new(poisoned));
            let inner = event_log::open_writer(log_dir, WAL).expect("open the wal");
            let writer = Self {
                inner,
                poisoned: Arc::clone(&flag),
            };
            (writer, flag)
        }
    }

    impl LogWriter for ToggledWriter {
        fn append(&mut self, event: &LogEvent) -> Result<LogOffset, WalError> {
            if self.poisoned.load(Ordering::SeqCst) {
                return Err(WalError::Poisoned);
            }
            self.inner.append(event)
        }

        fn log_id(&self) -> uuid::Uuid {
            self.inner.log_id()
        }

        fn head(&self) -> Option<LogOffset> {
            self.inner.head()
        }
    }

    /// A reader whose every scan fails.
    pub(crate) struct FailingReader;

    impl FailingReader {
        fn failed() -> LogStream {
            let error = ReaderError::Io(std::io::Error::other("injected read failure"));
            Box::pin(stream::once(async { Err(error) }))
        }
    }

    impl LogReader for FailingReader {
        fn scan(&self, _from: Option<LogOffset>) -> LogStream {
            Self::failed()
        }

        fn scan_through(&self, _from: Option<LogOffset>, _through: LogOffset) -> LogStream {
            Self::failed()
        }
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
        append_unapplied_node_in(log_dir, WAL, id, content);
    }

    /// `append_unapplied_node` through a writer with its own segment sizes.
    pub(crate) fn append_unapplied_node_in(
        log_dir: &Path,
        wal: WalConfig,
        id: &str,
        content: &str,
    ) {
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
        let mut writer = event_log::open_writer(log_dir, wal).expect("open the wal");
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
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use liam_log::event::LogPayload;
    use liam_log::wal::MANIFEST_NAME;
    use liam_store::{Error as StoreError, Millis, NewNode, NodeId, ReembedReport};

    use crate::test_logs::{capture, captured};

    use super::test_support::*;
    use super::*;

    async fn started(config: &Config, dir: &Path) -> anyhow::Result<StartedStore> {
        open_store_with_log(
            config,
            &database_in(dir),
            &log_dir_in(dir),
            embedder(),
            &Shutdown::default(),
        )
        .await
    }

    async fn start(dir: &Path) -> anyhow::Result<DefaultGraph> {
        started(&config_in(dir), dir)
            .await
            .map(|started| started.store)
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

    fn files_with_extension(log_dir: &Path, extension: &str) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(log_dir)
            .expect("list the log directory")
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(extension))
            .collect();
        names.sort();
        names
    }

    /// Polls until `condition` holds, failing the test at the deadline.
    async fn eventually(what: &str, condition: impl Fn() -> bool) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while !condition() {
            assert!(tokio::time::Instant::now() < deadline, "timed out: {what}");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    fn error_text(result: anyhow::Result<impl Sized>) -> String {
        format!("{:#}", result.map(|_| ()).expect_err("the call must fail"))
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
    async fn startup_opens_the_database_it_is_given_not_the_one_in_the_config() {
        // Arrange: the config names another database
        let dir = tempfile::tempdir().unwrap();
        let mut config = config_in(dir.path());
        config.database_path = dir.path().join("decoy.db").to_str().unwrap().to_string();

        // Act
        started(&config, dir.path())
            .await
            .expect("a fresh store must start");

        // Assert
        assert!(database_in(dir.path()).exists());
        assert!(!dir.path().join("decoy.db").exists());
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
    async fn the_configured_wal_size_reaches_the_writer_and_closed_segments_are_compacted() {
        // Arrange: a segment size of one byte closes the segment on every write
        let dir = tempfile::tempdir().unwrap();
        let log_dir = log_dir_in(dir.path());
        let mut config = config_in(dir.path());
        config.log.wal_segment_max_bytes = 1;
        let started = started(&config, dir.path()).await.expect("start");

        // Act
        for content in ["first", "second"] {
            started
                .store
                .insert(NewNode::now("fact", "label", content))
                .await
                .expect("insert");
        }

        // Assert: more than one segment exists at once, and the first closed
        // one ends up as Parquet with its WAL file removed
        let segments = files_with_extension(&log_dir, ".wal").len()
            + files_with_extension(&log_dir, ".parquet").len();
        assert!(segments > 1, "the size limit must have closed a segment");
        eventually("the first segment is compacted", || {
            !files_with_extension(&log_dir, ".parquet").is_empty()
                && !files_with_extension(&log_dir, ".wal")
                    .iter()
                    .any(|name| name.starts_with("00000000000000000000"))
        })
        .await;
        assert_eq!(node_writes(&logged_events(&log_dir).await), 2);
    }

    #[tokio::test]
    async fn startup_compacts_the_closed_segments_a_crash_left_behind() {
        // Arrange: two closed WAL segments and an open one, from a writer that
        // never announced them to a compactor
        let dir = tempfile::tempdir().unwrap();
        let log_dir = log_dir_in(dir.path());
        append_unapplied_node_in(&log_dir, SEGMENT_PER_EVENT, "node-a", "alpha");
        append_unapplied_node_in(&log_dir, SEGMENT_PER_EVENT, "node-b", "beta");
        assert_eq!(
            files_with_extension(&log_dir, ".wal").len(),
            3,
            "precondition"
        );

        // Act
        let graph = start(dir.path()).await.expect("start after the crash");

        // Assert: recovery finished before startup returned, and replay read
        // the Parquet it wrote
        assert_eq!(files_with_extension(&log_dir, ".parquet").len(), 2);
        assert_eq!(files_with_extension(&log_dir, ".wal").len(), 1);
        assert!(has_node(&graph, "node-a").await);
        assert!(has_node(&graph, "node-b").await);
    }

    #[tokio::test]
    async fn a_database_from_before_the_log_is_backfilled_at_startup() {
        // Arrange: two rows and no log
        let dir = tempfile::tempdir().unwrap();
        seed_unlogged(&database_in(dir.path()), &["first", "second"]).await;
        capture();

        // Act
        let graph = start(dir.path()).await.expect("an upgrade must start");

        // Assert: the log holds both rows, the gate is clear, writes are allowed
        let events = logged_events(&log_dir_in(dir.path())).await;
        assert_eq!(node_writes(&events), 2, "both rows must be in the log");
        assert_eq!(graph.needs_backfill().await.ok(), Some(false));
        assert!(
            captured().contains("predate the event log"),
            "{}",
            captured()
        );
        graph
            .insert(NewNode::now("fact", "label", "third"))
            .await
            .expect("writes must be allowed once the backfill completed");
    }

    #[tokio::test]
    async fn a_fresh_install_does_not_announce_a_backfill() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        capture();

        // Act
        start(dir.path()).await.expect("a fresh store must start");

        // Assert
        let text = captured();
        assert!(!text.contains("predate"), "{text}");
    }

    #[tokio::test]
    async fn a_log_first_store_reopened_has_its_backfill_gate_cleared_without_new_events() {
        // Arrange: a store written through its log from the start, so the
        // backfill was never run for it, then closed
        let dir = tempfile::tempdir().unwrap();
        let log_dir = log_dir_in(dir.path());
        {
            let store = open_database(&config_in(dir.path()), &database_in(dir.path()))
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
        capture();

        // Act
        start(dir.path()).await.expect("second start");

        // Assert: nothing new was logged, and the second start copied nothing
        assert_eq!(after_first.len(), 2);
        assert_eq!(logged_events(&log_dir).await, after_first);
        let text = captured();
        assert!(!text.contains("backfill"), "{text}");
    }

    #[tokio::test]
    async fn a_log_that_does_not_belong_to_the_database_stops_startup() {
        // Arrange: a database whose log directory was deleted, so the next
        // start opens a new log with a different id
        let dir = tempfile::tempdir().unwrap();
        database_whose_log_was_deleted(dir.path()).await;
        let other_dir = tempfile::tempdir().unwrap();
        let other_log = log_dir_in(other_dir.path());
        drop(standalone_log(&other_log));

        // Act
        let error = open_store_with_log(
            &config_in(dir.path()),
            &database_in(dir.path()),
            &other_log,
            embedder(),
            &Shutdown::default(),
        )
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

        // Act
        let result = open_store_with_log(
            &config_in(dir.path()),
            &database_in(dir.path()),
            &blocker.join("events"),
            embedder(),
            &Shutdown::default(),
        )
        .await;

        // Assert
        let message = error_text(result);
        assert!(message.contains("blocker"), "{message}");
    }

    #[tokio::test]
    async fn a_directory_of_other_files_is_refused_before_anything_is_written_to_it() {
        // Arrange: the configured directory holds someone's notes
        let dir = tempfile::tempdir().unwrap();
        let log_dir = log_dir_in(dir.path());
        std::fs::create_dir(&log_dir).unwrap();
        std::fs::write(log_dir.join("notes.txt"), b"mine").unwrap();

        // Act
        let result = start(dir.path()).await;

        // Assert: refused by name, and nothing was added to it or created beside it
        let message = error_text(result);
        assert!(message.contains(log_dir.to_str().unwrap()), "{message}");
        assert_eq!(
            std::fs::read_dir(&log_dir).unwrap().count(),
            1,
            "the directory must be left as it was"
        );
        assert!(!database_in(dir.path()).exists());
    }

    #[tokio::test]
    async fn a_missing_or_empty_directory_becomes_a_new_log() {
        for precreate in [false, true] {
            // Arrange
            let dir = tempfile::tempdir().unwrap();
            let log_dir = log_dir_in(dir.path());
            if precreate {
                std::fs::create_dir(&log_dir).unwrap();
            }

            // Act
            let result = start(dir.path()).await;

            // Assert
            assert!(result.is_ok(), "precreated {precreate}: {:?}", result.err());
            assert!(log_dir.join(MANIFEST_NAME).is_file());
        }
    }

    #[tokio::test]
    async fn a_mistyped_directory_for_a_logged_database_is_refused_and_left_empty() {
        // Arrange: a database written through its log, then a config pointing
        // at a directory that does not exist
        let dir = tempfile::tempdir().unwrap();
        drop(start(dir.path()).await.expect("first start"));
        let typo = dir.path().join("evnets");

        // Act
        let result = open_store_with_log(
            &config_in(dir.path()),
            &database_in(dir.path()),
            &typo,
            embedder(),
            &Shutdown::default(),
        )
        .await;

        // Assert: the refusal names both, and no log came into being there
        let message = error_text(result);
        assert!(message.contains("belongs to the log"), "{message}");
        assert!(message.contains("evnets"), "{message}");
        assert!(!typo.exists(), "no directory may have been created");
    }

    #[tokio::test]
    async fn a_second_start_on_a_log_in_use_is_refused_until_the_first_lets_go() {
        // Arrange: one daemon's store holds the log, and a second database is
        // configured to the same directory
        let dir = tempfile::tempdir().unwrap();
        let first = start(dir.path()).await.expect("first start");
        let other = dir.path().join("other.db");

        // Act
        let refused = open_store_with_log(
            &config_in(dir.path()),
            &other,
            &log_dir_in(dir.path()),
            embedder(),
            &Shutdown::default(),
        )
        .await;

        // Assert: refused, naming the log directory
        let message = error_text(refused);
        assert!(message.contains("held by another process"), "{message}");
        assert!(
            message.contains(log_dir_in(dir.path()).to_str().unwrap()),
            "{message}"
        );

        // Act: the first lets go
        drop(first);
        let again = start(dir.path()).await;

        // Assert
        assert!(again.is_ok(), "{:?}", again.err());
    }

    #[tokio::test]
    async fn a_bloom_rate_the_config_validation_would_refuse_stops_startup() {
        // Arrange: a config built in code skips the load time check
        let dir = tempfile::tempdir().unwrap();
        let mut config = config_in(dir.path());
        config.log.bloom_fpr = 2.0;

        // Act
        let result = started(&config, dir.path()).await;

        // Assert: startup sizes the filter from the config
        let message = error_text(result);
        assert!(message.contains("log.bloom_fpr"), "{message}");
    }

    // ---- the steps after the log is open ----

    /// A log over the directory for the database to attach to, whose writer
    /// refuses appends and whose reader works.
    fn log_whose_writer_fails(log_dir: &Path) -> SharedLog {
        let (writer, _flag) = ToggledWriter::open(log_dir, true);
        event_log::shared_log(
            writer,
            log_dir,
            liam_log::dedup::BloomConfig::default(),
            None,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn startup_fails_when_copying_the_old_rows_into_the_log_fails() {
        // Arrange: rows from before the log, and a writer that refuses them
        let dir = tempfile::tempdir().unwrap();
        let config = config_in(dir.path());
        seed_unlogged(&database_in(dir.path()), &["first"]).await;
        let store = open_database(&config, &database_in(dir.path()))
            .await
            .unwrap();
        let log = log_whose_writer_fails(&log_dir_in(dir.path()));

        // Act
        let result = open_store_on_log(
            store,
            &database_in(dir.path()),
            &log_dir_in(dir.path()),
            log,
            embedder(),
        )
        .await;

        // Assert
        let message = error_text(result);
        assert!(message.contains("copying the existing rows"), "{message}");
    }

    #[tokio::test]
    async fn startup_fails_when_replaying_the_log_fails() {
        // Arrange: a started store, then an event it has not applied, then a
        // reopen over a log whose reader fails
        let dir = tempfile::tempdir().unwrap();
        let config = config_in(dir.path());
        let log_dir = log_dir_in(dir.path());
        drop(start(dir.path()).await.expect("first start"));
        append_unapplied_node(&log_dir, "late-node", "from a crash");
        let writer = event_log::open_writer(&log_dir, config.log.wal()).unwrap();
        let log = log_over(writer, Arc::new(FailingReader), None);
        let store = open_database(&config, &database_in(dir.path()))
            .await
            .unwrap();

        // Act
        let result =
            open_store_on_log(store, &database_in(dir.path()), &log_dir, log, embedder()).await;

        // Assert
        let message = error_text(result);
        assert!(message.contains("replaying the event log"), "{message}");
    }

    // ---- the embedder ----

    struct FailingEmbedder;

    #[async_trait::async_trait]
    impl Embedder for FailingEmbedder {
        fn dims(&self) -> usize {
            DIMS
        }

        async fn embed(
            &self,
            _text: &str,
            _kind: liam_model::TextKind,
        ) -> liam_model::Result<Vec<f32>> {
            Err(liam_model::ModelError::Embed("injected failure".into()))
        }
    }

    #[tokio::test]
    async fn startup_embeds_the_rows_that_predate_the_log() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        seed_unlogged(&database_in(dir.path()), &["first", "second"]).await;

        // Act
        let graph = start(dir.path()).await.expect("an upgrade must start");

        // Assert: nothing is left without a vector
        let left = graph.reembed_missing().await.unwrap();
        assert_eq!(left, ReembedReport::default());
    }

    #[tokio::test]
    async fn startup_embeds_the_nodes_it_replays() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        drop(start(dir.path()).await.expect("first start"));
        append_unapplied_node(&log_dir_in(dir.path()), "late-node", "from a crash");

        // Act
        let graph = start(dir.path()).await.expect("second start");

        // Assert
        assert!(has_node(&graph, "late-node").await);
        let left = graph.reembed_missing().await.unwrap();
        assert_eq!(left, ReembedReport::default());
    }

    #[tokio::test]
    async fn startup_succeeds_with_an_embedder_that_fails_and_the_node_is_readable() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        drop(start(dir.path()).await.expect("first start"));
        append_unapplied_node(&log_dir_in(dir.path()), "late-node", "from a crash");

        // Act
        let result = open_store_with_log(
            &config_in(dir.path()),
            &database_in(dir.path()),
            &log_dir_in(dir.path()),
            Arc::new(FailingEmbedder),
            &Shutdown::default(),
        )
        .await;

        // Assert: the node is there, only its vector is owed
        let started = result.expect("a failing embedder must not stop startup");
        assert!(has_node(&started.store, "late-node").await);
    }

    // ---- through the real serve path ----

    use crate::cli::Mode;
    use tokio::net::UnixStream;

    /// A daemon serving in a task, stopped through its shutdown token.
    struct Daemon {
        shutdown: Shutdown,
        task: tokio::task::JoinHandle<anyhow::Result<()>>,
        socket: std::path::PathBuf,
    }

    impl Daemon {
        async fn start(config: &Config) -> Daemon {
            let shutdown = Shutdown::default();
            let task = tokio::spawn(crate::serve_until(
                Mode::Serve,
                config.clone(),
                shutdown.clone(),
            ));
            let socket = std::path::PathBuf::from(&config.socket_path);
            let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
            while UnixStream::connect(&socket).await.is_err() {
                assert!(!task.is_finished(), "the daemon stopped before serving");
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "the daemon never served"
                );
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            Daemon {
                shutdown,
                task,
                socket,
            }
        }

        async fn stop(self) -> anyhow::Result<()> {
            self.shutdown.token().cancel();
            tokio::time::timeout(Duration::from_secs(20), self.task)
                .await
                .expect("the daemon must stop")
                .expect("the daemon task must not panic")
        }
    }

    #[derive(Clone)]
    struct TestClient;

    impl rmcp::ClientHandler for TestClient {
        fn get_info(&self) -> rmcp::model::ClientInfo {
            rmcp::model::InitializeRequestParams::new(
                rmcp::model::ClientCapabilities::default(),
                rmcp::model::Implementation::new("startup-test", "0.0.0"),
            )
        }
    }

    async fn call_tool(socket: &Path, tool: &str, arguments: serde_json::Value) -> String {
        use rmcp::ServiceExt;
        let stream = UnixStream::connect(socket).await.expect("connect");
        let client = TestClient.serve(stream).await.expect("handshake");
        let arguments = arguments.as_object().expect("an object").clone();
        let result = client
            .peer()
            .call_tool(
                rmcp::model::CallToolRequestParams::new(tool.to_string()).with_arguments(arguments),
            )
            .await
            .expect("the tool call");
        let text = result
            .content
            .first()
            .and_then(|block| block.as_text())
            .map(|text| text.text.clone())
            .expect("text content");
        let _ = client.cancel().await;
        text
    }

    fn remember(content: &str) -> serde_json::Value {
        serde_json::json!({
            "kind": "fact",
            "label": "label",
            "content": content,
            "scope": null,
            "subject": null,
        })
    }

    fn recall(query: &str) -> serde_json::Value {
        serde_json::json!({ "query": query, "kind": null, "scope": null, "k": 10 })
    }

    #[tokio::test]
    async fn a_database_deleted_while_the_daemon_is_down_comes_back_from_the_log_on_restart() {
        // Arrange: a daemon takes a write over its socket and stops
        let dir = tempfile::tempdir().unwrap();
        let config = config_in(dir.path());
        let daemon = Daemon::start(&config).await;
        let reply = call_tool(
            &daemon.socket,
            "remember",
            remember("survives the database"),
        )
        .await;
        assert!(reply.starts_with("remembered"), "{reply}");
        daemon.stop().await.expect("a clean stop");
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", config.database_path));
        }

        // Act: a new daemon over the same log and no database
        let daemon = Daemon::start(&config).await;
        let found = call_tool(&daemon.socket, "recall", recall("survives the database")).await;
        daemon.stop().await.expect("a clean stop");

        // Assert
        assert!(found.contains("survives the database"), "{found}");
    }

    #[tokio::test]
    async fn without_a_configured_dir_the_log_is_created_beside_the_database() {
        // Arrange: no [log] dir, and a database in a nested directory
        let dir = tempfile::tempdir().unwrap();
        let mut config = config_in(dir.path());
        config.log.dir = None;

        // Act
        let daemon = Daemon::start(&config).await;
        daemon.stop().await.expect("a clean stop");

        // Assert
        assert!(dir.path().join("liam.log").join(MANIFEST_NAME).is_file());
    }

    #[tokio::test]
    async fn a_relative_log_dir_is_taken_from_the_database_directory() {
        // Arrange: the working directory is somewhere else entirely
        let dir = tempfile::tempdir().unwrap();
        let mut config = config_in(dir.path());
        config.log.dir = Some("events".to_string());

        // Act
        let daemon = Daemon::start(&config).await;
        daemon.stop().await.expect("a clean stop");

        // Assert
        assert!(log_dir_in(dir.path()).join(MANIFEST_NAME).is_file());
    }

    #[tokio::test]
    async fn the_store_lock_is_taken_before_the_log_directory_is_created() {
        // Arrange: another process holds the store
        let dir = tempfile::tempdir().unwrap();
        let config = config_in(dir.path());
        let _held = liam_daemon::storelock::StoreLock::acquire(&database_in(dir.path())).unwrap();

        // Act
        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            crate::serve_with_store(Mode::Serve, config),
        )
        .await
        .expect("serve_with_store must fail on the lock instead of serving");

        // Assert
        let message = error_text(outcome);
        assert!(message.contains("store"), "{message}");
        assert!(
            !log_dir_in(dir.path()).exists(),
            "the log must not be created"
        );
    }

    #[tokio::test]
    async fn a_second_daemon_on_the_same_log_directory_is_refused() {
        // Arrange: one daemon serving, and another configured with a different
        // database and socket but the same log directory
        let dir = tempfile::tempdir().unwrap();
        let first = Daemon::start(&config_in(dir.path())).await;
        let mut second = config_in(dir.path());
        second.database_path = dir.path().join("second.db").to_str().unwrap().to_string();
        second.socket_path = dir.path().join("second.sock").to_str().unwrap().to_string();

        // Act
        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            crate::serve_with_store(Mode::Serve, second),
        )
        .await
        .expect("the second daemon must fail instead of serving");

        // Assert
        let message = error_text(outcome);
        assert!(message.contains("held by another process"), "{message}");
        assert!(!dir.path().join("second.sock").exists());
        first.stop().await.expect("the first daemon is unaffected");
    }

    #[tokio::test]
    async fn the_daemon_never_starts_serving_when_the_log_check_fails() {
        // Arrange: the same foreign-log database, through the real serve path
        let dir = tempfile::tempdir().unwrap();
        database_whose_log_was_deleted(dir.path()).await;
        let other = dir.path().join("elsewhere");
        let mut config = config_in(dir.path());
        config.log.dir = Some(other.to_str().unwrap().to_string());
        drop(standalone_log(&other));
        crate::embedder_fingerprint::record(
            Path::new(&config.database_path),
            &crate::embedder_fingerprint::EmbedderFingerprint::from_config(&config),
        )
        .unwrap();

        // Act: a daemon that went on to serve would never return
        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            crate::serve_with_store(Mode::Serve, config),
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

    #[tokio::test]
    async fn a_poisoned_log_stops_the_daemon_with_a_failure() {
        // Arrange: a daemon whose store is about to meet a poisoned log. The
        // WAL goes bad the way a failed fsync leaves it.
        let dir = tempfile::tempdir().unwrap();
        let log_dir = log_dir_in(dir.path());
        let shutdown = Shutdown::default();
        let (writer, poisoned) = ToggledWriter::open(&log_dir, false);
        let log = log_over(
            writer,
            Arc::new(
                liam_log::reader::SequentialScanReader::local(&log_dir).expect("open the reader"),
            ),
            Some(shutdown.poison_hook()),
        );
        let store = open_database(&config_in(dir.path()), &database_in(dir.path()))
            .await
            .unwrap();
        let store = open_store_on_log(store, &database_in(dir.path()), &log_dir, log, embedder())
            .await
            .unwrap();
        poisoned.store(true, Ordering::SeqCst);
        capture();

        // Act
        let refused = store.insert(NewNode::now("fact", "label", "x")).await;

        // Assert: the write says why, and the daemon is told to stop failing
        assert!(
            matches!(refused, Err(StoreError::LogPoisoned)),
            "{refused:?}"
        );
        assert!(shutdown.token().is_cancelled());
        let failure = shutdown.failure().expect("the exit must be a failure");
        assert!(failure.to_string().contains("restart liamd"), "{failure}");
        assert!(captured().contains("ERROR"), "{}", captured());
    }
}
