// SPDX-License-Identifier: Apache-2.0
//! Opening a store on a log: `with_log` refuses a log that is not the one the
//! store's cursor belongs to, refuses a cursor the log has not reached, and
//! rebuilds the dedup filter from `log_hash_index` before any write is served.

use std::sync::atomic::{AtomicUsize, Ordering};

use liam_log::event::LogEvent;
use liam_log::wal::{WalConfig, WalError, WalWriter};
use liam_log::{LogOffset, LogWriter};
use uuid::Uuid;

use super::log_write::{count, cursor, fact_at, share};
use super::*;
use crate::DefaultGraph;

/// A log with a chosen identity and head that accepts no append.
struct FixedLog {
    log_id: Uuid,
    head: Option<LogOffset>,
}

impl FixedLog {
    fn new(head: Option<LogOffset>) -> Self {
        Self {
            log_id: Uuid::now_v7(),
            head,
        }
    }
}

impl LogWriter for FixedLog {
    fn append(&mut self, _event: &LogEvent) -> std::result::Result<LogOffset, WalError> {
        Err(WalError::Poisoned)
    }

    fn log_id(&self) -> Uuid {
        self.log_id
    }

    fn head(&self) -> Option<LogOffset> {
        self.head
    }
}

fn at(segment: u64, index: u64) -> LogOffset {
    LogOffset { segment, index }
}

fn db_path(dir: &TempDir) -> String {
    dir.path().join("graph.db").to_str().unwrap().to_owned()
}

async fn plain(path: &str) -> DefaultGraph {
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    DefaultGraph::open_with_clock(path, GraphConfig::new(8), clock)
        .await
        .unwrap()
}

async fn seed_cursor(g: &DefaultGraph, log_id: impl ToString, last: Option<LogOffset>) {
    let (segment, index) = match last {
        Some(offset) => (
            Value::Int(offset.segment as i64),
            Value::Int(offset.index as i64),
        ),
        None => (Value::Null, Value::Null),
    };
    g.backend
        .execute(
            "INSERT INTO log_cursor (id, log_id, last_segment, last_index) VALUES (1, ?1, ?2, ?3)",
            &[log_id.to_string().into(), segment, index],
        )
        .await
        .unwrap();
}

fn hash(n: u32) -> [u8; 32] {
    let mut hash = [0u8; 32];
    hash[..4].copy_from_slice(&n.to_be_bytes());
    hash[31] = 0xA5;
    hash
}

async fn index_hashes<B: Backend>(g: &Graph<B>, hashes: impl Iterator<Item = u32>) {
    for n in hashes {
        g.backend
            .execute(
                "INSERT INTO log_hash_index (content_hash, first_event_id, row_ids)
                 VALUES (?1, 'event', '[\"row\"]')",
                &[hash(n).to_vec().into()],
            )
            .await
            .unwrap();
    }
}

async fn knows(log: &SharedLog, hash: &[u8; 32]) -> bool {
    log.lock().await.bloom_might_contain(hash)
}

async fn wal_log(dir: &std::path::Path) -> SharedLog {
    let config = WalConfig {
        segment_max_bytes: 1 << 20,
        rotate_interval_secs: 3600,
    };
    share(WalWriter::open_with_system_clock(dir, config).expect("open wal"))
}

#[tokio::test]
async fn open_check_accepts_a_matching_log_id_with_the_cursor_at_the_head() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let writer = FixedLog::new(Some(at(2, 4)));
    let g = plain(&db_path(&dir)).await;
    seed_cursor(&g, writer.log_id, Some(at(2, 4))).await;

    // Act
    let opened = g.with_log(share(writer)).await;

    // Assert
    assert!(opened.is_ok(), "{:?}", opened.err());
}

#[tokio::test]
async fn open_check_refuses_a_log_id_that_differs_from_the_cursors_and_writes_nothing() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let other = Uuid::now_v7();
    let writer = FixedLog::new(Some(at(0, 0)));
    let log_id = writer.log_id;
    let g = plain(&db_path(&dir)).await;
    seed_cursor(&g, other, Some(at(0, 0))).await;
    index_hashes(&g, 0..1).await;
    let backend_view = plain(&db_path(&dir)).await;

    // Act
    let opened = g.with_log(share(writer)).await;

    // Assert
    match opened {
        Err(Error::LogIdMismatch { store, log }) => {
            assert_eq!((store, log), (other.to_string(), log_id));
        }
        other => panic!("expected LogIdMismatch, got {:?}", other.err()),
    }
    assert_eq!(
        cursor(&backend_view).await,
        Some((other.to_string(), Some((0, 0))))
    );
    assert_eq!(count(&backend_view, "nodes").await, 0);
    assert_eq!(count(&backend_view, "log_hash_index").await, 1);
}

#[tokio::test]
async fn open_check_refuses_a_cursor_beyond_the_log_head() {
    // Arrange: each case is a cursor and the head the log reports.
    let cases = [
        ("later index", at(1, 3), Some(at(1, 2))),
        ("later segment", at(2, 0), Some(at(1, 9))),
        ("empty log", at(0, 0), None),
    ];
    for (name, cursor_at, head) in cases {
        let dir = TempDir::new().unwrap();
        let writer = FixedLog::new(head);
        let g = plain(&db_path(&dir)).await;
        seed_cursor(&g, writer.log_id, Some(cursor_at)).await;

        // Act
        let opened = g.with_log(share(writer)).await;

        // Assert
        match opened {
            Err(Error::CursorBeyondLog {
                cursor,
                head: reported,
            }) => assert_eq!((cursor, reported), (cursor_at, head), "{name}"),
            other => panic!("{name}: expected CursorBeyondLog, got {:?}", other.err()),
        }
    }
}

#[tokio::test]
async fn open_check_accepts_a_cursor_behind_the_head_for_replay_to_close() {
    // Arrange: each case is a cursor and a head that is ahead of it.
    let cases = [
        ("same segment", Some(at(1, 1)), at(1, 4)),
        ("earlier segment", Some(at(0, 9)), at(1, 0)),
        ("nothing applied yet", None, at(0, 0)),
    ];
    for (name, cursor_at, head) in cases {
        let dir = TempDir::new().unwrap();
        let writer = FixedLog::new(Some(head));
        let g = plain(&db_path(&dir)).await;
        seed_cursor(&g, writer.log_id, cursor_at).await;

        // Act
        let opened = g.with_log(share(writer)).await;

        // Assert
        assert!(opened.is_ok(), "{name}: {:?}", opened.err());
    }
}

#[tokio::test]
async fn open_check_gives_a_fresh_store_a_cursor_row_with_the_log_id_and_no_offsets() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let writer = FixedLog::new(None);
    let log_id = writer.log_id.to_string();
    let g = plain(&db_path(&dir)).await;
    let before = cursor(&g).await;

    // Act
    let opened = g.with_log(share(writer)).await.unwrap();

    // Assert
    assert_eq!(before, None);
    assert_eq!(cursor(&opened).await, Some((log_id, None)));
}

#[tokio::test]
async fn open_check_keeps_a_cursorless_store_that_predates_a_log_with_history() {
    // Arrange: rows written before any log existed, and a log that has records.
    let dir = TempDir::new().unwrap();
    let g = plain(&db_path(&dir)).await;
    g.insert(fact_at("written before the log")).await.unwrap();
    let writer = FixedLog::new(Some(at(0, 3)));
    let log_id = writer.log_id.to_string();

    // Act
    let opened = g.with_log(share(writer)).await.unwrap();

    // Assert: the cursor is created with no offsets, and the rows are kept.
    assert_eq!(cursor(&opened).await, Some((log_id, None)));
    assert_eq!(count(&opened, "nodes").await, 1);
}

#[tokio::test]
async fn open_check_leaves_a_store_without_a_log_unchanged() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let g = plain(&db_path(&dir)).await;

    // Act
    let id = g.insert(fact_at("unlogged")).await;

    // Assert
    assert!(id.is_ok());
    assert_eq!(cursor(&g).await, None);
    assert_eq!(count(&g, "log_hash_index").await, 0);
}

#[tokio::test]
async fn a_reopened_store_on_the_same_wal_still_deduplicates() {
    // Arrange: X is written through a real WAL and the store is closed.
    let store_dir = TempDir::new().unwrap();
    let wal_dir = TempDir::new().unwrap();
    let path = db_path(&store_dir);
    let first_log = wal_log(wal_dir.path()).await;
    let g = plain(&path).await.with_log(first_log).await.unwrap();
    let first = g.insert(fact_at("X")).await.unwrap();
    drop(g);

    // Act: reopen on the same database and the same log, with a new filter.
    let reopened_log = wal_log(wal_dir.path()).await;
    let g = plain(&path)
        .await
        .with_log(Arc::clone(&reopened_log))
        .await
        .unwrap();
    let indexed = g
        .backend
        .query("SELECT content_hash FROM log_hash_index", &[])
        .await
        .unwrap();
    let second = g.insert(fact_at("X")).await.unwrap();

    // Assert: the filter knew X, so the index was consulted and decided.
    assert_eq!(indexed.len(), 1);
    assert_eq!(second, first);
    assert_eq!(count(&g, "nodes").await, 1);
}

#[tokio::test]
async fn the_filter_holds_every_indexed_hash_after_open() {
    // Arrange: hashes indexed before this process started.
    let dir = TempDir::new().unwrap();
    let g = plain(&db_path(&dir)).await;
    index_hashes(&g, 0..50).await;
    let log = share(FixedLog::new(None));

    // Act
    let _g = g.with_log(Arc::clone(&log)).await.unwrap();

    // Assert
    for n in 0..50 {
        assert!(knows(&log, &hash(n)).await, "hash {n} missing");
    }
}

/// Rows a single query may return while the filter is rebuilt: far below the
/// index size used here, so reading it whole would exceed the bound.
const PAGE_BOUND: usize = 1_000;
const INDEXED_ROWS: u32 = 3_000;

/// A backend that records the most rows any one `query` handed back.
struct Paged {
    inner: crate::backends::DefaultBackend,
    most_rows: AtomicUsize,
}

#[async_trait::async_trait]
impl Backend for Paged {
    async fn open(path: &str, read_pool_size: usize) -> Result<Self> {
        Ok(Self {
            inner: crate::backends::DefaultBackend::open(path, read_pool_size).await?,
            most_rows: AtomicUsize::new(0),
        })
    }
    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>> {
        let rows = self.inner.query(sql, params).await?;
        self.most_rows.fetch_max(rows.len(), Ordering::SeqCst);
        Ok(rows)
    }
    async fn execute(&self, sql: &str, params: &[Value]) -> Result<u64> {
        self.inner.execute(sql, params).await
    }
    async fn execute_batch(&self, sql: &str) -> Result<()> {
        self.inner.execute_batch(sql).await
    }
    async fn execute_atomic(&self, statements: &[(String, Vec<Value>)]) -> Result<()> {
        self.inner.execute_atomic(statements).await
    }
    fn vector_ddl(&self, dims: usize) -> String {
        self.inner.vector_ddl(dims)
    }
    async fn vector_upsert(&self, node_id: &str, embedding: &[f32]) -> Result<()> {
        self.inner.vector_upsert(node_id, embedding).await
    }
    async fn vector_delete(&self, node_id: &str) -> Result<()> {
        self.inner.vector_delete(node_id).await
    }
    async fn vector_search(
        &self,
        query: &[f32],
        k: usize,
        kind: Option<&str>,
        scope: Option<&str>,
        as_of: Millis,
    ) -> Result<Vec<NodeId>> {
        self.inner.vector_search(query, k, kind, scope, as_of).await
    }
    async fn vector_sweep_orphans(&self) -> Result<u64> {
        self.inner.vector_sweep_orphans().await
    }
    async fn begin(&self) -> Result<Box<dyn crate::backend::BackendTx + '_>> {
        self.inner.begin().await
    }
}

#[tokio::test]
async fn a_large_index_is_read_in_pages_while_the_filter_is_rebuilt() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let g = Graph::<Paged>::open(&db_path(&dir), GraphConfig::new(8))
        .await
        .unwrap();
    index_hashes(&g, 0..INDEXED_ROWS).await;
    let log = share(FixedLog::new(None));

    // Act
    let g = g.with_log(Arc::clone(&log)).await.unwrap();

    // Assert: every hash is in the filter, yet no query returned them all.
    for n in 0..INDEXED_ROWS {
        assert!(knows(&log, &hash(n)).await, "hash {n} missing");
    }
    let most = g.backend.most_rows.load(Ordering::SeqCst);
    assert!(
        most > 0 && most <= PAGE_BOUND,
        "the largest query returned {most} of {INDEXED_ROWS} rows"
    );
}
