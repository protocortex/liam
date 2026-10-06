// SPDX-License-Identifier: Apache-2.0
//! Helpers the log tests share: a log to hand to a graph, and reads of what the
//! store holds.

use std::sync::Mutex as StdMutex;

use futures_util::stream;
use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::reader::{LogReader, LogStream};
use liam_log::{LogOffset, LogWriter};

use super::*;

/// A reader over a log nothing replays from.
struct EmptyReader;

impl LogReader for EmptyReader {
    fn scan(&self, _from: Option<LogOffset>) -> LogStream {
        Box::pin(stream::empty())
    }

    fn scan_through(&self, _from: Option<LogOffset>, _through: LogOffset) -> LogStream {
        Box::pin(stream::empty())
    }
}

/// An embedder that gives each text a vector of its own, records what it was
/// asked to embed, and can be told to fail on one text.
pub(super) struct StubEmbedder {
    dims: usize,
    calls: StdMutex<Vec<String>>,
    failing_on: StdMutex<Option<String>>,
}

impl StubEmbedder {
    pub(super) fn new(dims: usize) -> Arc<Self> {
        Arc::new(Self {
            dims,
            calls: StdMutex::default(),
            failing_on: StdMutex::default(),
        })
    }

    /// One hot at a position taken from the text, so two texts get different
    /// vectors and the same text always gets the same one.
    pub(super) fn vector_for(dims: usize, text: &str) -> Vec<f32> {
        let hash = text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        });
        let mut vector = vec![0.0; dims];
        vector[(hash % dims as u64) as usize] = 1.0;
        vector
    }

    pub(super) fn fail_on(&self, text: Option<&str>) {
        *self.failing_on.lock().unwrap() = text.map(str::to_owned);
    }

    /// The texts embedded so far, sorted.
    pub(super) fn calls(&self) -> Vec<String> {
        let mut calls = self.calls.lock().unwrap().clone();
        calls.sort();
        calls
    }
}

#[async_trait::async_trait]
impl ContentEmbedder for StubEmbedder {
    async fn embed(&self, text: &str) -> std::result::Result<Vec<f32>, EmbedError> {
        self.calls.lock().unwrap().push(text.to_owned());
        if self.failing_on.lock().unwrap().as_deref() == Some(text) {
            return Err("injected embed failure".into());
        }
        Ok(Self::vector_for(self.dims, text))
    }
}

/// A backend that lets a test interfere with a re-embed pass where it reads
/// and writes, and otherwise defers to a real one.
pub(super) struct ReembedProbe {
    inner: crate::backends::DefaultBackend,
    listing_fails: std::sync::atomic::AtomicBool,
    extra_listed: StdMutex<Option<NodeId>>,
    unreadable: StdMutex<Option<String>>,
    stored_after_listing: StdMutex<Option<(String, Vec<f32>)>>,
}

impl ReembedProbe {
    pub(super) fn fail_listing(&self) {
        self.listing_fails
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Lists `id` as missing a vector although no such node exists.
    pub(super) fn list_also(&self, id: NodeId) {
        *self.extra_listed.lock().unwrap() = Some(id);
    }

    /// Fails the content read of `id`.
    pub(super) fn fail_content_read(&self, id: &NodeId) {
        *self.unreadable.lock().unwrap() = Some(id.as_str().to_owned());
    }

    /// Stores `vector` for `id` right after the next listing, as a live write
    /// landing between the listing and the repair would.
    pub(super) fn store_after_listing(&self, id: &NodeId, vector: Vec<f32>) {
        *self.stored_after_listing.lock().unwrap() = Some((id.as_str().to_owned(), vector));
    }
}

#[async_trait::async_trait]
impl Backend for ReembedProbe {
    async fn open(path: &str, read_pool_size: usize) -> Result<Self> {
        Ok(Self {
            inner: crate::backends::DefaultBackend::open(path, read_pool_size).await?,
            listing_fails: std::sync::atomic::AtomicBool::new(false),
            extra_listed: StdMutex::new(None),
            unreadable: StdMutex::new(None),
            stored_after_listing: StdMutex::new(None),
        })
    }
    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>> {
        let unreadable = self.unreadable.lock().unwrap().clone();
        let reads_it = unreadable.is_some_and(|id| {
            sql.starts_with("SELECT content FROM nodes")
                && params
                    .iter()
                    .any(|param| matches!(param, Value::Text(text) if *text == id))
        });
        if reads_it {
            return Err(Error::Backend("injected content read failure".to_string()));
        }
        self.inner.query(sql, params).await
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
    async fn vector_insert_if_absent(&self, node_id: &str, embedding: &[f32]) -> Result<bool> {
        self.inner.vector_insert_if_absent(node_id, embedding).await
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
    async fn nodes_missing_vectors(&self) -> Result<Vec<NodeId>> {
        if self.listing_fails.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(Error::Backend("injected listing failure".to_string()));
        }
        let mut listed = self.inner.nodes_missing_vectors().await?;
        listed.extend(self.extra_listed.lock().unwrap().clone());
        let landed = self.stored_after_listing.lock().unwrap().take();
        if let Some((id, vector)) = landed {
            self.inner.vector_upsert(&id, &vector).await?;
        }
        Ok(listed)
    }
    async fn vector_sweep_orphans(&self) -> Result<u64> {
        self.inner.vector_sweep_orphans().await
    }
    async fn begin(&self) -> Result<Box<dyn crate::backend::BackendTx + '_>> {
        self.inner.begin().await
    }
}

pub(super) fn shared(log: EventLog) -> SharedLog {
    Arc::new(tokio::sync::Mutex::new(log))
}

/// A log over `writer` that cannot be replayed from, with the default filter.
pub(super) fn share(writer: impl LogWriter + 'static) -> SharedLog {
    share_sized(writer, BloomConfig::default())
}

/// `share` with a filter configured for `config`, as an operator would set it.
pub(super) fn share_sized(writer: impl LogWriter + 'static, config: BloomConfig) -> SharedLog {
    shared(EventLog::new(
        Box::new(writer),
        Arc::new(EmptyReader),
        HashBloom::new(config),
    ))
}

pub(super) fn fact(content: &str) -> NewNode {
    NewNode::now("fact", "label", content)
        .with_producer("agent-a")
        .with_confidence(0.75)
}

/// `fact` with a supplied valid time, so source, trust, valid time, and ingest
/// time are four different values on the log record.
pub(super) fn fact_at(content: &str) -> NewNode {
    fact(content).with_valid_from(Millis(500))
}

pub(super) async fn count<B: Backend>(g: &Graph<B>, table: &str) -> i64 {
    let rows = g
        .backend
        .query(&format!("SELECT COUNT(*) FROM {table}"), &[])
        .await
        .unwrap();
    rows[0].get_i64(0).unwrap()
}

/// `None` when the cursor row does not exist yet; otherwise its log id and
/// last applied offset (`None` while the offsets are still NULL).
pub(super) async fn cursor<B: Backend>(g: &Graph<B>) -> Option<(String, Option<(i64, i64)>)> {
    let rows = g
        .backend
        .query(
            "SELECT log_id, last_segment, last_index FROM log_cursor",
            &[],
        )
        .await
        .unwrap();
    let row = rows.first()?;
    let offset = match (&row.0[1], &row.0[2]) {
        (Value::Int(segment), Value::Int(index)) => Some((*segment, *index)),
        _ => None,
    };
    Some((row.get_string(0).unwrap(), offset))
}

pub(super) fn offset_pair(offset: LogOffset) -> (i64, i64) {
    (offset.segment as i64, offset.index as i64)
}
