// SPDX-License-Identifier: Apache-2.0
//! Counting the results `recall` and `ask` return, and writing the counts to
//! the log in batches, off the request path.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use liam_daemon::consolidation::shared_store::SharedStore;
use liam_store::{Backend, Graph, LineageKey, Millis, NodeId, UsageEntry};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// A map that holds at most `capacity` keys and drops the oldest one to make
/// room, so a flood of distinct keys cannot grow it without bound.
pub struct BoundedMap<K, V> {
    capacity: usize,
    order: VecDeque<K>,
    entries: HashMap<K, V>,
    dropped: u64,
}

impl<K: Clone + Eq + Hash, V> BoundedMap<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            order: VecDeque::new(),
            entries: HashMap::new(),
            dropped: 0,
        }
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.entries.get_mut(key)
    }

    /// Sets `key`. A new key goes last in the order and, with the map full,
    /// displaces the oldest.
    pub fn insert(&mut self, key: K, value: V) {
        if let Some(existing) = self.entries.get_mut(&key) {
            *existing = value;
            return;
        }
        if self.entries.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
                self.dropped += 1;
            }
        }
        self.order.push_back(key.clone());
        self.entries.insert(key, value);
    }

    pub fn remove(&mut self, key: &K) {
        if self.entries.remove(key).is_some() {
            self.order.retain(|queued| queued != key);
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter()
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Keys displaced to make room since the map was made.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }
}

/// What the store offers a flush.
#[async_trait]
pub trait UsageSink: Send + Sync {
    async fn lineage_of(&self, node: &NodeId) -> liam_store::Result<Option<LineageKey>>;
    async fn usage_started(&self) -> liam_store::Result<bool>;
    async fn put_usage_logged(&self, day: i64, entries: &[UsageEntry]) -> liam_store::Result<()>;
}

#[async_trait]
impl<B: Backend> UsageSink for Graph<B> {
    async fn lineage_of(&self, node: &NodeId) -> liam_store::Result<Option<LineageKey>> {
        Graph::lineage_of(self, node).await
    }

    async fn usage_started(&self) -> liam_store::Result<bool> {
        Graph::usage_started(self).await
    }

    async fn put_usage_logged(&self, day: i64, entries: &[UsageEntry]) -> liam_store::Result<()> {
        Graph::put_usage_logged(self, day, entries).await
    }
}

/// The shared slot, so a flush always reaches the store a rollback last
/// reopened.
#[async_trait]
impl UsageSink for SharedStore {
    async fn lineage_of(&self, node: &NodeId) -> liam_store::Result<Option<LineageKey>> {
        let store = self.get().await.map_err(unavailable)?;
        store.lineage_of(node).await
    }

    async fn usage_started(&self) -> liam_store::Result<bool> {
        let store = self.get().await.map_err(unavailable)?;
        store.usage_started().await
    }

    async fn put_usage_logged(&self, day: i64, entries: &[UsageEntry]) -> liam_store::Result<()> {
        let store = self.get().await.map_err(unavailable)?;
        store.put_usage_logged(day, entries).await
    }
}

fn unavailable(error: anyhow::Error) -> liam_store::Error {
    liam_store::Error::Backend(error.to_string())
}

#[derive(Clone, Copy)]
struct Counted {
    hits: u32,
    last_access_at: Millis,
}

/// What one flush wrote.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlushReport {
    /// Lineages in the batch.
    pub entries: usize,
    /// Counted nodes whose lineage was gone, which are dropped.
    pub unresolved: usize,
}

/// Hits per node since the last flush, held in memory. Nothing here touches the
/// store, so a request pays a lock and a map update.
pub struct UsageRecorder {
    counts: Mutex<BoundedMap<NodeId, Counted>>,
}

impl UsageRecorder {
    pub fn new(max_tracked_items: usize) -> Self {
        Self {
            counts: Mutex::new(BoundedMap::new(max_tracked_items)),
        }
    }

    fn counts(&self) -> MutexGuard<'_, BoundedMap<NodeId, Counted>> {
        // A panic while counting leaves a count at worst a hit off.
        self.counts.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Counts one hit on each of `nodes` at `at`.
    pub fn record<'a>(&self, nodes: impl IntoIterator<Item = &'a NodeId>, at: Millis) {
        let mut counts = self.counts();
        for node in nodes {
            match counts.get_mut(node) {
                Some(counted) => {
                    counted.hits = counted.hits.saturating_add(1);
                    counted.last_access_at = counted.last_access_at.max(at);
                }
                None => counts.insert(
                    node.clone(),
                    Counted {
                        hits: 1,
                        last_access_at: at,
                    },
                ),
            }
        }
    }

    /// Nodes whose counts were displaced by newer ones.
    pub fn dropped(&self) -> u64 {
        self.counts().dropped()
    }

    /// Every counted node with its hits, by id, for tests.
    #[cfg(test)]
    pub fn hits(&self) -> Vec<(NodeId, u32)> {
        let mut hits: Vec<_> = self
            .pending()
            .into_iter()
            .map(|(node, counted)| (node, counted.hits))
            .collect();
        hits.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
        hits
    }

    fn pending(&self) -> Vec<(NodeId, Counted)> {
        self.counts()
            .iter()
            .map(|(node, counted)| (node.clone(), *counted))
            .collect()
    }

    /// What a flush wrote is taken off the counts, and what a read added while
    /// it ran is kept, so no hit is lost or counted twice.
    fn settle(&self, written: &[(NodeId, Counted)]) {
        let mut counts = self.counts();
        for (node, counted) in written {
            let Some(current) = counts.get_mut(node) else {
                continue;
            };
            current.hits = current.hits.saturating_sub(counted.hits);
            if current.hits == 0 {
                counts.remove(node);
            }
        }
    }

    /// Appends the counts, grouped by lineage, as one batch for the day of
    /// `now`. Nothing is written when nothing was counted. On failure the
    /// counts stay for the next flush.
    pub async fn flush(
        &self,
        sink: &dyn UsageSink,
        now: Millis,
    ) -> liam_store::Result<FlushReport> {
        let pending = self.pending();
        if pending.is_empty() {
            return Ok(FlushReport::default());
        }
        let mut grouped: BTreeMap<LineageKey, Counted> = BTreeMap::new();
        let mut unresolved = 0;
        for (node, counted) in &pending {
            let Some(lineage) = sink.lineage_of(node).await? else {
                unresolved += 1;
                continue;
            };
            let total = grouped.entry(lineage).or_insert(Counted {
                hits: 0,
                last_access_at: counted.last_access_at,
            });
            total.hits = total.hits.saturating_add(counted.hits);
            total.last_access_at = total.last_access_at.max(counted.last_access_at);
        }
        let entries: Vec<UsageEntry> = grouped
            .into_iter()
            .map(|(lineage_key, counted)| UsageEntry {
                lineage_key,
                hits: counted.hits,
                last_access_at: counted.last_access_at,
            })
            .collect();
        if !entries.is_empty() {
            sink.put_usage_logged(now.day_number(), &entries).await?;
        }
        self.settle(&pending);
        Ok(FlushReport {
            entries: entries.len(),
            unresolved,
        })
    }
}

/// Flushes a recorder on a timer until stopped, then once more.
pub struct UsageFlusher {
    recorder: Arc<UsageRecorder>,
    sink: Arc<dyn UsageSink>,
    period: Duration,
}

impl UsageFlusher {
    pub fn new(recorder: Arc<UsageRecorder>, sink: Arc<dyn UsageSink>, period: Duration) -> Self {
        Self {
            recorder,
            sink,
            period,
        }
    }

    /// Runs until `stop` is cancelled. The first thing it does is mark when
    /// counting began, if no batch says so yet, because a store that was never
    /// counted must not be judged on the absence of counts. The last thing is a
    /// flush, so the caller cancels `stop` only after the requests have drained.
    pub async fn run(self, stop: CancellationToken) {
        let mut marked = false;
        let mut ticks =
            tokio::time::interval_at(tokio::time::Instant::now() + self.period, self.period);
        ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        self.mark_start(&mut marked).await;
        loop {
            tokio::select! {
                () = stop.cancelled() => break,
                _ = ticks.tick() => {
                    self.mark_start(&mut marked).await;
                    self.flush("periodic").await;
                }
            }
        }
        self.mark_start(&mut marked).await;
        self.flush("shutdown").await;
        let dropped = self.recorder.dropped();
        if dropped > 0 {
            tracing::warn!(
                dropped,
                "usage counts were displaced because the map was full"
            );
        }
    }

    async fn mark_start(&self, marked: &mut bool) {
        if *marked {
            return;
        }
        let written = async {
            if !self.sink.usage_started().await? {
                self.sink
                    .put_usage_logged(Millis::now().day_number(), &[])
                    .await?;
            }
            Ok::<(), liam_store::Error>(())
        }
        .await;
        match written {
            Ok(()) => *marked = true,
            Err(error) => tracing::warn!(%error, "could not mark the start of usage counting"),
        }
    }

    async fn flush(&self, when: &'static str) {
        match self.recorder.flush(self.sink.as_ref(), Millis::now()).await {
            Ok(report) if report.entries > 0 => {
                tracing::debug!(when, entries = report.entries, "usage flushed");
            }
            Ok(_) => {}
            Err(error) => {
                tracing::warn!(when, %error, "usage flush failed; the counts are kept for the next one");
            }
        }
    }
}

/// A running flusher and what stops it.
pub struct UsageFlush {
    task: JoinHandle<()>,
    stop: CancellationToken,
}

impl UsageFlush {
    pub fn start(flusher: UsageFlusher) -> Self {
        let stop = CancellationToken::new();
        let task = tokio::spawn(flusher.run(stop.clone()));
        Self { task, stop }
    }

    /// Stops the flusher after its last flush, waiting up to `deadline` for it.
    /// Call it once the requests have drained, so their counts are in it.
    pub async fn stop(self, deadline: Duration) {
        self.stop.cancel();
        if tokio::time::timeout(deadline, self.task).await.is_err() {
            tracing::warn!("the usage flush is still running; its counts are lost at exit");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use liam_store::{FixedClock, NewNode};

    use super::super::test_support::logged_store;
    use super::*;

    const AT: Millis = Millis(5 * 86_400_000);
    const DIMS: usize = 8;

    fn id(text: &str) -> NodeId {
        NodeId::from_raw(text)
    }

    /// Records every batch it is given and can be made to fail.
    #[derive(Default)]
    struct StubSink {
        lineages: HashMap<String, String>,
        started: AtomicBool,
        failing: AtomicBool,
        batches: Mutex<Vec<(i64, Vec<UsageEntry>)>>,
    }

    impl StubSink {
        fn knowing(pairs: &[(&str, &str)]) -> Self {
            Self {
                lineages: pairs
                    .iter()
                    .map(|(node, lineage)| ((*node).to_string(), (*lineage).to_string()))
                    .collect(),
                ..Self::default()
            }
        }

        fn batches(&self) -> Vec<(i64, Vec<UsageEntry>)> {
            self.batches.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl UsageSink for StubSink {
        async fn lineage_of(&self, node: &NodeId) -> liam_store::Result<Option<LineageKey>> {
            Ok(self
                .lineages
                .get(node.as_str())
                .map(|lineage| LineageKey::from_raw(lineage.as_str())))
        }

        async fn usage_started(&self) -> liam_store::Result<bool> {
            Ok(self.started.load(Ordering::SeqCst))
        }

        async fn put_usage_logged(
            &self,
            day: i64,
            entries: &[UsageEntry],
        ) -> liam_store::Result<()> {
            if self.failing.load(Ordering::SeqCst) {
                return Err(liam_store::Error::Backend("the log is full".into()));
            }
            self.started.store(true, Ordering::SeqCst);
            self.batches.lock().unwrap().push((day, entries.to_vec()));
            Ok(())
        }
    }

    #[test]
    fn a_full_map_drops_the_oldest_key_and_counts_the_drop() {
        // Arrange
        let mut map = BoundedMap::new(2);
        map.insert("a", 1);
        map.insert("b", 2);

        // Act
        map.insert("c", 3);
        map.insert("b", 20);

        // Assert
        let mut keys: Vec<_> = map.iter().map(|(key, value)| (*key, *value)).collect();
        keys.sort();
        assert_eq!(keys, [("b", 20), ("c", 3)]);
        assert_eq!(map.dropped(), 1);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn updating_a_key_keeps_its_place_in_the_order() {
        // Arrange
        let mut map = BoundedMap::new(2);
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("a", 10);

        // Act
        map.insert("c", 3);

        // Assert: "a" was inserted first, so it is the one displaced
        assert!(map.get_mut(&"a").is_none());
        assert_eq!(map.dropped(), 1);
    }

    #[test]
    fn a_removed_key_frees_its_place() {
        // Arrange
        let mut map = BoundedMap::new(2);
        map.insert("a", 1);
        map.insert("b", 2);

        // Act
        map.remove(&"a");
        map.insert("c", 3);

        // Assert
        assert_eq!(map.dropped(), 0);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn more_distinct_nodes_than_the_bound_displace_the_oldest_and_count_it() {
        // Arrange
        let recorder = UsageRecorder::new(2);
        let (a, b, c) = (id("a"), id("b"), id("c"));

        // Act
        recorder.record([&a, &b, &c], AT);

        // Assert
        assert_eq!(recorder.dropped(), 1);
        assert_eq!(recorder.pending().len(), 2);
    }

    #[tokio::test]
    async fn a_flush_groups_by_lineage_sums_the_hits_and_keeps_the_latest_access() {
        // Arrange
        let sink = StubSink::knowing(&[("n1", "L1"), ("n2", "L1"), ("n3", "L2")]);
        let recorder = UsageRecorder::new(10);
        let (n1, n2, n3) = (id("n1"), id("n2"), id("n3"));
        recorder.record([&n1, &n2, &n3], Millis(AT.0 + 10));
        recorder.record([&n1], Millis(AT.0 + 30));
        recorder.record([&n2], Millis(AT.0 + 20));

        // Act
        let report = recorder.flush(&sink, AT).await.unwrap();

        // Assert
        let entry = |key: &str, hits: u32, at: i64| UsageEntry {
            lineage_key: LineageKey::from_raw(key),
            hits,
            last_access_at: Millis(AT.0 + at),
        };
        assert_eq!(
            report,
            FlushReport {
                entries: 2,
                unresolved: 0
            }
        );
        assert_eq!(
            sink.batches(),
            [(
                AT.day_number(),
                vec![entry("L1", 4, 30), entry("L2", 1, 10)]
            )]
        );
        assert!(recorder.pending().is_empty());
    }

    #[tokio::test]
    async fn two_flushes_with_identical_counts_are_two_batches() {
        // Arrange
        let sink = StubSink::knowing(&[("n1", "L1")]);
        let recorder = UsageRecorder::new(10);
        let n1 = id("n1");

        // Act
        recorder.record([&n1], AT);
        recorder.flush(&sink, AT).await.unwrap();
        recorder.record([&n1], AT);
        recorder.flush(&sink, AT).await.unwrap();

        // Assert
        let batches = sink.batches();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], batches[1]);
    }

    #[tokio::test]
    async fn a_flush_with_nothing_counted_writes_no_batch() {
        // Arrange
        let sink = StubSink::default();

        // Act
        let report = UsageRecorder::new(10).flush(&sink, AT).await.unwrap();

        // Assert
        assert_eq!(report, FlushReport::default());
        assert!(sink.batches().is_empty());
    }

    #[tokio::test]
    async fn a_node_whose_lineage_is_gone_is_dropped_and_counted() {
        // Arrange
        let sink = StubSink::knowing(&[("n1", "L1")]);
        let recorder = UsageRecorder::new(10);
        let (n1, gone) = (id("n1"), id("gone"));
        recorder.record([&n1, &gone], AT);

        // Act
        let report = recorder.flush(&sink, AT).await.unwrap();

        // Assert
        assert_eq!(
            report,
            FlushReport {
                entries: 1,
                unresolved: 1
            }
        );
        assert!(recorder.pending().is_empty());
    }

    #[tokio::test]
    async fn a_failed_flush_keeps_the_counts_and_the_next_one_writes_them_all() {
        // Arrange
        let sink = StubSink::knowing(&[("n1", "L1")]);
        let recorder = UsageRecorder::new(10);
        let n1 = id("n1");
        recorder.record([&n1, &n1], AT);
        sink.failing.store(true, Ordering::SeqCst);

        // Act
        let failed = recorder.flush(&sink, AT).await;
        recorder.record([&n1], AT);
        sink.failing.store(false, Ordering::SeqCst);
        let retried = recorder.flush(&sink, AT).await;

        // Assert
        assert!(failed.is_err());
        assert!(retried.is_ok());
        assert_eq!(sink.batches()[0].1[0].hits, 3);
    }

    #[tokio::test]
    async fn reads_racing_a_flush_are_neither_lost_nor_counted_twice() {
        // Arrange
        const READERS: usize = 8;
        const HITS_EACH: usize = 50;
        let sink = Arc::new(StubSink::knowing(&[("n1", "L1")]));
        let recorder = Arc::new(UsageRecorder::new(10));
        let barrier = Arc::new(tokio::sync::Barrier::new(READERS + 1));
        let readers: Vec<_> = (0..READERS)
            .map(|_| {
                let (recorder, barrier) = (recorder.clone(), barrier.clone());
                tokio::spawn(async move {
                    let n1 = id("n1");
                    barrier.wait().await;
                    for _ in 0..HITS_EACH {
                        recorder.record([&n1], AT);
                        tokio::task::yield_now().await;
                    }
                })
            })
            .collect();

        // Act
        barrier.wait().await;
        for _ in 0..HITS_EACH {
            recorder.flush(sink.as_ref(), AT).await.unwrap();
            tokio::task::yield_now().await;
        }
        for reader in readers {
            reader.await.unwrap();
        }
        recorder.flush(sink.as_ref(), AT).await.unwrap();

        // Assert
        let written: u32 = sink
            .batches()
            .iter()
            .flat_map(|(_, entries)| entries)
            .map(|entry| entry.hits)
            .sum();
        assert_eq!(written as usize, READERS * HITS_EACH);
    }

    #[tokio::test]
    async fn the_real_store_gets_one_entry_per_lineage_and_each_flush_adds_to_it() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = logged_store(dir.path(), Arc::new(FixedClock::new(AT)), DIMS).await;
        let first = store
            .insert(NewNode::now("fact", "ada", "ada's role").with_subject("ada"))
            .await
            .unwrap();
        let second = store
            .upsert_by(NewNode::now("fact", "ada", "ada's new role").with_subject("ada"))
            .await
            .unwrap();
        let recorder = UsageRecorder::new(10);

        // Act
        recorder.record([&first, &second], AT);
        let report = recorder.flush(&store, AT).await.unwrap();
        recorder.record([&second], AT);
        recorder.flush(&store, AT).await.unwrap();

        // Assert
        let snapshot = store.usage_snapshot(AT, 30).await.unwrap();
        let hits: i64 = snapshot
            .items
            .iter()
            .flat_map(|item| &item.hits_by_day)
            .map(|(_, hits)| hits)
            .sum();
        assert_eq!(report.entries, 1);
        assert_eq!(hits, 3);
    }

    #[tokio::test(start_paused = true)]
    async fn the_flusher_marks_the_start_then_flushes_each_period_and_once_more_when_stopped() {
        // Arrange
        let period = Duration::from_secs(60);
        let sink = Arc::new(StubSink::knowing(&[("n1", "L1")]));
        let recorder = Arc::new(UsageRecorder::new(10));
        let stop = CancellationToken::new();
        let task = tokio::spawn(
            UsageFlusher::new(recorder.clone(), sink.clone(), period).run(stop.clone()),
        );
        let n1 = id("n1");

        // Act
        tokio::time::sleep(Duration::from_secs(1)).await;
        let after_start = sink.batches();
        recorder.record([&n1], AT);
        tokio::time::sleep(period).await;
        let after_period = sink.batches();
        recorder.record([&n1], AT);
        stop.cancel();
        task.await.unwrap();

        // Assert
        assert_eq!(after_start.len(), 1);
        assert!(after_start[0].1.is_empty(), "the marker is an empty batch");
        assert!(after_period.len() >= 2);
        assert_eq!(after_period[1].1[0].hits, 1);
        let all = sink.batches();
        assert_eq!(
            all.last().unwrap().1[0].hits,
            1,
            "the stop flushed the rest"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_flusher_writes_no_marker_when_counting_has_started() {
        // Arrange
        let sink = Arc::new(StubSink::default());
        sink.started.store(true, Ordering::SeqCst);
        let stop = CancellationToken::new();
        let task = tokio::spawn(
            UsageFlusher::new(
                Arc::new(UsageRecorder::new(10)),
                sink.clone(),
                Duration::from_secs(60),
            )
            .run(stop.clone()),
        );

        // Act
        tokio::time::sleep(Duration::from_secs(1)).await;
        stop.cancel();
        task.await.unwrap();

        // Assert
        assert!(sink.batches().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn stopping_the_running_flusher_writes_what_was_counted_since_the_last_flush() {
        // Arrange
        let sink = Arc::new(StubSink::knowing(&[("n1", "L1")]));
        sink.started.store(true, Ordering::SeqCst);
        let recorder = Arc::new(UsageRecorder::new(10));
        let flush = UsageFlush::start(UsageFlusher::new(
            recorder.clone(),
            sink.clone(),
            Duration::from_secs(3600),
        ));
        tokio::time::sleep(Duration::from_secs(1)).await;
        recorder.record([&id("n1")], AT);

        // Act
        flush.stop(Duration::from_secs(2)).await;

        // Assert
        assert_eq!(sink.batches().len(), 1);
        assert_eq!(sink.batches()[0].1[0].hits, 1);
    }
}
