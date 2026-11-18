// SPDX-License-Identifier: Apache-2.0
//! The loop with a provider that answers in batches: one submit per window of
//! misses, collection into the cache, and every crash point and expiry in
//! between. The provider is a scripted double; nothing here opens a socket.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use liam_store::{batch_key, FixedClock, GraphConfig, Millis};
use tempfile::TempDir;

use super::failures::tracking_id;
use super::tests::{facts_for, input, inputs, FakeRam, Provider, Script, MODEL_SHA};
use super::*;
use crate::deletion::batch_store::{BatchState, BatchStore, EntryFailure};
use crate::deletion::send_ledger::SendLedger;
use crate::extraction::batch_flow::{BatchApi, BatchFault, Crashed, RemoteResult, RemoteState};

const PROVIDER: &str = "anthropic";
const SENT_AT: i64 = 1_000;
const WINDOW: LogOffset = LogOffset {
    segment: 0,
    index: 100,
};

type Behaviour = Box<dyn Fn(&str, &ExtractRequest) -> Option<RemoteResult> + Send + Sync>;

/// A provider that holds batches and answers them when told to.
struct Remote {
    batches: Mutex<Vec<Vec<(String, ExtractRequest)>>>,
    state: Mutex<RemoteState>,
    behaviour: Mutex<Behaviour>,
    polls: AtomicUsize,
    collects: AtomicUsize,
}

impl Remote {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            batches: Mutex::new(Vec::new()),
            state: Mutex::new(RemoteState::InProgress),
            behaviour: Mutex::new(Box::new(|_, request| {
                Some(RemoteResult::Answer(facts_for(&request.prompt)))
            })),
            polls: AtomicUsize::new(0),
            collects: AtomicUsize::new(0),
        })
    }

    fn set_state(&self, state: RemoteState) {
        *self.state.lock().unwrap() = state;
    }

    fn answer_with(
        &self,
        behaviour: impl Fn(&str, &ExtractRequest) -> Option<RemoteResult> + Send + Sync + 'static,
    ) {
        *self.behaviour.lock().unwrap() = Box::new(behaviour);
    }

    fn submits(&self) -> usize {
        self.batches.lock().unwrap().len()
    }

    fn requests_in(&self, batch: usize) -> Vec<String> {
        self.batches.lock().unwrap()[batch]
            .iter()
            .map(|(key, _)| key.clone())
            .collect()
    }
}

#[async_trait]
impl BatchApi for Remote {
    fn provider(&self) -> &str {
        PROVIDER
    }

    async fn submit(&self, requests: &[(String, ExtractRequest)]) -> Result<String> {
        let mut batches = self.batches.lock().unwrap();
        batches.push(requests.to_vec());
        Ok(format!("batch-{}", batches.len() - 1))
    }

    async fn poll(&self, _batch_id: &str) -> Result<RemoteState> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        Ok(*self.state.lock().unwrap())
    }

    async fn collect(&self, batch_id: &str) -> Result<Vec<(String, RemoteResult)>> {
        self.collects.fetch_add(1, Ordering::SeqCst);
        let index: usize = batch_id.trim_start_matches("batch-").parse().unwrap();
        let batches = self.batches.lock().unwrap();
        let behaviour = self.behaviour.lock().unwrap();
        Ok(batches[index]
            .iter()
            .filter_map(|(key, request)| behaviour(key, request).map(|r| (key.clone(), r)))
            .collect())
    }
}

struct Env {
    dir: TempDir,
    store: DefaultGraph,
    cache: Arc<CacheStore>,
    config: ExtractionConfig,
    permits: Semaphore,
    ram: FakeRam,
    stopping: CancellationToken,
    script: Arc<Script>,
    provider: Provider,
}

impl Env {
    async fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let store = Self::open_store(&dir, "liam.db").await;
        let cache = Arc::new(CacheStore::open(dir.path().join("cache.jsonl")).unwrap());
        let script = Script::honest();
        Self {
            provider: Provider::new(&script),
            script,
            store,
            cache,
            config: ExtractionConfig {
                ram_budget_mb: 1,
                batch_events: 2,
                ..ExtractionConfig::default()
            },
            permits: Semaphore::new(1),
            ram: FakeRam(std::sync::atomic::AtomicU64::new(u64::MAX)),
            stopping: CancellationToken::new(),
            dir,
        }
    }

    async fn open_store(dir: &TempDir, name: &str) -> DefaultGraph {
        let path = dir.path().join(name);
        DefaultGraph::open(path.to_str().unwrap(), GraphConfig::new(8))
            .await
            .unwrap()
    }

    /// A store restored to before any fact was written, as a rollback leaves it.
    async fn roll_back(&mut self) {
        self.store = Self::open_store(&self.dir, "restored.db").await;
    }

    fn batches(&self) -> BatchStore {
        BatchStore::new(self.dir.path().join("batches.json"))
    }

    fn ledger(&self) -> SendLedger {
        SendLedger::new(self.dir.path().join("sends.jsonl"))
    }

    /// A flow over `remote`, as a fresh process would build it.
    fn flow(&self, remote: &Arc<Remote>) -> BatchFlow {
        BatchFlow::new(
            remote.clone(),
            self.batches(),
            self.ledger(),
            Arc::clone(&self.cache),
            Arc::new(FixedClock::new(Millis(SENT_AT))),
        )
    }

    async fn run(&self, flow: &BatchFlow, events: &[ExtractionInput]) -> Result<ExtractionReport> {
        let identity = JsonExtractor::identity(self.script.as_ref());
        let deps = ExtractionDeps {
            provider: &self.provider,
            identity,
            cache: Arc::clone(&self.cache),
            store: &self.store,
            config: &self.config,
            permits: &self.permits,
            free_ram: &self.ram,
            write_source: "consolidation:test-run",
            deadline: Instant::now() + std::time::Duration::from_secs(60),
            stopping: &self.stopping,
            window_end: WINDOW,
            gate: None,
            batch: Some(flow),
        };
        extract_events(events, deps).await
    }

    async fn facts_about(&self, n: u64) -> usize {
        self.store
            .facts_valid_at(
                &format!("person{n}"),
                None,
                Millis(super::tests::READ_AT),
                10,
            )
            .await
            .unwrap()
            .facts
            .len()
    }

    fn table(&self) -> Vec<crate::deletion::batch_store::BatchEntry> {
        self.batches().load().unwrap()
    }
}

#[tokio::test]
async fn a_window_of_misses_is_one_submit_and_the_cursor_stops_before_the_pending_events() {
    // Arrange
    let env = Env::new().await;
    let remote = Remote::new();
    let flow = env.flow(&remote);

    // Act
    let report = env.run(&flow, &inputs(4)).await.unwrap();

    // Assert
    assert_eq!(remote.submits(), 1);
    assert_eq!(remote.requests_in(0).len(), 2);
    assert_eq!(report.requests_submitted, 2);
    assert_eq!(report.events_in_batch, 4);
    assert_eq!(report.completed_offset, None);
    assert_eq!(report.facts_written, 0);
    assert_eq!(env.provider.opens(), 0, "no direct model call");
    let ledger = env.ledger().load().unwrap();
    assert_eq!(ledger.len(), 2);
    assert!(ledger
        .iter()
        .all(|row| row.provider == PROVIDER && row.sent_at == SENT_AT));
    let table = env.table();
    assert!(table
        .iter()
        .all(|entry| entry.state == BatchState::Submitted));
}

#[tokio::test]
async fn a_second_run_while_the_batch_is_submitted_sends_nothing_and_keeps_the_cursor() {
    // Arrange
    let env = Env::new().await;
    let remote = Remote::new();
    env.run(&env.flow(&remote), &inputs(4)).await.unwrap();

    // Act
    let report = env.run(&env.flow(&remote), &inputs(4)).await.unwrap();

    // Assert
    assert_eq!(remote.submits(), 1);
    assert_eq!(remote.polls.load(Ordering::SeqCst), 1);
    assert_eq!(report.requests_submitted, 0);
    assert_eq!(report.events_in_batch, 4);
    assert_eq!(report.completed_offset, None);
}

#[tokio::test]
async fn an_ended_batch_is_collected_into_the_cache_and_served_with_no_provider_call() {
    // Arrange
    let env = Env::new().await;
    let remote = Remote::new();
    env.run(&env.flow(&remote), &inputs(4)).await.unwrap();
    remote.set_state(RemoteState::Ended);

    // Act
    let report = env.run(&env.flow(&remote), &inputs(4)).await.unwrap();
    let again = env.run(&env.flow(&remote), &inputs(4)).await.unwrap();

    // Assert
    assert_eq!(report.facts_written, 4);
    assert_eq!(report.completed_offset, Some(WINDOW));
    assert_eq!(remote.submits(), 1);
    assert_eq!(remote.collects.load(Ordering::SeqCst), 1);
    assert_eq!(env.script.calls(), 0);
    assert_eq!(env.provider.opens(), 0);
    assert_eq!(
        again.facts_written, 0,
        "the same facts are not written twice"
    );
    assert_eq!(env.facts_about(1).await, 1);
    assert!(env
        .table()
        .iter()
        .all(|entry| entry.state == BatchState::Collected));
}

#[tokio::test]
async fn a_rolled_back_run_replays_from_the_cache_with_no_provider_call() {
    // Arrange
    let mut env = Env::new().await;
    let remote = Remote::new();
    env.run(&env.flow(&remote), &inputs(4)).await.unwrap();
    remote.set_state(RemoteState::Ended);
    env.run(&env.flow(&remote), &inputs(4)).await.unwrap();
    let table_before = env.table();
    env.roll_back().await;

    // Act
    let report = env.run(&env.flow(&remote), &inputs(4)).await.unwrap();

    // Assert
    assert_eq!(report.facts_written, 4);
    assert_eq!(report.completed_offset, Some(WINDOW));
    assert_eq!(remote.submits(), 1);
    assert_eq!(remote.collects.load(Ordering::SeqCst), 1);
    assert_eq!(env.table(), table_before);
}

/// Where a crash row stops the flow: while submitting, or while collecting.
#[derive(Clone, Copy)]
enum Phase {
    Submit,
    Collect,
}

#[tokio::test]
async fn a_crash_at_every_point_resumes_with_the_stated_submits_and_collects() {
    // fault, phase, submits in all, provider collects in all, possible duplicate sends on resume (two requests in the window)
    let rows = [
        (BatchFault::BeforeSubmit, Phase::Submit, 1, 1, 2),
        (BatchFault::AfterSubmitBeforeTable, Phase::Submit, 2, 1, 2),
        (BatchFault::AfterTable, Phase::Submit, 1, 1, 0),
        (BatchFault::AfterCollectBeforeCache, Phase::Collect, 1, 2, 0),
        (BatchFault::PartwayThroughCache(1), Phase::Collect, 1, 2, 0),
        (
            BatchFault::AfterCacheBeforeCollected,
            Phase::Collect,
            1,
            1,
            0,
        ),
    ];
    for (fault, phase, submits, collects, duplicates) in rows {
        // Arrange
        let env = Env::new().await;
        let remote = Remote::new();
        let events = inputs(4);
        let mut resumed_duplicates = 0;

        // Act
        match phase {
            Phase::Submit => {
                let crashed = env
                    .run(&env.flow(&remote).stopping_at(fault), &events)
                    .await;
                assert!(is_crash(crashed, fault), "{fault:?}");
                resumed_duplicates = env
                    .run(&env.flow(&remote), &events)
                    .await
                    .unwrap()
                    .possible_duplicate_sends;
            }
            Phase::Collect => {
                env.run(&env.flow(&remote), &events).await.unwrap();
                remote.set_state(RemoteState::Ended);
                let crashed = env
                    .run(&env.flow(&remote).stopping_at(fault), &events)
                    .await;
                assert!(is_crash(crashed, fault), "{fault:?}");
            }
        }
        remote.set_state(RemoteState::Ended);
        let finished = env.run(&env.flow(&remote), &events).await.unwrap();

        // Assert
        assert_eq!(remote.submits(), submits, "{fault:?} submits");
        assert_eq!(
            remote.collects.load(Ordering::SeqCst),
            collects,
            "{fault:?} collects"
        );
        assert_eq!(resumed_duplicates, duplicates, "{fault:?} duplicates");
        assert_eq!(finished.completed_offset, Some(WINDOW), "{fault:?}");
        for person in 0..4 {
            assert_eq!(
                env.facts_about(person).await,
                1,
                "{fault:?} person {person}"
            );
        }
    }
}

fn is_crash(result: Result<ExtractionReport>, fault: BatchFault) -> bool {
    matches!(result, Err(error) if matches!(error.downcast_ref::<Crashed>(), Some(Crashed(at)) if *at == fault))
}

#[tokio::test]
async fn a_result_for_a_request_the_table_no_longer_holds_is_dropped() {
    // Arrange
    let env = Env::new().await;
    let remote = Remote::new();
    env.run(&env.flow(&remote), &inputs(2)).await.unwrap();
    remote.set_state(RemoteState::Ended);
    remote.answer_with(|key, request| {
        // The provider also returns an answer for a request that was deleted.
        let _ = key;
        Some(RemoteResult::Answer(facts_for(&request.prompt)))
    });
    let held = env.table()[0].cache_key.clone();
    let mut table = env.table();
    table[0].cache_key = "someone-else".to_owned();
    env.batches().replace(table).unwrap();

    // Act
    env.run(&env.flow(&remote), &inputs(2)).await.unwrap();

    // Assert
    assert!(
        !env.cache.contains(&held),
        "the purged request's answer is dropped"
    );
}

#[tokio::test]
async fn an_answer_that_does_not_validate_counts_toward_quarantine_and_is_not_cached() {
    // Arrange
    let env = Env::new().await;
    let remote = Remote::new();
    remote.answer_with(|_, _| Some(RemoteResult::Answer("not json".to_owned())));
    env.run(&env.flow(&remote), &inputs(1)).await.unwrap();
    remote.set_state(RemoteState::Ended);

    // Act
    let report = env.run(&env.flow(&remote), &inputs(1)).await.unwrap();

    // Assert
    let key = batch_key(PROMPT_VERSION, MODEL_SHA, &[tracking_id(None, "e0")]);
    assert_eq!(env.store.batch_failures(&key).await.unwrap(), 1);
    assert_eq!(
        remote.submits(),
        1,
        "a failed single event is not sent again in the same run"
    );
    assert_eq!(report.requests_submitted, 0);
    assert_eq!(env.facts_about(0).await, 0);
    assert!(env.table().is_empty(), "the counted failure is cleared");
}

#[tokio::test]
async fn a_refused_request_stops_the_scope_and_is_not_blamed_on_the_event() {
    // Arrange
    let env = Env::new().await;
    let remote = Remote::new();
    remote.answer_with(|_, _| Some(RemoteResult::Refused));
    env.run(&env.flow(&remote), &inputs(1)).await.unwrap();
    remote.set_state(RemoteState::Ended);

    // Act
    let report = env.run(&env.flow(&remote), &inputs(1)).await.unwrap();

    // Assert
    let key = batch_key(PROMPT_VERSION, MODEL_SHA, &[tracking_id(None, "e0")]);
    assert_eq!(env.store.batch_failures(&key).await.unwrap(), 0);
    assert_eq!(report.batches_failed, 1);
    assert_eq!(report.completed_offset, None);
    assert_eq!(remote.submits(), 1);
}

#[tokio::test]
async fn an_expired_batch_caches_what_it_finished_and_resends_the_rest_once() {
    // Arrange
    let env = Env::new().await;
    let remote = Remote::new();
    env.run(&env.flow(&remote), &inputs(4)).await.unwrap();
    let first = remote.requests_in(0);
    let missing = first[1].clone();
    let finished = first[0].clone();
    remote.set_state(RemoteState::Expired);
    let skip = missing.clone();
    remote.answer_with(move |key, request| {
        (key != skip).then(|| RemoteResult::Answer(facts_for(&request.prompt)))
    });

    // Act
    let report = env.run(&env.flow(&remote), &inputs(4)).await.unwrap();

    // Assert
    assert!(env.cache.contains(&finished));
    assert!(!env.cache.contains(&missing));
    assert_eq!(report.requests_submitted, 1);
    assert_eq!(remote.submits(), 2);
    assert_eq!(remote.requests_in(1), std::slice::from_ref(&missing));
    let entry = env
        .table()
        .into_iter()
        .find(|e| e.cache_key == missing)
        .unwrap();
    assert_eq!((entry.state, entry.attempts), (BatchState::Submitted, 2));
}

#[tokio::test]
async fn a_second_expiry_fails_the_request_and_sends_nothing_a_third_time() {
    // Arrange
    let env = Env::new().await;
    let remote = Remote::new();
    remote.answer_with(|_, _| None);
    env.run(&env.flow(&remote), &inputs(2)).await.unwrap();
    remote.set_state(RemoteState::Expired);
    env.run(&env.flow(&remote), &inputs(2)).await.unwrap();

    // Act
    let report = env.run(&env.flow(&remote), &inputs(2)).await.unwrap();

    // Assert
    assert_eq!(remote.submits(), 2, "one send and one resend");
    assert_eq!(report.requests_submitted, 0);
    assert_eq!(report.batches_failed, 1);
    assert_eq!(report.completed_offset, None);
    assert!(env
        .table()
        .iter()
        .all(|entry| entry.failure != Some(EntryFailure::Invalid)));
}

#[tokio::test]
async fn a_cancelled_request_is_sent_again_as_a_new_first_send() {
    // Arrange
    let env = Env::new().await;
    let remote = Remote::new();
    env.run(&env.flow(&remote), &inputs(2)).await.unwrap();
    let mut table = env.table();
    for entry in &mut table {
        entry.state = BatchState::Cancelled;
    }
    env.batches().replace(table).unwrap();

    // Act
    let report = env.run(&env.flow(&remote), &inputs(2)).await.unwrap();

    // Assert
    assert_eq!(report.requests_submitted, 1);
    assert_eq!(remote.submits(), 2);
    assert_eq!(report.possible_duplicate_sends, 0);
    assert_eq!(env.table()[0].attempts, 1);
}

#[tokio::test]
async fn a_batch_the_provider_cannot_be_asked_about_is_left_for_the_next_run() {
    // Arrange
    struct Down;
    #[async_trait]
    impl BatchApi for Down {
        fn provider(&self) -> &str {
            PROVIDER
        }
        async fn submit(&self, _: &[(String, ExtractRequest)]) -> Result<String> {
            bail!("the provider is down")
        }
        async fn poll(&self, _: &str) -> Result<RemoteState> {
            bail!("the provider is down")
        }
        async fn collect(&self, _: &str) -> Result<Vec<(String, RemoteResult)>> {
            bail!("the provider is down")
        }
    }
    let env = Env::new().await;
    let remote = Remote::new();
    env.run(&env.flow(&remote), &inputs(1)).await.unwrap();
    let down = BatchFlow::new(
        Arc::new(Down),
        env.batches(),
        env.ledger(),
        Arc::clone(&env.cache),
        Arc::new(FixedClock::new(Millis(SENT_AT))),
    );

    // Act
    let polled = env.run(&down, &inputs(1)).await.unwrap();
    let fresh = Env::new().await;
    let sent = fresh
        .run(
            &BatchFlow::new(
                Arc::new(Down),
                fresh.batches(),
                fresh.ledger(),
                Arc::clone(&fresh.cache),
                Arc::new(FixedClock::new(Millis(SENT_AT))),
            ),
            &inputs(1),
        )
        .await
        .unwrap();

    // Assert
    assert_eq!(polled.completed_offset, None);
    assert_eq!(polled.events_in_batch, 1);
    assert_eq!(sent.batches_failed, 1, "a failed submit is reported");
    assert_eq!(sent.requests_submitted, 0);
    assert!(fresh.table().is_empty());
    let _ = input(0, None);
}

#[tokio::test]
async fn a_delete_cancels_a_pending_batch_and_a_late_collect_writes_nothing() {
    use crate::deletion::batch_lineage::{BatchCanceller, BatchLineage, ProviderCopies};
    use crate::deletion::job::{DeleteJob, JobStore};
    use crate::deletion::DeleteTarget;
    use liam_store::{LineageStore, LineageTarget};

    // Arrange
    struct Cancelling(AtomicUsize);
    #[async_trait]
    impl BatchCanceller for Cancelling {
        async fn cancel(&self, _: &str, _: &str) -> Result<ProviderCopies> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(ProviderCopies::Deleted)
        }
    }
    let env = Env::new().await;
    let remote = Remote::new();
    env.run(&env.flow(&remote), &inputs(2)).await.unwrap();
    let key = env.table()[0].cache_key.clone();
    let jobs = JobStore::new(env.dir.path().join("jobs.json"));
    let job = DeleteJob::new(
        "job".to_owned(),
        DeleteTarget::Event {
            event_id: "e0".to_owned(),
        },
        0,
        1,
    );
    jobs.save(&job).unwrap();
    let canceller = Arc::new(Cancelling(AtomicUsize::new(0)));
    let lineage = BatchLineage::new(
        env.batches(),
        env.ledger(),
        JobStore::new(env.dir.path().join("jobs.json")),
        canceller.clone(),
    );
    let target = LineageTarget {
        job_id: "job".to_owned(),
        event_ids: vec!["e0".to_owned()],
        ..LineageTarget::default()
    };

    // Act
    lineage.purge(&target).await.unwrap();
    remote.set_state(RemoteState::Ended);
    env.run(&env.flow(&remote), &[]).await.unwrap();

    // Assert
    let left = lineage.verify(&target).await.unwrap();
    assert_eq!(canceller.0.load(Ordering::SeqCst), 1);
    assert!(left.is_empty(), "{left:?}");
    assert_eq!(remote.collects.load(Ordering::SeqCst), 0);
    assert!(!env.cache.contains(&key));
    assert!(jobs.get("job").unwrap().unwrap().notices.is_empty());
}
