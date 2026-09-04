// SPDX-License-Identifier: Apache-2.0
//! Cold-start concurrency tuning: benchmarked empirically, cached per model and backend.

use std::collections::VecDeque;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use liam_model::Llm;
use serde::{Deserialize, Serialize};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::ask::estimate_tokens;

/// Entries a `RollingWindow` keeps before the oldest is evicted.
const ROLLING_WINDOW_CAPACITY: usize = 50;

/// Fixed-capacity ring buffer of AIMD latency samples, shared process-wide
/// behind an `Arc<Mutex<_>>` so every connection's calls merge into one window.
pub(crate) struct RollingWindow {
    entries: VecDeque<(Duration, Duration)>,
    /// Calls ever recorded, unlike `entries.len()` which caps at
    /// `ROLLING_WINDOW_CAPACITY`; drives the AIMD evaluation trigger below.
    total_recorded: usize,
}

impl RollingWindow {
    pub(crate) fn new() -> Self {
        Self {
            entries: VecDeque::with_capacity(ROLLING_WINDOW_CAPACITY),
            total_recorded: 0,
        }
    }

    /// Records one call's queue wait and generation time, evicting the
    /// oldest entry first once the window is already full.
    pub(crate) fn record(&mut self, queue_wait: Duration, generation_time: Duration) {
        if self.entries.len() == ROLLING_WINDOW_CAPACITY {
            self.entries.pop_front();
        }
        self.entries.push_back((queue_wait, generation_time));
        self.total_recorded += 1;
    }

    /// True on every `ROLLING_WINDOW_CAPACITY`th call since construction:
    /// the AIMD evaluation trigger point.
    pub(crate) fn should_evaluate(&self) -> bool {
        self.total_recorded.is_multiple_of(ROLLING_WINDOW_CAPACITY)
    }

    /// Average end-to-end latency (`queue_wait + generation_time`) over
    /// whatever is currently in the window; zero when empty.
    pub(crate) fn average_latency(&self) -> Duration {
        if self.entries.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.entries.iter().map(|&(wait, gen)| wait + gen).sum();
        total / self.entries.len() as u32
    }

    /// Average queue wait alone, the numerator AIMD checks against
    /// generation time to decide whether waiting, not generating, dominates.
    pub(crate) fn average_queue_wait(&self) -> Duration {
        if self.entries.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.entries.iter().map(|&(wait, _)| wait).sum();
        total / self.entries.len() as u32
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Bundle of the `Arc`-shared AIMD state one evaluation needs, cloned
/// cheaply from `MemoryServer`'s own fields at the point of use.
#[derive(Clone)]
pub(crate) struct AimdHandles {
    pub(crate) generation_permits: Arc<Semaphore>,
    pub(crate) granted_capacity: Arc<AtomicUsize>,
    pub(crate) held_permit: Arc<Mutex<Option<OwnedSemaphorePermit>>>,
    pub(crate) evaluating: Arc<AtomicBool>,
    pub(crate) previous_window_average: Arc<Mutex<Option<Duration>>>,
}

/// Records one call's latency, spawning an AIMD evaluation of the window
/// that just closed every `ROLLING_WINDOW_CAPACITY`th call.
pub(crate) fn record_and_maybe_evaluate(
    window: &Arc<Mutex<RollingWindow>>,
    handles: AimdHandles,
    queue_wait: Duration,
    generation_time: Duration,
) {
    let averages = {
        let mut window = window.lock().expect("rolling window mutex poisoned");
        window.record(queue_wait, generation_time);
        window
            .should_evaluate()
            .then(|| (window.average_queue_wait(), window.average_latency()))
    };
    let Some((avg_queue_wait, avg_total)) = averages else {
        return;
    };
    tokio::spawn(async move {
        evaluate(handles, avg_queue_wait, avg_total, memory_ceiling()).await;
    });
}

/// Single-flight-guarded grow/shrink/no-op decision for one closed window.
/// Returns whether this call actually ran the decision, `false` if another evaluation was already in flight.
pub(crate) async fn evaluate(
    handles: AimdHandles,
    window_queue_wait: Duration,
    window_total: Duration,
    ceiling: usize,
) -> bool {
    if handles
        .evaluating
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return false;
    }
    decide(&handles, window_queue_wait, window_total, ceiling).await;
    handles.evaluating.store(false, Ordering::Release);
    true
}

/// As `evaluate`, but pauses right after the guard succeeds until `hooks.release` fires.
#[cfg(test)]
pub(crate) async fn evaluate_with_hooks(
    handles: AimdHandles,
    window_queue_wait: Duration,
    window_total: Duration,
    ceiling: usize,
    hooks: &EvaluationHooks,
) -> bool {
    if handles
        .evaluating
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return false;
    }
    hooks.reached_pause.notify_one();
    hooks.release.notified().await;
    hooks.decisions.fetch_add(1, Ordering::Relaxed);
    decide(&handles, window_queue_wait, window_total, ceiling).await;
    handles.evaluating.store(false, Ordering::Release);
    true
}

/// Test-only synchronization for overlapping two evaluation attempts
/// against the single-flight guard in `evaluate`.
#[cfg(test)]
pub(crate) struct EvaluationHooks {
    pub(crate) reached_pause: tokio::sync::Notify,
    pub(crate) release: tokio::sync::Notify,
    pub(crate) decisions: AtomicUsize,
}

#[cfg(test)]
impl EvaluationHooks {
    pub(crate) fn new() -> Self {
        Self {
            reached_pause: tokio::sync::Notify::new(),
            release: tokio::sync::Notify::new(),
            decisions: AtomicUsize::new(0),
        }
    }
}

/// A regression against the prior window shrinks; otherwise a queue-wait-
/// dominant window grows. `None` (first evaluation) always takes the grow check.
async fn decide(
    handles: &AimdHandles,
    window_queue_wait: Duration,
    window_total: Duration,
    ceiling: usize,
) {
    let previous = handles
        .previous_window_average
        .lock()
        .expect("previous window average mutex poisoned")
        .replace(window_total);

    match previous {
        Some(prev) if window_total > prev => shrink(handles).await,
        _ => {
            let generation_time = window_total
                .checked_sub(window_queue_wait)
                .unwrap_or(Duration::ZERO);
            if window_queue_wait > generation_time {
                grow(handles, ceiling);
            }
        }
    }
}

/// Releases a held-back permit if there is one; otherwise adds a new one
/// while under `ceiling`. Never both in the same call.
fn grow(handles: &AimdHandles, ceiling: usize) {
    let mut held = handles
        .held_permit
        .lock()
        .expect("held permit mutex poisoned");
    if held.take().is_some() {
        return;
    }
    drop(held);
    if handles.granted_capacity.load(Ordering::Relaxed) < ceiling {
        handles.generation_permits.add_permits(1);
        handles.granted_capacity.fetch_add(1, Ordering::Relaxed);
    }
}

/// Holds back one permit instead of releasing it, without touching
/// `granted_capacity`. A no-op if already shrunk or already at the floor.
async fn shrink(handles: &AimdHandles) {
    {
        let held = handles
            .held_permit
            .lock()
            .expect("held permit mutex poisoned");
        if held.is_some() || handles.granted_capacity.load(Ordering::Relaxed) <= 1 {
            return;
        }
    }
    if let Ok(permit) = handles.generation_permits.clone().acquire_owned().await {
        *handles
            .held_permit
            .lock()
            .expect("held permit mutex poisoned") = Some(permit);
    }
}

/// Measured KV-cache cost of one concurrent context, the same figure
/// `LlmConfig::max_concurrent_generations` documents.
const BYTES_PER_CONCURRENT_CONTEXT: u64 = 110 * 1024 * 1024;

/// Sane cap on benchmarked concurrency regardless of RAM, same "generous but
/// bounded" shape as `ask::MAX_ASK_EVIDENCE`.
const MAX_CONCURRENCY_CEILING: usize = 8;

/// Minimum throughput gain the next level must clear to be worth the memory.
const IMPROVEMENT_THRESHOLD: f64 = 0.10;

const BENCHMARK_SYSTEM: &str = "You are a helpful assistant.";
const BENCHMARK_PROMPT: &str = "Say one short sentence about the weather.";
const BENCHMARK_MAX_TOKENS: usize = 64;

const CACHE_FILE_NAME: &str = "concurrency_tuning.json";

/// Read available system RAM and turn it into a concurrency ceiling.
pub(crate) fn memory_ceiling() -> usize {
    let system = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram()),
    );
    compute_ceiling(system.available_memory())
}

/// At most half of `available_bytes` goes to this budget, and
/// `MAX_CONCURRENCY_CEILING` bounds it regardless of what that half implies.
fn compute_ceiling(available_bytes: u64) -> usize {
    let budget = available_bytes / 2;
    let by_memory = (budget / BYTES_PER_CONCURRENT_CONTEXT) as usize;
    by_memory.clamp(1, MAX_CONCURRENCY_CEILING)
}

/// Probes concurrency 1..=`ceiling`, stopping once the marginal gain drops
/// below `IMPROVEMENT_THRESHOLD`; returns the last level that cleared it.
pub(crate) async fn cold_start_benchmark(llm: &dyn Llm, ceiling: usize) -> usize {
    let ceiling = ceiling.max(1);
    let mut best_level = 1;
    let mut best_throughput = throughput_at(llm, 1).await;
    for level in 2..=ceiling {
        if best_throughput <= 0.0 {
            break;
        }
        let throughput = throughput_at(llm, level).await;
        let gain = (throughput - best_throughput) / best_throughput;
        if gain < IMPROVEMENT_THRESHOLD {
            break;
        }
        best_level = level;
        best_throughput = throughput;
    }
    best_level
}

/// Aggregate output tokens/sec running `level` concurrent completions.
async fn throughput_at(llm: &dyn Llm, level: usize) -> f64 {
    let calls: Vec<_> = (0..level)
        .map(|_| llm.complete_capped(BENCHMARK_SYSTEM, BENCHMARK_PROMPT, BENCHMARK_MAX_TOKENS))
        .collect();
    let start = tokio::time::Instant::now();
    let results = join_all(calls).await;
    let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);
    let total_tokens: usize = results
        .into_iter()
        .map(|r| match r {
            Ok(text) => llm
                .count_tokens(&text)
                .unwrap_or_else(|| estimate_tokens(&text)),
            Err(_) => 0,
        })
        .sum();
    total_tokens as f64 / elapsed
}

/// Hand-rolled join, driving a dynamic set of started futures to completion
/// concurrently, without a `futures`-crate dependency this crate lacks.
async fn join_all<T>(mut futures: Vec<Pin<Box<dyn Future<Output = T> + Send + '_>>>) -> Vec<T> {
    let mut results: Vec<Option<T>> = futures.iter().map(|_| None).collect();
    std::future::poll_fn(move |cx| {
        for (call, slot) in futures.iter_mut().zip(results.iter_mut()) {
            if slot.is_none() {
                if let std::task::Poll::Ready(value) = call.as_mut().poll(cx) {
                    *slot = Some(value);
                }
            }
        }
        if results.iter().all(Option::is_some) {
            std::task::Poll::Ready(std::mem::take(&mut results).into_iter().flatten().collect())
        } else {
            std::task::Poll::Pending
        }
    })
    .await
}

#[derive(Serialize, Deserialize)]
struct CacheEntry {
    model: String,
    backend: String,
    value: usize,
}

fn cache_path(cache_dir: &str) -> PathBuf {
    Path::new(cache_dir).join(CACHE_FILE_NAME)
}

/// `None` on a missing file, a parse error, or a fingerprint mismatch: all three mean "benchmark again".
pub(crate) fn load_cached(cache_dir: &str, model: &str, backend: &str) -> Option<usize> {
    let contents = std::fs::read_to_string(cache_path(cache_dir)).ok()?;
    let entry: CacheEntry = serde_json::from_str(&contents).ok()?;
    (entry.model == model && entry.backend == backend).then_some(entry.value)
}

/// Best-effort write: a failed write just costs a repeat benchmark next start.
pub(crate) fn save_cache(cache_dir: &str, model: &str, backend: &str, value: usize) {
    let entry = CacheEntry {
        model: model.to_string(),
        backend: backend.to_string(),
        value,
    };
    let Ok(json) = serde_json::to_string(&entry) else {
        return;
    };
    let path = cache_path(cache_dir);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = std::fs::write(&path, json) {
        tracing::warn!(error = %e, "failed to write concurrency tuning cache");
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    use async_trait::async_trait;
    use liam_model::Result;

    use super::*;

    const PLATEAU_OUTPUT: &str = "the weather is mild today";

    /// `Llm` double with a `tokio::time::pause`-driven latency knob keyed by
    /// in-flight call count.
    pub(crate) struct SequencedLatencyLlm {
        concurrency_latency: Box<dyn Fn(usize) -> Duration + Send + Sync>,
        in_flight: AtomicUsize,
    }

    impl SequencedLatencyLlm {
        pub(crate) fn new() -> Self {
            Self {
                concurrency_latency: Box::new(|_| Duration::ZERO),
                in_flight: AtomicUsize::new(0),
            }
        }

        /// Delay keyed by how many calls are in flight (1-based) when a call starts.
        pub(crate) fn with_concurrency_latency(
            mut self,
            f: impl Fn(usize) -> Duration + Send + Sync + 'static,
        ) -> Self {
            self.concurrency_latency = Box::new(f);
            self
        }
    }

    #[async_trait]
    impl Llm for SequencedLatencyLlm {
        async fn complete(&self, system: &str, prompt: &str) -> Result<String> {
            self.complete_capped(system, prompt, usize::MAX).await
        }

        async fn complete_capped(
            &self,
            _system: &str,
            _prompt: &str,
            _max_new_tokens: usize,
        ) -> Result<String> {
            let concurrency = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            let delay = (self.concurrency_latency)(concurrency);
            tokio::time::sleep(delay).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(PLATEAU_OUTPUT.to_string())
        }
    }

    #[tokio::test(start_paused = true)]
    async fn cold_start_benchmark_stops_at_the_concurrency_where_gains_plateau() {
        // Arrange: throughput improves through level 3; level 4 buys under
        // 10%, so the benchmark should stop at 3.
        let llm = SequencedLatencyLlm::new().with_concurrency_latency(|n| match n {
            1 => Duration::from_millis(100),
            2 => Duration::from_millis(110),
            3 => Duration::from_millis(120),
            _ => Duration::from_millis(150),
        });

        // Act
        let result = cold_start_benchmark(&llm, 5).await;

        // Assert
        assert_eq!(
            result, 3,
            "must stop at the plateau, not run to the ceiling"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cold_start_benchmark_never_regresses_below_the_safe_floor() {
        // Arrange: any concurrency beyond 1 is worse from the first extra
        // call, so nothing beats level 1.
        let llm = SequencedLatencyLlm::new().with_concurrency_latency(|n| match n {
            1 => Duration::from_millis(100),
            _ => Duration::from_millis(400),
        });

        // Act
        let result = cold_start_benchmark(&llm, 4).await;

        // Assert
        assert_eq!(result, 1, "must never return below the safe floor of 1");
    }

    #[test]
    fn load_cached_returns_the_stored_value_for_the_current_model_and_backend() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().to_str().expect("utf8 path");
        save_cache(cache_dir, "qwen3-1.7b", "llama.cpp/metal", 4);

        // Act
        let cached = load_cached(cache_dir, "qwen3-1.7b", "llama.cpp/metal");

        // Assert
        assert_eq!(cached, Some(4));
    }

    #[test]
    fn load_cached_returns_none_for_a_different_model_or_backend() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().to_str().expect("utf8 path");
        save_cache(cache_dir, "qwen3-1.7b", "llama.cpp/metal", 4);

        // Act
        let cached = load_cached(cache_dir, "qwen3-1.7b", "llama.cpp/cpu");

        // Assert
        assert_eq!(
            cached, None,
            "a backend mismatch must not return a stale value"
        );
    }

    struct AlwaysFailingLlm;

    #[async_trait]
    impl Llm for AlwaysFailingLlm {
        async fn complete(&self, _system: &str, _prompt: &str) -> Result<String> {
            Err(liam_model::ModelError::Llm("boom".into()))
        }
    }

    #[tokio::test(start_paused = true)]
    async fn cold_start_benchmark_stays_at_the_safe_floor_when_every_call_errors() {
        // Arrange: every probed level's calls all error, so throughput is
        // 0.0 at every level, including the level-1 baseline.
        let llm = AlwaysFailingLlm;

        // Act
        let result = cold_start_benchmark(&llm, 4).await;

        // Assert
        assert_eq!(
            result, 1,
            "a total outage during the benchmark must not climb past the safe floor"
        );
    }

    #[test]
    fn compute_ceiling_caps_at_the_maximum_regardless_of_available_ram() {
        // Arrange: far more RAM than the cap could ever need.
        let available = 1024u64 * 1024 * 1024 * 1024; // 1TB

        // Act
        let ceiling = compute_ceiling(available);

        // Assert
        assert_eq!(ceiling, 8);
    }

    #[test]
    fn compute_ceiling_scales_down_for_a_small_amount_of_ram() {
        // Arrange: half of 660MB budgeted at 110MB/context is exactly 3.
        let available = 660 * 1024 * 1024;

        // Act
        let ceiling = compute_ceiling(available);

        // Assert
        assert_eq!(ceiling, 3);
    }

    #[test]
    fn rolling_window_average_matches_hand_computed_expectation() {
        // Arrange: each of the 3 calls totals 100ms end-to-end.
        let mut window = RollingWindow::new();
        window.record(Duration::from_millis(10), Duration::from_millis(90));
        window.record(Duration::from_millis(20), Duration::from_millis(80));
        window.record(Duration::from_millis(30), Duration::from_millis(70));

        // Act
        let average = window.average_latency();

        // Assert
        assert_eq!(average, Duration::from_millis(100));
    }

    #[test]
    fn rolling_window_evicts_the_oldest_entries_once_full() {
        // Arrange: 10 slow calls, then 50 fast ones, past the 50 capacity.
        let mut window = RollingWindow::new();
        for _ in 0..10 {
            window.record(Duration::from_millis(500), Duration::from_millis(500));
        }
        for _ in 0..50 {
            window.record(Duration::from_millis(5), Duration::from_millis(5));
        }

        // Act
        let average = window.average_latency();

        // Assert: if any slow call still counted, the average could not be
        // this low; only the most recent 50 (the fast ones) contribute.
        assert_eq!(window.len(), 50);
        assert_eq!(average, Duration::from_millis(10));
    }

    #[test]
    fn rolling_window_record_stays_fast_under_a_thousand_calls() {
        // Arrange
        let mut window = RollingWindow::new();

        // Act
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            window.record(Duration::from_millis(1), Duration::from_millis(1));
        }
        let elapsed = start.elapsed();

        // Assert: a generous, CI-safe bound; widen it if it ever flakes on a
        // loaded runner, never delete the guard outright.
        assert!(
            elapsed < Duration::from_millis(50),
            "record took {elapsed:?} for 1000 calls"
        );
    }

    /// Fresh AIMD state with no prior evaluation and no held-back permit.
    fn test_handles(generation_permits: Arc<Semaphore>, granted_capacity: usize) -> AimdHandles {
        AimdHandles {
            generation_permits,
            granted_capacity: Arc::new(AtomicUsize::new(granted_capacity)),
            held_permit: Arc::new(Mutex::new(None)),
            evaluating: Arc::new(AtomicBool::new(false)),
            previous_window_average: Arc::new(Mutex::new(None)),
        }
    }

    /// Drives `count` concurrent acquire-generate-record cycles, the same
    /// shape `ask`'s `PermitTimer` and its permit acquire drive in production.
    async fn drive_window(
        llm: &SequencedLatencyLlm,
        permits: &Arc<Semaphore>,
        window: &Arc<Mutex<RollingWindow>>,
        count: usize,
    ) {
        let calls: Vec<Pin<Box<dyn Future<Output = ()> + Send + '_>>> = (0..count)
            .map(|_| {
                let permits = permits.clone();
                let window = window.clone();
                let call: Pin<Box<dyn Future<Output = ()> + Send + '_>> = Box::pin(async move {
                    let acquire_start = tokio::time::Instant::now();
                    let permit = permits.acquire_owned().await.expect("semaphore closed");
                    let queue_wait = acquire_start.elapsed();
                    let generation_start = tokio::time::Instant::now();
                    let _ = llm.complete_capped("s", "p", usize::MAX).await;
                    let generation_time = generation_start.elapsed();
                    drop(permit);
                    window
                        .lock()
                        .expect("rolling window mutex poisoned")
                        .record(queue_wait, generation_time);
                });
                call
            })
            .collect();
        join_all(calls).await;
    }

    /// The two figures `evaluate` needs from a window that just closed.
    fn window_averages(window: &Arc<Mutex<RollingWindow>>) -> (Duration, Duration) {
        let w = window.lock().expect("rolling window mutex poisoned");
        (w.average_queue_wait(), w.average_latency())
    }

    #[tokio::test(start_paused = true)]
    async fn aimd_grows_step_by_step_while_queue_wait_dominates() {
        // Arrange: flat generation regardless of concurrency, so a small
        // permit pool against 50 calls keeps queue-wait dominant each window.
        let llm =
            SequencedLatencyLlm::new().with_concurrency_latency(|_| Duration::from_millis(50));
        let permits = Arc::new(Semaphore::new(1));
        let window = Arc::new(Mutex::new(RollingWindow::new()));
        let handles = test_handles(permits.clone(), 1);
        let ceiling = 4;

        // Act / Assert: three windows, each should grow capacity by one
        for expected_capacity in [2usize, 3, 4] {
            drive_window(&llm, &permits, &window, ROLLING_WINDOW_CAPACITY).await;
            let (avg_queue_wait, avg_total) = window_averages(&window);
            let ran = evaluate(handles.clone(), avg_queue_wait, avg_total, ceiling).await;
            assert!(ran, "an uncontended evaluation must run");
            assert_eq!(
                handles.granted_capacity.load(Ordering::Relaxed),
                expected_capacity,
                "granted_capacity, not available_permits, is the source of truth"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn aimd_shrinks_when_a_grow_is_followed_by_a_contention_spike() {
        // Arrange: flat generation at concurrency 1, a severe spike from 2 on.
        let llm = SequencedLatencyLlm::new().with_concurrency_latency(|n| {
            if n <= 1 {
                Duration::from_millis(50)
            } else {
                Duration::from_millis(1500)
            }
        });
        let permits = Arc::new(Semaphore::new(1));
        let window = Arc::new(Mutex::new(RollingWindow::new()));
        let handles = test_handles(permits.clone(), 1);
        let ceiling = 4;

        // Act: window 1 is queue-wait dominant and grows to 2
        drive_window(&llm, &permits, &window, ROLLING_WINDOW_CAPACITY).await;
        let (qw1, total1) = window_averages(&window);
        evaluate(handles.clone(), qw1, total1, ceiling).await;
        assert_eq!(handles.granted_capacity.load(Ordering::Relaxed), 2);

        // Act: window 2 hits the spike the extra concurrency triggers
        drive_window(&llm, &permits, &window, ROLLING_WINDOW_CAPACITY).await;
        let (qw2, total2) = window_averages(&window);
        evaluate(handles.clone(), qw2, total2, ceiling).await;

        // Assert: it reverses the grow instead of only ever growing further
        assert_eq!(
            handles.granted_capacity.load(Ordering::Relaxed),
            2,
            "shrink does not touch granted_capacity"
        );
        assert!(
            handles
                .held_permit
                .lock()
                .expect("held permit lock")
                .is_some(),
            "a regression must shrink"
        );
        assert_eq!(permits.available_permits(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn aimd_releases_the_held_permit_instead_of_growing_on_top_of_it() {
        // Arrange: same grow-then-spike setup as the shrink scenario above.
        let llm = SequencedLatencyLlm::new().with_concurrency_latency(|n| {
            if n <= 1 {
                Duration::from_millis(50)
            } else {
                Duration::from_millis(1500)
            }
        });
        let permits = Arc::new(Semaphore::new(1));
        let window = Arc::new(Mutex::new(RollingWindow::new()));
        let handles = test_handles(permits.clone(), 1);
        let ceiling = 4;
        drive_window(&llm, &permits, &window, ROLLING_WINDOW_CAPACITY).await;
        let (qw1, total1) = window_averages(&window);
        evaluate(handles.clone(), qw1, total1, ceiling).await;
        drive_window(&llm, &permits, &window, ROLLING_WINDOW_CAPACITY).await;
        let (qw2, total2) = window_averages(&window);
        evaluate(handles.clone(), qw2, total2, ceiling).await;
        assert!(handles
            .held_permit
            .lock()
            .expect("held permit lock")
            .is_some());

        // Act: window 3 is back to one effective permit, flat generation
        // again, so queue-wait dominates with no regression versus window 2
        drive_window(&llm, &permits, &window, ROLLING_WINDOW_CAPACITY).await;
        let (qw3, total3) = window_averages(&window);
        let ran = evaluate(handles.clone(), qw3, total3, ceiling).await;

        // Assert: the held permit is released, not a new one added on top
        assert!(ran);
        assert!(handles
            .held_permit
            .lock()
            .expect("held permit lock")
            .is_none());
        assert_eq!(handles.granted_capacity.load(Ordering::Relaxed), 2);
        assert_eq!(permits.available_permits(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn aimd_never_changes_when_queue_wait_is_already_near_zero() {
        // Arrange: as many permits as calls, so nothing ever queues.
        let llm =
            SequencedLatencyLlm::new().with_concurrency_latency(|_| Duration::from_millis(10));
        let permits = Arc::new(Semaphore::new(ROLLING_WINDOW_CAPACITY));
        let window = Arc::new(Mutex::new(RollingWindow::new()));
        let handles = test_handles(permits.clone(), ROLLING_WINDOW_CAPACITY);
        let ceiling = ROLLING_WINDOW_CAPACITY + 4;

        // Act
        for _ in 0..3 {
            drive_window(&llm, &permits, &window, ROLLING_WINDOW_CAPACITY).await;
            let (avg_queue_wait, avg_total) = window_averages(&window);
            evaluate(handles.clone(), avg_queue_wait, avg_total, ceiling).await;
        }

        // Assert
        assert_eq!(
            handles.granted_capacity.load(Ordering::Relaxed),
            ROLLING_WINDOW_CAPACITY
        );
        assert!(handles
            .held_permit
            .lock()
            .expect("held permit lock")
            .is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn aimd_does_not_exceed_the_ceiling() {
        // Arrange: the same queue-wait-dominant curve as the grow scenario,
        // but capacity already sits at the ceiling.
        let llm =
            SequencedLatencyLlm::new().with_concurrency_latency(|_| Duration::from_millis(50));
        let permits = Arc::new(Semaphore::new(2));
        let window = Arc::new(Mutex::new(RollingWindow::new()));
        let handles = test_handles(permits.clone(), 2);
        let ceiling = 2;

        // Act
        for _ in 0..2 {
            drive_window(&llm, &permits, &window, ROLLING_WINDOW_CAPACITY).await;
            let (avg_queue_wait, avg_total) = window_averages(&window);
            evaluate(handles.clone(), avg_queue_wait, avg_total, ceiling).await;
        }

        // Assert
        assert_eq!(handles.granted_capacity.load(Ordering::Relaxed), 2);
        assert!(handles
            .held_permit
            .lock()
            .expect("held permit lock")
            .is_none());
        assert_eq!(permits.available_permits(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn aimd_first_evaluation_skips_the_did_it_help_check() {
        // Arrange: a dominant curve, but no window has ever been evaluated.
        let llm =
            SequencedLatencyLlm::new().with_concurrency_latency(|_| Duration::from_millis(50));
        let permits = Arc::new(Semaphore::new(1));
        let window = Arc::new(Mutex::new(RollingWindow::new()));
        let handles = test_handles(permits.clone(), 1);
        let ceiling = 4;
        assert!(handles
            .previous_window_average
            .lock()
            .expect("previous window lock")
            .is_none());

        // Act
        drive_window(&llm, &permits, &window, ROLLING_WINDOW_CAPACITY).await;
        let (avg_queue_wait, avg_total) = window_averages(&window);
        let ran = evaluate(handles.clone(), avg_queue_wait, avg_total, ceiling).await;

        // Assert: no panic, and a dominant first window still grows, which a
        // `0`-sentinel mistaken for a prior average would read as a regression.
        assert!(ran);
        assert_eq!(handles.granted_capacity.load(Ordering::Relaxed), 2);
        assert!(handles
            .held_permit
            .lock()
            .expect("held permit lock")
            .is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn aimd_single_flight_guard_lets_only_one_evaluation_run_at_a_time() {
        // Arrange: task 1 will pause right after acquiring the guard.
        let permits = Arc::new(Semaphore::new(4));
        let handles = test_handles(permits, 1);
        let hooks = Arc::new(EvaluationHooks::new());
        let ceiling = 4;
        let queue_wait = Duration::from_millis(100);
        let total = Duration::from_millis(150);

        let task1 = tokio::spawn({
            let handles = handles.clone();
            let hooks = hooks.clone();
            async move { evaluate_with_hooks(handles, queue_wait, total, ceiling, &hooks).await }
        });
        hooks.reached_pause.notified().await;

        // Act: task 2 only ever attempts the guard once task 1 is confirmed inside it
        let task2 = tokio::spawn({
            let handles = handles.clone();
            async move { evaluate(handles, queue_wait, total, ceiling).await }
        });
        let task2_ran = task2.await.expect("task 2 must not panic");
        assert!(!task2_ran, "task 2 must observe the guard and skip");

        hooks.release.notify_one();
        let task1_ran = task1.await.expect("task 1 must not panic");

        // Assert: exactly one evaluation ran the decision logic
        assert!(task1_ran);
        assert_eq!(hooks.decisions.load(Ordering::Relaxed), 1);
        assert_eq!(handles.granted_capacity.load(Ordering::Relaxed), 2);
    }
}
