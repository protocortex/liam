// SPDX-License-Identifier: Apache-2.0
//! Cold-start concurrency tuning: benchmarked empirically, cached per model and backend.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use liam_model::{Llm, Result};
use serde::{Deserialize, Serialize};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};
use tokio::sync::Semaphore;

use crate::ask::estimate_tokens;

/// Per-token KV-cache byte cost, derived from the measured 110MB figure
/// `LlmConfig::max_concurrent_generations` documents at the shipped default
/// of 8192 tokens, so a configured `context_tokens` scales the per-slot cost
/// instead of assuming that default forever.
const KV_CACHE_BYTES_PER_TOKEN: u64 = (110 * 1024 * 1024) / 8192;

/// Sane cap on benchmarked concurrency regardless of RAM, same "generous but
/// bounded" shape as `ask::MAX_ASK_EVIDENCE`.
const MAX_CONCURRENCY_CEILING: usize = 8;

/// Minimum throughput gain the next level must clear to be worth the memory.
const IMPROVEMENT_THRESHOLD: f64 = 0.10;

const BENCHMARK_SYSTEM: &str = "You are a helpful assistant.";
const BENCHMARK_PROMPT: &str = "Say one short sentence about the weather.";
const BENCHMARK_MAX_TOKENS: usize = 64;

const CACHE_FILE_NAME: &str = "concurrency_tuning.json";

/// Read available system RAM and turn it into a concurrency ceiling, scaled
/// to the real per-slot KV-cache cost at `context_tokens`.
pub(crate) fn memory_ceiling(context_tokens: usize) -> usize {
    let system = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram()),
    );
    compute_ceiling(system.available_memory(), context_tokens)
}

/// At most half of `available_bytes` goes to this budget, and
/// `MAX_CONCURRENCY_CEILING` bounds it regardless of what that half implies.
fn compute_ceiling(available_bytes: u64, context_tokens: usize) -> usize {
    let budget = available_bytes / 2;
    let bytes_per_context = KV_CACHE_BYTES_PER_TOKEN * context_tokens as u64;
    let by_memory = (budget / bytes_per_context) as usize;
    by_memory.clamp(1, MAX_CONCURRENCY_CEILING)
}

/// Probes concurrency 1..=`ceiling`, stopping once the marginal gain drops
/// below `IMPROVEMENT_THRESHOLD`; returns the last level that cleared it.
/// `generation_permits` is the SAME semaphore real requests acquire, so the
/// probing below counts against the one memory budget instead of adding
/// extra live contexts on top of it.
pub(crate) async fn cold_start_benchmark(
    llm: &dyn Llm,
    ceiling: usize,
    generation_permits: &Arc<Semaphore>,
) -> usize {
    let ceiling = ceiling.max(1);
    let mut best_level = 1;
    let mut best_throughput = throughput_at(llm, 1, generation_permits).await;
    for level in 2..=ceiling {
        if best_throughput <= 0.0 {
            break;
        }
        let throughput = throughput_at(llm, level, generation_permits).await;
        let gain = (throughput - best_throughput) / best_throughput;
        if gain < IMPROVEMENT_THRESHOLD {
            break;
        }
        best_level = level;
        best_throughput = throughput;
    }
    best_level
}

/// Aggregate output tokens/sec running `level` concurrent completions. Each
/// call waits for its own permit from `generation_permits` first, same as a
/// real request, so an undersized semaphore serializes these calls instead
/// of letting them bypass the budget it enforces.
async fn throughput_at(llm: &dyn Llm, level: usize, generation_permits: &Arc<Semaphore>) -> f64 {
    let calls: Vec<Pin<Box<dyn Future<Output = Result<String>> + Send + '_>>> = (0..level)
        .map(|_| {
            let permits = generation_permits.clone();
            Box::pin(async move {
                let _permit = permits
                    .acquire_owned()
                    .await
                    .expect("generation_permits semaphore is never closed");
                llm.complete_capped(BENCHMARK_SYSTEM, BENCHMARK_PROMPT, BENCHMARK_MAX_TOKENS)
                    .await
            }) as Pin<Box<dyn Future<Output = Result<String>> + Send + '_>>
        })
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
    context_tokens: usize,
    value: usize,
}

fn cache_path(cache_dir: &str) -> PathBuf {
    Path::new(cache_dir).join(CACHE_FILE_NAME)
}

/// `None` on a missing file, a parse error, or a fingerprint mismatch: all
/// three mean "benchmark again". `context_tokens` is part of the
/// fingerprint because it changes the real per-slot memory cost (see
/// `memory_ceiling`), so a config change that raises or lowers it must not
/// silently reuse a value benchmarked under the old cost.
pub(crate) fn load_cached(
    cache_dir: &str,
    model: &str,
    backend: &str,
    context_tokens: usize,
) -> Option<usize> {
    let contents = std::fs::read_to_string(cache_path(cache_dir)).ok()?;
    let entry: CacheEntry = serde_json::from_str(&contents).ok()?;
    (entry.model == model && entry.backend == backend && entry.context_tokens == context_tokens)
        .then_some(entry.value)
}

/// Best-effort write: a failed write just costs a repeat benchmark next start.
pub(crate) fn save_cache(
    cache_dir: &str,
    model: &str,
    backend: &str,
    value: usize,
    context_tokens: usize,
) {
    let entry = CacheEntry {
        model: model.to_string(),
        backend: backend.to_string(),
        context_tokens,
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
    use std::sync::atomic::{AtomicUsize, Ordering};
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
        let permits = Arc::new(Semaphore::new(5));

        // Act
        let result = cold_start_benchmark(&llm, 5, &permits).await;

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
        let permits = Arc::new(Semaphore::new(4));

        // Act
        let result = cold_start_benchmark(&llm, 4, &permits).await;

        // Assert
        assert_eq!(result, 1, "must never return below the safe floor of 1");
    }

    #[tokio::test(start_paused = true)]
    async fn throughput_at_serializes_calls_beyond_the_semaphores_capacity() {
        // Arrange: 1 permit for 2 concurrent calls, each taking 100ms, so a
        // benchmark honoring the semaphore must run them one after another.
        let llm =
            SequencedLatencyLlm::new().with_concurrency_latency(|_| Duration::from_millis(100));
        let permits = Arc::new(Semaphore::new(1));

        // Act
        let start = tokio::time::Instant::now();
        let throughput = throughput_at(&llm, 2, &permits).await;

        // Assert: serialized by the single permit takes 200ms, not the 100ms
        // genuine concurrency would; a benchmark bypassing the semaphore
        // would finish in 100ms instead.
        assert!(
            start.elapsed() >= Duration::from_millis(200),
            "an undersized semaphore must serialize the calls, not let them bypass it"
        );
        assert!(throughput > 0.0);
    }

    #[test]
    fn load_cached_returns_the_stored_value_for_the_current_model_and_backend() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().to_str().expect("utf8 path");
        save_cache(cache_dir, "qwen3-1.7b", "llama.cpp/metal", 4, 8192);

        // Act
        let cached = load_cached(cache_dir, "qwen3-1.7b", "llama.cpp/metal", 8192);

        // Assert
        assert_eq!(cached, Some(4));
    }

    #[test]
    fn load_cached_returns_none_for_a_different_model_or_backend() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().to_str().expect("utf8 path");
        save_cache(cache_dir, "qwen3-1.7b", "llama.cpp/metal", 4, 8192);

        // Act
        let cached = load_cached(cache_dir, "qwen3-1.7b", "llama.cpp/cpu", 8192);

        // Assert
        assert_eq!(
            cached, None,
            "a backend mismatch must not return a stale value"
        );
    }

    #[test]
    fn load_cached_returns_none_for_a_different_context_tokens() {
        // Arrange: a value cached while context_tokens was 8192.
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().to_str().expect("utf8 path");
        save_cache(cache_dir, "qwen3-1.7b", "llama.cpp/metal", 4, 8192);

        // Act: the operator raised context_tokens in liam.toml since then.
        let cached = load_cached(cache_dir, "qwen3-1.7b", "llama.cpp/metal", 4096);

        // Assert
        assert_eq!(
            cached, None,
            "a context_tokens change must force a fresh benchmark, not reuse a stale value"
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
        let permits = Arc::new(Semaphore::new(4));

        // Act
        let result = cold_start_benchmark(&llm, 4, &permits).await;

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
        let ceiling = compute_ceiling(available, 8192);

        // Assert
        assert_eq!(ceiling, 8);
    }

    #[test]
    fn compute_ceiling_scales_down_for_a_small_amount_of_ram() {
        // Arrange: half of 660MB budgeted at 110MB/context is exactly 3.
        let available = 660 * 1024 * 1024;

        // Act
        let ceiling = compute_ceiling(available, 8192);

        // Assert
        assert_eq!(ceiling, 3);
    }

    #[test]
    fn compute_ceiling_drops_when_context_tokens_doubles() {
        // Arrange: same 660MB as the small-RAM test, but a doubled context
        // window doubles the real per-slot KV-cache cost.
        let available = 660 * 1024 * 1024;

        // Act
        let ceiling = compute_ceiling(available, 16384);

        // Assert: 3 at 8192 tokens roughly halves under integer truncation.
        assert_eq!(
            ceiling, 1,
            "doubling context_tokens must lower the ceiling, not leave it unchanged"
        );
    }

    #[test]
    fn compute_ceiling_rises_when_context_tokens_halves() {
        // Arrange: same 660MB as the small-RAM test, but a halved context
        // window halves the real per-slot KV-cache cost.
        let available = 660 * 1024 * 1024;

        // Act
        let ceiling = compute_ceiling(available, 4096);

        // Assert: roughly double the 8192-token result of 3, still bounded.
        assert_eq!(
            ceiling, 6,
            "halving context_tokens must raise the ceiling, bounded by MAX_CONCURRENCY_CEILING"
        );
    }
}
