// SPDX-License-Identifier: Apache-2.0
//! Cold-start concurrency tuning: benchmarked empirically, cached per model and backend.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use liam_model::Llm;
use serde::{Deserialize, Serialize};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

use crate::ask::estimate_tokens;

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
}
