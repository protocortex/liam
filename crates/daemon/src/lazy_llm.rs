// SPDX-License-Identifier: Apache-2.0
//! An `Llm` that loads its model on the first call and lets go of it after an
//! idle period. A background daemon should not hold a gigabyte of weights for a
//! model it uses a few times a day; the first call after an unload pays the
//! load instead.
//!
//! The load runs as its own task, so a caller that gives up (an `ask` deadline,
//! a client that went away) does not cancel it: the model finishes loading and
//! the next call finds it ready. A call in flight holds its own handle to the
//! model, so an unload never pulls the model out from under it.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use liam_model::{Llm, ModelError};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// How often the idle task looks at the model. An unload is therefore late by
/// at most this much, which is small next to the idle limit.
pub const IDLE_CHECK_INTERVAL: Duration = Duration::from_secs(30);

/// Builds the model. Blocking and expensive, so it always runs off the async
/// runtime threads.
pub type Loader = Arc<dyn Fn() -> Result<Arc<dyn Llm>, ModelError> + Send + Sync>;

/// A source of the current time, so a test moves it by hand instead of sleeping.
pub trait Clock: Send + Sync {
    fn now(&self) -> Instant;
}

pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// What every caller waiting on one load receives.
type LoadResult = Result<Arc<dyn Llm>, ModelError>;

enum Slot {
    NotLoaded,
    Loading(watch::Receiver<Option<LoadResult>>),
    Loaded {
        llm: Arc<dyn Llm>,
        last_used: Instant,
        /// Calls that hold the model right now. The idle unload waits for zero.
        in_flight: usize,
    },
}

/// Held for the length of one call. Dropping it, even from a cancelled
/// future, ends the call and restarts the idle period.
struct CallGuard {
    state: Arc<State>,
}

impl Drop for CallGuard {
    fn drop(&mut self) {
        let now = self.state.clock.now();
        if let Slot::Loaded {
            last_used,
            in_flight,
            ..
        } = &mut *self.state.lock()
        {
            *in_flight = in_flight.saturating_sub(1);
            *last_used = now;
        }
    }
}

struct State {
    slot: Mutex<Slot>,
    loader: Loader,
    clock: Arc<dyn Clock>,
}

impl State {
    fn lock(&self) -> std::sync::MutexGuard<'_, Slot> {
        self.slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

pub struct LazyLlm {
    state: Arc<State>,
    backend: &'static str,
}

impl LazyLlm {
    /// `backend` is the label the loaded model will report, known before the
    /// load so startup logs and the tuning cache key do not need the model.
    pub fn new(loader: Loader, backend: &'static str) -> Self {
        Self::with_clock(loader, backend, Arc::new(SystemClock))
    }

    pub fn with_clock(loader: Loader, backend: &'static str, clock: Arc<dyn Clock>) -> Self {
        Self {
            state: Arc::new(State {
                slot: Mutex::new(Slot::NotLoaded),
                loader,
                clock,
            }),
            backend,
        }
    }

    /// Drop the model when it has been unused for `idle_unload_secs`. `0` never
    /// unloads. Returns whether it dropped the model.
    pub fn unload_if_idle(&self, idle_unload_secs: u64) -> bool {
        if idle_unload_secs == 0 {
            return false;
        }
        let limit = Duration::from_secs(idle_unload_secs);
        let released = {
            let mut slot = self.state.lock();
            match &*slot {
                Slot::Loaded {
                    last_used,
                    in_flight: 0,
                    ..
                } if self.state.clock.now().saturating_duration_since(*last_used) >= limit => {
                    std::mem::replace(&mut *slot, Slot::NotLoaded)
                }
                _ => return false,
            }
        };
        // Freeing the weights can take a moment, so it happens off the lock.
        drop(released);
        tracing::info!(idle_unload_secs, "llm unloaded after idle");
        true
    }

    /// Start the task that unloads an idle model every `IDLE_CHECK_INTERVAL`
    /// until `cancel` fires. `None` when `idle_unload_secs` is `0`, since there
    /// is nothing to watch.
    pub fn spawn_idle_unloader(
        self: &Arc<Self>,
        idle_unload_secs: u64,
        cancel: CancellationToken,
    ) -> Option<JoinHandle<()>> {
        if idle_unload_secs == 0 {
            return None;
        }
        let llm = Arc::clone(self);
        Some(tokio::spawn(async move {
            loop {
                tokio::select! {
                    () = cancel.cancelled() => break,
                    () = tokio::time::sleep(IDLE_CHECK_INTERVAL) => {
                        llm.unload_if_idle(idle_unload_secs);
                    }
                }
            }
        }))
    }

    /// The loaded model and a guard that keeps it from being unloaded, loading
    /// the model first when absent. Concurrent callers share one load.
    async fn acquire(&self) -> Result<(Arc<dyn Llm>, CallGuard), ModelError> {
        loop {
            let mut pending = {
                let mut slot = self.state.lock();
                match &mut *slot {
                    Slot::Loaded {
                        llm,
                        last_used,
                        in_flight,
                    } => {
                        *in_flight += 1;
                        *last_used = self.state.clock.now();
                        let guard = CallGuard {
                            state: Arc::clone(&self.state),
                        };
                        return Ok((Arc::clone(llm), guard));
                    }
                    Slot::Loading(pending) => pending.clone(),
                    Slot::NotLoaded => {
                        let pending = self.start_load();
                        *slot = Slot::Loading(pending.clone());
                        pending
                    }
                }
            };
            let outcome = pending
                .wait_for(|result| result.is_some())
                .await
                .map_err(|_| ModelError::Llm("the model load stopped before it finished".into()))?
                .clone();
            outcome.expect("wait_for returned a finished load")?;
            // Loaded: take the model through the slot so the call is counted.
        }
    }

    /// Run the loader on a blocking thread, as its own task. The task records
    /// the outcome in the slot itself, so it completes even when every caller
    /// that waited on it has given up.
    fn start_load(&self) -> watch::Receiver<Option<LoadResult>> {
        let (done, pending) = watch::channel(None);
        let state = Arc::clone(&self.state);
        tokio::spawn(async move {
            let started = Instant::now();
            let loader = Arc::clone(&state.loader);
            let result: LoadResult = match tokio::task::spawn_blocking(move || loader()).await {
                Ok(result) => result,
                Err(error) => Err(ModelError::Llm(format!(
                    "the model load task failed: {error}"
                ))),
            };
            *state.lock() = match &result {
                Ok(llm) => Slot::Loaded {
                    llm: Arc::clone(llm),
                    last_used: state.clock.now(),
                    in_flight: 0,
                },
                Err(_) => Slot::NotLoaded,
            };
            match &result {
                Ok(_) => tracing::info!(elapsed = ?started.elapsed(), "llm loaded"),
                Err(error) => tracing::warn!(%error, "llm load failed; the next call retries"),
            }
            // No receiver left means every waiter gave up; the slot is already set.
            let _ = done.send(Some(result));
        });
        pending
    }

    fn loaded(&self) -> Option<Arc<dyn Llm>> {
        match &*self.state.lock() {
            Slot::Loaded { llm, .. } => Some(Arc::clone(llm)),
            _ => None,
        }
    }
}

#[async_trait]
impl Llm for LazyLlm {
    async fn complete(&self, system: &str, prompt: &str) -> liam_model::Result<String> {
        let (llm, _call) = self.acquire().await?;
        llm.complete(system, prompt).await
    }

    async fn complete_capped(
        &self,
        system: &str,
        prompt: &str,
        max_new_tokens: usize,
    ) -> liam_model::Result<String> {
        let (llm, _call) = self.acquire().await?;
        llm.complete_capped(system, prompt, max_new_tokens).await
    }

    fn backend(&self) -> &'static str {
        self.backend
    }

    async fn warmup(&self) -> liam_model::Result<()> {
        let (llm, _call) = self.acquire().await?;
        llm.warmup().await
    }

    async fn ensure_loaded(&self) -> liam_model::Result<()> {
        self.acquire().await.map(|_| ())
    }

    /// Counts only while the model is loaded. Loading is blocking and this call
    /// is not async, so an unloaded model answers `None` and the caller uses
    /// its own estimate, as it does for any provider that cannot count.
    fn count_tokens(&self, text: &str) -> Option<usize> {
        self.loaded()?.count_tokens(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::future::join_all;
    use futures_util::FutureExt;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;

    const REPLY: &str = "inner reply";
    const IDLE_SECS: u64 = 300;
    const CALLERS: usize = 8;
    const TOKENS: usize = 7;

    struct FakeClock {
        base: Instant,
        offset: Mutex<Duration>,
    }

    impl FakeClock {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                base: Instant::now(),
                offset: Mutex::new(Duration::ZERO),
            })
        }

        fn advance(&self, by: Duration) {
            *self.offset.lock().unwrap() += by;
        }
    }

    impl Clock for FakeClock {
        fn now(&self) -> Instant {
            self.base + *self.offset.lock().unwrap()
        }
    }

    /// Answers with `REPLY` and flips `dropped` when the model is freed.
    struct Inner {
        dropped: Arc<AtomicBool>,
        gate: Option<Mutex<mpsc::Receiver<()>>>,
        started: Option<mpsc::Sender<()>>,
        hang: bool,
    }

    impl Drop for Inner {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl Llm for Inner {
        async fn complete(&self, _system: &str, _prompt: &str) -> liam_model::Result<String> {
            if let Some(started) = &self.started {
                started.send(()).unwrap();
            }
            if let Some(gate) = &self.gate {
                let gate = gate.lock().unwrap();
                tokio::task::block_in_place(|| gate.recv().unwrap());
            }
            if self.hang {
                std::future::pending::<()>().await;
            }
            Ok(REPLY.to_string())
        }

        fn count_tokens(&self, _text: &str) -> Option<usize> {
            Some(TOKENS)
        }
    }

    struct Fixture {
        lazy: Arc<LazyLlm>,
        loads: Arc<AtomicUsize>,
        dropped: Arc<AtomicBool>,
        clock: Arc<FakeClock>,
    }

    /// A loader that fails its first `failures` attempts and, when `load_gate`
    /// is set, blocks each attempt until the gate is released.
    fn fixture(failures: usize, load_gate: Option<mpsc::Receiver<()>>) -> Fixture {
        let loads = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicBool::new(false));
        let clock = FakeClock::new();
        let gate = Mutex::new(load_gate);
        let loader: Loader = {
            let loads = Arc::clone(&loads);
            let dropped = Arc::clone(&dropped);
            Arc::new(move || {
                let attempt = loads.fetch_add(1, Ordering::SeqCst);
                if let Some(gate) = gate.lock().unwrap().as_ref() {
                    gate.recv().ok();
                }
                if attempt < failures {
                    return Err(ModelError::Llm("no weights".into()));
                }
                dropped.store(false, Ordering::SeqCst);
                Ok(Arc::new(Inner {
                    dropped: Arc::clone(&dropped),
                    gate: None,
                    started: None,
                    hang: false,
                }) as Arc<dyn Llm>)
            })
        };
        Fixture {
            lazy: Arc::new(LazyLlm::with_clock(loader, "test", clock.clone())),
            loads,
            dropped,
            clock,
        }
    }

    fn loads(fixture: &Fixture) -> usize {
        fixture.loads.load(Ordering::SeqCst)
    }

    fn idle_for(fixture: &Fixture, secs: u64) {
        fixture.clock.advance(Duration::from_secs(secs));
    }

    #[tokio::test]
    async fn building_the_wrapper_loads_nothing() {
        // Arrange / Act
        let fixture = fixture(0, None);

        // Assert
        assert_eq!(loads(&fixture), 0);
        assert!(fixture.lazy.count_tokens("x").is_none());
    }

    #[tokio::test]
    async fn the_first_call_loads_once_and_a_second_call_reuses_the_model() {
        // Arrange
        let fixture = fixture(0, None);

        // Act
        let first = fixture.lazy.complete("s", "p").await.unwrap();
        let second = fixture.lazy.complete_capped("s", "p", 4).await.unwrap();

        // Assert
        assert_eq!((first.as_str(), second.as_str()), (REPLY, REPLY));
        assert_eq!(loads(&fixture), 1);
        assert_eq!(fixture.lazy.count_tokens("x"), Some(TOKENS));
    }

    #[tokio::test]
    async fn concurrent_first_calls_share_one_load() {
        // Arrange
        let (open, gate) = mpsc::channel();
        let fixture = fixture(0, Some(gate));
        let mut all = Box::pin(join_all(
            (0..CALLERS).map(|_| fixture.lazy.complete("s", "p")),
        ));

        // Act: one poll registers every caller on the load, which the gate
        // holds open, before dropping the sender lets the load finish.
        assert!((&mut all).now_or_never().is_none());
        drop(open);
        let replies = all.await;

        // Assert
        assert!(replies
            .iter()
            .all(|reply| matches!(reply.as_deref(), Ok(REPLY))));
        assert_eq!(loads(&fixture), 1);
    }

    #[tokio::test]
    async fn a_failed_load_reaches_the_callers_and_the_next_call_retries() {
        // Arrange
        let fixture = fixture(1, None);

        // Act
        let failed = fixture.lazy.complete("s", "p").await;
        let retried = fixture.lazy.complete("s", "p").await;

        // Assert
        assert!(failed.unwrap_err().to_string().contains("no weights"));
        assert_eq!(retried.unwrap(), REPLY);
        assert_eq!(loads(&fixture), 2);
    }

    #[tokio::test]
    async fn an_idle_model_is_dropped_and_the_next_call_loads_again() {
        // Arrange
        let fixture = fixture(0, None);
        fixture.lazy.complete("s", "p").await.unwrap();
        idle_for(&fixture, IDLE_SECS);

        // Act
        let unloaded = fixture.lazy.unload_if_idle(IDLE_SECS);

        // Assert
        assert!(unloaded);
        assert!(fixture.dropped.load(Ordering::SeqCst));
        assert!(fixture.lazy.count_tokens("x").is_none());
        fixture.lazy.complete("s", "p").await.unwrap();
        assert_eq!(loads(&fixture), 2);
    }

    #[tokio::test]
    async fn a_model_used_inside_the_limit_stays_loaded() {
        // Arrange
        let fixture = fixture(0, None);
        fixture.lazy.complete("s", "p").await.unwrap();
        idle_for(&fixture, IDLE_SECS - 1);

        // Act
        let unloaded = fixture.lazy.unload_if_idle(IDLE_SECS);

        // Assert
        assert!(!unloaded);
        assert!(!fixture.dropped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn a_zero_limit_never_unloads() {
        // Arrange
        let fixture = fixture(0, None);
        fixture.lazy.complete("s", "p").await.unwrap();
        idle_for(&fixture, IDLE_SECS * 100);

        // Act
        let unloaded = fixture.lazy.unload_if_idle(0);

        // Assert
        assert!(!unloaded);
        assert!(!fixture.dropped.load(Ordering::SeqCst));
        assert!(fixture
            .lazy
            .spawn_idle_unloader(0, CancellationToken::new())
            .is_none());
    }

    #[tokio::test]
    async fn warming_up_loads_the_model_and_it_still_unloads_when_idle() {
        // Arrange
        let fixture = fixture(0, None);

        // Act
        fixture.lazy.warmup().await.unwrap();
        idle_for(&fixture, IDLE_SECS);
        let unloaded = fixture.lazy.unload_if_idle(IDLE_SECS);

        // Assert
        assert_eq!(loads(&fixture), 1);
        assert!(unloaded);
        assert!(fixture.dropped.load(Ordering::SeqCst));
    }

    fn calls_in_flight(lazy: &LazyLlm) -> usize {
        match &*lazy.state.lock() {
            Slot::Loaded { in_flight, .. } => *in_flight,
            _ => 0,
        }
    }

    struct BlockingLazy {
        lazy: Arc<LazyLlm>,
        clock: Arc<FakeClock>,
        dropped: Arc<AtomicBool>,
        release: mpsc::Sender<()>,
        started: mpsc::Receiver<()>,
    }

    /// A lazy wrapper whose model blocks (or hangs) inside a call and reports
    /// when the call has started.
    fn blocking_lazy(hang: bool) -> BlockingLazy {
        let (release, gate) = mpsc::channel();
        let (started_tx, started) = mpsc::channel();
        let dropped = Arc::new(AtomicBool::new(false));
        let clock = FakeClock::new();
        let loader: Loader = {
            let dropped = Arc::clone(&dropped);
            let gate = Mutex::new(Some(gate));
            let started_tx = Mutex::new(Some(started_tx));
            Arc::new(move || {
                Ok(Arc::new(Inner {
                    dropped: Arc::clone(&dropped),
                    gate: gate
                        .lock()
                        .unwrap()
                        .take()
                        .filter(|_| !hang)
                        .map(Mutex::new),
                    started: started_tx.lock().unwrap().take(),
                    hang,
                }) as Arc<dyn Llm>)
            })
        };
        let lazy = Arc::new(LazyLlm::with_clock(loader, "test", clock.clone()));
        BlockingLazy {
            lazy,
            clock,
            dropped,
            release,
            started,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_call_in_flight_blocks_the_unload_until_it_ends() {
        // Arrange: a model whose call blocks until released.
        let BlockingLazy {
            lazy,
            clock,
            dropped,
            release,
            started,
        } = blocking_lazy(false);
        let caller = {
            let lazy = Arc::clone(&lazy);
            tokio::spawn(async move { lazy.complete("s", "p").await })
        };
        tokio::task::spawn_blocking(move || started.recv().unwrap())
            .await
            .unwrap();
        clock.advance(Duration::from_secs(IDLE_SECS));

        // Act: the tick runs while the call is still inside the model.
        let unloaded_during_call = lazy.unload_if_idle(IDLE_SECS);
        release.send(()).unwrap();
        let reply = caller.await.unwrap();
        let unloaded_at_once = lazy.unload_if_idle(IDLE_SECS);
        clock.advance(Duration::from_secs(IDLE_SECS));
        let unloaded_after_limit = lazy.unload_if_idle(IDLE_SECS);

        // Assert
        assert!(!unloaded_during_call, "a call is still using the model");
        assert_eq!(reply.unwrap(), REPLY);
        assert!(
            !unloaded_at_once,
            "the idle period restarts when the call ends"
        );
        assert!(unloaded_after_limit);
        assert!(dropped.load(Ordering::SeqCst));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_dropped_call_leaves_no_call_in_flight_and_the_model_unloads() {
        // Arrange: a call that never finishes, then cancelled mid-flight.
        let BlockingLazy {
            lazy,
            clock,
            dropped,
            started,
            ..
        } = blocking_lazy(true);
        let caller = {
            let lazy = Arc::clone(&lazy);
            tokio::spawn(async move { lazy.complete("s", "p").await })
        };
        tokio::task::spawn_blocking(move || started.recv().unwrap())
            .await
            .unwrap();
        let during = calls_in_flight(&lazy);

        // Act
        caller.abort();
        assert!(caller.await.unwrap_err().is_cancelled());
        clock.advance(Duration::from_secs(IDLE_SECS));
        let unloaded = lazy.unload_if_idle(IDLE_SECS);

        // Assert
        assert_eq!(during, 1);
        assert_eq!(calls_in_flight(&lazy), 0);
        assert!(unloaded);
        assert!(dropped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn a_load_error_keeps_its_variant() {
        // Arrange
        let loader: Loader = Arc::new(|| Err(ModelError::Embed("bad vectors".into())));
        let lazy = LazyLlm::new(loader, "test");

        // Act
        let error = lazy.ensure_loaded().await.unwrap_err();

        // Assert
        assert!(matches!(error, ModelError::Embed(message) if message == "bad vectors"));
    }

    #[tokio::test]
    async fn a_load_goes_on_after_its_caller_gives_up() {
        // Arrange
        let (open, gate) = mpsc::channel();
        let fixture = fixture(0, Some(gate));
        let abandoned = fixture.lazy.complete("s", "p");
        let mut abandoned = Box::pin(abandoned);
        assert!((&mut abandoned).now_or_never().is_none());

        // Act: drop the caller mid-load, then let the load finish.
        drop(abandoned);
        drop(open);
        let reply = fixture.lazy.complete("s", "p").await;

        // Assert
        assert_eq!(reply.unwrap(), REPLY);
        assert_eq!(loads(&fixture), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn the_idle_task_unloads_on_its_interval_and_stops_on_cancel() {
        // Arrange
        let fixture = fixture(0, None);
        fixture.lazy.complete("s", "p").await.unwrap();
        let cancel = CancellationToken::new();
        let task = fixture
            .lazy
            .spawn_idle_unloader(IDLE_SECS, cancel.clone())
            .expect("a nonzero limit starts the task");
        idle_for(&fixture, IDLE_SECS);

        // Act
        tokio::time::sleep(IDLE_CHECK_INTERVAL + Duration::from_secs(1)).await;
        cancel.cancel();
        task.await.unwrap();

        // Assert
        assert!(fixture.dropped.load(Ordering::SeqCst));
    }
}
