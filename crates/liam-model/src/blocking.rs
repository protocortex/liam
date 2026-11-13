// SPDX-License-Identifier: Apache-2.0
//! Runs a sync model call on a blocking thread.

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::error::{ModelError, Result};

/// Run `call` on a blocking thread with exclusive access to `model`, so the
/// async runtime stays free.
///
/// The owned async lock is taken before the blocking job exists, so a caller
/// that gives up waiting never queues a job behind the lock. A failed join
/// becomes `error(message)`, so each caller keeps its own `ModelError` variant.
pub(crate) async fn with_model_blocking<T, R>(
    model: Arc<Mutex<T>>,
    error: fn(String) -> ModelError,
    call: impl FnOnce(&mut T) -> Result<R> + Send + 'static,
) -> Result<R>
where
    T: Send + 'static,
    R: Send + 'static,
{
    let mut guard = model.lock_owned().await;
    tokio::task::spawn_blocking(move || call(&mut guard))
        .await
        .map_err(|e| error(e.to_string()))?
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::sync::Mutex;

    use super::*;

    const INITIAL: i32 = 1;
    const THIRD_CALL_VALUE: i32 = 7;
    const SECOND_CALL_VALUE: i32 = 3;
    const SET_BEFORE_PANIC: i32 = 5;

    #[tokio::test]
    async fn returns_the_call_result_with_exclusive_access_to_the_model() {
        // Arrange
        let model = Arc::new(Mutex::new(INITIAL));

        // Act
        let out = with_model_blocking(model.clone(), ModelError::Embed, |m| {
            *m += 1;
            Ok(*m)
        })
        .await;

        // Assert
        assert_eq!(out.unwrap(), INITIAL + 1);
        assert_eq!(*model.lock().await, INITIAL + 1);
    }

    #[tokio::test]
    async fn passes_the_calls_own_error_through() {
        // Arrange
        let model = Arc::new(Mutex::new(0));

        // Act
        let err = with_model_blocking(model, ModelError::Embed, |_| -> Result<()> {
            Err(ModelError::Rerank("boom".into()))
        })
        .await
        .unwrap_err();

        // Assert
        assert_eq!(err.to_string(), "reranker: boom");
    }

    #[tokio::test]
    async fn a_panicking_call_maps_to_the_callers_error_variant() {
        // Arrange
        let model = Arc::new(Mutex::new(0));

        // Act
        let err = with_model_blocking(model, ModelError::Embed, |_| -> Result<()> {
            panic!("model crashed")
        })
        .await
        .unwrap_err();

        // Assert
        assert!(err.to_string().starts_with("embedder: "), "{err}");
    }

    #[tokio::test]
    async fn a_panic_leaves_the_lock_free_and_the_model_as_the_call_left_it() {
        // Arrange
        let model = Arc::new(Mutex::new(0));
        let _ = with_model_blocking(model.clone(), ModelError::Embed, |m| -> Result<()> {
            *m = SET_BEFORE_PANIC;
            panic!("model crashed")
        })
        .await;

        // Act
        let next = with_model_blocking(model, ModelError::Embed, |m| Ok(*m)).await;

        // Assert
        assert_eq!(next.unwrap(), SET_BEFORE_PANIC);
    }

    #[tokio::test]
    async fn a_dropped_waiter_never_runs_its_closure() {
        // Arrange
        let model = Arc::new(Mutex::new(0));
        let guard = model.clone().lock_owned().await;
        let runs = Arc::new(AtomicUsize::new(0));
        let second_runs = runs.clone();

        // Act
        {
            let second = with_model_blocking(model.clone(), ModelError::Embed, move |_| {
                second_runs.fetch_add(1, Ordering::SeqCst);
                Ok(SECOND_CALL_VALUE)
            });
            tokio::pin!(second);
            // The lock is the first await in the call, so one poll parks it in the queue.
            let still_pending = tokio::select! {
                biased;
                _ = &mut second => false,
                _ = std::future::ready(()) => true,
            };
            assert!(still_pending, "the call must wait on the held lock");
        }
        drop(guard);
        // The lock queues waiters in FIFO order, so a leftover waiter would run first.
        let third = with_model_blocking(model, ModelError::Embed, |_| Ok(THIRD_CALL_VALUE)).await;

        // Assert
        assert_eq!(third.unwrap(), THIRD_CALL_VALUE);
        assert_eq!(runs.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_queued_waiter_runs_once_after_the_lock_is_released() {
        // Arrange
        let model = Arc::new(Mutex::new(0));
        let guard = model.clone().lock_owned().await;
        let runs = Arc::new(AtomicUsize::new(0));
        let second_runs = runs.clone();
        let second = with_model_blocking(model, ModelError::Embed, move |_| {
            second_runs.fetch_add(1, Ordering::SeqCst);
            Ok(SECOND_CALL_VALUE)
        });
        tokio::pin!(second);
        let still_pending = tokio::select! {
            biased;
            _ = &mut second => false,
            _ = std::future::ready(()) => true,
        };
        assert!(still_pending, "the call must wait on the held lock");

        // Act
        drop(guard);
        let out = second.await;

        // Assert
        assert_eq!(out.unwrap(), SECOND_CALL_VALUE);
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }
}
