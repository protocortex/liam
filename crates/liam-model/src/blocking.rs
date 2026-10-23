// SPDX-License-Identifier: Apache-2.0
//! Runs a sync model call on a blocking thread (feature `local`).

use std::sync::{Arc, Mutex};

use crate::error::{ModelError, Result};

/// Lock `model` and run `call` on a blocking thread so the async runtime stays
/// free. Lock poisoning and a failed join become `error(message)`, so each
/// caller keeps its own `ModelError` variant.
pub(crate) async fn with_model_blocking<T, R>(
    model: Arc<Mutex<T>>,
    error: fn(String) -> ModelError,
    call: impl FnOnce(&mut T) -> Result<R> + Send + 'static,
) -> Result<R>
where
    T: Send + 'static,
    R: Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let mut guard = model
            .lock()
            .map_err(|_| error("model lock poisoned".into()))?;
        call(&mut guard)
    })
    .await
    .map_err(|e| error(e.to_string()))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn returns_the_call_result_with_exclusive_access_to_the_model() {
        // Arrange
        let model = Arc::new(Mutex::new(1));

        // Act
        let out = with_model_blocking(model.clone(), ModelError::Embed, |m| {
            *m += 1;
            Ok(*m)
        })
        .await;

        // Assert
        assert_eq!(out.unwrap(), 2);
        assert_eq!(*model.lock().unwrap(), 2);
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
    async fn a_poisoned_lock_maps_to_the_callers_error_variant() {
        // Arrange
        let model = Arc::new(Mutex::new(0));
        let poisoner = model.clone();
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.lock().unwrap();
            panic!("poison the lock");
        })
        .join();

        // Act
        let err = with_model_blocking(model, ModelError::Rerank, |_| Ok(()))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(err.to_string(), "reranker: model lock poisoned");
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
}
