// SPDX-License-Identifier: Apache-2.0
//! A hosted provider's batch API, seen from the extraction loop. The key is
//! read when the loop first needs the provider, so a run that touches no batch
//! never reads it.

use std::sync::{Arc, Mutex, PoisonError};

use anyhow::Result;
use async_trait::async_trait;

use super::batch::{BatchOutcome, BatchRequest, BatchStatus, HostedBatch};
use crate::config::Provider;
use crate::extraction::batch_flow::{BatchApi, RemoteResult, RemoteState};
use crate::extraction::client::ExtractRequest;

type Open = dyn Fn() -> Result<Arc<dyn HostedBatch>> + Send + Sync;

pub struct HostedBatchApi {
    provider: Provider,
    open: Box<Open>,
    opened: Mutex<Option<Arc<dyn HostedBatch>>>,
}

impl HostedBatchApi {
    pub fn new(
        provider: Provider,
        open: impl Fn() -> Result<Arc<dyn HostedBatch>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            provider,
            open: Box::new(open),
            opened: Mutex::new(None),
        }
    }

    fn batch(&self) -> Result<Arc<dyn HostedBatch>> {
        let mut opened = self.opened.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(batch) = opened.as_ref() {
            return Ok(Arc::clone(batch));
        }
        let batch = (self.open)()?;
        *opened = Some(Arc::clone(&batch));
        Ok(batch)
    }
}

#[async_trait]
impl BatchApi for HostedBatchApi {
    fn provider(&self) -> &str {
        self.provider.name()
    }

    async fn submit(&self, requests: &[(String, ExtractRequest)]) -> Result<String> {
        let requests: Vec<BatchRequest> = requests
            .iter()
            .map(|(cache_key, request)| BatchRequest {
                cache_key: cache_key.clone(),
                system: request.system.clone(),
                prompt: request.prompt.clone(),
                max_tokens: request.max_tokens,
                temperature: request.temperature,
                seed: request.seed,
                schema: Some(request.schema.clone()),
            })
            .collect();
        Ok(self.batch()?.submit(&requests).await?)
    }

    async fn poll(&self, batch_id: &str) -> Result<RemoteState> {
        Ok(match self.batch()?.poll(batch_id).await? {
            BatchStatus::InProgress => RemoteState::InProgress,
            BatchStatus::Ended => RemoteState::Ended,
            BatchStatus::Expired => RemoteState::Expired,
            BatchStatus::Failed => RemoteState::Failed,
            BatchStatus::Cancelled => RemoteState::Cancelled,
        })
    }

    async fn collect(&self, batch_id: &str) -> Result<Vec<(String, RemoteResult)>> {
        Ok(self
            .batch()?
            .collect(batch_id)
            .await?
            .into_iter()
            .map(|result| {
                let outcome = match result.outcome {
                    BatchOutcome::Answer(reply) => RemoteResult::Answer(reply.text),
                    BatchOutcome::Failed(_) => RemoteResult::Refused,
                    BatchOutcome::Expired => RemoteResult::Expired,
                    BatchOutcome::Cancelled => RemoteResult::Cancelled,
                };
                (result.cache_key, outcome)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TokenUsage;
    use crate::hosted::batch::BatchResult;
    use crate::hosted::dialect::ChatReply;
    use crate::hosted::model::ModelCallError;

    /// Returns fixed answers and remembers what it was sent.
    struct Fixed {
        sent: Mutex<Vec<BatchRequest>>,
        status: BatchStatus,
    }

    #[async_trait]
    impl HostedBatch for Fixed {
        async fn submit(&self, requests: &[BatchRequest]) -> Result<String, ModelCallError> {
            self.sent.lock().unwrap().extend_from_slice(requests);
            Ok("batch-1".to_owned())
        }
        async fn poll(&self, _: &str) -> Result<BatchStatus, ModelCallError> {
            Ok(self.status)
        }
        async fn collect(&self, _: &str) -> Result<Vec<BatchResult>, ModelCallError> {
            let reply = ChatReply {
                text: "{}".to_owned(),
                model: "m".to_owned(),
                usage: TokenUsage::default(),
            };
            Ok(vec![
                BatchResult {
                    cache_key: "a".into(),
                    outcome: BatchOutcome::Answer(reply),
                },
                BatchResult {
                    cache_key: "b".into(),
                    outcome: BatchOutcome::Failed(ModelCallError::Status(400)),
                },
                BatchResult {
                    cache_key: "c".into(),
                    outcome: BatchOutcome::Expired,
                },
                BatchResult {
                    cache_key: "d".into(),
                    outcome: BatchOutcome::Cancelled,
                },
            ])
        }
        async fn cancel(&self, _: &str) -> Result<crate::deletion::ProviderCopies, ModelCallError> {
            unreachable!("the loop never cancels")
        }
    }

    fn request() -> ExtractRequest {
        ExtractRequest {
            system: "sys".to_owned(),
            prompt: "prompt".to_owned(),
            schema: serde_json::json!({"type": "object"}),
            source_event_ids: vec!["e1".to_owned()],
            schema_hash: "h".to_owned(),
            temperature: 0.0,
            seed: 0,
            max_tokens: 128,
        }
    }

    #[tokio::test]
    async fn requests_results_and_states_map_between_the_loop_and_the_provider() {
        // Arrange
        let fixed = Arc::new(Fixed {
            sent: Mutex::new(Vec::new()),
            status: BatchStatus::Expired,
        });
        let handle = Arc::clone(&fixed);
        let api = HostedBatchApi::new(Provider::OpenAi, move || {
            Ok(handle.clone() as Arc<dyn HostedBatch>)
        });

        // Act
        let id = api
            .submit(&[("key-1".to_owned(), request())])
            .await
            .unwrap();
        let state = api.poll(&id).await.unwrap();
        let results = api.collect(&id).await.unwrap();

        // Assert
        let sent = fixed.sent.lock().unwrap();
        assert_eq!(api.provider(), "openai");
        assert_eq!(id, "batch-1");
        assert_eq!(state, RemoteState::Expired);
        assert_eq!(sent[0].cache_key, "key-1");
        assert_eq!(sent[0].max_tokens, 128);
        assert!(sent[0].schema.is_some());
        assert_eq!(
            results,
            vec![
                ("a".to_owned(), RemoteResult::Answer("{}".to_owned())),
                ("b".to_owned(), RemoteResult::Refused),
                ("c".to_owned(), RemoteResult::Expired),
                ("d".to_owned(), RemoteResult::Cancelled),
            ]
        );
    }

    #[tokio::test]
    async fn the_provider_is_opened_once_and_only_when_first_used() {
        // Arrange
        let opens = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&opens);
        let api = HostedBatchApi::new(Provider::Gemini, move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(Arc::new(Fixed {
                sent: Mutex::new(Vec::new()),
                status: BatchStatus::Ended,
            }) as Arc<dyn HostedBatch>)
        });
        let before = opens.load(std::sync::atomic::Ordering::SeqCst);

        // Act
        api.poll("b").await.unwrap();
        api.poll("b").await.unwrap();

        // Assert
        assert_eq!(before, 0);
        assert_eq!(opens.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_provider_that_cannot_be_opened_is_an_error() {
        // Arrange
        let api = HostedBatchApi::new(Provider::Gemini, || anyhow::bail!("no key"));

        // Act
        let error = api.poll("b").await.unwrap_err();

        // Assert
        assert!(error.to_string().contains("no key"));
    }
}
