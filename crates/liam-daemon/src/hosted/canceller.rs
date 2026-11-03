// SPDX-License-Identifier: Apache-2.0
//! The hosted side of a delete: cancels a batch at its provider and deletes
//! what the provider lets it.

use std::sync::Arc;

use async_trait::async_trait;

use super::batch::HostedBatch;
use crate::config::Provider;
use crate::deletion::{BatchCanceller, ProviderCopies};

type Open = dyn Fn(Provider) -> anyhow::Result<Arc<dyn HostedBatch>> + Send + Sync;

pub struct HostedCanceller {
    open: Box<Open>,
}

impl HostedCanceller {
    pub fn new(
        open: impl Fn(Provider) -> anyhow::Result<Arc<dyn HostedBatch>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            open: Box::new(open),
        }
    }
}

#[async_trait]
impl BatchCanceller for HostedCanceller {
    /// A provider that cannot be opened (no key, no pin) cannot be asked, so
    /// its copy is reported as kept rather than stalling the delete; a call
    /// that fails is an error, and the job retries it.
    async fn cancel(&self, provider: &str, batch_id: &str) -> anyhow::Result<ProviderCopies> {
        let Some(provider) = Provider::from_name(provider).filter(|p| *p != Provider::Local) else {
            anyhow::bail!("the batch table names an unknown provider {provider}");
        };
        let batch = match (self.open)(provider) {
            Ok(batch) => batch,
            Err(error) => {
                tracing::error!(
                    provider = provider.name(),
                    error = %format!("{error:#}"),
                    "a hosted batch could not be cancelled, so the provider keeps its copy"
                );
                return Ok(ProviderCopies::Kept);
            }
        };
        Ok(batch.cancel(batch_id).await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hosted::batch::{BatchRequest, BatchResult, BatchStatus};
    use crate::hosted::model::ModelCallError;

    /// Answers every cancel with a fixed result.
    struct Fixed(Result<ProviderCopies, u16>);

    #[async_trait]
    impl HostedBatch for Fixed {
        async fn submit(&self, _: &[BatchRequest]) -> Result<String, ModelCallError> {
            unreachable!("a delete never submits")
        }
        async fn poll(&self, _: &str) -> Result<BatchStatus, ModelCallError> {
            unreachable!("a delete never polls")
        }
        async fn collect(&self, _: &str) -> Result<Vec<BatchResult>, ModelCallError> {
            unreachable!("a delete never collects")
        }
        async fn cancel(&self, _: &str) -> Result<ProviderCopies, ModelCallError> {
            self.0.map_err(ModelCallError::Status)
        }
    }

    #[tokio::test]
    async fn the_named_provider_cancels_and_its_answer_is_passed_on() {
        // Arrange
        let canceller = HostedCanceller::new(|provider| {
            assert_eq!(provider, Provider::OpenAi);
            Ok(Arc::new(Fixed(Ok(ProviderCopies::Deleted))))
        });

        // Act
        let copies = canceller.cancel("openai", "batch_1").await.unwrap();

        // Assert
        assert_eq!(copies, ProviderCopies::Deleted);
    }

    #[tokio::test]
    async fn a_provider_that_cannot_be_opened_leaves_its_copy_and_a_failed_call_is_an_error() {
        // Arrange
        let unopened = HostedCanceller::new(|_| anyhow::bail!("no key"));
        let failing = HostedCanceller::new(|_| Ok(Arc::new(Fixed(Err(500)))));

        // Act
        let kept = unopened.cancel("anthropic", "b").await.unwrap();
        let failed = failing.cancel("anthropic", "b").await.unwrap_err();

        // Assert
        assert_eq!(kept, ProviderCopies::Kept);
        assert!(failed.to_string().contains("500"), "{failed}");
    }

    #[tokio::test]
    async fn a_provider_the_table_cannot_name_is_an_error() {
        // Arrange
        let canceller = HostedCanceller::new(|_| unreachable!("nothing to open"));

        // Act
        let unknown = canceller.cancel("acme", "b").await.unwrap_err();
        let local = canceller.cancel("local", "b").await.unwrap_err();

        // Assert
        assert!(unknown.to_string().contains("acme"), "{unknown}");
        assert!(local.to_string().contains("local"), "{local}");
    }
}
