// SPDX-License-Identifier: Apache-2.0
//! Anthropic's Message Batches API.

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{
    answer, json, json_lines, path_safe, text_of, BatchOutcome, BatchRequest, BatchResult,
    BatchStatus, HostedBatch, Wire,
};
use crate::config::ModelPin;
use crate::deletion::ProviderCopies;
use crate::hosted::anthropic::HOST;
use crate::hosted::client::HostedClient;
use crate::hosted::cost::CostRecorder;
use crate::hosted::dialect::Dialect;
use crate::hosted::model::ModelCallError;
use crate::hosted::secrets::SecretString;
use crate::hosted::transport::HttpTransport;

const BATCHES_PATH: &str = "/v1/messages/batches";

pub struct AnthropicBatch<T: HttpTransport> {
    wire: Wire<T>,
}

impl<T: HttpTransport> AnthropicBatch<T> {
    pub fn new(
        client: HostedClient<T>,
        pin: ModelPin,
        key: SecretString,
        costs: CostRecorder,
    ) -> Self {
        let dialect: Box<dyn Dialect> = Box::new(crate::hosted::anthropic::Anthropic);
        Self {
            wire: Wire::new(client, dialect, pin, key, costs),
        }
    }

    fn url(id: &str) -> Result<String, ModelCallError> {
        Ok(format!("https://{HOST}{BATCHES_PATH}/{}", path_safe(id)?))
    }
}

#[async_trait]
impl<T: HttpTransport + 'static> HostedBatch for AnthropicBatch<T> {
    async fn submit(&self, requests: &[BatchRequest]) -> Result<String, ModelCallError> {
        let entries: Vec<Value> = requests
            .iter()
            .map(|request| {
                json!({
                    "custom_id": request.cache_key,
                    "params": self.wire.dialect.body(self.wire.model(), &request.call()),
                })
            })
            .collect();
        let response = self
            .wire
            .submit(
                &format!("https://{HOST}{BATCHES_PATH}"),
                &json!({ "requests": entries }),
            )
            .await?;
        let body = json(&response)?;
        let id = text_of(&body["id"], "the reply names no batch")?;
        Ok(path_safe(id)?.to_owned())
    }

    async fn poll(&self, id: &str) -> Result<BatchStatus, ModelCallError> {
        let body = json(&self.wire.get(&Self::url(id)?).await?)?;
        match text_of(&body["processing_status"], "the batch has no status")? {
            "in_progress" | "canceling" => Ok(BatchStatus::InProgress),
            "ended" => Ok(BatchStatus::Ended),
            _ => Err(ModelCallError::Malformed("the batch status is unknown")),
        }
    }

    async fn collect(&self, id: &str) -> Result<Vec<BatchResult>, ModelCallError> {
        let response = self
            .wire
            .get(&format!("{}/results", Self::url(id)?))
            .await?;
        json_lines(&response)?
            .iter()
            .map(|line| {
                let cache_key = text_of(&line["custom_id"], "a result has no custom id")?;
                let result = &line["result"];
                let outcome = match text_of(&result["type"], "a result has no type")? {
                    "succeeded" => answer(&self.wire, &result["message"]),
                    "errored" => BatchOutcome::Failed(ModelCallError::Malformed(
                        "the provider could not run the request",
                    )),
                    "canceled" => BatchOutcome::Cancelled,
                    "expired" => BatchOutcome::Expired,
                    _ => return Err(ModelCallError::Malformed("a result type is unknown")),
                };
                Ok(BatchResult {
                    cache_key: cache_key.to_owned(),
                    outcome,
                })
            })
            .collect()
    }

    async fn cancel(&self, id: &str) -> Result<ProviderCopies, ModelCallError> {
        let url = Self::url(id)?;
        self.wire.stop(&format!("{url}/cancel")).await?;
        self.wire.remove(&url).await
    }
}
