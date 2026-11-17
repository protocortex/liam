// SPDX-License-Identifier: Apache-2.0
//! Google's Gemini batch mode, with the requests sent inline.

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{
    answer, json, path_safe, text_of, BatchOutcome, BatchRequest, BatchResult, BatchStatus,
    HostedBatch, Wire,
};
use crate::config::ModelPin;
use crate::deletion::ProviderCopies;
use crate::hosted::client::HostedClient;
use crate::hosted::cost::CostRecorder;
use crate::hosted::dialect::Dialect;
use crate::hosted::gemini::HOST;
use crate::hosted::model::ModelCallError;
use crate::hosted::secrets::SecretString;
use crate::hosted::transport::HttpTransport;

const API_PREFIX: &str = "v1beta";
/// A Gemini batch is named `batches/<id>`.
const NAME_PREFIX: &str = "batches/";
const DISPLAY_NAME: &str = "liam";

pub struct GeminiBatch<T: HttpTransport> {
    wire: Wire<T>,
}

impl<T: HttpTransport> GeminiBatch<T> {
    pub fn new(
        client: HostedClient<T>,
        pin: ModelPin,
        key: SecretString,
        costs: CostRecorder,
    ) -> Self {
        let dialect: Box<dyn Dialect> = Box::new(crate::hosted::gemini::Gemini);
        Self {
            wire: Wire::new(client, dialect, pin, key, costs),
        }
    }

    fn url(name: &str) -> Result<String, ModelCallError> {
        let id = name
            .strip_prefix(NAME_PREFIX)
            .ok_or(ModelCallError::Malformed(
                "the batch name is not batches/<id>",
            ))?;
        Ok(format!(
            "https://{HOST}/{API_PREFIX}/{NAME_PREFIX}{}",
            path_safe(id)?
        ))
    }
}

#[async_trait]
impl<T: HttpTransport + 'static> HostedBatch for GeminiBatch<T> {
    async fn submit(&self, requests: &[BatchRequest]) -> Result<String, ModelCallError> {
        let entries: Vec<Value> = requests
            .iter()
            .map(|request| {
                json!({
                    "request": self.wire.dialect.body(self.wire.model(), &request.call()),
                    "metadata": { "key": request.cache_key },
                })
            })
            .collect();
        let url = format!(
            "https://{HOST}/{API_PREFIX}/models/{}:batchGenerateContent",
            self.wire.model()
        );
        let body = json!({
            "batch": {
                "display_name": DISPLAY_NAME,
                "input_config": { "requests": { "requests": entries } },
            }
        });
        let reply = json(&self.wire.submit(&url, &body).await?)?;
        let name = text_of(&reply["name"], "the reply names no batch")?;
        Self::url(name)?;
        Ok(name.to_owned())
    }

    async fn poll(&self, id: &str) -> Result<BatchStatus, ModelCallError> {
        let body = json(&self.wire.get(&Self::url(id)?).await?)?;
        match text_of(&body["metadata"]["state"], "the batch has no state")? {
            "BATCH_STATE_PENDING" | "BATCH_STATE_RUNNING" => Ok(BatchStatus::InProgress),
            "BATCH_STATE_SUCCEEDED" => Ok(BatchStatus::Ended),
            "BATCH_STATE_FAILED" => Ok(BatchStatus::Failed),
            "BATCH_STATE_CANCELLED" => Ok(BatchStatus::Cancelled),
            "BATCH_STATE_EXPIRED" => Ok(BatchStatus::Expired),
            _ => Err(ModelCallError::Malformed("the batch state is unknown")),
        }
    }

    async fn collect(&self, id: &str) -> Result<Vec<BatchResult>, ModelCallError> {
        let body = json(&self.wire.get(&Self::url(id)?).await?)?;
        let inlined = body["response"]["inlinedResponses"]["inlinedResponses"]
            .as_array()
            .ok_or(ModelCallError::Malformed(
                "the batch holds no inline results",
            ))?;
        inlined
            .iter()
            .map(|entry| {
                let cache_key = text_of(&entry["metadata"]["key"], "a result has no key")?;
                let outcome = if entry["error"].is_null() {
                    answer(&self.wire, &entry["response"])
                } else {
                    BatchOutcome::Failed(ModelCallError::Malformed(
                        "the provider could not run the request",
                    ))
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
        self.wire.stop(&format!("{url}:cancel")).await?;
        self.wire.remove(&url).await
    }
}
