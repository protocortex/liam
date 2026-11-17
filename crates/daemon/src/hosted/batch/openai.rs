// SPDX-License-Identifier: Apache-2.0
//! OpenAI's Batch API. The requests go up as a JSON Lines file first, and the
//! results come back as files.

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
use crate::hosted::model::ModelCallError;
use crate::hosted::multipart::Form;
use crate::hosted::openai::HOST;
use crate::hosted::secrets::SecretString;
use crate::hosted::transport::HttpTransport;

const CHAT_PATH: &str = "/v1/chat/completions";
const FILES_PATH: &str = "/v1/files";
const BATCHES_PATH: &str = "/v1/batches";
const UPLOAD_PURPOSE: &str = "batch";
const UPLOAD_NAME: &str = "requests.jsonl";
const JSONL_TYPE: &str = "application/jsonl";
const COMPLETION_WINDOW: &str = "24h";
const LINE_END: u8 = b'\n';
const NOT_FOUND: u16 = 404;
const HTTP_OK_RANGE: std::ops::Range<u16> = 200..300;

pub struct OpenAiBatch<T: HttpTransport> {
    wire: Wire<T>,
}

impl<T: HttpTransport> OpenAiBatch<T> {
    pub fn new(
        client: HostedClient<T>,
        pin: ModelPin,
        key: SecretString,
        costs: CostRecorder,
    ) -> Self {
        let dialect: Box<dyn Dialect> = Box::new(crate::hosted::openai::OpenAi);
        Self {
            wire: Wire::new(client, dialect, pin, key, costs),
        }
    }

    fn batch_url(id: &str) -> Result<String, ModelCallError> {
        Ok(format!("https://{HOST}{BATCHES_PATH}/{}", path_safe(id)?))
    }

    fn file_url(id: &str) -> Result<String, ModelCallError> {
        Ok(format!("https://{HOST}{FILES_PATH}/{}", path_safe(id)?))
    }

    fn jsonl(&self, requests: &[BatchRequest]) -> Result<Vec<u8>, ModelCallError> {
        let mut file = Vec::new();
        for request in requests {
            let line = json!({
                "custom_id": request.cache_key,
                "method": "POST",
                "url": CHAT_PATH,
                "body": self.wire.dialect.body(self.wire.model(), &request.call()),
            });
            serde_json::to_writer(&mut file, &line)
                .map_err(|_| ModelCallError::Malformed("request"))?;
            file.push(LINE_END);
        }
        Ok(file)
    }

    /// The batch record: its status and the files it names.
    async fn record(&self, id: &str) -> Result<Value, ModelCallError> {
        json(&self.wire.get(&Self::batch_url(id)?).await?)
    }

    async fn file_lines(&self, file_id: &str) -> Result<Vec<Value>, ModelCallError> {
        let url = format!("{}/content", Self::file_url(file_id)?);
        super::json_lines(&self.wire.get(&url).await?)
    }

    /// The results in one file. An output file holds answers; an error file
    /// holds requests the provider refused.
    fn results(&self, lines: &[Value]) -> Result<Vec<BatchResult>, ModelCallError> {
        lines
            .iter()
            .map(|line| {
                let cache_key = text_of(&line["custom_id"], "a result has no custom id")?;
                let response = &line["response"];
                let status = response["status_code"]
                    .as_u64()
                    .and_then(|code| u16::try_from(code).ok());
                let outcome = match status {
                    Some(code) if HTTP_OK_RANGE.contains(&code) => {
                        answer(&self.wire, &response["body"])
                    }
                    Some(code) => BatchOutcome::Failed(ModelCallError::Status(code)),
                    None => BatchOutcome::Failed(ModelCallError::Malformed(
                        "the provider could not run the request",
                    )),
                };
                Ok(BatchResult {
                    cache_key: cache_key.to_owned(),
                    outcome,
                })
            })
            .collect()
    }
}

fn file_ids(record: &Value) -> Vec<&str> {
    ["input_file_id", "output_file_id", "error_file_id"]
        .iter()
        .filter_map(|field| record[*field].as_str())
        .collect()
}

#[async_trait]
impl<T: HttpTransport + 'static> HostedBatch for OpenAiBatch<T> {
    async fn submit(&self, requests: &[BatchRequest]) -> Result<String, ModelCallError> {
        let form = Form::new().text("purpose", UPLOAD_PURPOSE).file(
            "file",
            UPLOAD_NAME,
            JSONL_TYPE,
            self.jsonl(requests)?,
        );
        let upload = json(
            &self
                .wire
                .post_form(&format!("https://{HOST}{FILES_PATH}"), form)
                .await?,
        )?;
        let file_id = path_safe(text_of(&upload["id"], "the upload names no file")?)?;
        let created = self
            .wire
            .post(
                &format!("https://{HOST}{BATCHES_PATH}"),
                &json!({
                    "input_file_id": file_id,
                    "endpoint": CHAT_PATH,
                    "completion_window": COMPLETION_WINDOW,
                }),
            )
            .await;
        let id = match created.and_then(|response| json(&response)) {
            Ok(body) => text_of(&body["id"], "the reply names no batch").map(str::to_owned),
            Err(error) => Err(error),
        };
        match id {
            Ok(id) => Ok(path_safe(&id)?.to_owned()),
            Err(error) => {
                // The upload is an orphan now; a failed delete only leaves it
                // to the provider's retention.
                let _ = self.wire.remove(&Self::file_url(file_id)?).await;
                Err(error)
            }
        }
    }

    async fn poll(&self, id: &str) -> Result<BatchStatus, ModelCallError> {
        let record = self.record(id).await?;
        match text_of(&record["status"], "the batch has no status")? {
            "validating" | "in_progress" | "finalizing" | "cancelling" => {
                Ok(BatchStatus::InProgress)
            }
            "completed" => Ok(BatchStatus::Ended),
            "failed" => Ok(BatchStatus::Failed),
            "expired" => Ok(BatchStatus::Expired),
            "cancelled" => Ok(BatchStatus::Cancelled),
            _ => Err(ModelCallError::Malformed("the batch status is unknown")),
        }
    }

    async fn collect(&self, id: &str) -> Result<Vec<BatchResult>, ModelCallError> {
        let record = self.record(id).await?;
        let mut collected = Vec::new();
        for field in ["output_file_id", "error_file_id"] {
            if let Some(file_id) = record[field].as_str() {
                collected.extend(self.results(&self.file_lines(file_id).await?)?);
            }
        }
        Ok(collected)
    }

    async fn cancel(&self, id: &str) -> Result<ProviderCopies, ModelCallError> {
        let url = Self::batch_url(id)?;
        let response = self.wire.get_raw(&url).await?;
        if response.status == NOT_FOUND {
            return Ok(ProviderCopies::Deleted);
        }
        let record = json(&super::accepted(response)?)?;
        let running = matches!(
            text_of(&record["status"], "the batch has no status")?,
            "validating" | "in_progress" | "finalizing"
        );
        if running {
            self.wire.stop(&format!("{url}/cancel")).await?;
        }
        let mut copies = ProviderCopies::Deleted;
        for file_id in file_ids(&record) {
            if self.wire.remove(&Self::file_url(file_id)?).await? == ProviderCopies::Kept {
                copies = ProviderCopies::Kept;
            }
        }
        // A running batch writes its output file only once it has stopped, so
        // that file is out of reach now.
        if running {
            copies = ProviderCopies::Kept;
        }
        Ok(copies)
    }
}
