// SPDX-License-Identifier: Apache-2.0
//! The local summariser keeps the model's behaviour and counts its calls.

use std::sync::Arc;

use liam_store::{FixedClock, Millis};

use super::*;
use crate::metrics::sink::{Labels, MemorySink, MetricId, MetricsSink};
use crate::summary::test_support::ScriptedLlm;

const MODEL: &str = "llama:model/file.gguf";
const REPLY: &str = "A short summary.";

fn identity() -> ModelIdentity {
    ModelIdentity {
        model_id: MODEL.to_owned(),
        model_sha256: String::new(),
    }
}

fn request() -> SummaryRequest {
    SummaryRequest {
        system: "system".to_owned(),
        prompt: "prompt".to_owned(),
        source_event_ids: vec!["e1".to_owned()],
        max_tokens: 16,
    }
}

#[tokio::test]
async fn the_local_summariser_returns_the_models_reply_with_no_usage() {
    // Arrange
    let llm = Arc::new(ScriptedLlm::replying(&[REPLY]));
    let summariser = LocalSummariser::new(llm.clone(), identity());

    // Act
    let response = summariser.summarise(&request()).await.unwrap();

    // Assert
    assert_eq!(
        response,
        SummaryResponse {
            text: REPLY.to_owned(),
            usage: None
        }
    );
    assert_eq!(llm.prompts(), [("system".to_owned(), "prompt".to_owned())]);
    assert_eq!(summariser.identity(), &identity());
}

#[tokio::test]
async fn a_failed_local_call_is_an_error_and_is_not_counted() {
    // Arrange
    let sink = Arc::new(MemorySink::new(Arc::new(FixedClock::new(Millis(0)))));
    let metrics = Arc::new(Metrics::new(
        sink.clone(),
        Arc::new(FixedClock::new(Millis(0))),
        1,
    ));
    let summariser = LocalSummariser::new(Arc::new(ScriptedLlm::failing("oom")), identity())
        .with_metrics(metrics);

    // Act
    let error = summariser.summarise(&request()).await.unwrap_err();

    // Assert
    assert_eq!(error.to_string(), "llm: oom");
    assert!(sink.snapshot().readings.is_empty());
}

#[tokio::test]
async fn each_local_call_is_counted_under_the_local_provider_at_no_cost() {
    // Arrange
    let sink = Arc::new(MemorySink::new(Arc::new(FixedClock::new(Millis(0)))));
    let metrics = Arc::new(Metrics::new(
        sink.clone(),
        Arc::new(FixedClock::new(Millis(0))),
        1,
    ));
    let llm = Arc::new(ScriptedLlm::replying(&[REPLY, REPLY]));
    let summariser = LocalSummariser::new(llm, identity()).with_metrics(metrics);

    // Act
    summariser.summarise(&request()).await.unwrap();
    summariser.summarise(&request()).await.unwrap();

    // Assert
    let readings = sink.snapshot().readings;
    let value = |metric| readings.iter().find(|r| r.metric == metric).unwrap().value;
    assert_eq!(value(MetricId::ModelCalls), 2.0);
    assert_eq!(value(MetricId::ModelCostMicroDollars), 0.0);
    assert!(readings
        .iter()
        .all(|r| matches!(r.labels, Labels::Model { .. })));
}
