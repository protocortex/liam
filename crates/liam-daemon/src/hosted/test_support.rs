// SPDX-License-Identifier: Apache-2.0
//! A transport double and a model builder for the adapter tests. Nothing here
//! opens a socket.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use liam_store::{FixedClock, Millis};

use super::client::HostedClient;
use super::cost::CostRecorder;
use super::dialect::Dialect;
use super::model::HostedModel;
use super::secrets::SecretString;
use super::transport::{HttpRequest, HttpResponse, HttpTransport, TransportError};
use crate::config::{HostedConfig, ModelPin, Price, PricesConfig, Provider};
use crate::metrics::recorder::Metrics;
use crate::metrics::sink::{MemorySink, MetricsSink};

pub const KEY: &str = "sk-test-key-0123456789";
pub const PRICE_VERSION: &str = "test-2026-11";

/// One recorded request, with the secrets read out for assertions.
#[derive(Debug, Clone)]
pub struct Seen {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: serde_json::Value,
}

/// Answers each call from a script and records what it was sent.
#[derive(Clone, Default)]
pub struct FakeTransport {
    answers: Arc<Mutex<VecDeque<Result<HttpResponse, TransportError>>>>,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl FakeTransport {
    pub fn answering(status: u16, body: &str) -> Self {
        let fake = Self::default();
        fake.push(status, body);
        fake
    }

    pub fn push(&self, status: u16, body: &str) {
        self.answers.lock().unwrap().push_back(Ok(HttpResponse {
            status,
            body: body.as_bytes().to_vec(),
        }));
    }

    pub fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl HttpTransport for FakeTransport {
    async fn post_json(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
        self.seen.lock().unwrap().push(Seen {
            url: request.url.clone(),
            headers: request
                .headers
                .iter()
                .map(|(name, value)| (name.clone(), value.expose().to_owned()))
                .collect(),
            body: serde_json::from_slice(&request.body).expect("a JSON request"),
        });
        self.answers
            .lock()
            .unwrap()
            .pop_front()
            .expect("a scripted answer")
    }
}

pub fn price() -> Price {
    Price {
        input_micro_per_mtok: 1_000_000,
        output_micro_per_mtok: 5_000_000,
        cached_input_micro_per_mtok: 100_000,
        batch_input_micro_per_mtok: 500_000,
        batch_output_micro_per_mtok: 2_500_000,
    }
}

pub fn metrics() -> (Arc<Metrics>, Arc<MemorySink>) {
    let clock = Arc::new(FixedClock::new(Millis(0)));
    let sink = Arc::new(MemorySink::new(clock.clone()));
    let metrics = Arc::new(Metrics::new(sink.clone() as Arc<dyn MetricsSink>, clock, 1));
    (metrics, sink)
}

/// A model for `provider` pinned to `pin_id`, over `transport`, with room for
/// responses up to `max_response_bytes`.
pub fn model(
    provider: Provider,
    dialect: Box<dyn Dialect>,
    pin_id: &str,
    host: &str,
    transport: FakeTransport,
    max_response_bytes: usize,
) -> (HostedModel<FakeTransport>, Arc<MemorySink>) {
    let config = HostedConfig {
        enabled: true,
        allowed_hosts: vec![host.to_owned()],
        max_response_bytes,
        ..HostedConfig::default()
    };
    let (metrics, sink) = metrics();
    let prices = PricesConfig {
        version: PRICE_VERSION.to_owned(),
        models: [(pin_id.to_owned(), price())].into(),
    };
    let model = HostedModel::new(
        HostedClient::new(transport, &config),
        dialect,
        ModelPin::new(provider.name(), pin_id).unwrap(),
        SecretString::new(KEY),
        CostRecorder::new(metrics, prices),
    );
    (model, sink)
}
