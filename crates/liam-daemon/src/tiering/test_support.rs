// SPDX-License-Identifier: Apache-2.0
//! Doubles and builders shared by the tiering tests.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use liam_log::dedup::BloomConfig;
use liam_model::{Embedder, ModelError, TextKind};
use liam_store::{DefaultGraph, FixedClock, GraphConfig, NodeId, OriginClaim, OriginVerdict};

use super::verify::TierVerifier;
use crate::config::LogConfig;
use crate::extraction::sidecar::FreeRam;

/// An in-memory store on an event log under `dir`, read through `clock`.
pub async fn logged_store(dir: &Path, clock: Arc<FixedClock>, dims: usize) -> DefaultGraph {
    let log_dir = dir.join("events");
    let writer = liam_daemon::event_log::open_writer(&log_dir, LogConfig::default().wal()).unwrap();
    let log =
        liam_daemon::event_log::shared_log(writer, &log_dir, BloomConfig::default(), None).unwrap();
    DefaultGraph::open_with_clock(":memory:", GraphConfig::new(dims), clock)
        .await
        .unwrap()
        .with_log(log)
        .await
        .unwrap()
}

/// A vector whose component `j` is positive when bit `j` of `pattern` is set,
/// so its sign signature is `pattern`. `MockEmbedder` gives all-positive
/// components, which would make every signature the same.
pub fn signed(pattern: u8, dims: usize) -> Vec<f32> {
    (0..dims)
        .map(|bit| if pattern >> bit & 1 == 1 { 1.0 } else { -1.0 })
        .collect()
}

/// Maps a text to the vector it embeds to; any other text is an error.
pub struct ScriptedEmbedder {
    dims: usize,
    vectors: HashMap<String, Vec<f32>>,
    calls: AtomicUsize,
}

impl ScriptedEmbedder {
    pub fn new(dims: usize, vectors: &[(&str, Vec<f32>)]) -> Self {
        Self {
            dims,
            vectors: vectors
                .iter()
                .map(|(text, vector)| ((*text).to_string(), vector.clone()))
                .collect(),
            calls: AtomicUsize::new(0),
        }
    }

    /// How many texts it was asked to embed.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Embedder for ScriptedEmbedder {
    fn dims(&self) -> usize {
        self.dims
    }

    async fn embed(&self, text: &str, _kind: TextKind) -> liam_model::Result<Vec<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.vectors
            .get(text)
            .cloned()
            .ok_or_else(|| ModelError::Embed(format!("no scripted vector for {text:?}")))
    }
}

/// Counts the calls of an embedder it wraps and records the kind of each.
pub struct CountingEmbedder<E> {
    inner: E,
    kinds: Mutex<Vec<TextKind>>,
}

impl<E> CountingEmbedder<E> {
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            kinds: Mutex::new(Vec::new()),
        }
    }

    pub fn kinds(&self) -> Vec<TextKind> {
        self.kinds.lock().unwrap().clone()
    }
}

#[async_trait]
impl<E: Embedder> Embedder for CountingEmbedder<E> {
    fn dims(&self) -> usize {
        self.inner.dims()
    }

    async fn embed(&self, text: &str, kind: TextKind) -> liam_model::Result<Vec<f32>> {
        self.kinds.lock().unwrap().push(kind);
        self.inner.embed(text, kind).await
    }
}

/// Fails its `nth` call (1-based) and, when made slow, waits that long in every
/// call, so a deadline can be tripped between moves.
pub struct FailOnNthEmbedder<E> {
    inner: E,
    nth: usize,
    delay: Option<Duration>,
    calls: AtomicUsize,
}

impl<E> FailOnNthEmbedder<E> {
    pub fn new(inner: E, nth: usize) -> Self {
        Self {
            inner,
            nth,
            delay: None,
            calls: AtomicUsize::new(0),
        }
    }

    pub fn slow(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }
}

#[async_trait]
impl<E: Embedder> Embedder for FailOnNthEmbedder<E> {
    fn dims(&self) -> usize {
        self.inner.dims()
    }

    async fn embed(&self, text: &str, kind: TextKind) -> liam_model::Result<Vec<f32>> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(delay) = self.delay {
            tokio::time::sleep(delay).await;
        }
        if call == self.nth {
            return Err(ModelError::Embed("injected failure".to_string()));
        }
        self.inner.embed(text, kind).await
    }
}

/// Judges every node `Verified` unless told otherwise, and counts its calls.
#[derive(Default)]
pub struct StubVerifier {
    verdicts: Mutex<HashMap<NodeId, OriginVerdict>>,
    calls: AtomicUsize,
    claims_seen: AtomicUsize,
}

impl StubVerifier {
    pub fn judging(&self, node: &NodeId, verdict: OriginVerdict) {
        self.verdicts.lock().unwrap().insert(node.clone(), verdict);
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    pub fn claims_seen(&self) -> usize {
        self.claims_seen.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl TierVerifier for StubVerifier {
    async fn verify(
        &self,
        claims: &[OriginClaim],
    ) -> liam_store::Result<HashMap<NodeId, OriginVerdict>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.claims_seen.fetch_add(claims.len(), Ordering::SeqCst);
        let set = self.verdicts.lock().unwrap();
        Ok(claims
            .iter()
            .map(|claim| {
                let verdict = set
                    .get(&claim.node_id)
                    .copied()
                    .unwrap_or(OriginVerdict::Verified);
                (claim.node_id.clone(), verdict)
            })
            .collect())
    }
}

/// Free memory a test sets.
pub struct FixedRam(pub AtomicU64);

impl FixedRam {
    pub fn plenty() -> Self {
        Self(AtomicU64::new(u64::MAX))
    }
}

impl FreeRam for FixedRam {
    fn free_mb(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}
