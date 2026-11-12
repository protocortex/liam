// SPDX-License-Identifier: Apache-2.0
//! Test doubles shared by the `mcp` test modules.

use liam_model::{Embedder, TextKind};

/// Always-errors `Reranker`, so `ask` must fall back to identity order
/// (via `Reranker::order`'s default `unwrap_or_else`) instead of panicking.
pub(super) struct FailingReranker;
#[async_trait::async_trait]
impl liam_model::Reranker for FailingReranker {
    async fn scores(&self, _q: &str, _d: &[String]) -> liam_model::Result<Vec<f32>> {
        Err(liam_model::ModelError::Rerank("boom".into()))
    }
}

/// Wraps a real `Embedder`, recording every `(text, kind)` it is asked to
/// embed, so a test can assert how many calls one handler made and in
/// which role.
/// `MockEmbedder` itself has no counter of its own; this follows the
/// same small-custom-double pattern as `FailingLlm` in the `mcp.rs` tests
/// and `FailingReranker` in this file, delegating the real work to the
/// wrapped embedder instead of faking a result.
pub(super) struct CountingEmbedder {
    inner: liam_model::MockEmbedder,
    calls: std::sync::Mutex<Vec<(String, TextKind)>>,
}

impl CountingEmbedder {
    pub(super) fn new(dims: usize) -> Self {
        Self {
            inner: liam_model::MockEmbedder::new(dims),
            calls: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub(super) fn call_count(&self) -> usize {
        self.calls().len()
    }

    pub(super) fn calls(&self) -> Vec<(String, TextKind)> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl Embedder for CountingEmbedder {
    fn dims(&self) -> usize {
        self.inner.dims()
    }
    async fn embed(&self, text: &str, kind: TextKind) -> liam_model::Result<Vec<f32>> {
        self.calls.lock().unwrap().push((text.to_string(), kind));
        self.inner.embed(text, kind).await
    }
}

/// An `Embedder` whose every call fails, counting the calls.
pub(super) struct ErroringEmbedder {
    dims: usize,
    calls: std::sync::atomic::AtomicUsize,
}

impl ErroringEmbedder {
    pub(super) fn new(dims: usize) -> Self {
        Self {
            dims,
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub(super) fn call_count(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl Embedder for ErroringEmbedder {
    fn dims(&self) -> usize {
        self.dims
    }
    async fn embed(&self, _text: &str, _kind: TextKind) -> liam_model::Result<Vec<f32>> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Err(liam_model::ModelError::Embed("boom".into()))
    }
}

/// Waits on a shared gate before it counts and answers a call, so a test can
/// park an embed and check whether dropping the future ever reached the count.
pub(super) struct GatedEmbedder {
    inner: liam_model::MockEmbedder,
    gate: std::sync::Arc<tokio::sync::Mutex<()>>,
    calls: std::sync::atomic::AtomicUsize,
}

impl GatedEmbedder {
    pub(super) fn new(dims: usize, gate: std::sync::Arc<tokio::sync::Mutex<()>>) -> Self {
        Self {
            inner: liam_model::MockEmbedder::new(dims),
            gate,
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub(super) fn call_count(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl Embedder for GatedEmbedder {
    fn dims(&self) -> usize {
        self.inner.dims()
    }
    async fn embed(&self, text: &str, kind: TextKind) -> liam_model::Result<Vec<f32>> {
        let _held = self.gate.lock().await;
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.embed(text, kind).await
    }
}
