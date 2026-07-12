//! Embedding: turn text into a vector. The store never does this; the caller
//! (typically the daemon) embeds, then passes vectors into the store.

use async_trait::async_trait;

use crate::error::Result;

#[async_trait]
pub trait Embedder: Send + Sync {
    fn dims(&self) -> usize;
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    /// Batch by default calls `embed`; adapters may override for efficiency.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t).await?);
        }
        Ok(out)
    }
}

/// Deterministic hashing embedder for tests: no model, stable output.
pub struct MockEmbedder {
    dims: usize,
}

impl MockEmbedder {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

#[async_trait]
impl Embedder for MockEmbedder {
    fn dims(&self) -> usize {
        self.dims
    }
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = vec![0.0f32; self.dims];
        for (i, b) in text.bytes().enumerate() {
            v[i % self.dims] += (b as f32) / 255.0;
        }
        Ok(v)
    }
}

/// In-process embedder over fastembed-rs (feature `local`). Runs a Qwen3
/// embedding model with candle; no server, offline once the model is cached.
/// The sync fastembed call runs on a blocking thread so the async runtime stays
/// free. `truncate_dims` is the MRL length (the store's `embedding_dims`).
///
/// VERSION CHECK: confirm against fastembed v5 with the `qwen3` feature:
/// `Qwen3TextEmbedding::from_hf(model_id, &device, DType, dims)` and its
/// `embed(&[&str]) -> Result<Vec<Vec<f32>>>` signature.
#[cfg(feature = "local")]
pub struct FastEmbedEmbedder {
    model: std::sync::Arc<std::sync::Mutex<fastembed::Qwen3TextEmbedding>>,
    dims: usize,
}

#[cfg(feature = "local")]
impl FastEmbedEmbedder {
    /// Load a model by Hugging Face id (e.g. "Qwen/Qwen3-Embedding-0.6B"),
    /// truncated to `truncate_dims` via Matryoshka.
    pub fn load(model_id: &str, truncate_dims: usize) -> Result<Self> {
        let device = candle_core::Device::Cpu;
        let model = fastembed::Qwen3TextEmbedding::from_hf(
            model_id,
            &device,
            candle_core::DType::F32,
            truncate_dims,
        )
        .map_err(|e| crate::error::ModelError::Embed(e.to_string()))?;
        Ok(Self { model: std::sync::Arc::new(std::sync::Mutex::new(model)), dims: truncate_dims })
    }
}

#[cfg(feature = "local")]
#[async_trait]
impl Embedder for FastEmbedEmbedder {
    fn dims(&self) -> usize {
        self.dims
    }
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let model = self.model.clone();
        let query = format!("query: {text}");
        let out = tokio::task::spawn_blocking(move || {
            let m = model.lock().map_err(|_| crate::error::ModelError::Embed("model lock poisoned".into()))?;
            let vecs = m
                .embed(&[query.as_str()])
                .map_err(|e| crate::error::ModelError::Embed(e.to_string()))?;
            vecs.into_iter()
                .next()
                .ok_or_else(|| crate::error::ModelError::Embed("empty embedding".into()))
        })
        .await
        .map_err(|e| crate::error::ModelError::Embed(e.to_string()))??;
        Ok(out)
    }
}
