// SPDX-License-Identifier: Apache-2.0
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

/// Tokenizer input cap, in tokens. This is fastembed's own `max_length`
/// parameter on the tokenizer, unrelated to the output embedding width
/// below (an earlier version of this file conflated the two, see
/// `mrl_truncate`). 8192 comfortably holds any realistic remembered fact or
/// episode item, well under Qwen3-Embedding's 32,768-token native context,
/// so it bounds worst-case latency without silently cutting off normal
/// content.
#[cfg(feature = "local")]
const EMBED_MAX_INPUT_TOKENS: usize = 8192;

/// Shrink an already-normalized embedding to its first `dims` entries and
/// re-normalize to a unit vector.
///
/// Matryoshka-trained models (Qwen3-Embedding among them) are trained so a
/// truncated prefix of the full vector is still a meaningful embedding, but
/// fastembed 5.17.3 does not implement that truncation: `Qwen3TextEmbedding
/// ::embed` always returns the model's full native width (1024 for the
/// 0.6B default), and the argument that looks like an MRL length
/// (`from_hf`'s 4th parameter) is actually the tokenizer's own `max_length`,
/// which caps the *input*, not the output. Passing the store's configured
/// `embedding_dims` into that parameter silently truncated long input
/// instead of shrinking the output, and every real embedding call then
/// failed the store's dimension check (expected `embedding_dims`, got the
/// model's native width). This function does the truncation fastembed does
/// not, so `FastEmbedEmbedder::dims()` and the actual vector length agree.
///
/// Pure (no model, no candle, no fastembed): compiled whenever `local` is
/// enabled (its real caller) or under `cfg(test)` (so its own unit tests
/// below build and run without the `local` feature or any downloaded
/// weights).
#[cfg(any(test, feature = "local"))]
fn mrl_truncate(full: &[f32], dims: usize) -> Vec<f32> {
    if dims >= full.len() {
        return full.to_vec();
    }
    let mut out = full[..dims].to_vec();
    let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut out {
            *x /= norm;
        }
    }
    out
}

/// Reject a requested Matryoshka output width the model cannot honor: zero
/// (silently produces empty vectors, since `mrl_truncate` has nothing to
/// truncate to and nothing to renormalize) or larger than the model's native
/// width (Matryoshka only shrinks). Pure and gated the same as `mrl_truncate`,
/// so both bounds are unit tested without a real model.
#[cfg(any(test, feature = "local"))]
fn validate_dims(model_id: &str, dims: usize, native_dims: usize) -> Result<()> {
    if dims == 0 {
        return Err(crate::error::ModelError::Embed(format!(
            "embedding_dims must be greater than 0, got 0 for {model_id}"
        )));
    }
    if dims > native_dims {
        return Err(crate::error::ModelError::Embed(format!(
            "embedding_dims ({dims}) exceeds {model_id}'s native output width \
             ({native_dims}); Matryoshka truncation can only shrink an embedding, \
             not extend it"
        )));
    }
    Ok(())
}

/// In-process embedder over fastembed-rs (feature `local`). Runs a Qwen3
/// embedding model with candle; no server, offline once the model is cached.
/// The sync fastembed call runs on a blocking thread so the async runtime stays
/// free. `dims` is the output embedding width after Matryoshka truncation
/// (see `mrl_truncate`); it must not exceed the model's native hidden size.
#[cfg(feature = "local")]
pub struct FastEmbedEmbedder {
    model: std::sync::Arc<std::sync::Mutex<fastembed::Qwen3TextEmbedding>>,
    dims: usize,
}

#[cfg(feature = "local")]
impl FastEmbedEmbedder {
    /// Load a model by Hugging Face id (e.g. "Qwen/Qwen3-Embedding-0.6B"),
    /// truncating its output to `dims` via Matryoshka (see `mrl_truncate`).
    ///
    /// Confirmed against fastembed v5.17.3's `qwen3` feature:
    /// `Qwen3TextEmbedding::from_hf(model_id, &device, DType, max_length)`,
    /// where `max_length` bounds tokenizer input, not output width, and
    /// `embed(&[&str]) -> Result<Vec<Vec<f32>>>`, which always returns the
    /// model's native hidden size.
    pub fn load(model_id: &str, dims: usize) -> Result<Self> {
        let device = candle_core::Device::Cpu;
        let model = fastembed::Qwen3TextEmbedding::from_hf(
            model_id,
            &device,
            candle_core::DType::F32,
            EMBED_MAX_INPUT_TOKENS,
        )
        .map_err(|e| crate::error::ModelError::Embed(e.to_string()))?;
        validate_dims(model_id, dims, model.config().hidden_size)?;
        Ok(Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(model)),
            dims,
        })
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
        let dims = self.dims;
        let out = tokio::task::spawn_blocking(move || {
            let m = model
                .lock()
                .map_err(|_| crate::error::ModelError::Embed("model lock poisoned".into()))?;
            let vecs = m
                .embed(&[query.as_str()])
                .map_err(|e| crate::error::ModelError::Embed(e.to_string()))?;
            let full = vecs
                .into_iter()
                .next()
                .ok_or_else(|| crate::error::ModelError::Embed("empty embedding".into()))?;
            Ok(mrl_truncate(&full, dims))
        })
        .await
        .map_err(|e| crate::error::ModelError::Embed(e.to_string()))??;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{mrl_truncate, validate_dims};

    #[test]
    fn mrl_truncate_shrinks_and_renormalizes() {
        // Arrange: a unit vector in 4 dims (3-4-... style triple: 3/5, 4/5).
        let full = vec![0.6, 0.8, 0.0, 0.0];

        // Act
        let truncated = mrl_truncate(&full, 2);

        // Assert: same 2 leading values, re-normalized to unit length (they
        // already summed to 1 here, so truncation is a no-op on the values,
        // but the function must not skip renormalization based on that).
        assert_eq!(truncated.len(), 2);
        let norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "expected unit norm, got {norm}");
        assert!((truncated[0] - 0.6).abs() < 1e-6);
        assert!((truncated[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn mrl_truncate_renormalizes_when_prefix_alone_is_not_unit_length() {
        // Arrange: full vector is unit length, but its first 2 entries alone
        // are not, so truncating without renormalizing would return a
        // sub-unit vector that silently degrades cosine similarity.
        let full = vec![0.5, 0.5, 0.5, 0.5];

        // Act
        let truncated = mrl_truncate(&full, 2);

        // Assert
        let norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "expected unit norm, got {norm}");
    }

    #[test]
    fn mrl_truncate_is_a_no_op_when_dims_meets_or_exceeds_the_input() {
        // Arrange / Act / Assert: equal length passes through unchanged...
        let full = vec![0.1, 0.2, 0.3];
        assert_eq!(mrl_truncate(&full, 3), full);
        // ...and a requested width larger than the input cannot be granted,
        // so the caller gets everything there is rather than a panic.
        assert_eq!(mrl_truncate(&full, 10), full);
    }

    #[test]
    fn mrl_truncate_does_not_divide_by_zero_on_an_all_zero_prefix() {
        // Arrange: pathological input (should not occur from a real model,
        // but the function must not panic if it ever does).
        let full = vec![0.0, 0.0, 0.7, 0.7];

        // Act
        let truncated = mrl_truncate(&full, 2);

        // Assert: zero prefix stays zero rather than NaN from a 0/0 divide.
        assert_eq!(truncated, vec![0.0, 0.0]);
    }

    #[test]
    fn validate_dims_accepts_a_width_within_the_native_size() {
        assert!(validate_dims("test-model", 768, 1024).is_ok());
    }

    #[test]
    fn validate_dims_accepts_the_native_size_exactly() {
        // Boundary: no truncation needed at all, still a valid request.
        assert!(validate_dims("test-model", 1024, 1024).is_ok());
    }

    #[test]
    fn validate_dims_rejects_zero() {
        // A zero width would make mrl_truncate return an empty vector for
        // every embed call, so this must fail at load instead of silently
        // shipping unusable embeddings.
        let err = validate_dims("test-model", 0, 1024)
            .unwrap_err()
            .to_string();
        assert!(err.contains("greater than 0"), "unexpected message: {err}");
    }

    #[test]
    fn validate_dims_rejects_a_width_larger_than_native() {
        let err = validate_dims("test-model", 2048, 1024)
            .unwrap_err()
            .to_string();
        assert!(err.contains("exceeds"), "unexpected message: {err}");
    }
}
