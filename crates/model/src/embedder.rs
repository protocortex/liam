// SPDX-License-Identifier: Apache-2.0
//! Embedding: turn text into a vector. The store never does this; the caller
//! (typically the daemon) embeds, then passes vectors into the store.

use async_trait::async_trait;

use crate::error::Result;

/// The role a text plays in retrieval. Asymmetric models embed a search query
/// differently from the content it is matched against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TextKind {
    Query,
    Document,
}

#[async_trait]
pub trait Embedder: Send + Sync {
    fn dims(&self) -> usize;
    /// The engine and device that run the model, for example `onnx/cpu`.
    /// Reported by the benchmark.
    fn backend(&self) -> &str {
        "unspecified"
    }
    async fn embed(&self, text: &str, kind: TextKind) -> Result<Vec<f32>>;
    /// Batch by default calls `embed`; adapters may override for efficiency.
    async fn embed_batch(&self, texts: &[String], kind: TextKind) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t, kind).await?);
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
    async fn embed(&self, text: &str, _kind: TextKind) -> Result<Vec<f32>> {
        let mut v = vec![0.0f32; self.dims];
        for (i, b) in text.bytes().enumerate() {
            v[i % self.dims] += (b as f32) / 255.0;
        }
        Ok(v)
    }
}

/// Shrink an already-normalized embedding to its first `dims` entries and
/// re-normalize to a unit vector.
///
/// Matryoshka-trained models are trained so a truncated prefix of the full
/// vector is still a meaningful embedding, but the ONNX runtime always returns
/// the model's full native width, so the cut is done here.
pub(crate) fn mrl_truncate(full: &[f32], dims: usize) -> Vec<f32> {
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
/// width (Matryoshka only shrinks).
pub(crate) fn validate_dims(model_id: &str, dims: usize, native_dims: usize) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::{mrl_truncate, validate_dims, Embedder, MockEmbedder, TextKind};

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

    #[tokio::test]
    async fn mock_embedder_ignores_the_text_kind() {
        // Arrange
        let embedder = MockEmbedder::new(8);
        let text = "same text, two roles";

        // Act
        let as_query = embedder.embed(text, TextKind::Query).await.unwrap();
        let as_document = embedder.embed(text, TextKind::Document).await.unwrap();

        // Assert
        assert_eq!(as_query, as_document);
    }

    /// Records the kind each item was embedded with.
    struct KindSpy(std::sync::Mutex<Vec<(String, TextKind)>>);

    #[async_trait::async_trait]
    impl Embedder for KindSpy {
        fn dims(&self) -> usize {
            1
        }
        async fn embed(&self, text: &str, kind: TextKind) -> crate::error::Result<Vec<f32>> {
            self.0.lock().unwrap().push((text.to_string(), kind));
            Ok(vec![0.0])
        }
    }

    #[tokio::test]
    async fn default_embed_batch_passes_the_kind_to_every_item() {
        for kind in [TextKind::Query, TextKind::Document] {
            // Arrange
            let spy = KindSpy(std::sync::Mutex::new(Vec::new()));
            let texts = vec!["alpha".to_string(), "beta".to_string()];

            // Act
            let batch = spy.embed_batch(&texts, kind).await.unwrap();

            // Assert
            assert_eq!(batch.len(), 2);
            assert_eq!(
                *spy.0.lock().unwrap(),
                vec![("alpha".to_string(), kind), ("beta".to_string(), kind)]
            );
        }
    }
}
