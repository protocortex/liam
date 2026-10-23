// SPDX-License-Identifier: Apache-2.0
//! BGE-M3 dense embedder (feature `local`).
//!
//! Loads BAAI/bge-m3 through fastembed's ONNX `TextEmbedding` path, which is
//! independent of the Qwen3 candle loader in `embedder.rs`. Per the model card,
//! dense retrieval uses the final-layer CLS token and an L2 normalisation, and
//! neither queries nor documents take an instruction prefix. fastembed applies
//! the CLS pooling (`get_default_pooling_method`) and the L2 normalisation
//! (`text_embedding/output.rs`) itself, so this module only owns the loading,
//! the input convention and the fixed width.

#[cfg(any(test, feature = "local"))]
use crate::error::{ModelError, Result};
#[cfg(any(test, feature = "local"))]
use crate::TextKind;
#[cfg(feature = "local")]
use async_trait::async_trait;

/// The Hugging Face id config selects this embedder with.
pub const BGE_M3_MODEL_ID: &str = "BAAI/bge-m3";

/// BGE-M3's output width. It has no Matryoshka truncation, so it is also the
/// only `embedding_dims` this embedder accepts.
#[cfg(any(test, feature = "local"))]
const BGE_M3_DIMS: usize = 1024;

/// The exact string handed to the model for `text`. BGE-M3 dense retrieval
/// uses no instruction on either side; matching on `kind` keeps a future
/// `TextKind` variant from silently inheriting that.
#[cfg(any(test, feature = "local"))]
fn bge_m3_input(text: &str, kind: TextKind) -> &str {
    match kind {
        TextKind::Query | TextKind::Document => text,
    }
}

#[cfg(any(test, feature = "local"))]
fn validate_bge_m3_dims(dims: usize) -> Result<()> {
    if dims != BGE_M3_DIMS {
        return Err(ModelError::Embed(format!(
            "embedding_dims ({dims}) does not match {BGE_M3_MODEL_ID}'s fixed output width \
             ({BGE_M3_DIMS}); it cannot be truncated or extended"
        )));
    }
    Ok(())
}

/// In-process BGE-M3 dense embedder. The sync ONNX call runs on a blocking
/// thread so the async runtime stays free.
#[cfg(feature = "local")]
pub struct BgeM3Embedder {
    model: std::sync::Arc<std::sync::Mutex<fastembed::TextEmbedding>>,
}

#[cfg(feature = "local")]
impl BgeM3Embedder {
    /// Load BGE-M3 into `cache_dir`. `dims` must be 1024, the model's own width.
    pub fn load(dims: usize, cache_dir: &str) -> Result<Self> {
        validate_bge_m3_dims(dims)?;
        let options = fastembed::TextInitOptions::new(fastembed::EmbeddingModel::BGEM3)
            .with_cache_dir(std::path::PathBuf::from(cache_dir))
            .with_max_length(crate::embedder::EMBED_MAX_INPUT_TOKENS);
        let model = fastembed::TextEmbedding::try_new(options)
            .map_err(|e| ModelError::Embed(format!("load {BGE_M3_MODEL_ID}: {e}")))?;
        Ok(Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(model)),
        })
    }
}

#[cfg(feature = "local")]
#[async_trait]
impl crate::Embedder for BgeM3Embedder {
    fn dims(&self) -> usize {
        BGE_M3_DIMS
    }

    async fn embed(&self, text: &str, kind: TextKind) -> Result<Vec<f32>> {
        let texts = [text.to_string()];
        let vectors = self.embed_batch(&texts, kind).await?;
        vectors
            .into_iter()
            .next()
            .ok_or_else(|| ModelError::Embed("empty embedding".into()))
    }

    async fn embed_batch(&self, texts: &[String], kind: TextKind) -> Result<Vec<Vec<f32>>> {
        let model = self.model.clone();
        let inputs: Vec<String> = texts
            .iter()
            .map(|t| bge_m3_input(t, kind).to_string())
            .collect();
        tokio::task::spawn_blocking(move || {
            let mut m = model
                .lock()
                .map_err(|_| ModelError::Embed("model lock poisoned".into()))?;
            m.embed(&inputs, None)
                .map_err(|e| ModelError::Embed(e.to_string()))
        })
        .await
        .map_err(|e| ModelError::Embed(e.to_string()))?
    }
}

#[cfg(test)]
mod tests {
    use super::{bge_m3_input, validate_bge_m3_dims, BGE_M3_DIMS};
    use crate::TextKind;

    #[test]
    fn bge_m3_input_leaves_text_unprefixed_for_every_kind() {
        for kind in [TextKind::Query, TextKind::Document] {
            // Arrange
            let text = "where do I live";

            // Act
            let input = bge_m3_input(text, kind);

            // Assert
            assert_eq!(input, text, "kind: {kind:?}");
        }
    }

    /// Downloads about 2 GB of weights: run explicitly with
    /// `cargo test -p liam-model --features local -- --ignored bge_m3`.
    #[cfg(feature = "local")]
    #[tokio::test]
    #[ignore = "downloads BGE-M3 weights; see the doc comment for the run command"]
    async fn bge_m3_loads_and_embeds_unit_vectors_of_its_native_width() {
        use crate::Embedder;

        // Arrange
        let cache_dir = std::env::temp_dir().join("liam-bge-m3-test-cache");
        let embedder = super::BgeM3Embedder::load(BGE_M3_DIMS, cache_dir.to_str().unwrap())
            .expect("load BGE-M3");

        // Act
        let vector = embedder
            .embed("where do I live", TextKind::Query)
            .await
            .expect("embed after load");

        // Assert
        assert_eq!(embedder.dims(), BGE_M3_DIMS);
        assert_eq!(vector.len(), BGE_M3_DIMS);
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "expected unit norm, got {norm}");
    }

    #[test]
    fn validate_bge_m3_dims_accepts_the_native_width() {
        assert!(validate_bge_m3_dims(BGE_M3_DIMS).is_ok());
    }

    #[test]
    fn validate_bge_m3_dims_rejects_any_other_width() {
        for dims in [0, 512, 2048] {
            // Act
            let err = validate_bge_m3_dims(dims).unwrap_err().to_string();

            // Assert: BGE-M3 has no Matryoshka truncation, so only its own width fits.
            assert!(err.contains(&dims.to_string()), "message: {err}");
            assert!(err.contains("1024"), "message: {err}");
        }
    }
}
