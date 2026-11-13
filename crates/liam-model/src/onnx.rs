// SPDX-License-Identifier: Apache-2.0
//! Small ONNX sentence encoders.
//!
//! Each model in `ONNX_MODELS` loads through fastembed's ONNX `TextEmbedding`
//! path, the same one BGE-M3 uses. fastembed owns the pooling and the L2
//! normalisation per model; this module owns what it leaves to the caller: the
//! query and document prefixes, the input token cap, and the output width.
//! Adding a model is one row in the table plus its fastembed variant.

use crate::blocking::with_model_blocking;
use crate::error::{ModelError, Result};
use crate::TextKind;
use async_trait::async_trait;

/// Inputs per ONNX run. fastembed's default of 256 pads the whole batch to its
/// longest input; a small batch bounds that.
const EMBED_BATCH_SIZE: usize = 16;

/// The retrieval query instruction of the BGE and Arctic model cards. Passages
/// take no prefix.
const BGE_STYLE_QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

/// Everything the loader needs to know about one ONNX model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxModelSpec {
    /// The Hugging Face id config selects the model with.
    pub id: &'static str,
    /// The model's own output width.
    pub native_dims: usize,
    /// Whether a prefix of the vector is itself a usable embedding, so
    /// `embedding_dims` may be smaller than `native_dims`.
    pub matryoshka: bool,
    /// The longest input, in tokens, the model's position table covers.
    pub max_tokens: usize,
    /// Written before a search query. Empty when the model takes none.
    pub query_prefix: &'static str,
    /// Written before stored text. Empty when the model takes none.
    pub document_prefix: &'static str,
    /// The fastembed variant that loads it.
    pub model: fastembed::EmbeddingModel,
}

/// All of the models this module loads, in the order the docs list them.
pub const ONNX_MODELS: &[OnnxModelSpec] = &[
    OnnxModelSpec {
        id: "sentence-transformers/all-MiniLM-L6-v2",
        native_dims: 384,
        matryoshka: false,
        max_tokens: 256,
        query_prefix: "",
        document_prefix: "",
        model: fastembed::EmbeddingModel::AllMiniLML6V2,
    },
    OnnxModelSpec {
        id: "sentence-transformers/all-MiniLM-L12-v2",
        native_dims: 384,
        matryoshka: false,
        max_tokens: 128,
        query_prefix: "",
        document_prefix: "",
        model: fastembed::EmbeddingModel::AllMiniLML12V2,
    },
    OnnxModelSpec {
        id: "BAAI/bge-small-en-v1.5",
        native_dims: 384,
        matryoshka: false,
        max_tokens: 512,
        query_prefix: BGE_STYLE_QUERY_PREFIX,
        document_prefix: "",
        model: fastembed::EmbeddingModel::BGESmallENV15,
    },
    OnnxModelSpec {
        id: "Snowflake/snowflake-arctic-embed-xs",
        native_dims: 384,
        matryoshka: false,
        max_tokens: 512,
        query_prefix: BGE_STYLE_QUERY_PREFIX,
        document_prefix: "",
        model: fastembed::EmbeddingModel::SnowflakeArcticEmbedXS,
    },
    OnnxModelSpec {
        id: "Snowflake/snowflake-arctic-embed-s",
        native_dims: 384,
        matryoshka: false,
        max_tokens: 512,
        query_prefix: BGE_STYLE_QUERY_PREFIX,
        document_prefix: "",
        model: fastembed::EmbeddingModel::SnowflakeArcticEmbedS,
    },
    OnnxModelSpec {
        id: "nomic-ai/nomic-embed-text-v1.5",
        native_dims: 768,
        matryoshka: true,
        max_tokens: 8192,
        query_prefix: "search_query: ",
        document_prefix: "search_document: ",
        model: fastembed::EmbeddingModel::NomicEmbedTextV15,
    },
    OnnxModelSpec {
        id: "intfloat/multilingual-e5-small",
        native_dims: 384,
        matryoshka: false,
        max_tokens: 512,
        query_prefix: "query: ",
        document_prefix: "passage: ",
        model: fastembed::EmbeddingModel::MultilingualE5Small,
    },
];

/// The spec whose id is `model`. Hugging Face ids are case-insensitive, so the
/// match is too.
pub fn onnx_model_spec(model: &str) -> Option<&'static OnnxModelSpec> {
    ONNX_MODELS
        .iter()
        .find(|spec| spec.id.eq_ignore_ascii_case(model))
}

/// The exact string handed to the model for `text` in its `kind` role.
fn onnx_input(spec: &OnnxModelSpec, text: &str, kind: TextKind) -> String {
    let prefix = match kind {
        TextKind::Query => spec.query_prefix,
        TextKind::Document => spec.document_prefix,
    };
    format!("{prefix}{text}")
}

/// The output width `dims` may take: any width up to the native one for a
/// Matryoshka model, only the native one for the rest.
fn validate_onnx_dims(spec: &OnnxModelSpec, dims: usize) -> Result<()> {
    if spec.matryoshka {
        return crate::embedder::validate_dims(spec.id, dims, spec.native_dims);
    }
    if dims != spec.native_dims {
        return Err(ModelError::Embed(format!(
            "embedding_dims ({dims}) does not match {}'s fixed output width ({}); \
             it cannot be truncated or extended",
            spec.id, spec.native_dims
        )));
    }
    Ok(())
}

/// In-process ONNX embedder for one `OnnxModelSpec`. The sync ONNX call runs on
/// a blocking thread so the async runtime stays free.
pub struct OnnxEmbedder {
    model: std::sync::Arc<tokio::sync::Mutex<fastembed::TextEmbedding>>,
    spec: &'static OnnxModelSpec,
    dims: usize,
}

impl OnnxEmbedder {
    /// Load `spec` into `cache_dir`, producing vectors `dims` wide.
    pub fn load(spec: &'static OnnxModelSpec, dims: usize, cache_dir: &str) -> Result<Self> {
        validate_onnx_dims(spec, dims)?;
        if let Some(hf_home) =
            crate::bge::hf_home_overriding(std::env::var("HF_HOME").ok().as_deref(), cache_dir)
        {
            tracing::warn!(
                hf_home,
                cache_dir,
                "HF_HOME is set, so {} is stored under HF_HOME, not embedder.cache_dir",
                spec.id
            );
        }
        let options = fastembed::TextInitOptions::new(spec.model.clone())
            .with_cache_dir(std::path::PathBuf::from(cache_dir))
            .with_max_length(spec.max_tokens);
        let model = fastembed::TextEmbedding::try_new(options)
            .map_err(|e| ModelError::Embed(format!("load {}: {e}", spec.id)))?;
        Ok(Self {
            model: std::sync::Arc::new(tokio::sync::Mutex::new(model)),
            spec,
            dims,
        })
    }
}

#[async_trait]
impl crate::Embedder for OnnxEmbedder {
    fn dims(&self) -> usize {
        self.dims
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
        let spec = self.spec;
        let dims = self.dims;
        let inputs: Vec<String> = texts.iter().map(|t| onnx_input(spec, t, kind)).collect();
        with_model_blocking(model, ModelError::Embed, move |m| {
            let full = m
                .embed(&inputs, Some(EMBED_BATCH_SIZE))
                .map_err(|e| ModelError::Embed(e.to_string()))?;
            Ok(full
                .iter()
                .map(|v| crate::embedder::mrl_truncate(v, dims))
                .collect())
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::{
        onnx_input, onnx_model_spec, validate_onnx_dims, OnnxModelSpec, BGE_STYLE_QUERY_PREFIX,
        ONNX_MODELS,
    };
    use crate::TextKind;

    fn spec(id: &str) -> &'static OnnxModelSpec {
        onnx_model_spec(id).expect("a listed model")
    }

    #[test]
    fn model_ids_are_unique_and_never_the_other_loaders() {
        for (i, a) in ONNX_MODELS.iter().enumerate() {
            // Arrange / Act / Assert
            assert!(a.id != crate::BGE_M3_MODEL_ID && a.id != crate::QWEN3_EMBEDDING_MODEL_ID);
            for b in &ONNX_MODELS[i + 1..] {
                assert!(!a.id.eq_ignore_ascii_case(b.id), "duplicate {}", a.id);
            }
        }
    }

    #[test]
    fn every_spec_has_a_positive_width_and_input_cap() {
        for spec in ONNX_MODELS {
            assert!(spec.native_dims > 0 && spec.max_tokens > 0, "{}", spec.id);
        }
    }

    #[test]
    fn lookup_is_case_insensitive_and_rejects_unknown_ids() {
        // Arrange / Act / Assert
        assert_eq!(spec("baai/BGE-SMALL-en-v1.5").id, "BAAI/bge-small-en-v1.5");
        assert!(onnx_model_spec("BAAI/bge-small").is_none());
    }

    #[test]
    fn queries_get_the_query_prefix_and_documents_the_document_prefix() {
        // Arrange
        let bge = spec("BAAI/bge-small-en-v1.5");
        let nomic = spec("nomic-ai/nomic-embed-text-v1.5");
        let minilm = spec("sentence-transformers/all-MiniLM-L6-v2");

        // Act / Assert
        assert_eq!(
            onnx_input(bge, "q", TextKind::Query),
            format!("{BGE_STYLE_QUERY_PREFIX}q")
        );
        assert_eq!(onnx_input(bge, "d", TextKind::Document), "d");
        assert_eq!(onnx_input(nomic, "q", TextKind::Query), "search_query: q");
        assert_eq!(
            onnx_input(nomic, "d", TextKind::Document),
            "search_document: d"
        );
        for kind in [TextKind::Query, TextKind::Document] {
            assert_eq!(onnx_input(minilm, "t", kind), "t");
        }
    }

    #[test]
    fn a_fixed_width_model_accepts_only_its_native_width() {
        // Arrange
        let bge = spec("BAAI/bge-small-en-v1.5");

        // Act
        let err = validate_onnx_dims(bge, 256).unwrap_err().to_string();

        // Assert
        assert!(validate_onnx_dims(bge, bge.native_dims).is_ok());
        assert!(err.contains("256") && err.contains("384"), "{err}");
    }

    #[test]
    fn a_matryoshka_model_accepts_any_width_up_to_native() {
        // Arrange
        let nomic = spec("nomic-ai/nomic-embed-text-v1.5");

        // Act / Assert
        assert!(validate_onnx_dims(nomic, 256).is_ok());
        assert!(validate_onnx_dims(nomic, nomic.native_dims).is_ok());
        assert!(validate_onnx_dims(nomic, nomic.native_dims + 1).is_err());
        assert!(validate_onnx_dims(nomic, 0).is_err());
    }

    /// Downloads about 130 MB: run explicitly with
    /// `cargo test -p liam-model -- --ignored onnx`.
    #[tokio::test]
    #[ignore = "downloads BGE-small weights; see the doc comment for the run command"]
    async fn onnx_small_model_scores_a_related_pair_above_an_unrelated_pair() {
        use crate::Embedder;

        // Arrange
        let bge = spec("BAAI/bge-small-en-v1.5");
        let cache_dir = std::env::temp_dir().join("liam-onnx-test-cache");
        let embedder = super::OnnxEmbedder::load(bge, bge.native_dims, cache_dir.to_str().unwrap())
            .expect("load bge-small");
        let texts = [
            "where do I live".to_string(),
            "I live in Lisbon, Portugal".to_string(),
            "The mitochondria is the powerhouse of the cell".to_string(),
        ];

        // Act
        let vectors = embedder
            .embed_batch(&texts, TextKind::Document)
            .await
            .expect("embed batch");

        // Assert
        let cosine = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert_eq!(vectors[0].len(), bge.native_dims);
        assert!(cosine(&vectors[0], &vectors[1]) > cosine(&vectors[0], &vectors[2]));
    }

    #[test]
    fn load_rejects_a_non_native_width_before_touching_the_cache() {
        // Arrange
        let bge = spec("BAAI/bge-small-en-v1.5");
        let cache_dir = std::env::temp_dir().join("liam-onnx-dims-test-cache");
        let _ = std::fs::remove_dir_all(&cache_dir);

        // Act
        let result = super::OnnxEmbedder::load(bge, 128, cache_dir.to_str().unwrap());

        // Assert
        assert!(result.is_err());
        assert!(!cache_dir.exists(), "no download before dims validation");
    }
}
