// SPDX-License-Identifier: Apache-2.0
//! Small ONNX sentence encoders.
//!
//! Each model in `ONNX_MODELS` loads through fastembed's ONNX `TextEmbedding`
//! path, the same one BGE-M3 uses. fastembed owns the pooling and the L2
//! normalisation per model; this module owns what it leaves to the caller: the
//! query and document prefixes, the input token cap, and the output width.
//! Adding a model is one row in the table, naming either its fastembed variant
//! or a pinned download for a model fastembed does not list.

use crate::blocking::with_model_blocking;
use crate::download_notice;
use crate::error::{ModelError, Result};
use crate::onnx_pinned::{self, PinnedOnnxModel, ARCTIC_EMBED_M_V2};
use crate::TextKind;
use async_trait::async_trait;

/// Inputs per ONNX run on the batched path. fastembed's default of 256 pads the
/// whole batch to its longest input; a small batch bounds that.
const EMBED_BATCH_SIZE: usize = 16;

/// The model accepts 8192 tokens. Stored facts are short, so 512 bounds the
/// latency of a long input without cutting normal ones.
const ARCTIC_M_V2_MAX_TOKENS: usize = 512;

/// The retrieval query instruction of the BGE and Arctic model cards. Passages
/// take no prefix.
const BGE_STYLE_QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

/// The embedding model `embedder.model` selects when the config names none.
pub const DEFAULT_EMBEDDER_MODEL_ID: &str = "Snowflake/snowflake-arctic-embed-m-v2.0";

/// Where a model's files and loader come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OnnxSource {
    /// A variant fastembed downloads and loads itself.
    Fastembed(fastembed::EmbeddingModel),
    /// Files pinned by size and sha256, downloaded and checked here.
    Pinned(&'static PinnedOnnxModel),
}

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
    pub source: OnnxSource,
}

impl OnnxModelSpec {
    /// Whether one model call may hold several texts. Dynamic quantisation
    /// scales its data range to each batch, so a text's vector would depend on
    /// the texts beside it.
    pub fn allows_batching(&self) -> bool {
        match &self.source {
            OnnxSource::Fastembed(_) => true,
            OnnxSource::Pinned(pinned) => {
                !matches!(pinned.quantization, fastembed::QuantizationMode::Dynamic)
            }
        }
    }
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
        source: OnnxSource::Fastembed(fastembed::EmbeddingModel::AllMiniLML6V2),
    },
    OnnxModelSpec {
        id: "sentence-transformers/all-MiniLM-L12-v2",
        native_dims: 384,
        matryoshka: false,
        max_tokens: 128,
        query_prefix: "",
        document_prefix: "",
        source: OnnxSource::Fastembed(fastembed::EmbeddingModel::AllMiniLML12V2),
    },
    OnnxModelSpec {
        id: "BAAI/bge-small-en-v1.5",
        native_dims: 384,
        matryoshka: false,
        max_tokens: 512,
        query_prefix: BGE_STYLE_QUERY_PREFIX,
        document_prefix: "",
        source: OnnxSource::Fastembed(fastembed::EmbeddingModel::BGESmallENV15),
    },
    OnnxModelSpec {
        id: "Snowflake/snowflake-arctic-embed-xs",
        native_dims: 384,
        matryoshka: false,
        max_tokens: 512,
        query_prefix: BGE_STYLE_QUERY_PREFIX,
        document_prefix: "",
        source: OnnxSource::Fastembed(fastembed::EmbeddingModel::SnowflakeArcticEmbedXS),
    },
    OnnxModelSpec {
        id: "Snowflake/snowflake-arctic-embed-s",
        native_dims: 384,
        matryoshka: false,
        max_tokens: 512,
        query_prefix: BGE_STYLE_QUERY_PREFIX,
        document_prefix: "",
        source: OnnxSource::Fastembed(fastembed::EmbeddingModel::SnowflakeArcticEmbedS),
    },
    OnnxModelSpec {
        id: "nomic-ai/nomic-embed-text-v1.5",
        native_dims: 768,
        matryoshka: true,
        max_tokens: 8192,
        query_prefix: "search_query: ",
        document_prefix: "search_document: ",
        source: OnnxSource::Fastembed(fastembed::EmbeddingModel::NomicEmbedTextV15),
    },
    OnnxModelSpec {
        id: "intfloat/multilingual-e5-small",
        native_dims: 384,
        matryoshka: false,
        max_tokens: 512,
        query_prefix: "query: ",
        document_prefix: "passage: ",
        source: OnnxSource::Fastembed(fastembed::EmbeddingModel::MultilingualE5Small),
    },
    OnnxModelSpec {
        id: DEFAULT_EMBEDDER_MODEL_ID,
        native_dims: 768,
        matryoshka: true,
        max_tokens: ARCTIC_M_V2_MAX_TOKENS,
        query_prefix: "query: ",
        document_prefix: "",
        source: OnnxSource::Pinned(&ARCTIC_EMBED_M_V2),
    },
];

/// The spec whose id is `model`. Hugging Face ids are case-insensitive, so the
/// match is too.
pub fn onnx_model_spec(model: &str) -> Option<&'static OnnxModelSpec> {
    ONNX_MODELS
        .iter()
        .find(|spec| spec.id.eq_ignore_ascii_case(model))
}

/// The `HF_HOME` value that redirects `spec`'s download away from `cache_dir`.
/// A pinned model downloads through an explicit cache directory, which
/// `HF_HOME` cannot override; only fastembed's own download honours it.
fn hf_home_override<'a>(
    spec: &OnnxModelSpec,
    hf_home: Option<&'a str>,
    cache_dir: &str,
) -> Option<&'a str> {
    match spec.source {
        OnnxSource::Fastembed(_) => crate::bge::hf_home_overriding(hf_home, cache_dir),
        OnnxSource::Pinned(_) => None,
    }
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
            hf_home_override(spec, std::env::var("HF_HOME").ok().as_deref(), cache_dir)
        {
            tracing::warn!(
                hf_home,
                cache_dir,
                "HF_HOME is set, so {} is stored under HF_HOME, not embedder.cache_dir",
                spec.id
            );
        }
        let model = match &spec.source {
            OnnxSource::Fastembed(variant) => {
                if let Ok(info) = fastembed::TextEmbedding::get_model_info(variant) {
                    download_notice::log_unless_cached(
                        download_notice::hub_repo_cached(cache_dir, &info.model_code),
                        spec.id,
                        cache_dir,
                        None,
                    );
                }
                let options = fastembed::TextInitOptions::new(variant.clone())
                    .with_cache_dir(std::path::PathBuf::from(cache_dir))
                    .with_max_length(spec.max_tokens);
                fastembed::TextEmbedding::try_new(options)
                    .map_err(|e| ModelError::embed_failed(format!("load {}", spec.id), e))?
            }
            OnnxSource::Pinned(pinned) => onnx_pinned::load_pinned(
                pinned,
                |file| {
                    download_notice::hub_file_cached(
                        cache_dir,
                        pinned.repo,
                        pinned.revision,
                        file.remote_path,
                    )
                },
                |bytes| download_notice::log_unless_cached(false, spec.id, cache_dir, Some(bytes)),
                |file| onnx_pinned::fetch_from_hub(pinned, cache_dir, file),
                crate::verify_pinned_file,
                |paths| onnx_pinned::fastembed_session(pinned, spec.max_tokens, paths),
            )?,
        };
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
            let full = embed_inputs(&inputs, spec.allows_batching(), |chunk, batch_size| {
                m.embed(chunk, batch_size)
                    .map_err(|e| ModelError::embed_failed("embedding failed", e))
            })?;
            Ok(full
                .iter()
                .map(|v| crate::embedder::mrl_truncate(v, dims))
                .collect())
        })
        .await
    }
}

/// Runs `embed` over `inputs`, in order. When batching is not allowed each text
/// gets its own call, so its vector never depends on its neighbours.
fn embed_inputs<E>(inputs: &[String], batching: bool, mut embed: E) -> Result<Vec<Vec<f32>>>
where
    E: FnMut(&[String], Option<usize>) -> Result<Vec<Vec<f32>>>,
{
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    if batching {
        return embed(inputs, Some(EMBED_BATCH_SIZE));
    }
    let mut vectors = Vec::with_capacity(inputs.len());
    for input in inputs {
        vectors.extend(embed(std::slice::from_ref(input), None)?);
    }
    Ok(vectors)
}

#[cfg(test)]
mod tests {
    use super::{
        embed_inputs, hf_home_override, onnx_input, onnx_model_spec, validate_onnx_dims,
        OnnxModelSpec, BGE_STYLE_QUERY_PREFIX, EMBED_BATCH_SIZE, ONNX_MODELS,
    };
    use crate::TextKind;

    const TEXT_COUNT: usize = 40;
    const FAILING_TEXT_INDEX: usize = 2;
    const EMBED_EPSILON: f32 = 1e-4;
    const BATCH_TEXT_COUNT: usize = 20;

    fn texts(count: usize) -> Vec<String> {
        (0..count).map(|i| i.to_string()).collect()
    }

    /// Records each call as (texts in the call, batch size) and answers every
    /// text with a one-element vector holding its number.
    fn recording(
        calls: &mut Vec<(usize, Option<usize>)>,
    ) -> impl FnMut(&[String], Option<usize>) -> crate::Result<Vec<Vec<f32>>> + '_ {
        move |chunk, batch_size| {
            calls.push((chunk.len(), batch_size));
            Ok(chunk.iter().map(|t| vec![t.parse().unwrap()]).collect())
        }
    }

    #[test]
    fn a_dynamic_spec_embeds_one_text_per_call_and_keeps_the_order() {
        // Arrange
        assert!(!spec(super::DEFAULT_EMBEDDER_MODEL_ID).allows_batching());
        let inputs = texts(TEXT_COUNT);
        let mut calls = Vec::new();

        // Act
        let vectors = embed_inputs(&inputs, false, recording(&mut calls)).unwrap();

        // Assert
        assert_eq!(calls.len(), TEXT_COUNT);
        assert!(calls.iter().all(|(len, _)| *len == 1), "{calls:?}");
        let expected: Vec<Vec<f32>> = (0..TEXT_COUNT).map(|i| vec![i as f32]).collect();
        assert_eq!(vectors, expected);
    }

    #[test]
    fn a_non_dynamic_spec_embeds_in_one_batched_call() {
        // Arrange
        assert!(spec("BAAI/bge-small-en-v1.5").allows_batching());
        let inputs = texts(TEXT_COUNT);
        let mut calls = Vec::new();

        // Act
        let vectors = embed_inputs(&inputs, true, recording(&mut calls)).unwrap();

        // Assert
        assert_eq!(calls, vec![(TEXT_COUNT, Some(EMBED_BATCH_SIZE))]);
        assert_eq!(vectors.len(), TEXT_COUNT);
    }

    #[test]
    fn the_spec_decides_whether_embed_inputs_batches() {
        for (id, expected_calls) in [
            (super::DEFAULT_EMBEDDER_MODEL_ID, TEXT_COUNT),
            ("BAAI/bge-small-en-v1.5", 1),
        ] {
            // Arrange
            let batching = spec(id).allows_batching();
            let inputs = texts(TEXT_COUNT);
            let mut calls = Vec::new();

            // Act
            embed_inputs(&inputs, batching, recording(&mut calls)).unwrap();

            // Assert
            assert_eq!(calls.len(), expected_calls, "{id}: {calls:?}");
        }
    }

    #[test]
    fn no_fastembed_spec_that_allows_batching_is_dynamically_quantised() {
        for spec in ONNX_MODELS {
            // Arrange
            let super::OnnxSource::Fastembed(variant) = &spec.source else {
                continue;
            };

            // Act
            let mode = fastembed::TextEmbedding::get_quantization_mode(variant);

            // Assert
            assert!(
                !(spec.allows_batching() && matches!(mode, fastembed::QuantizationMode::Dynamic)),
                "{} is dynamically quantised: make allows_batching return false for that spec",
                spec.id
            );
        }
    }

    #[test]
    fn an_empty_input_returns_no_vectors_and_makes_no_call() {
        for batching in [false, true] {
            // Arrange
            let mut calls = Vec::new();

            // Act
            let vectors = embed_inputs(&[], batching, recording(&mut calls)).unwrap();

            // Assert
            assert!(vectors.is_empty());
            assert!(calls.is_empty(), "{calls:?}");
        }
    }

    #[test]
    fn an_error_on_the_third_text_is_returned_and_stops_the_run() {
        // Arrange
        let inputs = texts(TEXT_COUNT);
        let mut calls = 0;

        // Act
        let result = embed_inputs(&inputs, false, |chunk, _| {
            if calls == FAILING_TEXT_INDEX {
                return Err(crate::ModelError::Embed("boom".into()));
            }
            calls += 1;
            Ok(chunk.iter().map(|_| vec![0.0]).collect())
        });

        // Assert
        assert!(matches!(result, Err(crate::ModelError::Embed(m)) if m == "boom"));
        assert_eq!(calls, FAILING_TEXT_INDEX);
    }

    fn spec(id: &str) -> &'static OnnxModelSpec {
        onnx_model_spec(id).expect("a listed model")
    }

    #[test]
    fn hf_home_is_reported_for_a_fastembed_download_but_not_a_pinned_one() {
        // Arrange
        let fastembed = spec("BAAI/bge-small-en-v1.5");
        let pinned = spec(crate::DEFAULT_EMBEDDER_MODEL_ID);

        // Act
        let for_fastembed = hf_home_override(fastembed, Some("/hf"), "/cache");
        let for_pinned = hf_home_override(pinned, Some("/hf"), "/cache");

        // Assert
        assert_eq!(for_fastembed, Some("/hf"));
        assert_eq!(for_pinned, None);
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
    fn arctic_is_the_default_and_loads_from_pinned_files() {
        // Arrange / Act
        let arctic = spec(super::DEFAULT_EMBEDDER_MODEL_ID);

        // Assert
        assert_eq!(arctic.id, "Snowflake/snowflake-arctic-embed-m-v2.0");
        assert_eq!(arctic.native_dims, 768);
        assert!(arctic.matryoshka);
        assert_eq!(arctic.max_tokens, 512);
        assert_eq!(
            arctic.source,
            super::OnnxSource::Pinned(&super::ARCTIC_EMBED_M_V2)
        );
    }

    #[test]
    fn arctic_prefixes_queries_and_leaves_documents_bare() {
        // Arrange
        let arctic = spec(super::DEFAULT_EMBEDDER_MODEL_ID);

        // Act / Assert
        assert_eq!(onnx_input(arctic, "q", TextKind::Query), "query: q");
        assert_eq!(onnx_input(arctic, "d", TextKind::Document), "d");
    }

    #[test]
    fn a_768_wide_arctic_vector_is_cut_to_384_and_renormalised() {
        // Arrange
        let arctic = spec(super::DEFAULT_EMBEDDER_MODEL_ID);
        let unit = 1.0 / (arctic.native_dims as f32).sqrt();
        let full = vec![unit; arctic.native_dims];

        // Act
        let cut = crate::embedder::mrl_truncate(&full, 384);

        // Assert
        assert!(validate_onnx_dims(arctic, 384).is_ok());
        assert_eq!(cut.len(), 384);
        let norm = cut.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "{norm}");
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

    /// Uses the pinned files in the directory `LIAM_MODEL_CACHE` names and
    /// downloads the missing ones (311 MB). Run with
    /// `LIAM_MODEL_CACHE=/path/to/cache cargo test -p liam-model -- --ignored arctic`.
    #[tokio::test]
    #[ignore = "downloads the arctic weights; see the doc comment for the run command"]
    async fn arctic_ranks_the_relevant_document_first() {
        use crate::Embedder;

        // Arrange
        let cache_dir = std::env::var("LIAM_MODEL_CACHE").expect("set LIAM_MODEL_CACHE");
        let arctic = spec(super::DEFAULT_EMBEDDER_MODEL_ID);
        let embedder = super::OnnxEmbedder::load(arctic, 384, &cache_dir).expect("load arctic");
        let documents = [
            "The mitochondria is the powerhouse of the cell".to_string(),
            "I live in Lisbon, Portugal".to_string(),
            "The stock market closed higher on Tuesday".to_string(),
        ];

        // Act
        let query = embedder
            .embed("where do I live", TextKind::Query)
            .await
            .expect("embed query");
        let vectors = embedder
            .embed_batch(&documents, TextKind::Document)
            .await
            .expect("embed documents");
        let many: Vec<String> = (0..BATCH_TEXT_COUNT)
            .map(|i| format!("{} {i}", documents[i % documents.len()]))
            .collect();
        let many_vectors = embedder
            .embed_batch(&many, TextKind::Document)
            .await
            .expect("embed more documents than one batch");
        let alone = embedder
            .embed(&many[0], TextKind::Document)
            .await
            .expect("embed one document");

        // Assert
        let cosine = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let scores: Vec<f32> = vectors.iter().map(|v| cosine(&query, v)).collect();
        assert_eq!(query.len(), 384);
        assert!(scores[1] > scores[0] && scores[1] > scores[2], "{scores:?}");
        assert_eq!(many_vectors.len(), BATCH_TEXT_COUNT);
        let drift = many_vectors[0]
            .iter()
            .zip(&alone)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(drift < EMBED_EPSILON, "{drift}");
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
