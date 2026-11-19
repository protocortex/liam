// SPDX-License-Identifier: Apache-2.0
//! The embedder: Snowflake arctic-embed-m-v2.0 as an int8 ONNX model on the
//! CPU, loaded through fastembed.
//!
//! The model files are pinned by revision, size and sha256 (see
//! `onnx_pinned`). fastembed owns the pooling and the L2 normalisation; this
//! module owns what it leaves to the caller: the query prefix, the input token
//! cap and the Matryoshka cut of the output width.

use crate::blocking::with_model_blocking;
use crate::download_notice::{self, NoticeCaller};
use crate::error::{ModelError, Result};
use crate::onnx_pinned::{self, PinnedFile, PinnedOnnxModel, PinnedPaths, ARCTIC_EMBED_M_V2};
use crate::TextKind;
use async_trait::async_trait;
use std::path::{Path, PathBuf};

/// The Hugging Face id of the one embedding model, which is what
/// `embedder.model` may name.
pub const EMBEDDER_MODEL_ID: &str = "Snowflake/snowflake-arctic-embed-m-v2.0";

/// The model's own output width. A shorter one is cut from it.
const NATIVE_DIMS: usize = 768;

/// The model accepts 8192 tokens. Stored facts are short, so 512 bounds the
/// latency of a long input without cutting normal ones.
const MAX_TOKENS: usize = 512;

/// The retrieval query instruction of the model card. Passages take no prefix.
const QUERY_PREFIX: &str = "query: ";

/// What `Embedder::backend` reports for the ONNX runtime on the CPU.
const BACKEND_LABEL: &str = "onnx/cpu";

/// The exact string handed to the model for `text` in its `kind` role.
fn model_input(text: &str, kind: TextKind) -> String {
    match kind {
        TextKind::Query => format!("{QUERY_PREFIX}{text}"),
        TextKind::Document => text.to_string(),
    }
}

/// In-process embedder for arctic-embed-m-v2.0. The sync ONNX call runs on a
/// blocking thread so the async runtime stays free.
pub struct ArcticEmbedder {
    model: std::sync::Arc<tokio::sync::Mutex<fastembed::TextEmbedding>>,
    dims: usize,
}

impl ArcticEmbedder {
    /// Load the model into `cache_dir`, producing vectors `dims` wide.
    pub fn load(dims: usize, cache_dir: &str, caller: NoticeCaller) -> Result<Self> {
        crate::embedder::validate_dims(EMBEDDER_MODEL_ID, dims, NATIVE_DIMS)?;
        let model = load_session(
            cache_dir,
            caller,
            |pinned, file| onnx_pinned::fetch_from_hub(pinned, cache_dir, file),
            crate::verify_pinned_file,
            onnx_pinned::fastembed_session,
        )?;
        Ok(Self {
            model: std::sync::Arc::new(tokio::sync::Mutex::new(model)),
            dims,
        })
    }
}

/// Fetches and verifies the pinned files, then builds the session with the
/// token cap. The fetch, verify and session steps are parameters so a test can
/// run the path without a model.
fn load_session<T>(
    cache_dir: &str,
    caller: NoticeCaller,
    fetch: impl Fn(&PinnedOnnxModel, &PinnedFile) -> Result<PathBuf>,
    verify: impl Fn(&Path, u64, &str) -> Result<()>,
    session: impl FnOnce(&PinnedOnnxModel, usize, PinnedPaths) -> Result<T>,
) -> Result<T> {
    let pinned = &ARCTIC_EMBED_M_V2;
    onnx_pinned::load_pinned(
        pinned,
        |file| {
            download_notice::hub_file_cached(
                cache_dir,
                pinned.repo,
                pinned.revision,
                file.remote_path,
            )
        },
        |bytes| {
            download_notice::log_unless_cached(
                false,
                EMBEDDER_MODEL_ID,
                cache_dir,
                Some(bytes),
                caller,
            )
        },
        |file| fetch(pinned, file),
        verify,
        |paths| session(pinned, MAX_TOKENS, paths),
    )
}

#[async_trait]
impl crate::Embedder for ArcticEmbedder {
    fn dims(&self) -> usize {
        self.dims
    }

    fn backend(&self) -> &str {
        BACKEND_LABEL
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
        let dims = self.dims;
        let inputs: Vec<String> = texts.iter().map(|t| model_input(t, kind)).collect();
        with_model_blocking(model, ModelError::Embed, move |m| {
            let full = embed_one_by_one(&inputs, |input| {
                m.embed(std::slice::from_ref(input), None)
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

/// Runs `embed` over `inputs`, in order, one text per call. The model is
/// dynamically quantised, which scales its data range to each call, so a text
/// embedded beside others would get a different vector than one embedded alone.
fn embed_one_by_one<E>(inputs: &[String], mut embed: E) -> Result<Vec<Vec<f32>>>
where
    E: FnMut(&String) -> Result<Vec<Vec<f32>>>,
{
    let mut vectors = Vec::with_capacity(inputs.len());
    for input in inputs {
        vectors.extend(embed(input)?);
    }
    Ok(vectors)
}

#[cfg(test)]
mod tests {
    use super::{embed_one_by_one, model_input, EMBEDDER_MODEL_ID, NATIVE_DIMS};
    use crate::TextKind;

    const TEXT_COUNT: usize = 40;
    const FAILING_TEXT_INDEX: usize = 2;
    const CUT_DIMS: usize = 384;
    const UNIT_NORM_TOLERANCE: f32 = 1e-5;

    fn texts(count: usize) -> Vec<String> {
        (0..count).map(|i| i.to_string()).collect()
    }

    #[test]
    fn each_text_is_embedded_in_its_own_call_and_the_order_is_kept() {
        // Arrange
        let inputs = texts(TEXT_COUNT);
        let mut calls = 0;

        // Act
        let vectors = embed_one_by_one(&inputs, |text| {
            calls += 1;
            Ok(vec![vec![text.parse().unwrap()]])
        })
        .unwrap();

        // Assert
        assert_eq!(calls, TEXT_COUNT);
        let expected: Vec<Vec<f32>> = (0..TEXT_COUNT).map(|i| vec![i as f32]).collect();
        assert_eq!(vectors, expected);
    }

    #[test]
    fn an_empty_input_returns_no_vectors_and_makes_no_call() {
        // Arrange
        let mut calls = 0;

        // Act
        let vectors = embed_one_by_one(&[], |_| {
            calls += 1;
            Ok(Vec::new())
        })
        .unwrap();

        // Assert
        assert!(vectors.is_empty());
        assert_eq!(calls, 0);
    }

    #[test]
    fn an_error_on_the_third_text_is_returned_and_stops_the_run() {
        // Arrange
        let inputs = texts(TEXT_COUNT);
        let mut calls = 0;

        // Act
        let result = embed_one_by_one(&inputs, |_| {
            if calls == FAILING_TEXT_INDEX {
                return Err(crate::ModelError::Embed("boom".into()));
            }
            calls += 1;
            Ok(vec![vec![0.0]])
        });

        // Assert
        assert!(matches!(result, Err(crate::ModelError::Embed(m)) if m == "boom"));
        assert_eq!(calls, FAILING_TEXT_INDEX);
    }

    #[test]
    fn queries_get_the_query_prefix_and_documents_stay_bare() {
        // Arrange / Act / Assert
        assert_eq!(model_input("q", TextKind::Query), "query: q");
        assert_eq!(model_input("d", TextKind::Document), "d");
    }

    #[test]
    fn a_native_width_vector_is_cut_to_the_configured_width_and_renormalised() {
        // Arrange
        let unit = 1.0 / (NATIVE_DIMS as f32).sqrt();
        let full = vec![unit; NATIVE_DIMS];

        // Act
        let cut = crate::embedder::mrl_truncate(&full, CUT_DIMS);

        // Assert
        assert_eq!(cut.len(), CUT_DIMS);
        let norm = cut.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < UNIT_NORM_TOLERANCE, "{norm}");
    }

    #[test]
    fn load_rejects_a_width_beyond_the_native_one_before_touching_the_cache() {
        // Arrange
        let cache_dir = tempfile::tempdir().unwrap();
        let missing = cache_dir.path().join("never-created");

        // Act
        let too_wide = super::ArcticEmbedder::load(
            NATIVE_DIMS + 1,
            missing.to_str().unwrap(),
            crate::NoticeCaller::Daemon,
        );
        let zero =
            super::ArcticEmbedder::load(0, missing.to_str().unwrap(), crate::NoticeCaller::Daemon);

        // Assert
        let message = too_wide.err().unwrap().to_string();
        assert!(message.contains(EMBEDDER_MODEL_ID), "{message}");
        assert!(zero.is_err());
        assert!(!missing.exists(), "no download before dims validation");
    }

    #[test]
    fn the_model_identity_and_pinned_source_are_the_expected_ones() {
        // Arrange
        let pinned = &super::ARCTIC_EMBED_M_V2;

        // Act / Assert
        assert_eq!(EMBEDDER_MODEL_ID, "Snowflake/snowflake-arctic-embed-m-v2.0");
        assert_eq!(pinned.repo, "Snowflake/snowflake-arctic-embed-m-v2.0");
        assert_eq!(NATIVE_DIMS, 768);
        assert_eq!(super::MAX_TOKENS, 512);
        assert_eq!(pinned.revision, "95c2741480856aa9666782eb4afe11959938017f");
        assert_eq!(pinned.onnx.name, "model_int8.onnx");
        assert_eq!(pinned.onnx.size, 310_916_060);
    }

    #[test]
    fn the_load_path_gives_the_session_factory_the_token_cap_and_the_arctic_spec() {
        // Arrange
        let cache_dir = tempfile::tempdir().unwrap();
        let mut seen = None;

        // Act
        let result = super::load_session(
            cache_dir.path().to_str().unwrap(),
            crate::NoticeCaller::Daemon,
            |_, file| Ok(std::path::PathBuf::from(file.name)),
            |_, _, _| Ok(()),
            |pinned, max_tokens, _| {
                seen = Some((pinned.repo, pinned.revision, max_tokens));
                Ok(())
            },
        );

        // Assert
        assert!(result.is_ok());
        assert_eq!(
            seen,
            Some((
                "Snowflake/snowflake-arctic-embed-m-v2.0",
                "95c2741480856aa9666782eb4afe11959938017f",
                512
            ))
        );
    }
}
