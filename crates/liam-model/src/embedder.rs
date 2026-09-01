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

/// Parse `config.json` bytes as a Qwen3 embedding config. Sharded and
/// Qwen3-VL checkpoints nest their fields under `text_config` instead of at
/// the top level, so they fail this parse rather than being silently
/// mis-loaded; see `load`'s scope note on why they are not ported.
#[cfg(feature = "local")]
fn parse_qwen3_config(model_id: &str, bytes: &[u8]) -> Result<fastembed::Qwen3Config> {
    serde_json::from_slice(bytes).map_err(|e| {
        crate::error::ModelError::Embed(format!(
            "{model_id}'s config.json does not parse as a Qwen3 embedding config ({e}); \
             sharded and Qwen3-VL checkpoints are not supported by this loader"
        ))
    })
}

/// The weight-fetch failure message: named so its wording is pinned by a
/// test, unlike this file's usual inline `ModelError::Embed` formatting.
#[cfg(feature = "local")]
fn weight_fetch_error(model_id: &str, cause: &str) -> crate::error::ModelError {
    crate::error::ModelError::Embed(format!(
        "failed to fetch model.safetensors for {model_id}: {cause}; if {model_id} is a \
         sharded or Qwen3-VL checkpoint, this loader does not support that shape"
    ))
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
    /// Load a model by Hugging Face id (e.g. "Qwen/Qwen3-Embedding-0.6B") into
    /// `cache_dir`, truncating its output to `dims` via Matryoshka (see
    /// `mrl_truncate`).
    ///
    /// Ported from fastembed v5.17.3's `Qwen3TextEmbedding::from_hf`
    /// (`qwen3.rs:1010-1087`), which builds its `hf_hub` API with no cache
    /// directory, so its weights always land under the OS default
    /// `~/.cache/huggingface/hub` regardless of this project's configured
    /// `embedder.cache_dir`. This body is the same steps with an explicit
    /// `with_cache_dir`, only a single unsharded `model.safetensors` and a
    /// plain (non-VL) `Qwen3Config` supported; see `parse_qwen3_config` and
    /// `weight_fetch_error` for the two rejected shapes.
    pub fn load(model_id: &str, dims: usize, cache_dir: &str) -> Result<Self> {
        let device = candle_core::Device::Cpu;
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(std::path::PathBuf::from(cache_dir))
            .build()
            .map_err(|e| {
                crate::error::ModelError::Embed(format!("hf-hub api for {model_id}: {e}"))
            })?;
        let repo = api.model(model_id.to_string());

        let config_path = repo.get("config.json").map_err(|e| {
            crate::error::ModelError::Embed(format!("fetch config.json for {model_id}: {e}"))
        })?;
        let config_bytes = std::fs::read(&config_path).map_err(|e| {
            crate::error::ModelError::Embed(format!("read config.json for {model_id}: {e}"))
        })?;
        let cfg = parse_qwen3_config(model_id, &config_bytes)?;
        validate_dims(model_id, dims, cfg.hidden_size)?;

        let weight_path = repo
            .get("model.safetensors")
            .map_err(|e| weight_fetch_error(model_id, &e.to_string()))?;

        // SAFETY: hf-hub stores each blob at a content-addressed path never
        // mutated in place, so no concurrent write can invalidate this mmap.
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[weight_path],
                candle_core::DType::F32,
                &device,
            )
        }
        .map_err(|e| {
            crate::error::ModelError::Embed(format!("load weights for {model_id}: {e}"))
        })?;
        let qwen3_model = fastembed::Qwen3Model::new(cfg, vb).map_err(|e| {
            crate::error::ModelError::Embed(format!("build model for {model_id}: {e}"))
        })?;

        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
            crate::error::ModelError::Embed(format!("fetch tokenizer.json for {model_id}: {e}"))
        })?;
        let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            crate::error::ModelError::Embed(format!("load tokenizer for {model_id}: {e}"))
        })?;
        let _ = tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Left,
            ..Default::default()
        }));
        let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: EMBED_MAX_INPUT_TOKENS,
            ..Default::default()
        }));

        let model = fastembed::Qwen3TextEmbedding::new(qwen3_model, tokenizer);
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
    #[cfg(feature = "local")]
    use super::{parse_qwen3_config, weight_fetch_error};

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

    /// A minimal, well-formed Qwen3 embedding config: every field
    /// `fastembed::Qwen3Config` requires without a `#[serde(default)]` or
    /// `Option` type, so this parses on its own and also nests under
    /// `text_config` to build a VL-shaped fixture below.
    #[cfg(feature = "local")]
    const VALID_QWEN3_CONFIG_JSON: &str = r#"{
        "attention_bias": false,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "max_position_embeddings": 32768,
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": true,
        "vocab_size": 151936
    }"#;

    #[cfg(feature = "local")]
    #[test]
    fn parse_qwen3_config_parses_a_well_formed_config() {
        // Arrange / Act
        let cfg = parse_qwen3_config("test-model", VALID_QWEN3_CONFIG_JSON.as_bytes())
            .expect("a well-formed Qwen3 config must parse");

        // Assert
        assert_eq!(cfg.hidden_size, 1024);
    }

    #[cfg(feature = "local")]
    #[test]
    fn parse_qwen3_config_rejects_unsupported_shapes_naming_the_model() {
        // Arrange: malformed JSON, and a VL-shaped config nesting every field
        // under text_config instead of at the top level, the real shape of a
        // Qwen3-VL checkpoint's config.json.
        let malformed = b"not json";
        let vl_shaped = format!(r#"{{"text_config": {VALID_QWEN3_CONFIG_JSON}}}"#);

        // Act
        let malformed_err = parse_qwen3_config("test-model", malformed)
            .unwrap_err()
            .to_string();
        let vl_err = parse_qwen3_config("Qwen/Qwen3-VL-Embedding-2B", vl_shaped.as_bytes())
            .unwrap_err()
            .to_string();

        // Assert: both name the model id and state the same limitation.
        assert!(
            malformed_err.contains("test-model"),
            "message: {malformed_err}"
        );
        assert!(
            malformed_err.contains("sharded"),
            "message: {malformed_err}"
        );
        assert!(
            vl_err.contains("Qwen/Qwen3-VL-Embedding-2B"),
            "message: {vl_err}"
        );
        assert!(vl_err.contains("VL"), "message: {vl_err}");

        // ...for different reasons: malformed bytes never parse as JSON at
        // all, the VL fixture is valid JSON missing required fields.
        assert!(
            !malformed_err.contains("missing field"),
            "message: {malformed_err}"
        );
        assert!(vl_err.contains("missing field"), "message: {vl_err}");
    }

    #[cfg(feature = "local")]
    #[test]
    fn weight_fetch_error_names_the_model_the_cause_and_the_limitation() {
        // Arrange / Act
        let err = weight_fetch_error("test-model", "404 not found").to_string();

        // Assert
        assert!(err.contains("test-model"), "message: {err}");
        assert!(err.contains("404 not found"), "message: {err}");
        assert!(err.contains("sharded"), "message: {err}");
    }

    /// Real-tier regression pin for issue #112: a fresh temp dir per run
    /// means this can only pass if `with_cache_dir` genuinely took effect,
    /// since the old bare `from_hf` call would leave it empty. Gated and
    /// ignored the same way as `tool_eval.rs`'s and `retrieval_eval.rs`'s own
    /// real-tier tests: `cargo test -p liam-model --features local -- --ignored`.
    #[cfg(feature = "local")]
    #[tokio::test]
    #[ignore = "downloads embedder weights; see module doc for the run command"]
    async fn load_downloads_weights_into_the_configured_cache_dir() {
        use crate::Embedder;

        // Arrange: removed on drop, even on an early panic below.
        struct CleanupOnDrop(std::path::PathBuf);
        impl Drop for CleanupOnDrop {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let cache_dir = CleanupOnDrop(std::env::temp_dir().join(format!(
            "liam-embedder-cache-dir-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock")
                .as_nanos()
        )));
        let model_id = "Qwen/Qwen3-Embedding-0.6B";
        let native_dims = 1024;

        // Act
        let embedder =
            super::FastEmbedEmbedder::load(model_id, native_dims, cache_dir.0.to_str().unwrap())
                .expect("load real embedder into a fresh cache dir");

        // Assert: the fetched files landed somewhere under the fresh cache dir.
        for file in ["config.json", "model.safetensors", "tokenizer.json"] {
            assert!(
                file_exists_under(&cache_dir.0, file),
                "{file} should exist under {:?} after load",
                cache_dir.0
            );
        }
        let vector = embedder
            .embed("hello world")
            .await
            .expect("embed after load");
        assert_eq!(vector.len(), native_dims);
    }

    #[cfg(feature = "local")]
    fn file_exists_under(dir: &std::path::Path, name: &str) -> bool {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return false;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if file_exists_under(&path, name) {
                    return true;
                }
            } else if path.file_name().and_then(|n| n.to_str()) == Some(name) {
                return true;
            }
        }
        false
    }
}
