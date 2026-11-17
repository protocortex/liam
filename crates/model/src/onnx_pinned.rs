// SPDX-License-Identifier: Apache-2.0
//! ONNX embedders that fastembed has no built-in variant for.
//!
//! A `PinnedOnnxModel` names one Hugging Face repo at one revision and lists
//! every file the loader needs with its size and sha256. The files download
//! into the embedder cache, are checked against the pins, and only then reach
//! the ONNX session, so a truncated or swapped file never gets loaded.
//!
//! After the first run a size and mtime marker stands in for the hash, and the session reads
//! the file by path later, so the cache directory is trusted to the daemon's own user: a process
//! with write access to it can replace a file that was already verified.

use std::path::{Path, PathBuf};

use crate::error::{ModelError, Result};

/// One file of a pinned model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedFile {
    /// The file name inside the model directory.
    pub name: &'static str,
    /// The path of the file inside the Hugging Face repo.
    pub remote_path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

const fn file(
    name: &'static str,
    remote_path: &'static str,
    size: u64,
    sha256: &'static str,
) -> PinnedFile {
    PinnedFile {
        name,
        remote_path,
        size,
        sha256,
    }
}

/// A model pinned to one Hugging Face revision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedOnnxModel {
    pub repo: &'static str,
    pub revision: &'static str,
    pub onnx: PinnedFile,
    pub tokenizer: PinnedFile,
    pub tokenizer_config: PinnedFile,
    pub special_tokens: PinnedFile,
    pub config: PinnedFile,
    /// How the hidden states turn into one vector.
    pub pooling: fastembed::Pooling,
    pub quantization: fastembed::QuantizationMode,
}

impl PinnedOnnxModel {
    /// Every file the loader needs.
    pub fn files(&self) -> [&PinnedFile; 5] {
        [
            &self.onnx,
            &self.tokenizer,
            &self.tokenizer_config,
            &self.special_tokens,
            &self.config,
        ]
    }
}

const ARCTIC_M_V2_REVISION: &str = "95c2741480856aa9666782eb4afe11959938017f";

/// Snowflake arctic-embed-m-v2.0 as the int8 ONNX export, which is 311 MB
/// where the fp32 export is over 1 GB.
pub const ARCTIC_EMBED_M_V2: PinnedOnnxModel = PinnedOnnxModel {
    repo: "Snowflake/snowflake-arctic-embed-m-v2.0",
    revision: ARCTIC_M_V2_REVISION,
    onnx: file(
        "model_int8.onnx",
        "onnx/model_int8.onnx",
        310_916_060,
        "03d923bb1850ebdccb068e2f3abd8aa43fe81c50d07d037ef103fe3d0fb78e3b",
    ),
    tokenizer: file(
        "tokenizer.json",
        "tokenizer.json",
        17_083_009,
        "f1cc44ad7faaeec47241864835473fd5403f2da94673f3f764a77ebcb0a803ec",
    ),
    tokenizer_config: file(
        "tokenizer_config.json",
        "tokenizer_config.json",
        1_340,
        "6f00514620aff01ba8b7291b2394e98daca5be264cb743805232d9ae27494b2a",
    ),
    special_tokens: file(
        "special_tokens_map.json",
        "special_tokens_map.json",
        964,
        "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
    ),
    config: file(
        "config.json",
        "config.json",
        971,
        "2e0b386e7a826903b06c301f8d4dde2ac6279f7ccd3ea11f3911019a1ee0bc05",
    ),
    pooling: fastembed::Pooling::Cls,
    quantization: fastembed::QuantizationMode::Dynamic,
};

/// Where each verified file of a `PinnedOnnxModel` sits on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedPaths {
    pub onnx: PathBuf,
    pub tokenizer: PathBuf,
    pub tokenizer_config: PathBuf,
    pub special_tokens: PathBuf,
    pub config: PathBuf,
}

fn embed_error(message: String) -> ModelError {
    ModelError::Embed(message)
}

/// Fetches and verifies every file of `model`, then hands the paths to
/// `factory`. `fetch` returns the local path of a file (downloading it on a
/// miss) and `verify` takes that path, the pinned size and the pinned sha256.
/// A file that is missing or fails `verify` stops the load before `factory`
/// runs.
pub(crate) fn load_pinned<T>(
    model: &PinnedOnnxModel,
    fetch: impl Fn(&PinnedFile) -> Result<PathBuf>,
    verify: impl Fn(&Path, u64, &str) -> Result<()>,
    factory: impl FnOnce(PinnedPaths) -> Result<T>,
) -> Result<T> {
    let resolve = |pinned: &PinnedFile| -> Result<PathBuf> {
        let path = fetch(pinned)?;
        verify(&path, pinned.size, pinned.sha256)?;
        Ok(path)
    };
    let paths = PinnedPaths {
        onnx: resolve(&model.onnx)?,
        tokenizer: resolve(&model.tokenizer)?,
        tokenizer_config: resolve(&model.tokenizer_config)?,
        special_tokens: resolve(&model.special_tokens)?,
        config: resolve(&model.config)?,
    };
    factory(paths)
}

/// Downloads one pinned file from the hub into `cache_dir`, a no-op once cached.
pub(crate) fn fetch_from_hub(
    model: &PinnedOnnxModel,
    cache_dir: &str,
    pinned: &PinnedFile,
) -> Result<PathBuf> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(PathBuf::from(cache_dir))
        .build()
        .map_err(|e| embed_error(format!("hf-hub api for {}: {e}", model.repo)))?;
    api.repo(hf_hub::Repo::with_revision(
        model.repo.to_string(),
        hf_hub::RepoType::Model,
        model.revision.to_string(),
    ))
    .get(pinned.remote_path)
    .map_err(|e| {
        embed_error(format!(
            "fetch {} from {}: {e}",
            pinned.remote_path, model.repo
        ))
    })
}

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| embed_error(format!("read {}: {e}", path.display())))
}

/// Builds the fastembed session from verified files.
pub(crate) fn fastembed_session(
    model: &PinnedOnnxModel,
    max_tokens: usize,
    paths: PinnedPaths,
) -> Result<fastembed::TextEmbedding> {
    let tokenizer_files = fastembed::TokenizerFiles {
        tokenizer_file: read(&paths.tokenizer)?,
        config_file: read(&paths.config)?,
        special_tokens_map_file: read(&paths.special_tokens)?,
        tokenizer_config_file: read(&paths.tokenizer_config)?,
    };
    let user_model = fastembed::UserDefinedEmbeddingModel::new(read(&paths.onnx)?, tokenizer_files)
        .with_pooling(model.pooling.clone())
        .with_quantization(model.quantization);
    let options = fastembed::InitOptionsUserDefined::new().with_max_length(max_tokens);
    fastembed::TextEmbedding::try_new_from_user_defined(user_model, options)
        .map_err(|e| embed_error(format!("load {}: {e}", model.repo)))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    const SHA256_HEX_LEN: usize = 64;

    fn is_lowercase_hex(text: &str) -> bool {
        text.len() == SHA256_HEX_LEN && text.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f'))
    }

    #[test]
    fn every_pinned_file_has_a_plain_name_a_size_and_a_lowercase_sha256() {
        for pinned in ARCTIC_EMBED_M_V2.files() {
            // Assert
            assert!(
                !pinned.name.contains('/') && !pinned.name.is_empty(),
                "{}",
                pinned.name
            );
            assert!(pinned.size > 0, "{}", pinned.name);
            assert!(is_lowercase_hex(pinned.sha256), "{}", pinned.name);
        }
        assert_eq!(ARCTIC_EMBED_M_V2.revision.len(), 40);
    }

    #[test]
    fn a_missing_file_refuses_before_the_session_factory_runs() {
        // Arrange
        let factory_calls = Cell::new(0);

        // Act
        let result = load_pinned(
            &ARCTIC_EMBED_M_V2,
            |pinned| Err(ModelError::Embed(format!("{} is missing", pinned.name))),
            |_, _, _| Ok(()),
            |_| {
                factory_calls.set(factory_calls.get() + 1);
                Ok(())
            },
        );

        // Assert
        assert!(result.unwrap_err().to_string().contains("is missing"));
        assert_eq!(factory_calls.get(), 0);
    }

    #[test]
    fn a_file_of_the_wrong_size_refuses_before_the_session_factory_runs() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let stray = dir.path().join("model_int8.onnx");
        std::fs::write(&stray, b"too short").unwrap();
        let factory_calls = Cell::new(0);

        // Act
        let result = load_pinned(
            &ARCTIC_EMBED_M_V2,
            |_| Ok(stray.clone()),
            crate::verify_pinned_file,
            |_| {
                factory_calls.set(factory_calls.get() + 1);
                Ok(())
            },
        );

        // Assert
        let message = result.unwrap_err().to_string();
        assert!(message.contains("bytes, expected"), "{message}");
        assert_eq!(factory_calls.get(), 0);
    }

    #[test]
    fn verified_files_reach_the_factory_once() {
        // Arrange
        let factory_calls = Cell::new(0);

        // Act
        let paths = load_pinned(
            &ARCTIC_EMBED_M_V2,
            |pinned| Ok(PathBuf::from(pinned.name)),
            |_, _, _| Ok(()),
            |paths| {
                factory_calls.set(factory_calls.get() + 1);
                Ok(paths)
            },
        )
        .unwrap();

        // Assert
        assert_eq!(paths.onnx, PathBuf::from("model_int8.onnx"));
        assert_eq!(paths.tokenizer, PathBuf::from("tokenizer.json"));
        assert_eq!(factory_calls.get(), 1);
    }
}
