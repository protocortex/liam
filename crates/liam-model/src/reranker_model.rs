// SPDX-License-Identifier: Apache-2.0
//! Pinned ONNX cross-encoder rerankers that load from a local directory.
//!
//! Each spec in `RERANKER_MODELS` names a Hugging Face repo at one revision and
//! lists every file the loader needs with its size and sha256. Loading checks
//! that each file exists and has its pinned size; `verify_installed` adds the
//! sha256 check.

use std::fs;
use std::path::{Path, PathBuf};

use async_trait::async_trait;

use crate::error::{ModelError, Result};
use crate::reranker::{DocScorer, ModelReranker};
use crate::Reranker;

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

/// A reranker pinned to one Hugging Face revision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankerModelSpec {
    /// The id config selects the model with.
    pub id: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub files: &'static [PinnedFile],
}

pub const ONNX_FILE_NAME: &str = "model_int8.onnx";
pub const TOKENIZER_FILE_NAME: &str = "tokenizer.json";
pub const TOKENIZER_CONFIG_FILE_NAME: &str = "tokenizer_config.json";
pub const SPECIAL_TOKENS_FILE_NAME: &str = "special_tokens_map.json";
pub const CONFIG_FILE_NAME: &str = "config.json";

const ONNX_REMOTE_PATH: &str = "onnx/model_int8.onnx";

pub const GTE_RERANKER_ID: &str = "gte-multilingual-reranker-base";
pub const BGE_RERANKER_V2_M3_ID: &str = "bge-reranker-v2-m3";

// Both repos ship the same special tokens map.
const SPECIAL_TOKENS_SIZE: u64 = 964;
const SPECIAL_TOKENS_SHA256: &str =
    "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835";

pub(crate) const fn file(
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

const GTE_FILES: &[PinnedFile] = &[
    file(
        ONNX_FILE_NAME,
        ONNX_REMOTE_PATH,
        340_858_200,
        "ccf51dba7f8aa9205753761cfaa68c55f741792501463a3bf25d7e5bcdac7c35",
    ),
    file(
        TOKENIZER_FILE_NAME,
        TOKENIZER_FILE_NAME,
        17_082_999,
        "3ffb37461c391f096759f4a9bbbc329da0f36952f88bab061fcf84940c022e98",
    ),
    file(
        TOKENIZER_CONFIG_FILE_NAME,
        TOKENIZER_CONFIG_FILE_NAME,
        1_340,
        "6f00514620aff01ba8b7291b2394e98daca5be264cb743805232d9ae27494b2a",
    ),
    file(
        SPECIAL_TOKENS_FILE_NAME,
        SPECIAL_TOKENS_FILE_NAME,
        SPECIAL_TOKENS_SIZE,
        SPECIAL_TOKENS_SHA256,
    ),
    file(
        CONFIG_FILE_NAME,
        CONFIG_FILE_NAME,
        1_578,
        "dfa5713436ecb4616eaa576795c8d3efd1f03122031a1ad4973d0b6b7e7edfd3",
    ),
];

const BGE_FILES: &[PinnedFile] = &[
    file(
        ONNX_FILE_NAME,
        ONNX_REMOTE_PATH,
        570_727_094,
        "912fc1215c2dbff6499700534bd8d31253af01573861abbfc43afd1fab6cce5d",
    ),
    file(
        TOKENIZER_FILE_NAME,
        TOKENIZER_FILE_NAME,
        17_082_900,
        "8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3",
    ),
    file(
        TOKENIZER_CONFIG_FILE_NAME,
        TOKENIZER_CONFIG_FILE_NAME,
        1_203,
        "b87c8703482b0300d3da30e201519aa641f6a450f5eb5bf1e624afbf70c74d80",
    ),
    file(
        SPECIAL_TOKENS_FILE_NAME,
        SPECIAL_TOKENS_FILE_NAME,
        SPECIAL_TOKENS_SIZE,
        SPECIAL_TOKENS_SHA256,
    ),
    file(
        CONFIG_FILE_NAME,
        CONFIG_FILE_NAME,
        848,
        "122e922dcfed6503c8721e6fe1daf090340c3d95ca7f3aa3a72730b321a51cfd",
    ),
];

/// The rerankers this module can load.
pub const RERANKER_MODELS: &[RerankerModelSpec] = &[
    RerankerModelSpec {
        id: GTE_RERANKER_ID,
        repo: "onnx-community/gte-multilingual-reranker-base",
        revision: "ee64367e35a2db0da46bb6497e13a18f8bd585cb",
        files: GTE_FILES,
    },
    RerankerModelSpec {
        id: BGE_RERANKER_V2_M3_ID,
        repo: "onnx-community/bge-reranker-v2-m3-ONNX",
        revision: "6f5ff65298512715a1e669753bc754d2bc8f367b",
        files: BGE_FILES,
    },
];

/// The spec for a config model id, `None` when this module does not load it.
pub fn reranker_model_spec(id: &str) -> Option<&'static RerankerModelSpec> {
    RERANKER_MODELS.iter().find(|spec| spec.id == id)
}

/// How a loaded model is sized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RerankerSettings {
    /// ONNX intra-op threads. 0 picks the performance cores.
    pub threads: usize,
    /// The longest input, in tokens, the model reads.
    pub max_tokens: usize,
}

/// What a session factory needs to build a scorer from verified files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionRequest {
    pub dir: PathBuf,
    pub threads: usize,
    pub max_tokens: usize,
}

fn rerank_error(message: String) -> ModelError {
    ModelError::Rerank(message)
}

/// The thread count to run with. A nonzero `configured` wins; otherwise the
/// performance cores, then the available parallelism. A reported count of 0 is
/// not a usable answer, so it falls through.
pub fn resolve_threads(
    configured: usize,
    performance_cores: impl Fn() -> Option<usize>,
    available_parallelism: impl Fn() -> usize,
) -> usize {
    if configured > 0 {
        return configured;
    }
    performance_cores()
        .filter(|n| *n > 0)
        .unwrap_or_else(available_parallelism)
}

fn real_threads(configured: usize) -> usize {
    resolve_threads(configured, crate::hardware::performance_cores, || {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    })
}

/// Refuses a model directory whose files are missing, not regular files, or
/// the wrong size. A symlink is refused so a pinned file cannot point outside
/// the directory.
fn check_sizes(dir: &Path, spec: &RerankerModelSpec) -> Result<()> {
    for pinned in spec.files {
        let path = dir.join(pinned.name);
        let metadata = fs::symlink_metadata(&path).map_err(|e| {
            let message = format!("{} of model {}: {e}", pinned.name, spec.id);
            if e.kind() == std::io::ErrorKind::NotFound {
                ModelError::RerankFilesNotFound(message)
            } else {
                rerank_error(message)
            }
        })?;
        if !metadata.is_file() {
            return Err(rerank_error(format!(
                "{} of model {} is not a regular file",
                pinned.name, spec.id
            )));
        }
        let size = metadata.len();
        if size != pinned.size {
            return Err(rerank_error(format!(
                "{} of model {} is {size} bytes, expected {}",
                pinned.name, spec.id, pinned.size
            )));
        }
    }
    Ok(())
}

/// The full check: every file's size, then its hash through `verify`, which
/// takes the path, the pinned size and the pinned sha256 (`verify_pinned_file`
/// in production). It is the entry point an installer calls once files are in
/// place. A file whose size is wrong never reaches `verify`.
///
/// # Errors
///
/// Fails on the first missing file, wrong size or failed `verify`.
pub fn verify_installed(
    dir: &Path,
    spec: &RerankerModelSpec,
    verify: impl Fn(&Path, u64, &str) -> Result<()>,
) -> Result<()> {
    check_sizes(dir, spec)?;
    for pinned in spec.files {
        verify(&dir.join(pinned.name), pinned.size, pinned.sha256)?;
    }
    Ok(())
}

fn read_file(dir: &Path, name: &str) -> Result<Vec<u8>> {
    fs::read(dir.join(name)).map_err(|e| rerank_error(format!("read {name}: {e}")))
}

/// Builds the fastembed ONNX session from the files in `request.dir`.
fn fastembed_session(request: SessionRequest) -> Result<fastembed::TextRerank> {
    let tokenizer_files = fastembed::TokenizerFiles {
        tokenizer_file: read_file(&request.dir, TOKENIZER_FILE_NAME)?,
        config_file: read_file(&request.dir, CONFIG_FILE_NAME)?,
        special_tokens_map_file: read_file(&request.dir, SPECIAL_TOKENS_FILE_NAME)?,
        tokenizer_config_file: read_file(&request.dir, TOKENIZER_CONFIG_FILE_NAME)?,
    };
    let model = fastembed::UserDefinedRerankingModel::new(
        request.dir.join(ONNX_FILE_NAME),
        tokenizer_files,
    );
    let options = fastembed::RerankInitOptionsUserDefined::new()
        .with_max_length(request.max_tokens)
        .with_intra_threads(request.threads);
    fastembed::TextRerank::try_new_from_user_defined(model, options)
        .map_err(|e| rerank_error(e.to_string()))
}

/// A pinned ONNX reranker loaded from a local model directory.
pub struct UserDefinedReranker {
    inner: Box<dyn Reranker>,
}

impl UserDefinedReranker {
    /// Loads the model in `dir`. Start-up checks that every file exists and has
    /// its pinned size; `verify_installed` performs the sha256 check, which
    /// start-up skips because hashing the model takes too long.
    ///
    /// # Errors
    ///
    /// Fails with `RerankFilesNotFound` when a file is absent, and with `Rerank`
    /// when a file is not a regular file or the wrong size, or the session
    /// cannot be built.
    pub fn load(dir: &Path, spec: &RerankerModelSpec, settings: RerankerSettings) -> Result<Self> {
        Self::load_with(dir, spec, settings, fastembed_session)
    }

    /// `load` with the session factory injected. It runs only after the size
    /// checks pass.
    pub(crate) fn load_with<S: DocScorer>(
        dir: &Path,
        spec: &RerankerModelSpec,
        settings: RerankerSettings,
        factory: impl FnOnce(SessionRequest) -> Result<S>,
    ) -> Result<Self> {
        check_sizes(dir, spec)?;
        let threads = real_threads(settings.threads);
        tracing::info!(model = spec.id, threads, "loading the reranker model");
        let scorer = factory(SessionRequest {
            dir: dir.to_path_buf(),
            threads,
            max_tokens: settings.max_tokens,
        })?;
        Ok(Self {
            inner: Box::new(ModelReranker::new(scorer)),
        })
    }
}

#[async_trait]
impl Reranker for UserDefinedReranker {
    async fn scores(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        self.inner.scores(query, docs).await
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    const SHA256_HEX_LEN: usize = 64;
    const MAX_TOKENS: usize = 512;
    const SETTINGS: RerankerSettings = RerankerSettings {
        threads: 3,
        max_tokens: MAX_TOKENS,
    };

    struct FakeScorer;

    impl DocScorer for FakeScorer {
        fn score(&mut self, _query: &str, docs: &[String]) -> Result<Vec<f32>> {
            Ok(docs.iter().map(|d| d.len() as f32).collect())
        }
    }

    fn gte() -> &'static RerankerModelSpec {
        reranker_model_spec(GTE_RERANKER_ID).unwrap()
    }

    /// Sparse files of exactly the pinned sizes, so no real data is written.
    fn model_dir(spec: &RerankerModelSpec) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for f in spec.files {
            let file = fs::File::create(dir.path().join(f.name)).unwrap();
            file.set_len(f.size).unwrap();
        }
        dir
    }

    fn resize(dir: &Path, name: &str, size: u64) {
        let file = fs::OpenOptions::new()
            .write(true)
            .open(dir.join(name))
            .unwrap();
        file.set_len(size).unwrap();
    }

    fn load_counting(dir: &Path) -> (Result<UserDefinedReranker>, usize) {
        let calls = Cell::new(0);
        let loaded = UserDefinedReranker::load_with(dir, gte(), SETTINGS, |_| {
            calls.set(calls.get() + 1);
            Ok(FakeScorer)
        });
        (loaded, calls.get())
    }

    fn rerank_message(result: Result<UserDefinedReranker>) -> String {
        match result {
            Err(ModelError::Rerank(message)) => message,
            Err(other) => panic!("expected a rerank error, got {other}"),
            Ok(_) => panic!("expected a refusal"),
        }
    }

    #[test]
    fn a_file_of_the_wrong_size_is_refused_before_the_factory_runs() {
        // Arrange
        let dir = model_dir(gte());
        resize(dir.path(), TOKENIZER_FILE_NAME, 10);

        // Act
        let (loaded, factory_calls) = load_counting(dir.path());

        // Assert
        assert!(rerank_message(loaded).contains(TOKENIZER_FILE_NAME));
        assert_eq!(factory_calls, 0);
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_to_a_correctly_sized_file_is_refused_before_the_factory_runs() {
        // Arrange
        let dir = model_dir(gte());
        let target = dir.path().join("elsewhere");
        fs::rename(dir.path().join(CONFIG_FILE_NAME), &target).unwrap();
        std::os::unix::fs::symlink(&target, dir.path().join(CONFIG_FILE_NAME)).unwrap();

        // Act
        let (loaded, factory_calls) = load_counting(dir.path());

        // Assert
        assert!(rerank_message(loaded).contains(CONFIG_FILE_NAME));
        assert_eq!(factory_calls, 0);
    }

    #[test]
    fn a_missing_file_is_refused_before_the_factory_runs() {
        // Arrange
        let dir = model_dir(gte());
        fs::remove_file(dir.path().join(CONFIG_FILE_NAME)).unwrap();

        // Act
        let (loaded, factory_calls) = load_counting(dir.path());

        // Assert
        assert!(matches!(
            loaded,
            Err(ModelError::RerankFilesNotFound(m)) if m.contains(CONFIG_FILE_NAME)
        ));
        assert_eq!(factory_calls, 0);
    }

    #[test]
    fn files_of_the_pinned_sizes_run_the_factory_once_without_hashing() {
        // Arrange
        let dir = model_dir(gte());

        // Act
        let (loaded, factory_calls) = load_counting(dir.path());

        // Assert
        assert!(loaded.is_ok());
        assert_eq!(factory_calls, 1);
    }

    #[test]
    fn the_factory_gets_the_directory_the_settings_and_the_resolved_threads() {
        // Arrange
        let dir = model_dir(gte());
        let seen = Cell::new(None);

        // Act
        UserDefinedReranker::load_with(dir.path(), gte(), SETTINGS, |request| {
            seen.set(Some((request.threads, request.max_tokens)));
            assert_eq!(request.dir, dir.path());
            Ok(FakeScorer)
        })
        .unwrap();

        // Assert
        assert_eq!(seen.get(), Some((3, MAX_TOKENS)));
    }

    #[test]
    fn a_truncated_file_with_the_right_prefix_is_refused_on_size_without_hashing() {
        // Arrange
        let dir = model_dir(gte());
        let onnx = &gte().files[0];
        resize(dir.path(), onnx.name, onnx.size - 1);
        let hash_calls = Cell::new(0);

        // Act
        let result = verify_installed(dir.path(), gte(), |_, _, _| {
            hash_calls.set(hash_calls.get() + 1);
            Ok(())
        });

        // Assert
        assert!(matches!(result, Err(ModelError::Rerank(m)) if m.contains(ONNX_FILE_NAME)));
        assert_eq!(hash_calls.get(), 0);
    }

    #[test]
    fn a_full_check_hashes_each_file_once() {
        // Arrange
        let dir = model_dir(gte());
        let hash_calls = Cell::new(0);

        // Act
        let result = verify_installed(dir.path(), gte(), |path, size, sha256| {
            hash_calls.set(hash_calls.get() + 1);
            assert!(path.starts_with(dir.path()));
            assert!(gte()
                .files
                .iter()
                .any(|f| f.size == size && f.sha256 == sha256));
            Ok(())
        });

        // Assert
        assert!(result.is_ok());
        assert_eq!(hash_calls.get(), gte().files.len());
    }

    #[test]
    fn a_failed_hash_fails_the_full_check() {
        // Arrange
        let dir = model_dir(gte());

        // Act
        let result = verify_installed(dir.path(), gte(), |_, _, _| {
            Err(ModelError::Rerank("bad hash".into()))
        });

        // Assert
        assert!(matches!(result, Err(ModelError::Rerank(m)) if m == "bad hash"));
    }

    #[tokio::test]
    async fn a_loaded_model_returns_one_score_per_document_in_order() {
        // Arrange
        let dir = model_dir(gte());
        let (loaded, _) = load_counting(dir.path());
        let docs = vec!["a".to_string(), "bbb".to_string(), "cc".to_string()];

        // Act
        let scores = loaded.unwrap().scores("q", &docs).await.unwrap();

        // Assert
        assert_eq!(scores, vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn threads_resolve_by_priority() {
        // Arrange
        const AVAILABLE: usize = 8;
        let cases: [(usize, Option<usize>, usize); 5] = [
            (0, Some(4), 4),
            (0, None, AVAILABLE),
            (3, Some(4), 3),
            (3, None, 3),
            (0, Some(0), AVAILABLE),
        ];

        for (configured, performance, expected) in cases {
            // Act
            let threads = resolve_threads(configured, || performance, || AVAILABLE);

            // Assert
            assert_eq!(
                threads, expected,
                "configured {configured}, cores {performance:?}"
            );
        }
    }

    /// Loads the real gte model and ranks three documents.
    ///
    /// Run with the model directory a downloader filled:
    /// `LIAM_RERANKER_MODEL_DIR=/path/to/gte cargo test -p liam-model -j 2 -- --ignored real_gte`
    #[tokio::test]
    #[ignore = "needs the model files in LIAM_RERANKER_MODEL_DIR"]
    async fn real_gte_model_ranks_the_relevant_document_first() {
        // Arrange
        let dir = std::env::var("LIAM_RERANKER_MODEL_DIR").expect("set LIAM_RERANKER_MODEL_DIR");
        let settings = RerankerSettings {
            threads: 0,
            max_tokens: MAX_TOKENS,
        };
        let reranker = UserDefinedReranker::load(Path::new(&dir), gte(), settings).unwrap();
        let docs = vec![
            "Bananas are rich in potassium.".to_string(),
            "The capital of France is Paris.".to_string(),
            "Rust has a borrow checker.".to_string(),
        ];

        // Act
        let scores = reranker
            .scores("What is the capital of France?", &docs)
            .await
            .unwrap();

        // Assert
        assert_eq!(scores.len(), docs.len());
        assert!(scores.iter().all(|s| s.is_finite()));
        let best = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(best, 1);
    }

    #[test]
    fn known_ids_resolve_and_an_unknown_id_does_not() {
        // Arrange
        let ids = [GTE_RERANKER_ID, BGE_RERANKER_V2_M3_ID];

        // Act
        let found: Vec<_> = ids.iter().map(|id| reranker_model_spec(id)).collect();
        let missing = reranker_model_spec("bge-reranker-base");

        // Assert
        for (id, spec) in ids.iter().zip(found) {
            assert_eq!(spec.unwrap().id, *id);
        }
        assert!(missing.is_none());
    }

    #[test]
    fn every_pinned_file_is_well_formed() {
        for spec in RERANKER_MODELS {
            // Arrange
            let names: Vec<_> = spec.files.iter().map(|f| f.name).collect();

            // Act and Assert
            assert!(names.contains(&ONNX_FILE_NAME), "{} has no model", spec.id);
            for f in spec.files {
                assert!(
                    !f.name.is_empty() && !f.name.contains('/') && !f.name.contains(".."),
                    "{} is not a plain file name",
                    f.name
                );
                assert!(f.size > 0, "{} has no size", f.name);
                assert_eq!(f.sha256.len(), SHA256_HEX_LEN, "{}", f.name);
                assert!(
                    f.sha256
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                    "{} hash is not lowercase hex",
                    f.name
                );
            }
        }
    }
}
