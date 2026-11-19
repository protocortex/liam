// SPDX-License-Identifier: Apache-2.0
//! Config path resolution and model construction, shared by `liamd` and `liam`.
//!
//! The daemon calls these at startup to build what it serves with. The CLI
//! calls the same functions from `liam fetch-models` to download and verify
//! those models ahead of time. Sharing them is the point: a fetch that
//! resolved a cache directory or a model id differently from the
//! daemon would report success and still leave the daemon unable to start.
//!
//! Every loader here is synchronous, so the CLI needs no async runtime to
//! drive them.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use liam_model::{ArcticEmbedder, Embedder, MockEmbedder, NoticeCaller, TextKind};
use liam_store::{ContentEmbedder, EmbedError};

use crate::config::{self, Config};

/// Expands `~` in a configured path. Nothing below the config layer
/// understands a home directory: the socket API would create a directory
/// literally named `~`, libSQL would create a database inside one, and the
/// model loaders would download gigabytes into one.
///
/// A tilde path with no `HOME` set is an error rather than a default,
/// because the silent version is worse than useless: an empty home turns
/// `~/.liam/liamd.sock` into `/.liam/liamd.sock`, at the filesystem root,
/// which fails later with a permission error naming a path the operator
/// never configured. Sparse environments are a real case here, since a
/// launchd job only has the variables its plist declares.
///
/// `key` names the config field so the error tells the operator which line
/// to fix rather than making them guess which path was at fault.
pub fn resolve_config_path(key: &str, value: &str) -> anyhow::Result<std::path::PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    Ok(std::path::PathBuf::from(resolve_path_with_home(
        key, value, &home,
    )?))
}

/// The same rule as `resolve_config_path`, yielding a `String` because the
/// model loaders take `&str` paths rather than `Path`, and taking `home` as an
/// argument so it is testable without mutating the process environment (which
/// would race the other tests in this binary).
pub fn resolve_path_with_home(key: &str, value: &str, home: &str) -> anyhow::Result<String> {
    if home.is_empty() && value.starts_with('~') {
        anyhow::bail!(
            "{key} is {value:?} but HOME is not set, so `~` cannot be expanded. \
             Set HOME, or write an absolute {key} in your config."
        );
    }
    Ok(config::expand_tilde(value, home))
}

/// The directory of the event log for the database at `database`: `[log] dir`
/// when set, else the default beside the database. The configured one has `~`
/// expanded and, when relative, is taken from the database's directory so the
/// log does not move with the working directory. `liamd` and `liam rebuild`
/// both go through this so they open the same log.
pub fn resolve_log_dir(config: &Config, database: &std::path::Path) -> anyhow::Result<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    resolve_log_dir_with_home(config, database, &home)
}

/// `resolve_log_dir` with `home` passed in, so it is testable without
/// mutating the process environment.
fn resolve_log_dir_with_home(
    config: &Config,
    database: &std::path::Path,
    home: &str,
) -> anyhow::Result<PathBuf> {
    match &config.log.dir {
        Some(dir) => {
            let dir = PathBuf::from(resolve_path_with_home("log.dir", dir, home)?);
            Ok(match database.parent() {
                Some(parent) if dir.is_relative() => parent.join(dir),
                _ => dir,
            })
        }
        None => Ok(config::default_log_dir(database)),
    }
}

/// The directory consolidation runs keep their snapshots in: `[consolidation]
/// snapshot_dir` when set, else the default beside the database. Resolved like
/// the log directory, so `liamd` and `liam consolidate` agree.
pub fn resolve_snapshot_dir(
    config: &Config,
    database: &std::path::Path,
) -> anyhow::Result<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    resolve_snapshot_dir_with_home(config, database, &home)
}

fn resolve_snapshot_dir_with_home(
    config: &Config,
    database: &std::path::Path,
    home: &str,
) -> anyhow::Result<PathBuf> {
    match &config.consolidation.snapshot_dir {
        Some(dir) => {
            let dir = PathBuf::from(resolve_path_with_home(
                "consolidation.snapshot_dir",
                dir,
                home,
            )?);
            Ok(match database.parent() {
                Some(parent) if dir.is_relative() => parent.join(dir),
                _ => dir,
            })
        }
        None => Ok(config::default_snapshot_dir(database)),
    }
}

/// Lets the store embed the content of nodes a replay restored, with the same
/// embedder the daemon serves with.
pub struct StoreEmbedder(pub Arc<dyn Embedder>);

#[async_trait]
impl ContentEmbedder for StoreEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(self.0.embed(text, TextKind::Document).await?)
    }
}

/// The `embedder.provider` value that selects the in-process models.
pub const LOCAL_PROVIDER: &str = "local";

/// The provider value that selects the mock embedder, for tests and dev.
pub const MOCK_PROVIDER: &str = "mock";

/// Choose the embedder from config. `mock` is for tests and dev; the `local`
/// provider loads the arctic ONNX model in-process. Any other name is an error,
/// so a typo cannot quietly select the mock.
pub fn build_embedder(config: &Config) -> anyhow::Result<Arc<dyn Embedder>> {
    build_embedder_for(config, NoticeCaller::Daemon)
}

/// `build_embedder`, where `caller` decides what the download notice says.
pub fn build_embedder_for(
    config: &Config,
    caller: NoticeCaller,
) -> anyhow::Result<Arc<dyn Embedder>> {
    if config.embedder.provider == LOCAL_PROVIDER {
        return build_local_embedder(config, caller);
    }
    if config.embedder.provider == MOCK_PROVIDER {
        return Ok(Arc::new(MockEmbedder::new(config.embedding_dims)));
    }
    anyhow::bail!(
        "embedder.provider = {:?} is not one of {LOCAL_PROVIDER:?}, {MOCK_PROVIDER:?}",
        config.embedder.provider
    )
}

fn build_local_embedder(
    config: &Config,
    caller: NoticeCaller,
) -> anyhow::Result<Arc<dyn Embedder>> {
    let home = std::env::var("HOME").unwrap_or_default();
    let cache_dir =
        resolve_path_with_home("embedder.cache_dir", &config.embedder.cache_dir, &home)?;
    Ok(Arc::new(ArcticEmbedder::load(
        config.embedding_dims,
        &cache_dir,
        caller,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// The arctic model's own output width.
    const ARCTIC_NATIVE_DIMS: usize = 768;

    /// The model cache dir was not being expanded, while `socket_path` and
    /// `database_path` always were. Left raw, hf-hub treats `~/.liam/models` as
    /// a relative path and downloads into a directory named `~`, under the
    /// launchd job's `WorkingDirectory`. The shipped mock defaults hid this
    /// because a mock embedder never opens the cache dir.
    #[test]
    fn a_tilde_model_cache_dir_expands_to_the_home_directory() {
        assert_eq!(
            resolve_path_with_home("embedder.cache_dir", "~/.liam/models", "/home/alice").unwrap(),
            "/home/alice/.liam/models"
        );
    }

    /// The local provider must reach the arctic loader: its width check fails
    /// before any download, so this runs with no network and leaves the cache
    /// empty.
    #[test]
    fn the_local_provider_loads_the_arctic_embedder() {
        // Arrange
        let cache = tempfile::tempdir().unwrap();
        let mut config = Config::default();
        config.embedder.provider = LOCAL_PROVIDER.to_string();
        config.embedder.cache_dir = cache.path().to_string_lossy().into_owned();
        config.embedding_dims = ARCTIC_NATIVE_DIMS + 1;

        // Act
        let error = match build_embedder(&config) {
            Ok(_) => panic!("a width beyond the model's own must be refused"),
            Err(error) => error.to_string(),
        };

        // Assert
        assert!(error.contains(liam_model::EMBEDDER_MODEL_ID), "{error}");
        assert_eq!(std::fs::read_dir(cache.path()).unwrap().count(), 0);
    }

    /// An absolute cache dir is already correct and must be passed through
    /// untouched, so an operator can point at a shared model directory.
    #[test]
    fn an_absolute_model_cache_dir_is_left_alone() {
        assert_eq!(
            resolve_path_with_home("embedder.cache_dir", "/opt/liam/models", "/home/alice")
                .unwrap(),
            "/opt/liam/models"
        );
    }

    /// A launchd job only gets the variables its plist declares, so an empty
    /// HOME is a real case here. Failing names the offending config field,
    /// because the silent version resolves to `/.liam/models` at the
    /// filesystem root and fails later with a path the operator never wrote.
    #[test]
    fn a_tilde_model_cache_dir_without_home_is_an_error_naming_the_field() {
        let error = resolve_path_with_home("embedder.cache_dir", "~/.liam/models", "")
            .expect_err("a tilde path with no HOME must not be silently resolved");
        let message = error.to_string();
        assert!(
            message.contains("embedder.cache_dir"),
            "error should name the config field: {message}"
        );
        assert!(
            message.contains("HOME"),
            "error should say what is missing: {message}"
        );
    }

    struct FailingEmbedder;

    #[async_trait]
    impl Embedder for FailingEmbedder {
        fn dims(&self) -> usize {
            8
        }

        async fn embed(&self, _text: &str, _kind: TextKind) -> liam_model::error::Result<Vec<f32>> {
            Err(liam_model::error::ModelError::Embed("no model".into()))
        }
    }

    #[tokio::test]
    async fn the_store_embedder_returns_the_vector_its_embedder_made() {
        // Arrange
        let embedder = Arc::new(MockEmbedder::new(8));
        let expected = embedder.embed("alpha", TextKind::Document).await.unwrap();

        // Act
        let vector = StoreEmbedder(embedder).embed("alpha").await.unwrap();

        // Assert
        assert_eq!(vector, expected);
        assert_eq!(vector.len(), 8);
    }

    #[tokio::test]
    async fn the_store_embedder_embeds_as_a_document() {
        // Arrange
        struct KindRecorder(std::sync::Mutex<Vec<TextKind>>);

        #[async_trait]
        impl Embedder for KindRecorder {
            fn dims(&self) -> usize {
                8
            }

            async fn embed(
                &self,
                _text: &str,
                kind: TextKind,
            ) -> liam_model::error::Result<Vec<f32>> {
                self.0.lock().unwrap().push(kind);
                Ok(vec![0.0; 8])
            }
        }
        let recorder = Arc::new(KindRecorder(std::sync::Mutex::new(Vec::new())));

        // Act
        StoreEmbedder(recorder.clone())
            .embed("alpha")
            .await
            .unwrap();

        // Assert
        assert_eq!(*recorder.0.lock().unwrap(), vec![TextKind::Document]);
    }

    #[tokio::test]
    async fn the_store_embedder_passes_an_embed_failure_on_as_an_error() {
        // Act
        let error = StoreEmbedder(Arc::new(FailingEmbedder))
            .embed("alpha")
            .await
            .expect_err("a failing embedder must not yield a vector");

        // Assert
        assert!(error.to_string().contains("no model"), "{error}");
    }

    #[test]
    fn the_default_config_builds_only_an_embedder() {
        // Arrange
        let config = Config::default();

        // Act
        let embedder: Arc<dyn Embedder> = build_embedder(&config).unwrap();

        // Assert
        assert_eq!(embedder.dims(), config.embedding_dims);
    }

    #[test]
    fn an_unknown_embedder_provider_is_an_error_not_a_mock() {
        let mut config = Config::default();
        config.embedder.provider = "loca".to_string();

        let message = build_embedder(&config).err().unwrap().to_string();

        assert!(
            message.contains("loca") && message.contains("local"),
            "{message}"
        );
    }

    fn config_with_snapshot_dir(dir: Option<&str>) -> Config {
        let mut config = Config::default();
        config.consolidation.snapshot_dir = dir.map(str::to_string);
        config
    }

    #[test]
    fn an_unset_snapshot_dir_is_the_default_beside_the_database() {
        let config = config_with_snapshot_dir(None);

        let dir =
            resolve_snapshot_dir_with_home(&config, Path::new("/data/work.db"), "/h").unwrap();

        assert_eq!(dir, PathBuf::from("/data/work.snapshots"));
    }

    #[test]
    fn a_snapshot_dir_expands_the_tilde_and_resolves_a_relative_path_from_the_database() {
        for (configured, expected) in [
            ("~/snaps", "/home/alice/snaps"),
            ("snaps", "/data/snaps"),
            ("/var/snaps", "/var/snaps"),
        ] {
            let config = config_with_snapshot_dir(Some(configured));

            let dir =
                resolve_snapshot_dir_with_home(&config, Path::new("/data/work.db"), "/home/alice")
                    .unwrap();

            assert_eq!(dir, PathBuf::from(expected), "configured {configured}");
        }
    }

    // ---- the log directory ----

    fn config_with_log_dir(log_dir: Option<&str>) -> Config {
        let mut config = Config::default();
        config.log.dir = log_dir.map(str::to_string);
        config
    }

    #[test]
    fn without_a_configured_dir_the_log_sits_beside_the_database() {
        // Arrange
        let config = config_with_log_dir(None);

        // Act
        let dir =
            resolve_log_dir_with_home(&config, Path::new("/data/work.db"), "/home/alice").unwrap();

        // Assert
        assert_eq!(dir, PathBuf::from("/data/work.log"));
    }

    #[test]
    fn a_configured_dir_beats_the_default() {
        // Arrange
        let config = config_with_log_dir(Some("/var/liam/events"));

        // Act
        let dir =
            resolve_log_dir_with_home(&config, Path::new("/data/work.db"), "/home/alice").unwrap();

        // Assert
        assert_eq!(dir, PathBuf::from("/var/liam/events"));
    }

    #[test]
    fn a_tilde_in_the_configured_dir_expands_to_the_home_directory() {
        // Arrange
        let config = config_with_log_dir(Some("~/events"));

        // Act
        let dir =
            resolve_log_dir_with_home(&config, Path::new("/data/work.db"), "/home/alice").unwrap();

        // Assert
        assert_eq!(dir, PathBuf::from("/home/alice/events"));
    }

    #[test]
    fn a_tilde_dir_without_home_is_an_error_naming_the_key() {
        // Arrange
        let config = config_with_log_dir(Some("~/events"));

        // Act
        let error = resolve_log_dir_with_home(&config, Path::new("/data/work.db"), "")
            .expect_err("a tilde dir with no HOME must not be silently resolved");

        // Assert
        assert!(error.to_string().contains("log.dir"), "{error}");
    }

    #[test]
    fn a_relative_dir_is_taken_from_the_database_directory() {
        // Arrange
        let config = config_with_log_dir(Some("events"));

        // Act
        let dir =
            resolve_log_dir_with_home(&config, Path::new("/data/work.db"), "/home/alice").unwrap();

        // Assert
        assert_eq!(dir, PathBuf::from("/data/events"));
    }

    const BATCH_TEXT_COUNT: usize = 20;
    const EMBED_EPSILON: f32 = 1e-4;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Loads the real model through `build_embedder`. It reads the pinned files
    /// from the directory `LIAM_MODEL_CACHE` names and downloads the missing
    /// ones (311 MB). Run with
    /// `LIAM_MODEL_CACHE=/path/to/cache cargo test -p liam-daemon --lib -- --ignored arctic_ranks`.
    #[tokio::test]
    #[ignore = "needs the arctic weights (311 MB) in LIAM_MODEL_CACHE or a download"]
    async fn arctic_ranks_the_relevant_document_first() {
        // Arrange
        let mut config = Config::default();
        config.embedder.provider = LOCAL_PROVIDER.to_string();
        config.embedder.cache_dir =
            std::env::var("LIAM_MODEL_CACHE").expect("set LIAM_MODEL_CACHE");
        let embedder = build_embedder(&config).expect("load arctic");
        let documents = [
            "The mitochondria is the powerhouse of the cell".to_string(),
            "I live in Lisbon, Portugal".to_string(),
            "The stock market closed higher on Tuesday".to_string(),
        ];
        let many: Vec<String> = (0..BATCH_TEXT_COUNT)
            .map(|i| format!("{} {i}", documents[i % documents.len()]))
            .collect();

        // Act
        let query = embedder
            .embed("where do I live", TextKind::Query)
            .await
            .expect("embed query");
        let vectors = embedder
            .embed_batch(&documents, TextKind::Document)
            .await
            .expect("embed documents");
        let many_vectors = embedder
            .embed_batch(&many, TextKind::Document)
            .await
            .expect("embed more documents than one call");
        let alone = embedder
            .embed(&many[0], TextKind::Document)
            .await
            .expect("embed one document");

        // Assert
        let scores: Vec<f32> = vectors.iter().map(|v| cosine(&query, v)).collect();
        assert_eq!(query.len(), config.embedding_dims);
        assert!(scores[1] > scores[0] && scores[1] > scores[2], "{scores:?}");
        assert_eq!(many_vectors.len(), BATCH_TEXT_COUNT);
        let drift = many_vectors[0]
            .iter()
            .zip(&alone)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(drift < EMBED_EPSILON, "{drift}");
    }
}
