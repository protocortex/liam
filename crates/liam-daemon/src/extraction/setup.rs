// SPDX-License-Identifier: Apache-2.0
//! Builds the extraction part the daemon runs, from its configuration and the
//! files beside the database.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use liam_log::reader::SequentialScanReader;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use super::cache::{cache_path, CacheStore};
use super::hooks::ExtractionHooks;
use super::identity::{GgufFile, IdentitySource};
use super::provider::{pidfile_path, SidecarProvider, SystemStarter};
use super::sidecar::{SidecarGuard, SystemRam};
use crate::config::ExtractionConfig;

/// What the daemon knows that the configuration does not say.
pub struct DaemonFiles<'a> {
    pub database_path: &'a Path,
    pub log_dir: &'a Path,
    /// The daemon's generation permits, shared with the other model users.
    pub permits: Arc<Semaphore>,
    pub stopping: CancellationToken,
}

/// Clears a model server an earlier daemon left behind. A killed daemon cannot
/// stop its child, so this runs once at startup, whether or not extraction is
/// on. Without `llama_server_path` there is no executable to check a process
/// against, so a pidfile is left alone.
pub fn sweep_stale_sidecar(config: &ExtractionConfig, database_path: &Path) {
    let pidfile = pidfile_path(database_path);
    if config.llama_server_path.is_none() && pidfile.exists() {
        warn!("extraction: a model server may be left running, but extraction.llama_server_path is not set to find it");
    }
    let outcome = SidecarGuard::sweep_stale(config, &pidfile);
    debug!(
        ?outcome,
        "extraction: swept for a model server left by an earlier daemon"
    );
}

/// The extraction part for `config`. The model file is not read here: the
/// first model pass hashes it, so a start with extraction on stays quick.
///
/// # Errors
///
/// Fails when the response cache or the event log cannot be opened.
pub fn build(config: &ExtractionConfig, files: DaemonFiles<'_>) -> anyhow::Result<ExtractionHooks> {
    let identity: Arc<dyn IdentitySource> = Arc::new(GgufFile::new(
        config.model_gguf.as_deref().map(PathBuf::from),
    ));
    let starter = SystemStarter {
        pidfile: pidfile_path(files.database_path),
    };
    let cache = CacheStore::open(cache_path(files.database_path))
        .context("could not open the extraction response cache")?;
    let reader = SequentialScanReader::local(files.log_dir)
        .context("could not open the event log for extraction")?;
    Ok(ExtractionHooks {
        config: config.clone(),
        reader: Arc::new(reader),
        provider: Arc::new(SidecarProvider::new(Arc::new(starter), identity.clone())),
        identity,
        cache: Arc::new(cache),
        permits: files.permits,
        free_ram: Arc::new(SystemRam),
        stopping: files.stopping,
    })
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::extraction::identity::hash_file_for;

    fn built(config: &ExtractionConfig, dir: &TempDir) -> ExtractionHooks {
        let files = DaemonFiles {
            database_path: &dir.path().join("liam.db"),
            log_dir: dir.path(),
            permits: Arc::new(Semaphore::new(1)),
            stopping: CancellationToken::new(),
        };
        build(config, files).unwrap()
    }

    fn config_with_model(path: &Path) -> ExtractionConfig {
        ExtractionConfig {
            model_gguf: Some(path.to_string_lossy().into_owned()),
            ..ExtractionConfig::default()
        }
    }

    #[tokio::test]
    async fn the_hooks_name_the_configured_model_file_when_asked() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let model = dir.path().join("model.gguf");
        std::fs::write(&model, b"weights").unwrap();
        let hooks = built(&config_with_model(&model), &dir);

        // Act
        let identity = hooks.identity.identity().await.unwrap();

        // Assert
        assert_eq!(identity.model_id, "model.gguf");
    }

    #[tokio::test]
    async fn an_unreadable_model_fails_when_asked_not_when_built() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let config = config_with_model(&dir.path().join("absent.gguf"));

        // Act
        let hooks = built(&config, &dir);

        // Assert
        assert!(hooks.identity.identity().await.is_err());
    }

    #[test]
    fn building_the_hooks_does_not_read_the_model() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let model = dir.path().join("model.gguf");
        std::fs::write(&model, b"weights").unwrap();

        // Act
        built(&config_with_model(&model), &dir);

        // Assert
        assert!(!hash_file_for(&model).exists());
    }

    #[test]
    fn a_leftover_pidfile_is_cleared_only_when_the_server_path_is_known() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let database = dir.path().join("liam.db");
        let pidfile = pidfile_path(&database);
        let unset = ExtractionConfig::default();
        let set = ExtractionConfig {
            llama_server_path: Some("/nonexistent/llama-server".into()),
            ..ExtractionConfig::default()
        };
        std::fs::write(&pidfile, "not a record").unwrap();

        // Act
        sweep_stale_sidecar(&unset, &database);
        let kept = pidfile.exists();
        sweep_stale_sidecar(&set, &database);

        // Assert
        assert!(kept);
        assert!(!pidfile.exists());
    }
}
