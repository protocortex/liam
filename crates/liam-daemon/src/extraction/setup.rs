// SPDX-License-Identifier: Apache-2.0
//! Builds the extraction part the daemon runs, from its configuration and the
//! files beside the database.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use liam_log::reader::SequentialScanReader;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

use super::cache::{cache_path, CacheStore};
use super::client::ModelIdentity;
use super::hooks::ExtractionHooks;
use super::identity::model_identity;
use super::provider::{pidfile_path, SidecarProvider, SystemStarter};
use super::sidecar::stale::SweepOutcome;
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
/// stop its child, so this runs once at startup, whether or not extraction is on.
pub fn sweep_stale_sidecar(config: &ExtractionConfig, database_path: &Path) {
    let outcome = SidecarGuard::sweep_stale(config, &pidfile_path(database_path));
    if outcome != SweepOutcome::NoPidfile {
        info!(
            ?outcome,
            "extraction: swept a model server left by an earlier daemon"
        );
    }
}

/// The extraction part for `config`. Reading the model file for its hash can
/// take seconds the first time, so it runs off the async threads.
///
/// # Errors
///
/// Fails when the response cache or the event log cannot be opened.
pub async fn build(
    config: &ExtractionConfig,
    files: DaemonFiles<'_>,
) -> anyhow::Result<ExtractionHooks> {
    let identity = if config.llm_enabled {
        read_identity(config).await
    } else {
        None
    };
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

/// The identity of the configured model file, or `None` with the reason
/// logged, which fails the model pass of each run until the file can be read.
async fn read_identity(config: &ExtractionConfig) -> Option<ModelIdentity> {
    let Some(gguf) = config.model_gguf.as_deref().map(PathBuf::from) else {
        error!("extraction: model_gguf is not set, so the model pass cannot run");
        return None;
    };
    match tokio::task::spawn_blocking(move || model_identity(&gguf)).await {
        Ok(Ok(identity)) => Some(identity),
        Ok(Err(source)) => {
            error!(kind = ?source.kind(), "extraction: could not read the model file, so the model pass cannot run");
            None
        }
        Err(source) => {
            error!(%source, "extraction: reading the model file did not finish");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    async fn built(config: &ExtractionConfig, dir: &TempDir) -> ExtractionHooks {
        let files = DaemonFiles {
            database_path: &dir.path().join("liam.db"),
            log_dir: dir.path(),
            permits: Arc::new(Semaphore::new(1)),
            stopping: CancellationToken::new(),
        };
        build(config, files).await.unwrap()
    }

    fn config_with_model(path: &Path) -> ExtractionConfig {
        ExtractionConfig {
            model_gguf: Some(path.to_string_lossy().into_owned()),
            ..ExtractionConfig::default()
        }
    }

    #[tokio::test]
    async fn a_readable_model_gives_the_hooks_its_identity() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let model = dir.path().join("model.gguf");
        std::fs::write(&model, b"weights").unwrap();

        // Act
        let hooks = built(&config_with_model(&model), &dir).await;

        // Assert
        let identity = hooks.identity.expect("the model was read");
        assert_eq!(identity.model_id, "model.gguf");
    }

    #[tokio::test]
    async fn an_unreadable_model_leaves_the_identity_unset_and_the_build_succeeds() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let config = config_with_model(&dir.path().join("absent.gguf"));

        // Act
        let hooks = built(&config, &dir).await;

        // Assert
        assert_eq!(hooks.identity, None);
    }

    #[tokio::test]
    async fn rules_only_extraction_reads_no_model() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let model = dir.path().join("model.gguf");
        std::fs::write(&model, b"weights").unwrap();
        let config = ExtractionConfig {
            llm_enabled: false,
            ..config_with_model(&model)
        };

        // Act
        let hooks = built(&config, &dir).await;

        // Assert
        assert_eq!(hooks.identity, None);
    }
}
