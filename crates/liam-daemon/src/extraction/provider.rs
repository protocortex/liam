// SPDX-License-Identifier: Apache-2.0
//! The model provider that runs the local `llama-server`: it starts the
//! sidecar, points a client at it, and hands the guard to the session, so the
//! server stops when the run drops the session.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;

use super::client::{ModelIdentity, SidecarClient};
use super::extractor::{ExtractorProvider, ExtractorSession};
use super::sidecar::{SidecarError, SidecarGuard};
use crate::config::ExtractionConfig;

/// Appended to the database path to name the file that records the running
/// sidecar, so a later daemon can clear one a crash left behind.
const PIDFILE_SUFFIX: &str = ".sidecar.pid";

pub fn pidfile_path(database_path: &Path) -> PathBuf {
    let mut name = database_path.as_os_str().to_owned();
    name.push(PIDFILE_SUFFIX);
    PathBuf::from(name)
}

/// Starts a ready sidecar. Tests start a fake server in place of the real one.
#[async_trait]
pub trait SidecarStarter: Send + Sync {
    async fn start(&self, config: &ExtractionConfig) -> Result<SidecarGuard, SidecarError>;
}

/// Starts the `llama-server` the configuration names.
pub struct SystemStarter {
    pub pidfile: PathBuf,
}

#[async_trait]
impl SidecarStarter for SystemStarter {
    async fn start(&self, config: &ExtractionConfig) -> Result<SidecarGuard, SidecarError> {
        SidecarGuard::start(config, &self.pidfile).await
    }
}

pub struct SidecarProvider {
    starter: Arc<dyn SidecarStarter>,
    /// The model the server is started on, which the client reports. `None`
    /// when the model file could not be read, so no server is started.
    identity: Option<ModelIdentity>,
}

impl SidecarProvider {
    pub fn new(starter: Arc<dyn SidecarStarter>, identity: Option<ModelIdentity>) -> Self {
        Self { starter, identity }
    }
}

#[async_trait]
impl ExtractorProvider for SidecarProvider {
    async fn open(&self, config: &ExtractionConfig) -> anyhow::Result<ExtractorSession> {
        let identity = self
            .identity
            .clone()
            .context("the model file could not be read, so the server is not started")?;
        let guard = self.starter.start(config).await?;
        let client = SidecarClient::new(
            guard.addr(),
            guard.api_key().to_owned(),
            Duration::from_secs(config.request_timeout_secs),
            identity,
        )?;
        let alive = guard.alive_handle();
        Ok(ExtractorSession {
            extractor: Arc::new(client),
            alive: Box::new(move || alive.is_alive()),
            _keepalive: Some(Box::new(guard)),
        })
    }
}
