// SPDX-License-Identifier: Apache-2.0
//! What a delete would destroy, asked of the store a daemon is serving. It
//! changes nothing, and it gives back its handle on the store before it ends,
//! because a rollback needs the slot's handle to be the last one.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use liam_daemon::consolidation::shared_store::SharedStore;
use liam_daemon::models::resolve_snapshot_dir;
use liam_log::crypto::KeyRegistry;
use liam_log::reader::LogReader;

use super::memory::NodeState;
use super::registry::LineageRegistry;
use super::service::{preview, DeleteError, Preview};
use super::target::DeleteTarget;
use crate::config::Config;

pub struct DeletePreview {
    database_path: PathBuf,
    snapshot_dir: PathBuf,
    /// A reader that opens sealed records.
    reader: Arc<dyn LogReader>,
    keys: Arc<KeyRegistry>,
    states: Vec<Arc<dyn NodeState>>,
}

impl DeletePreview {
    pub fn new(
        config: &Config,
        database_path: &Path,
        reader: Arc<dyn LogReader>,
        keys: Arc<KeyRegistry>,
        states: Vec<Arc<dyn NodeState>>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            database_path: database_path.to_path_buf(),
            snapshot_dir: resolve_snapshot_dir(config, database_path)?,
            reader,
            keys,
            states,
        })
    }

    /// What deleting `target` would destroy, from the log and every store.
    pub async fn read(
        &self,
        shared: &SharedStore,
        target: &DeleteTarget,
    ) -> Result<Preview, DeleteError> {
        // Declared in this order so the registry and the clone are gone before
        // the guard that holds a rollback back is released.
        let slot = shared.get_handle().await?;
        let store = Arc::clone(&slot);
        let registry = LineageRegistry::for_database(
            &self.database_path,
            &self.snapshot_dir,
            Arc::clone(&store),
            self.states.clone(),
        )?;
        preview(&store, self.reader.as_ref(), &self.keys, &registry, target).await
    }
}
