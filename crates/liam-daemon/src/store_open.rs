// SPDX-License-Identifier: Apache-2.0
//! Opening the database and putting it on its event log, which `liamd`, a
//! rollback's reopen and `liam consolidate` all do the same way.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use liam_model::Embedder;
use liam_store::{DefaultGraph, GraphConfig, SharedLog};

use crate::config::Config;
use crate::consolidation::run::StoreReopener;
use crate::models::StoreEmbedder;

pub async fn open_database(config: &Config, database_path: &Path) -> anyhow::Result<DefaultGraph> {
    let path = database_path
        .to_str()
        .context("the database path is not UTF-8")?;
    let graph = GraphConfig::new(config.embedding_dims).with_read_pool_size(config.read_pool_size);
    Ok(DefaultGraph::open(path, graph).await?)
}

/// Attaches `log` to `store`, copies in the rows that predate it and replays
/// what the store has not applied.
pub async fn open_store_on_log(
    store: DefaultGraph,
    database_path: &Path,
    log_dir: &Path,
    log: SharedLog,
    embedder: Arc<dyn Embedder>,
) -> anyhow::Result<DefaultGraph> {
    // The embedder goes in before the replay so restored nodes get vectors.
    let store = store
        .with_log(log)
        .await
        .with_context(|| {
            format!(
                "the event log in {} cannot be used with the database {}; the cause is below",
                log_dir.display(),
                database_path.display()
            )
        })?
        .with_embedder(Arc::new(StoreEmbedder(embedder)));

    if store.needs_backfill().await? {
        let report = store.backfill_log_from_projection().await.context(
            "copying the existing rows into the event log failed; fix the cause and \
                 restart, the copy resumes where it stopped",
        )?;
        tracing::info!(?report, "backfill completed");
    }
    let report = store
        .catch_up()
        .await
        .context("replaying the event log into the database failed")?;
    tracing::info!(?report, "event log caught up");
    Ok(store)
}

/// Reopens the database on the log already open, after a rollback replaced its
/// file.
pub struct LogReopener {
    pub config: Config,
    pub database_path: PathBuf,
    pub log_dir: PathBuf,
    pub log: SharedLog,
    pub embedder: Arc<dyn Embedder>,
}

#[async_trait]
impl StoreReopener for LogReopener {
    async fn reopen(&self) -> anyhow::Result<DefaultGraph> {
        let store = open_database(&self.config, &self.database_path).await?;
        open_store_on_log(
            store,
            &self.database_path,
            &self.log_dir,
            self.log.clone(),
            self.embedder.clone(),
        )
        .await
    }
}
