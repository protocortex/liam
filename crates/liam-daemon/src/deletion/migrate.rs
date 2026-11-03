// SPDX-License-Identifier: Apache-2.0
//! Seals a log written before encryption was turned on.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use liam_daemon::event_log;
use liam_log::envelope::Sealer;
use liam_log::migrate::{seal_log, MigrateReport, NoFault, NodeKeyHints};
use liam_store::{DefaultGraph, GraphConfig};

use crate::config::Config;

/// Seals every plaintext event of the log in `log_dir`. The caller holds the
/// store lock.
pub async fn encrypt_log(
    config: &Config,
    database_path: &Path,
    log_dir: &Path,
) -> anyhow::Result<MigrateReport> {
    anyhow::ensure!(
        config.encryption.enabled(),
        "encrypt-log needs [encryption] master_key in the configuration"
    );
    let registry = event_log::open_key_registry(log_dir, &config.encryption)?
        .context("the key registry could not be opened")?;
    let lineage_nodes = lineage_nodes(config, database_path).await?;
    let log_dir = log_dir.to_path_buf();
    let sealer = Sealer::new(Arc::clone(&registry));
    tokio::task::spawn_blocking(move || {
        seal_log(
            &log_dir,
            &sealer,
            &registry,
            &mut NodeKeyHints::new(lineage_nodes),
            &NoFault,
        )
    })
    .await
    .context("the migration task failed")?
    .context("the log could not be sealed")
}

/// A node of every lineage the store knows, so tier, usage and feedback entries
/// are sealed under their subject's key. An absent store has none.
async fn lineage_nodes(
    config: &Config,
    database_path: &Path,
) -> anyhow::Result<HashMap<String, String>> {
    if !database_path.exists() {
        return Ok(HashMap::new());
    }
    let path = database_path
        .to_str()
        .context("the database path is not valid text")?;
    let store = DefaultGraph::open(path, GraphConfig::new(config.embedding_dims))
        .await
        .context("could not read the lineages from the database")?;
    let nodes = store.one_node_per_lineage().await?;
    Ok(nodes
        .into_iter()
        .map(|(lineage, node)| (lineage.as_str().to_owned(), node.as_str().to_owned()))
        .collect())
}
