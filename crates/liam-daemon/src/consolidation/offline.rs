// SPDX-License-Identifier: Apache-2.0
//! `liam consolidate`: one consolidation run over a store no daemon holds, and
//! a look at the last run's manifest. The run has no model behind its
//! maintenance, so resynthesis stays with the daemon's scheduled runs.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use liam_log::wal::MANIFEST_NAME;
use liam_model::Embedder;

use super::hooks::ConsolidationHookSet;
use super::maintenance::StoreMaintenance;
use super::manifest::{manifest_path_for, RunManifest, RunOutcome};
use super::run::{Consolidator, Parts, StoreProbes};
use super::settings::{identity, run_settings};
use super::shared_store::SharedStore;
use crate::config::Config;
use crate::embedder_fingerprint::{self, EmbedderFingerprint};
use crate::event_log;
use crate::models::{self, StoreEmbedder};
use crate::store_open::{open_database, open_store_on_log, LogReopener};
use crate::storelock::StoreLock;

/// Runs the job once over `database` and prints how it ended to `out`. The
/// caller turns a rollback or a failed restore into a failing exit code.
///
/// # Errors
///
/// Fails before any change when a daemon holds the store, the config file is
/// missing, the log is not there, or the embedder differs from the one the
/// store was built with. Fails after one when the run itself cannot go on; the
/// next call resumes it.
pub async fn run_offline(
    config: &Config,
    config_path: &Path,
    database: &Path,
    embedder: Arc<dyn Embedder>,
    out: &mut dyn Write,
) -> anyhow::Result<RunOutcome> {
    anyhow::ensure!(
        config_path.exists(),
        "the config file {} does not exist, so the built-in defaults would run the job with the \
         mock embedder; pass --config or set LIAM_CONFIG",
        config_path.display()
    );
    let log_dir = models::resolve_log_dir(config, database)?;
    anyhow::ensure!(
        event_log::holds_log(&log_dir),
        "{} is not an event log, it has no {MANIFEST_NAME} file; start liamd once to create it, \
         or set [log] dir",
        log_dir.display()
    );
    let _lock = StoreLock::acquire(database)
        .context("liam consolidate needs the store to itself: stop liamd, then rerun")?;
    let fingerprint =
        embedder_fingerprint::verify(database, &EmbedderFingerprint::from_config(config))?;

    let writer = event_log::open_writer(&log_dir, config.log.wal())?;
    let log = event_log::shared_log(writer, &log_dir, config.log.bloom()?, None)?;
    let store = open_database(config, database).await?;
    let store = open_store_on_log(store, database, &log_dir, log.clone(), embedder.clone()).await?;
    if let Some(fingerprint) = fingerprint {
        fingerprint.commit()?;
    }

    let parts = Parts {
        embedder: Arc::new(StoreEmbedder(embedder.clone())),
        hooks: ConsolidationHookSet::default(),
        maintenance: Arc::new(StoreMaintenance {
            policy: config.gc_policy(),
        }),
        reopener: Arc::new(LogReopener {
            config: config.clone(),
            database_path: database.to_path_buf(),
            log_dir,
            log,
            embedder,
        }),
        probes: Arc::new(StoreProbes),
    };
    let consolidator = Consolidator::new(
        database.to_path_buf(),
        run_settings(config, database)?,
        identity(config),
        parts,
    );
    let run = consolidator.consolidate(&SharedStore::new(store)).await?;
    writeln!(out, "{}", describe(&run))?;
    run.outcome
        .context("the run ended without recording an outcome")
}

/// Prints the last run recorded for `database`, finished or not. Reads one
/// file and takes no lock, so it works while the daemon runs.
///
/// # Errors
///
/// Fails when the manifest exists and cannot be read.
pub fn print_status(database: &Path, out: &mut dyn Write) -> anyhow::Result<()> {
    match RunManifest::load(&manifest_path_for(database))? {
        None => writeln!(out, "no consolidation run is recorded for this database")?,
        Some(run) => {
            writeln!(out, "{}", describe(&run))?;
            if run.outcome.is_none() {
                writeln!(
                    out,
                    "it will resume at its next step with `liam consolidate`"
                )?;
            }
        }
    }
    Ok(())
}

fn describe(run: &RunManifest) -> String {
    let state = match run.outcome {
        Some(RunOutcome::Committed) => "committed",
        Some(RunOutcome::RolledBack) => "rolled back, the store is as it was",
        Some(RunOutcome::RestoreFailed) => "restore failed, the store needs manual inspection",
        None => "unfinished",
    };
    format!(
        "consolidation run {}: {state}\nsteps done: {}",
        run.run_id,
        run.completed_steps.join(", ")
    )
}

#[cfg(test)]
mod tests;
