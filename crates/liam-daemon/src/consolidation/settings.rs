// SPDX-License-Identifier: Apache-2.0
//! The run settings both binaries build from `liam.toml`.

use std::path::Path;

use super::run::{Identity, RunSettings};
use crate::config::Config;
use crate::embedder_fingerprint::EmbedderFingerprint;
use crate::models::resolve_snapshot_dir;

/// The settings of a run over the database at `database`.
///
/// # Errors
///
/// Fails when the snapshot directory has a `~` and `HOME` is not set.
pub fn run_settings(config: &Config, database: &Path) -> anyhow::Result<RunSettings> {
    let section = &config.consolidation;
    Ok(RunSettings {
        probe_recall_margin: section.probe_recall_margin,
        dedup_similarity_threshold: section.dedup_similarity_threshold,
        snapshot_dir: resolve_snapshot_dir(config, database)?,
        probe_count: section.probe_count,
    })
}

/// The models a run is made under, as `liam.toml` configures them.
pub fn identity(config: &Config) -> Identity {
    Identity {
        embedder: EmbedderFingerprint::from_config(config),
        llm_model: format!("{}/{}", config.llm.model, config.llm.gguf_file),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_identity_names_the_configured_embedder_and_llm() {
        // Arrange
        let config = Config::default();

        // Act
        let identity = identity(&config);

        // Assert
        assert_eq!(identity.embedder, EmbedderFingerprint::from_config(&config));
        assert_eq!(
            identity.llm_model,
            format!("{}/{}", config.llm.model, config.llm.gguf_file)
        );
    }

    #[test]
    fn the_settings_carry_the_configured_values_and_the_default_directory() {
        // Arrange
        let mut config = Config::default();
        config.consolidation.probe_count = 7;
        config.consolidation.probe_recall_margin = 0.1;

        // Act
        let settings = run_settings(&config, Path::new("/data/work.db")).unwrap();

        // Assert
        assert_eq!(settings.probe_count, 7);
        assert_eq!(settings.probe_recall_margin, 0.1);
        assert_eq!(settings.dedup_similarity_threshold, 0.95);
        assert_eq!(
            settings.snapshot_dir,
            Path::new("/data/work.snapshots").to_path_buf()
        );
    }
}
