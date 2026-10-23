// SPDX-License-Identifier: Apache-2.0
//! The record of one consolidation run: what produced it and which steps have
//! finished, so a crash or a sleep resumes the run instead of restarting it.

use std::path::{Path, PathBuf};

use liam_log::wal::{write_atomically, OsFileOps};
use liam_store::Millis;
use serde::{Deserialize, Serialize};

use crate::embedder_fingerprint::EmbedderFingerprint;

/// Appended to the database path to name the manifest file.
const MANIFEST_SUFFIX: &str = ".consolidation.json";

/// Layout version of the manifest file, written as `version`. A file with any
/// other version is refused rather than guessed at.
const FORMAT_VERSION: u32 = 1;

/// How a finished run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunOutcome {
    Committed,
    RolledBack,
    RestoreFailed,
}

/// Everything needed to reproduce and resume a run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunManifest {
    pub run_id: String,
    pub seed: u64,
    pub embedder: EmbedderFingerprint,
    pub llm_model_fingerprint: String,
    pub created_at: Millis,
    pub snapshot_log_offset: Option<u64>,
    pub completed_steps: Vec<String>,
    pub outcome: Option<RunOutcome>,
}

/// The file's contents: the manifest under a layout version.
#[derive(Serialize, Deserialize)]
struct Sidecar {
    version: u32,
    #[serde(flatten)]
    manifest: RunManifest,
}

/// Where the manifest for the database at `database_path` lives.
pub fn manifest_path_for(database_path: &Path) -> PathBuf {
    let mut path = database_path.as_os_str().to_owned();
    path.push(MANIFEST_SUFFIX);
    PathBuf::from(path)
}

impl RunManifest {
    /// A manifest for a run that has not started any step.
    pub fn new(
        run_id: String,
        seed: u64,
        embedder: EmbedderFingerprint,
        llm_model_fingerprint: String,
        created_at: Millis,
    ) -> Self {
        Self {
            run_id,
            seed,
            embedder,
            llm_model_fingerprint,
            created_at,
            snapshot_log_offset: None,
            completed_steps: Vec::new(),
            outcome: None,
        }
    }

    /// Writes the manifest to `path`, replacing any earlier one.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let sidecar = Sidecar {
            version: FORMAT_VERSION,
            manifest: self.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&sidecar)?;
        write_atomically(path, &bytes, &mut OsFileOps).map_err(|source| {
            anyhow::anyhow!(
                "could not save the consolidation manifest at {}: {source}",
                path.display()
            )
        })
    }

    /// Reads the manifest at `path`, or `None` when no run has left one. A
    /// file that cannot be read or parsed is an error, never "absent":
    /// starting over would discard the checkpoints of an interrupted run.
    pub fn load(path: &Path) -> anyhow::Result<Option<Self>> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => anyhow::bail!(
                "the consolidation manifest {} could not be read: {source}",
                path.display()
            ),
        };
        let sidecar: Sidecar = serde_json::from_str(&text).map_err(|source| {
            anyhow::anyhow!(
                "the consolidation manifest {} could not be parsed: {source}",
                path.display()
            )
        })?;
        anyhow::ensure!(
            sidecar.version == FORMAT_VERSION,
            "the consolidation manifest {} has version {}, but this build reads version \
             {FORMAT_VERSION}",
            path.display(),
            sidecar.version
        );
        Ok(Some(sidecar.manifest))
    }

    /// Records `step` as finished and saves at once, so a crash loses at most
    /// the step in progress.
    pub fn mark_step_complete(&mut self, step: &str, path: &Path) -> anyhow::Result<()> {
        if !self.is_complete(step) {
            self.completed_steps.push(step.to_owned());
        }
        self.save(path)
    }

    /// The index in `ordered_steps` of the first step not yet completed.
    pub fn resume_point(&self, ordered_steps: &[&str]) -> usize {
        ordered_steps
            .iter()
            .position(|step| !self.is_complete(step))
            .unwrap_or(ordered_steps.len())
    }

    fn is_complete(&self, step: &str) -> bool {
        self.completed_steps.iter().any(|done| done == step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint() -> EmbedderFingerprint {
        EmbedderFingerprint {
            provider: "mock".into(),
            model: "mock-model".into(),
            dims: 8,
            prompt_scheme: 1,
        }
    }

    fn manifest() -> RunManifest {
        RunManifest::new(
            "run-1".into(),
            42,
            fingerprint(),
            "llm/model.gguf".into(),
            Millis(1_000),
        )
    }

    const STEPS: [&str; 5] = ["snapshot", "dedup", "maintenance", "hooks", "probes"];

    #[test]
    fn a_populated_manifest_round_trips_through_a_file() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = manifest_path_for(&dir.path().join("liam.db"));
        let mut original = manifest();
        original.snapshot_log_offset = Some(7);
        original.completed_steps = vec!["snapshot".into()];
        original.outcome = Some(RunOutcome::RolledBack);

        // Act
        original.save(&path).unwrap();
        let loaded = RunManifest::load(&path).unwrap();

        // Assert
        assert_eq!(loaded, Some(original));
    }

    #[test]
    fn a_new_manifest_starts_with_no_steps_and_no_outcome() {
        // Arrange / Act
        let fresh = manifest();

        // Assert
        assert!(fresh.completed_steps.is_empty());
        assert_eq!(fresh.outcome, None);
        assert_eq!(fresh.snapshot_log_offset, None);
    }

    #[test]
    fn loading_without_a_manifest_file_finds_no_run() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = manifest_path_for(&dir.path().join("liam.db"));

        // Act
        let loaded = RunManifest::load(&path).unwrap();

        // Assert
        assert_eq!(loaded, None);
    }

    #[test]
    fn a_corrupt_manifest_is_an_error_not_an_absent_run() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = manifest_path_for(&dir.path().join("liam.db"));
        std::fs::write(&path, b"{not json").unwrap();

        // Act
        let result = RunManifest::load(&path);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn a_manifest_with_an_unknown_layout_version_is_refused() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = manifest_path_for(&dir.path().join("liam.db"));
        manifest().save(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, text.replace("\"version\": 1", "\"version\": 99")).unwrap();

        // Act
        let result = RunManifest::load(&path);

        // Assert
        assert!(result.unwrap_err().to_string().contains("version 99"));
    }

    #[test]
    fn marking_a_step_complete_persists_it_immediately() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = manifest_path_for(&dir.path().join("liam.db"));
        let mut run = manifest();

        // Act
        run.mark_step_complete("snapshot", &path).unwrap();

        // Assert
        let on_disk = RunManifest::load(&path).unwrap().unwrap();
        assert_eq!(on_disk.completed_steps, vec!["snapshot".to_string()]);
    }

    #[test]
    fn marking_the_same_step_twice_records_it_once() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = manifest_path_for(&dir.path().join("liam.db"));
        let mut run = manifest();
        run.mark_step_complete("snapshot", &path).unwrap();

        // Act
        run.mark_step_complete("snapshot", &path).unwrap();

        // Assert
        assert_eq!(run.completed_steps, vec!["snapshot".to_string()]);
    }

    #[test]
    fn resuming_skips_completed_steps_and_starts_at_the_first_incomplete() {
        // Arrange
        let mut run = manifest();
        run.completed_steps = vec!["snapshot".into(), "dedup".into()];

        // Act
        let point = run.resume_point(&STEPS);

        // Assert
        assert_eq!(point, 2);
    }

    #[test]
    fn a_fresh_run_resumes_at_zero() {
        // Arrange
        let run = manifest();

        // Act
        let point = run.resume_point(&STEPS);

        // Assert
        assert_eq!(point, 0);
    }

    #[test]
    fn a_run_with_every_step_complete_resumes_past_the_end() {
        // Arrange
        let mut run = manifest();
        run.completed_steps = STEPS.iter().map(|s| s.to_string()).collect();

        // Act
        let point = run.resume_point(&STEPS);

        // Assert
        assert_eq!(point, STEPS.len());
    }
}
