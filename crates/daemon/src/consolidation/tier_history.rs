// SPDX-License-Identifier: Apache-2.0
//! What rolled back runs leave for the tier step of later ones: how many in a
//! row moved items, and which lineages they moved. It lives beside the
//! database rather than in a run's manifest, because a rollback does not
//! restore it and the next run starts from a new manifest.

use std::path::{Path, PathBuf};

use liam_log::wal::{write_atomically, OsFileOps};
use serde::{Deserialize, Serialize};

/// Appended to the database path to name the history file.
const HISTORY_SUFFIX: &str = ".tier-history.json";

/// Layout version of the file. Any other is refused.
const FORMAT_VERSION: u32 = 1;

/// A lineage left alone for some more runs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cooldown {
    pub lineage_key: String,
    pub runs_left: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TierHistory {
    /// Runs in a row, up to the latest, that moved items and were rolled back.
    pub consecutive_rollbacks: u32,
    pub cooldown: Vec<Cooldown>,
}

#[derive(Serialize, Deserialize)]
struct Sidecar {
    version: u32,
    #[serde(flatten)]
    history: TierHistory,
}

impl TierHistory {
    /// A run that moved `moved` was rolled back. A run that moved nothing has
    /// nothing to blame and changes no history.
    pub fn after_rollback(&mut self, moved: &[String], cooldown_runs: u32) {
        if moved.is_empty() {
            return;
        }
        self.consecutive_rollbacks += 1;
        for lineage_key in moved {
            self.cooldown
                .retain(|entry| &entry.lineage_key != lineage_key);
            self.cooldown.push(Cooldown {
                lineage_key: lineage_key.clone(),
                runs_left: cooldown_runs,
            });
        }
        self.cooldown.retain(|entry| entry.runs_left > 0);
    }

    /// A run that ran the tier step was committed: every cooldown counts one
    /// run down, and a run that moved items ends the streak of rollbacks.
    pub fn after_commit(&mut self, moved_any: bool) {
        for entry in &mut self.cooldown {
            entry.runs_left -= 1;
        }
        self.cooldown.retain(|entry| entry.runs_left > 0);
        if moved_any {
            self.consecutive_rollbacks = 0;
        }
    }
}

/// Where the history for the database at `database_path` lives.
pub fn history_path_for(database_path: &Path) -> PathBuf {
    let mut path = database_path.as_os_str().to_owned();
    path.push(HISTORY_SUFFIX);
    PathBuf::from(path)
}

/// Writes `history` to `path`, replacing any earlier one.
pub fn save(path: &Path, history: &TierHistory) -> anyhow::Result<()> {
    let sidecar = Sidecar {
        version: FORMAT_VERSION,
        history: history.clone(),
    };
    let bytes = serde_json::to_vec(&sidecar)?;
    write_atomically(path, &bytes, &mut OsFileOps).map_err(|source| {
        anyhow::anyhow!(
            "could not save the tier history at {}: {source}",
            path.display()
        )
    })
}

/// Reads the history at `path`; no file means no rollback has left any.
///
/// # Errors
///
/// Fails when the file exists and cannot be read, parsed or understood.
pub fn load(path: &Path) -> anyhow::Result<TierHistory> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            return Ok(TierHistory::default())
        }
        Err(source) => anyhow::bail!(
            "the tier history {} could not be read: {source}",
            path.display()
        ),
    };
    let sidecar: Sidecar = serde_json::from_str(&text).map_err(|source| {
        anyhow::anyhow!(
            "the tier history {} could not be parsed: {source}",
            path.display()
        )
    })?;
    anyhow::ensure!(
        sidecar.version == FORMAT_VERSION,
        "the tier history {} has version {}, but this build reads version {FORMAT_VERSION}",
        path.display(),
        sidecar.version
    );
    Ok(sidecar.history)
}

#[cfg(test)]
mod tests {
    use super::*;

    const COOLDOWN_RUNS: u32 = 3;

    fn keys(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn a_rolled_back_run_that_moved_items_starts_a_cooldown_and_a_streak() {
        // Arrange
        let mut history = TierHistory::default();

        // Act
        history.after_rollback(&keys(&["a", "b"]), COOLDOWN_RUNS);

        // Assert
        assert_eq!(history.consecutive_rollbacks, 1);
        assert_eq!(history.cooldown.len(), 2);
        assert!(history
            .cooldown
            .iter()
            .all(|c| c.runs_left == COOLDOWN_RUNS));
    }

    #[test]
    fn a_rolled_back_run_that_moved_nothing_changes_nothing() {
        // Arrange
        let mut history = TierHistory::default();

        // Act
        history.after_rollback(&[], COOLDOWN_RUNS);

        // Assert
        assert_eq!(history, TierHistory::default());
    }

    #[test]
    fn a_second_rollback_restarts_the_cooldown_of_a_lineage_it_moved_again() {
        // Arrange
        let mut history = TierHistory::default();
        history.after_rollback(&keys(&["a"]), COOLDOWN_RUNS);
        history.after_commit(false);

        // Act
        history.after_rollback(&keys(&["a"]), COOLDOWN_RUNS);

        // Assert
        assert_eq!(history.consecutive_rollbacks, 2);
        assert_eq!(history.cooldown.len(), 1);
        assert_eq!(history.cooldown[0].runs_left, COOLDOWN_RUNS);
    }

    #[test]
    fn commits_count_a_cooldown_down_until_it_is_gone() {
        // Arrange
        let mut history = TierHistory::default();
        history.after_rollback(&keys(&["a"]), 2);

        // Act
        history.after_commit(false);
        let after_one = history.cooldown.len();
        history.after_commit(false);

        // Assert
        assert_eq!(after_one, 1);
        assert!(history.cooldown.is_empty());
        assert_eq!(history.consecutive_rollbacks, 1);
    }

    #[test]
    fn a_commit_that_moved_items_ends_the_streak() {
        // Arrange
        let mut history = TierHistory::default();
        history.after_rollback(&keys(&["a"]), COOLDOWN_RUNS);

        // Act
        history.after_commit(true);

        // Assert
        assert_eq!(history.consecutive_rollbacks, 0);
    }

    #[test]
    fn a_zero_cooldown_leaves_no_entry() {
        // Arrange
        let mut history = TierHistory::default();

        // Act
        history.after_rollback(&keys(&["a"]), 0);

        // Assert
        assert!(history.cooldown.is_empty());
        assert_eq!(history.consecutive_rollbacks, 1);
    }

    #[test]
    fn a_saved_history_loads_back_equal_and_no_file_is_empty() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = history_path_for(&dir.path().join("liam.db"));
        let mut history = TierHistory::default();
        history.after_rollback(&keys(&["a"]), COOLDOWN_RUNS);
        let before = load(&path).unwrap();

        // Act
        save(&path, &history).unwrap();

        // Assert
        assert_eq!(before, TierHistory::default());
        assert_eq!(load(&path).unwrap(), history);
    }

    #[test]
    fn a_corrupt_history_is_an_error_not_a_fresh_start() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = history_path_for(&dir.path().join("liam.db"));
        std::fs::write(&path, b"{not json").unwrap();

        // Act and Assert
        assert!(load(&path).is_err());
    }
}
