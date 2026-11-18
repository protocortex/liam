// SPDX-License-Identifier: Apache-2.0
//! The one line logged before a model download starts, so a first run that
//! waits on the network says why.

use std::path::{Path, PathBuf};

const BYTES_PER_MB: u64 = 1_000_000;
const FETCH_MODELS_HINT: &str = "`liam fetch-models` downloads it ahead of time";
/// hf-hub names a repo directory `models--<org>--<name>` inside the cache.
const HF_REPO_DIR_PREFIX: &str = "models--";

/// The text of the notice. `size_bytes` is left out when the size is not
/// pinned.
pub(crate) fn notice(model: &str, cache_dir: &str, size_bytes: Option<u64>) -> String {
    let size = size_bytes
        .map(|bytes| format!(" ({} MB)", bytes.div_ceil(BYTES_PER_MB)))
        .unwrap_or_default();
    format!("model {model} is not in {cache_dir}, downloading it from the hub{size}; {FETCH_MODELS_HINT}")
}

/// Hands the notice to `emit` only when the files are not cached already.
pub(crate) fn announce_unless_cached(
    cached: bool,
    model: &str,
    cache_dir: &str,
    size_bytes: Option<u64>,
    emit: impl FnOnce(String),
) {
    if !cached {
        emit(notice(model, cache_dir, size_bytes));
    }
}

/// Logs the notice at info level unless the files are cached already.
pub(crate) fn log_unless_cached(
    cached: bool,
    model: &str,
    cache_dir: &str,
    size_bytes: Option<u64>,
) {
    announce_unless_cached(cached, model, cache_dir, size_bytes, |line| {
        tracing::info!("{line}");
    });
}

/// Whether hf-hub holds `file` of `repo` at `revision` under `cache_dir`.
pub(crate) fn hub_file_cached(cache_dir: &str, repo: &str, revision: &str, file: &str) -> bool {
    hf_hub::Cache::new(PathBuf::from(cache_dir))
        .repo(hf_hub::Repo::with_revision(
            repo.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ))
        .get(file)
        .is_some()
}

/// Whether `cache_dir` has any directory for `repo`. A coarse check for the
/// loaders that pick their own files.
pub(crate) fn hub_repo_cached(cache_dir: &str, repo: &str) -> bool {
    Path::new(cache_dir)
        .join(format!("{HF_REPO_DIR_PREFIX}{}", repo.replace('/', "--")))
        .is_dir()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_model_emits_one_line_with_model_dir_size_and_hint() {
        // Arrange
        let mut lines = Vec::new();

        // Act
        announce_unless_cached(false, "org/m", "/cache", Some(639_150_592), |l| {
            lines.push(l)
        });

        // Assert
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("org/m"));
        assert!(lines[0].contains("/cache"));
        assert!(lines[0].contains("(640 MB)"));
        assert!(lines[0].contains("liam fetch-models"));
    }

    #[test]
    fn an_unknown_size_is_left_out() {
        // Arrange / Act
        let line = notice("org/m", "/cache", None);

        // Assert
        assert!(!line.contains("MB"));
    }

    #[test]
    fn a_cached_model_emits_nothing() {
        // Arrange
        let mut lines = Vec::new();

        // Act
        announce_unless_cached(true, "org/m", "/cache", None, |l| lines.push(l));

        // Assert
        assert!(lines.is_empty());
    }

    #[test]
    fn a_repo_directory_counts_as_cached() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().to_str().unwrap();
        assert!(!hub_repo_cached(cache, "org/m"));
        std::fs::create_dir(dir.path().join("models--org--m")).unwrap();

        // Act / Assert
        assert!(hub_repo_cached(cache, "org/m"));
    }
}
