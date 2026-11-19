// SPDX-License-Identifier: Apache-2.0
//! The one line logged before a model download starts, so a first run that
//! waits on the network says why.

use std::path::PathBuf;

const BYTES_PER_MB: u64 = 1_000_000;
const FETCH_MODELS_HINT: &str = "`liam fetch-models` downloads it ahead of time";

/// Who is loading the model, which decides whether the notice points at
/// `liam fetch-models`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoticeCaller {
    /// The daemon or another command that loads the model to use it.
    Daemon,
    /// `liam fetch-models`, which is the command the hint names.
    FetchModels,
}

/// The text of the notice. `size_bytes` is left out when the size is not
/// pinned, and the hint is left out for `liam fetch-models` itself.
pub(crate) fn notice(
    model: &str,
    cache_dir: &str,
    size_bytes: Option<u64>,
    caller: NoticeCaller,
) -> String {
    let size = size_bytes
        .map(|bytes| format!(" ({} MB)", bytes.div_ceil(BYTES_PER_MB)))
        .unwrap_or_default();
    let hint = match caller {
        NoticeCaller::Daemon => format!("; {FETCH_MODELS_HINT}"),
        NoticeCaller::FetchModels => String::new(),
    };
    format!("model {model} is not in {cache_dir}, downloading it from the hub{size}{hint}")
}

/// Hands the notice to `emit` only when the files are not cached already.
pub(crate) fn announce_unless_cached(
    cached: bool,
    model: &str,
    cache_dir: &str,
    size_bytes: Option<u64>,
    caller: NoticeCaller,
    emit: impl FnOnce(String),
) {
    if !cached {
        emit(notice(model, cache_dir, size_bytes, caller));
    }
}

/// Logs the notice at info level unless the files are cached already.
pub(crate) fn log_unless_cached(
    cached: bool,
    model: &str,
    cache_dir: &str,
    size_bytes: Option<u64>,
    caller: NoticeCaller,
) {
    announce_unless_cached(cached, model, cache_dir, size_bytes, caller, |line| {
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

#[cfg(test)]
mod tests {
    use super::*;

    const REPO: &str = "org/m";
    const REVISION: &str = "main";
    const COMMIT: &str = "abc123";

    /// Lays out `files` of `REPO` the way hf-hub caches them.
    fn cache_files(cache: &std::path::Path, files: &[&str]) {
        let repo = cache.join("models--org--m");
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs").join(REVISION), COMMIT).unwrap();
        for file in files {
            let path = repo.join("snapshots").join(COMMIT).join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"x").unwrap();
        }
    }

    #[test]
    fn a_missing_model_emits_one_line_with_model_dir_size_and_hint() {
        // Arrange
        let mut lines = Vec::new();

        // Act
        announce_unless_cached(
            false,
            "org/m",
            "/cache",
            Some(639_150_592),
            NoticeCaller::Daemon,
            |l| lines.push(l),
        );

        // Assert
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("org/m"));
        assert!(lines[0].contains("/cache"));
        assert!(lines[0].contains("(640 MB)"));
        assert!(lines[0].contains("liam fetch-models"));
    }

    #[test]
    fn fetch_models_gets_the_notice_without_the_hint_about_itself() {
        // Arrange / Act
        let line = notice("org/m", "/cache", None, NoticeCaller::FetchModels);

        // Assert
        assert!(line.contains("downloading it from the hub"));
        assert!(!line.contains("fetch-models"), "{line}");
    }

    #[test]
    fn an_unknown_size_is_left_out() {
        // Arrange / Act
        let line = notice("org/m", "/cache", None, NoticeCaller::Daemon);

        // Assert
        assert!(!line.contains("MB"));
    }

    #[test]
    fn a_cached_model_emits_nothing() {
        // Arrange
        let mut lines = Vec::new();

        // Act
        announce_unless_cached(true, "org/m", "/cache", None, NoticeCaller::Daemon, |l| {
            lines.push(l)
        });

        // Assert
        assert!(lines.is_empty());
    }

    #[test]
    fn a_file_missing_from_a_repo_directory_is_not_cached() {
        // Arrange: an interrupted download left the repo directory and one file
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().to_str().unwrap();
        cache_files(dir.path(), &["config.json"]);

        // Act
        let cached = hub_file_cached(cache, REPO, REVISION, "model.onnx");

        // Assert
        assert!(!cached);
    }

    #[test]
    fn a_file_in_the_repo_directory_is_cached() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().to_str().unwrap();
        cache_files(dir.path(), &["config.json", "onnx/model.onnx"]);

        // Act
        let cached = hub_file_cached(cache, REPO, REVISION, "onnx/model.onnx");

        // Assert
        assert!(cached);
    }
}
