// SPDX-License-Identifier: Apache-2.0
//! The one line logged before a model download starts, so a first run that
//! waits on the network says why.

use std::path::PathBuf;

const BYTES_PER_MB: u64 = 1_000_000;
const FETCH_MODELS_HINT: &str = "`liam fetch-models` downloads it ahead of time";
/// The revision fastembed and the candle loader fetch: they ask hf-hub for the
/// repo's default branch.
pub(crate) const HUB_DEFAULT_REVISION: &str = "main";

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

/// Whether every one of `files` of `repo` is cached. A repo directory left by
/// an interrupted download does not count, because the model is only whole when
/// each file it loads is there.
pub(crate) fn hub_files_cached<'a>(
    cache_dir: &str,
    repo: &str,
    files: impl IntoIterator<Item = &'a str>,
) -> bool {
    files
        .into_iter()
        .all(|file| hub_file_cached(cache_dir, repo, HUB_DEFAULT_REVISION, file))
}

/// The directory fastembed downloads into: `HF_HOME` when it is set, ahead of
/// the configured `cache_dir`.
pub(crate) fn fastembed_cache_dir<'a>(hf_home: Option<&'a str>, cache_dir: &'a str) -> &'a str {
    crate::bge::hf_home_overriding(hf_home, cache_dir).unwrap_or(cache_dir)
}

/// Whether the fastembed built-in model `info` is cached, looking in the
/// directory fastembed uses.
pub(crate) fn fastembed_model_cached(
    hf_home: Option<&str>,
    cache_dir: &str,
    info: &fastembed::ModelInfo<fastembed::EmbeddingModel>,
) -> bool {
    let files = std::iter::once(&info.model_file)
        .chain(&info.additional_files)
        .map(String::as_str);
    hub_files_cached(
        fastembed_cache_dir(hf_home, cache_dir),
        &info.model_code,
        files,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPO: &str = "org/m";
    const COMMIT: &str = "abc123";

    /// Lays out `files` of `REPO` the way hf-hub caches them.
    fn cache_files(cache: &std::path::Path, files: &[&str]) {
        let repo = cache.join("models--org--m");
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs").join(HUB_DEFAULT_REVISION), COMMIT).unwrap();
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
    fn a_repo_directory_with_a_file_missing_is_not_cached() {
        // Arrange: an interrupted download left the repo directory and one file
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().to_str().unwrap();
        cache_files(dir.path(), &["config.json"]);

        // Act
        let cached = hub_files_cached(cache, REPO, ["config.json", "model.onnx"]);

        // Assert
        assert!(!cached);
    }

    #[test]
    fn a_repo_with_every_file_is_cached() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().to_str().unwrap();
        cache_files(dir.path(), &["config.json", "onnx/model.onnx"]);

        // Act
        let cached = hub_files_cached(cache, REPO, ["config.json", "onnx/model.onnx"]);

        // Assert
        assert!(cached);
    }

    #[test]
    fn hf_home_is_the_directory_to_check_when_it_is_set() {
        // Arrange / Act / Assert
        assert_eq!(fastembed_cache_dir(Some("/hf"), "/cache"), "/hf");
        assert_eq!(fastembed_cache_dir(None, "/cache"), "/cache");
        assert_eq!(fastembed_cache_dir(Some(""), "/cache"), "/cache");
        assert_eq!(fastembed_cache_dir(Some("/cache"), "/cache"), "/cache");
    }

    #[test]
    fn a_model_cached_under_hf_home_is_found_there_and_not_in_the_cache_dir() {
        // Arrange: the files are under HF_HOME, and the configured cache is empty
        let hf_home = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        cache_files(hf_home.path(), &["config.json"]);
        let mut info = fastembed_info();
        info.model_code = REPO.to_string();
        info.model_file = "config.json".to_string();
        info.additional_files.clear();

        // Act
        let with_hf_home = fastembed_model_cached(
            hf_home.path().to_str(),
            cache.path().to_str().unwrap(),
            &info,
        );
        let without = fastembed_model_cached(None, cache.path().to_str().unwrap(), &info);

        // Assert
        assert!(with_hf_home);
        assert!(!without);
    }

    fn fastembed_info() -> fastembed::ModelInfo<fastembed::EmbeddingModel> {
        fastembed::TextEmbedding::get_model_info(&fastembed::EmbeddingModel::BGEM3)
            .expect("BGE-M3 is listed")
            .clone()
    }
}
