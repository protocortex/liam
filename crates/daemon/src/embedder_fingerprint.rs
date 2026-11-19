// SPDX-License-Identifier: Apache-2.0
//! Records which embedder wrote a store's vectors and refuses to start with a
//! different one.
//!
//! `node_vectors` has a fixed column width and holds vectors that only make
//! sense next to others from the same provider, model and prompting. Changing
//! `embedder.provider`, `embedder.model` or `embedding_dims` in `liam.toml`
//! would otherwise write incompatible vectors beside the old ones with nothing
//! to flag it.
//!
//! The record is a sidecar file beside the database, not a table: the store
//! knows nothing about models, and the record must survive `liam rebuild`,
//! which clears the store's derived tables. A start verifies it before the
//! store opens and records it after, so a start that fails to open the store
//! leaves nothing behind.

use std::path::{Path, PathBuf};

use liam_log::wal::{write_atomically, OsFileOps};
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::models::MOCK_PROVIDER;

/// Version of the convention that turns text into model input (prefixes and
/// the like). Bump it whenever that convention changes for a model, since the
/// model and dimensions stay the same while old and new vectors diverge.
/// Scheme 1: Qwen3 queries got a `query: ` prefix, stored text nothing.
/// Scheme 2: Qwen3 queries use the model's `Instruct: <task>\nQuery:<query>`
/// format, stored text nothing.
pub const PROMPT_SCHEME: u32 = 2;

/// The scheme every store written before fingerprints existed was embedded
/// under. It stays 0 however far `PROMPT_SCHEME` moves.
const LEGACY_PROMPT_SCHEME: u32 = 0;

/// The model every store written before fingerprints held vectors from. It is
/// fixed, whatever embedder runs now, because those stores never change.
const LEGACY_MODEL_ID: &str = "Qwen/Qwen3-Embedding-0.6B";

/// The width of those stores' vectors, the default `embedding_dims` then.
const LEGACY_DIMS: usize = 768;

/// Appended to the database path to name the sidecar file.
const FINGERPRINT_SUFFIX: &str = ".embedder.json";

/// Layout version of the sidecar file, written as `version`. A file with any
/// other version is refused rather than guessed at.
const FORMAT_VERSION: u32 = 1;

/// What produced a store's vectors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedderFingerprint {
    pub provider: String,
    pub model: String,
    pub dims: usize,
    pub prompt_scheme: u32,
}

impl EmbedderFingerprint {
    /// Whether vectors recorded under `self` serve `current`. Mock vectors do not depend on a
    /// model, so two mock fingerprints compare on dims and prompt scheme only.
    fn is_compatible_with(&self, current: &Self) -> bool {
        let both_mock = self.provider == MOCK_PROVIDER && current.provider == MOCK_PROVIDER;
        if both_mock {
            self.dims == current.dims && self.prompt_scheme == current.prompt_scheme
        } else {
            self == current
        }
    }
}

impl EmbedderFingerprint {
    /// The fingerprint the configured embedder would write.
    pub fn from_config(config: &Config) -> Self {
        Self {
            provider: config.embedder.provider.clone(),
            model: config.embedder.model.clone(),
            dims: config.embedding_dims,
            prompt_scheme: PROMPT_SCHEME,
        }
    }

    /// What a store with vectors but no record is assumed to hold: the model
    /// and width that were the default before fingerprints, under the scheme
    /// that predates them. Only the provider follows the configuration.
    fn legacy_of(current: &Self) -> Self {
        Self {
            provider: current.provider.clone(),
            model: LEGACY_MODEL_ID.to_string(),
            dims: LEGACY_DIMS,
            prompt_scheme: LEGACY_PROMPT_SCHEME,
        }
    }
}

/// The file's contents: the fingerprint under a layout version.
#[derive(Serialize, Deserialize)]
struct Sidecar {
    version: u32,
    #[serde(flatten)]
    fingerprint: EmbedderFingerprint,
}

/// Where the fingerprint for the database at `database_path` lives.
pub fn fingerprint_path_for(database_path: &Path) -> PathBuf {
    let mut path = database_path.as_os_str().to_owned();
    path.push(FINGERPRINT_SUFFIX);
    PathBuf::from(path)
}

/// A fingerprint that verification found missing and is still to be written.
#[derive(Debug)]
#[must_use = "a start that never records leaves the store unguarded"]
pub struct PendingRecord {
    database_path: PathBuf,
    fingerprint: EmbedderFingerprint,
}

impl PendingRecord {
    /// Writes the record. Call it once the store has opened.
    pub fn commit(self) -> anyhow::Result<()> {
        record(&self.database_path, &self.fingerprint)
    }
}

/// Requires the stored fingerprint to match `current`. Run before the store
/// opens, and creates no file.
///
/// With no record, a database file that holds data is a legacy store, assumed
/// to hold the vectors of the pre fingerprint default model and width under the
/// pre fingerprint prompt scheme; a missing or empty file is a fresh install,
/// which will record `current`. Either comes back as a [`PendingRecord`] to commit after the
/// store opens. A record that cannot be read or parsed is an error, never
/// "absent": overwriting it would erase the evidence of a model change.
pub fn verify(
    database_path: &Path,
    current: &EmbedderFingerprint,
) -> anyhow::Result<Option<PendingRecord>> {
    let path = fingerprint_path_for(database_path);
    let (assumed, recorded) = match std::fs::read_to_string(&path) {
        Ok(text) => (None, read_sidecar(&path, &text)?),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            if !holds_data(database_path)? {
                return Ok(Some(pending(database_path, current)));
            }
            let legacy = EmbedderFingerprint::legacy_of(current);
            (Some(pending(database_path, &legacy)), legacy)
        }
        Err(source) => anyhow::bail!(
            "the embedder fingerprint {} could not be read: {source}",
            path.display()
        ),
    };
    if recorded.is_compatible_with(current) {
        return Ok(assumed);
    }
    anyhow::bail!(
        "{}",
        mismatch_message(&path, assumed.is_some(), &recorded, current, database_path)
    )
}

/// Whether the database holds data but has no record, which is how a store
/// from before fingerprints looks. Creates no file.
pub fn predates_fingerprints(database_path: &Path) -> anyhow::Result<bool> {
    Ok(!fingerprint_path_for(database_path).exists() && holds_data(database_path)?)
}

/// Writes `fingerprint` as the record for the database, replacing any other.
/// `liam rebuild` calls it once it has re-embedded the store.
pub fn record(database_path: &Path, fingerprint: &EmbedderFingerprint) -> anyhow::Result<()> {
    let path = fingerprint_path_for(database_path);
    let sidecar = Sidecar {
        version: FORMAT_VERSION,
        fingerprint: fingerprint.clone(),
    };
    let bytes = serde_json::to_vec_pretty(&sidecar)?;
    write_atomically(&path, &bytes, &mut OsFileOps).map_err(|source| {
        anyhow::anyhow!(
            "could not record the embedder fingerprint at {}: {source}",
            path.display()
        )
    })?;
    tracing::info!(
        provider = %fingerprint.provider,
        model = %fingerprint.model,
        dims = fingerprint.dims,
        prompt_scheme = fingerprint.prompt_scheme,
        "recorded the embedder fingerprint"
    );
    Ok(())
}

fn pending(database_path: &Path, fingerprint: &EmbedderFingerprint) -> PendingRecord {
    PendingRecord {
        database_path: database_path.to_path_buf(),
        fingerprint: fingerprint.clone(),
    }
}

/// Whether the database file exists and is not empty. A start that failed
/// after creating the file but before writing anything leaves it empty.
fn holds_data(database_path: &Path) -> anyhow::Result<bool> {
    match std::fs::metadata(database_path) {
        Ok(metadata) => Ok(metadata.len() > 0),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(source) => anyhow::bail!(
            "the database {} could not be inspected: {source}",
            database_path.display()
        ),
    }
}

fn read_sidecar(path: &Path, text: &str) -> anyhow::Result<EmbedderFingerprint> {
    let sidecar: Sidecar = serde_json::from_str(text).map_err(|source| {
        anyhow::anyhow!(
            "the embedder fingerprint {} could not be parsed: {source}. \
             Fix or restore the file; it is not overwritten, because doing so \
             could hide a change of embedder.",
            path.display()
        )
    })?;
    anyhow::ensure!(
        sidecar.version == FORMAT_VERSION,
        "the embedder fingerprint {} has version {}, but this build reads version \
         {FORMAT_VERSION}. Use the liam build that wrote it, or restore the file.",
        path.display(),
        sidecar.version
    );
    Ok(sidecar.fingerprint)
}

fn mismatch_message(
    path: &Path,
    legacy: bool,
    recorded: &EmbedderFingerprint,
    configured: &EmbedderFingerprint,
    database_path: &Path,
) -> String {
    let origin = if legacy {
        format!(
            "has no embedder record at {fingerprint}, so it is taken to have been built with",
            fingerprint = path.display()
        )
    } else {
        format!(
            "recorded in {fingerprint} that it was built with",
            fingerprint = path.display()
        )
    };
    format!(
        "the embedder fingerprint does not match. The store at {database} {origin} provider \
         {recorded_provider:?}, model {recorded_model:?}, {recorded_dims} dimensions, prompt \
         scheme {recorded_scheme}. liam.toml asks for provider {configured_provider:?}, model \
         {configured_model:?}, {configured_dims} dimensions, prompt scheme {configured_scheme}. \
         To switch to the configured embedder, stop liamd and run `liam rebuild --database \
         {database}`. It re-embeds every node and records the new embedder. The old \
         vectors can only be served by the liam build that wrote them. Deleting {fingerprint} \
         does not help, the old vectors would remain. If you do nothing, liamd refuses to start \
         and the store is not changed.",
        database = database_path.display(),
        recorded_provider = recorded.provider,
        recorded_model = recorded.model,
        recorded_dims = recorded.dims,
        recorded_scheme = recorded.prompt_scheme,
        configured_provider = configured.provider,
        configured_model = configured.model,
        configured_dims = configured.dims,
        configured_scheme = configured.prompt_scheme,
        fingerprint = path.display(),
    )
}

#[cfg(test)]
mod io_error_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// Ids of embedders that no longer exist, as an old store recorded them.
    const QWEN3_MODEL_ID: &str = "Qwen/Qwen3-Embedding-0.6B";
    const QWEN3_GGUF_MODEL_ID: &str = "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0";

    #[test]
    fn the_refusal_is_one_line_in_the_order_problem_fix_consequence() {
        // Arrange
        let recorded = fingerprint("model-a", 768, 1);
        let configured = fingerprint("model-b", 384, 1);
        let database = Path::new("/data/liam.db");

        // Act
        let text = mismatch_message(
            &fingerprint_path_for(database),
            false,
            &recorded,
            &configured,
            database,
        );

        // Assert
        assert!(!text.contains('\n'), "{text}");
        let problem = text.find("does not match").unwrap();
        let fix = text.find("liam rebuild --database /data/liam.db").unwrap();
        let consequence = text.find("If you do nothing").unwrap();
        assert!(problem < fix && fix < consequence, "{text}");
    }

    fn fingerprint(model: &str, dims: usize, prompt_scheme: u32) -> EmbedderFingerprint {
        EmbedderFingerprint {
            provider: "local".to_string(),
            model: model.to_string(),
            dims,
            prompt_scheme,
        }
    }

    fn read(path: &Path) -> EmbedderFingerprint {
        let text = std::fs::read_to_string(path).unwrap();
        read_sidecar(path, &text).unwrap()
    }

    /// A database file that holds data, as one written by an earlier build does.
    fn existing_database(dir: &Path) -> PathBuf {
        let database = dir.join("liam.db");
        std::fs::write(&database, b"existing vectors").unwrap();
        database
    }

    /// Verifies and, as a successful start would, records.
    fn start(database: &Path, current: &EmbedderFingerprint) -> anyhow::Result<()> {
        if let Some(pending) = verify(database, current)? {
            pending.commit()?;
        }
        Ok(())
    }

    #[test]
    fn a_fresh_install_records_the_current_fingerprint_after_the_store_opens() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("liam.db");
        let current = fingerprint("model-a", 768, 1);

        // Act
        let pending = verify(&database, &current).unwrap();

        // Assert: nothing is written until the caller commits
        assert!(!fingerprint_path_for(&database).exists());
        pending
            .expect("a fresh install has a record to write")
            .commit()
            .unwrap();
        assert_eq!(read(&fingerprint_path_for(&database)), current);
    }

    #[test]
    fn an_empty_database_file_is_a_fresh_install() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("liam.db");
        std::fs::write(&database, b"").unwrap();
        let current = fingerprint("model-a", 768, 1);

        // Act
        start(&database, &current).unwrap();

        // Assert
        assert_eq!(read(&fingerprint_path_for(&database)), current);
    }

    #[test]
    fn a_legacy_store_is_recorded_under_prompt_scheme_0_when_that_is_still_current() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let database = existing_database(dir.path());
        let current = fingerprint(LEGACY_MODEL_ID, LEGACY_DIMS, 0);

        // Act
        start(&database, &current).unwrap();

        // Assert
        assert_eq!(read(&fingerprint_path_for(&database)), current);
    }

    #[test]
    fn a_legacy_store_is_refused_once_the_prompt_scheme_has_moved_on() {
        // Arrange: a store from before fingerprints, and a build whose scheme is 1
        let dir = tempfile::tempdir().unwrap();
        let database = existing_database(dir.path());
        let sidecar = fingerprint_path_for(&database);

        // Act
        let error = verify(&database, &fingerprint("model-a", 768, 1))
            .expect_err("scheme 0 vectors must not be taken for scheme 1")
            .to_string();

        // Assert
        assert!(error.contains("prompt scheme 0"), "{error}");
        assert!(error.contains("prompt scheme 1"), "{error}");
        assert!(error.contains(&sidecar.display().to_string()), "{error}");
        assert!(!sidecar.exists(), "a refusal must write nothing");
    }

    #[test]
    fn a_store_with_data_and_no_record_is_refused_under_the_default_embedder() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let database = existing_database(dir.path());
        let sidecar = fingerprint_path_for(&database);
        let mut config = Config::default();
        config.embedder.provider = "local".to_string();
        let current = EmbedderFingerprint {
            prompt_scheme: LEGACY_PROMPT_SCHEME,
            ..EmbedderFingerprint::from_config(&config)
        };

        // Act
        let error = verify(&database, &current)
            .expect_err("the default model must not read the pre fingerprint vectors")
            .to_string();

        // Assert
        assert!(error.contains("liam rebuild"), "{error}");
        assert!(error.contains(LEGACY_MODEL_ID), "{error}");
        assert!(error.contains(liam_model::EMBEDDER_MODEL_ID), "{error}");
        assert!(!sidecar.exists(), "a refusal must write nothing");
    }

    #[test]
    fn a_store_with_data_and_no_record_is_refused_at_other_dimensions() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let database = existing_database(dir.path());
        let current = fingerprint(LEGACY_MODEL_ID, LEGACY_DIMS / 2, LEGACY_PROMPT_SCHEME);

        // Act
        let result = verify(&database, &current);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn an_empty_store_opens_under_any_model() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("liam.db");
        std::fs::write(&database, b"").unwrap();
        let current = fingerprint(liam_model::EMBEDDER_MODEL_ID, 384, PROMPT_SCHEME);

        // Act
        start(&database, &current).unwrap();

        // Assert
        assert_eq!(read(&fingerprint_path_for(&database)), current);
    }

    #[test]
    fn only_a_database_with_data_and_no_record_predates_fingerprints() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let absent = dir.path().join("absent.db");
        let legacy = existing_database(dir.path());
        let recorded = dir.path().join("recorded.db");
        std::fs::write(&recorded, b"existing vectors").unwrap();
        record(&recorded, &fingerprint("model-a", 768, 1)).unwrap();

        // Act
        let answers = [&absent, &legacy, &recorded].map(|db| predates_fingerprints(db).unwrap());

        // Assert
        assert_eq!(answers, [false, true, false]);
    }

    #[test]
    fn a_store_recorded_under_the_query_prefix_scheme_is_refused_by_this_build() {
        // Arrange: a store whose queries were embedded with the `query: ` prefix
        let dir = tempfile::tempdir().unwrap();
        let database = existing_database(dir.path());
        let prefix_scheme = fingerprint("model-a", 768, 1);
        record(&database, &prefix_scheme).unwrap();

        // Act
        let error = verify(&database, &fingerprint("model-a", 768, PROMPT_SCHEME))
            .expect_err("scheme 1 vectors must be rebuilt under the current scheme")
            .to_string();

        // Assert
        assert!(error.contains("prompt scheme 1"), "{error}");
        assert!(error.contains("liam rebuild"), "{error}");
    }

    #[test]
    fn a_build_with_the_current_scheme_refuses_a_store_embedded_before_fingerprints() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let database = existing_database(dir.path());

        // Act
        let error = verify(&database, &fingerprint("model-a", 768, PROMPT_SCHEME))
            .expect_err("the current scheme must not be taken for the legacy one")
            .to_string();

        // Assert
        assert!(error.contains("prompt scheme 0"), "{error}");
    }

    #[test]
    fn a_deleted_record_next_to_a_populated_store_is_treated_as_legacy() {
        // Arrange: a record was written, then deleted
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("liam.db");
        start(&database, &fingerprint("model-a", 768, 1)).unwrap();
        std::fs::write(&database, b"vectors").unwrap();
        std::fs::remove_file(fingerprint_path_for(&database)).unwrap();

        // Act
        let error = verify(&database, &fingerprint("model-a", 768, 1))
            .expect_err("the old scheme cannot be vouched for")
            .to_string();

        // Assert
        assert!(error.contains("prompt scheme 0"), "{error}");
    }

    #[test]
    fn a_matching_fingerprint_starts_normally_and_has_nothing_to_record() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let database = existing_database(dir.path());
        let current = fingerprint("model-a", 768, 0);
        record(&database, &current).unwrap();

        // Act
        let pending = verify(&database, &current).unwrap();

        // Assert
        assert!(pending.is_none());
        assert_eq!(read(&fingerprint_path_for(&database)), current);
    }

    #[test]
    fn a_differing_field_on_an_existing_store_refuses_to_start_naming_both_fingerprints() {
        let recorded = fingerprint("model-a", 768, 0);
        let mut other_provider = recorded.clone();
        other_provider.provider = "mock".to_string();
        for configured in [
            fingerprint("model-b", 768, 0),
            fingerprint("model-a", 1024, 0),
            fingerprint("model-a", 384, 0),
            fingerprint("model-a", 768, 1),
            other_provider,
        ] {
            // Arrange: a real upgrade, the database already exists
            let dir = tempfile::tempdir().unwrap();
            let database = existing_database(dir.path());
            let sidecar = fingerprint_path_for(&database);
            record(&database, &recorded).unwrap();

            // Act
            let error = verify(&database, &configured)
                .expect_err("a different embedder must not start")
                .to_string();

            // Assert
            for phrase in [
                format!("provider {:?}", recorded.provider),
                format!("model {:?}", recorded.model),
                format!("{} dimensions", recorded.dims),
                format!("prompt scheme {}", recorded.prompt_scheme),
                format!("provider {:?}", configured.provider),
                format!("model {:?}", configured.model),
                format!("{} dimensions", configured.dims),
                format!("prompt scheme {}", configured.prompt_scheme),
                sidecar.display().to_string(),
                format!("liam rebuild --database {}", database.display()),
            ] {
                assert!(error.contains(&phrase), "{phrase} missing from: {error}");
            }
            assert_eq!(read(&sidecar), recorded);
        }
    }

    #[test]
    fn a_change_of_provider_alone_is_refused() {
        // Arrange: same model name, mock to local
        let dir = tempfile::tempdir().unwrap();
        let database = existing_database(dir.path());
        let mut mock = fingerprint("model-a", 768, 0);
        mock.provider = "mock".to_string();
        record(&database, &mock).unwrap();

        // Act
        let result = verify(&database, &fingerprint("model-a", 768, 0));

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn an_unparseable_fingerprint_fails_distinctly_and_is_left_untouched() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("liam.db");
        let path = fingerprint_path_for(&database);
        std::fs::write(&path, "{\"model\": \"model-a\", ").unwrap();

        // Act
        let error = verify(&database, &fingerprint("model-a", 768, 0))
            .expect_err("a corrupt record must not be treated as absent")
            .to_string();

        // Assert
        assert!(error.contains("could not be parsed"), "{error}");
        assert!(error.contains(&path.display().to_string()), "{error}");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "{\"model\": \"model-a\", "
        );
    }

    #[test]
    fn a_record_with_another_version_is_refused() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("liam.db");
        let path = fingerprint_path_for(&database);
        std::fs::write(
            &path,
            r#"{"version":2,"provider":"local","model":"m","dims":8,"prompt_scheme":0}"#,
        )
        .unwrap();

        // Act
        let error = verify(&database, &fingerprint("m", 8, 0))
            .expect_err("an unknown layout must not be guessed at")
            .to_string();

        // Assert
        assert!(error.contains("version 2"), "{error}");
    }

    #[test]
    fn the_record_is_a_versioned_file_with_the_documented_suffix() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("liam.db");

        // Act
        record(&database, &fingerprint("m", 8, 0)).unwrap();

        // Assert
        let path = dir.path().join("liam.db.embedder.json");
        assert_eq!(fingerprint_path_for(&database), path);
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(json["version"], 1);
        assert!(!dir.path().join("liam.db.embedder.json.tmp").exists());
    }

    #[test]
    fn an_old_mock_record_is_accepted_under_the_new_default_model_and_a_local_one_is_refused() {
        // Arrange
        let old_default = QWEN3_MODEL_ID;
        let new_default = liam_model::EMBEDDER_MODEL_ID;
        let mock_with = |model: &str| EmbedderFingerprint {
            provider: MOCK_PROVIDER.to_string(),
            ..fingerprint(model, 384, PROMPT_SCHEME)
        };
        let mock_dir = tempfile::tempdir().unwrap();
        let mock_db = mock_dir.path().join("liam.db");
        start(&mock_db, &mock_with(old_default)).unwrap();
        std::fs::write(&mock_db, b"existing vectors").unwrap();
        let local_dir = tempfile::tempdir().unwrap();
        let local_db = local_dir.path().join("liam.db");
        start(&local_db, &fingerprint(old_default, 384, PROMPT_SCHEME)).unwrap();
        std::fs::write(&local_db, b"existing vectors").unwrap();

        // Act
        let mock = verify(&mock_db, &mock_with(new_default));
        let mock_dims = verify(
            &mock_db,
            &EmbedderFingerprint {
                dims: 768,
                ..mock_with(new_default)
            },
        );
        let local = verify(&local_db, &fingerprint(new_default, 384, PROMPT_SCHEME));

        // Assert
        assert!(mock.is_ok());
        assert!(mock_dims.is_err());
        assert!(local.unwrap_err().to_string().contains("liam rebuild"));
    }

    #[test]
    fn a_store_made_with_another_embedder_is_refused_under_arctic_with_a_rebuild_hint() {
        // Arrange
        let local = |model: &str| {
            let mut config = Config::default();
            config.embedder.provider = "local".to_string();
            config.embedder.model = model.to_string();
            EmbedderFingerprint::from_config(&config)
        };
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("liam.db");
        start(&database, &local(QWEN3_GGUF_MODEL_ID)).unwrap();
        std::fs::write(&database, b"existing vectors").unwrap();

        // Act
        let error = verify(&database, &local(liam_model::EMBEDDER_MODEL_ID))
            .expect_err("arctic must not read another embedder's vectors")
            .to_string();

        // Assert
        assert!(error.contains("liam rebuild"), "{error}");
        assert!(error.contains("snowflake-arctic-embed-m-v2.0"), "{error}");
        assert!(error.contains(QWEN3_GGUF_MODEL_ID), "{error}");
    }

    #[test]
    fn the_fingerprint_comes_from_the_configured_embedder() {
        // Arrange
        let mut config = Config::default();
        config.embedder.provider = "local".to_string();
        config.embedder.model = "model-x".to_string();
        config.embedding_dims = 384;

        // Act
        let fingerprint = EmbedderFingerprint::from_config(&config);

        // Assert
        assert_eq!(
            fingerprint,
            self::fingerprint("model-x", 384, PROMPT_SCHEME)
        );
    }
}
