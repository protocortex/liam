// SPDX-License-Identifier: Apache-2.0
//! Assembling the event log over a directory, shared by `liamd` and `liam
//! rebuild` so both open a log the same way.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use liam_log::crypto::{FileMaster, KeyRegistry, MasterKeyProvider, REGISTRY_FILE};
use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::envelope::{OpeningReader, Sealer};
use liam_log::reader::{LogReader, SequentialScanReader};
use liam_log::wal::{SystemClock, WalConfig, WalWriter, LOCK_NAME, MANIFEST_NAME};
use liam_log::LogWriter;
use liam_store::{EventLog, PoisonHook, SharedLog};

use crate::config::{EncryptionConfig, MasterKeySource};

/// Where the keychain keeps the master key.
#[cfg(target_os = "macos")]
const KEYCHAIN_SERVICE: &str = "liam";
#[cfg(target_os = "macos")]
const KEYCHAIN_ACCOUNT: &str = "log-master-key";

/// Opens the log directory for writing, creating it on first use. The writer
/// keeps the directory locked until it is dropped.
pub fn open_writer(log_dir: &Path, wal: WalConfig) -> anyhow::Result<WalWriter<SystemClock>> {
    WalWriter::open_with_system_clock(log_dir, wal)
        .with_context(|| format!("could not open the log in {}", log_dir.display()))
}

/// The key registry of the log in `log_dir`, or `None` when the log is plain.
/// A log that already holds a registry cannot be opened without
/// `[encryption]`, because its sealed records would otherwise read as empty.
/// With a key file that does not exist yet, a log with no registry gets a new
/// key file; once a registry exists a missing file is an error, since a new key
/// could not open it.
pub fn open_key_registry(
    log_dir: &Path,
    encryption: &EncryptionConfig,
) -> anyhow::Result<Option<Arc<KeyRegistry>>> {
    let has_registry = log_dir.join(REGISTRY_FILE).is_file();
    let Some(source) = encryption.master_key else {
        anyhow::ensure!(
            !has_registry,
            "the log in {} has a key registry, so its records may be sealed; set              [encryption] master_key to open it",
            log_dir.display()
        );
        return Ok(None);
    };
    let master = master_key(source, encryption, has_registry)?;
    let registry = KeyRegistry::open(log_dir, master)
        .with_context(|| format!("could not open the key registry in {}", log_dir.display()))?;
    Ok(Some(Arc::new(registry)))
}

fn master_key(
    source: MasterKeySource,
    encryption: &EncryptionConfig,
    has_registry: bool,
) -> anyhow::Result<Arc<dyn MasterKeyProvider>> {
    match source {
        MasterKeySource::File => {
            let path = Path::new(
                encryption
                    .key_file
                    .as_deref()
                    .context("encryption.key_file is required when master_key = \"file\"")?,
            );
            if !has_registry && !path.exists() {
                tracing::info!(path = %path.display(), "making the master key file for a new key registry");
                FileMaster::generate(path)
                    .with_context(|| format!("could not make the key file {}", path.display()))?;
            }
            let master = FileMaster::open(path)
                .with_context(|| format!("could not read the key file {}", path.display()))?;
            Ok(Arc::new(master))
        }
        MasterKeySource::Keychain => keychain_master(),
    }
}

#[cfg(target_os = "macos")]
fn keychain_master() -> anyhow::Result<Arc<dyn MasterKeyProvider>> {
    use liam_log::crypto::{KeychainItems, KeychainMaster, SystemKeychain};

    let items: Arc<dyn KeychainItems> = Arc::new(SystemKeychain);
    let master = KeychainMaster::open_or_create(&items, KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT)
        .context("could not read the master key from the keychain")?;
    Ok(Arc::new(master))
}

#[cfg(not(target_os = "macos"))]
fn keychain_master() -> anyhow::Result<Arc<dyn MasterKeyProvider>> {
    anyhow::bail!("the keychain is only available on macOS; set encryption.master_key = \"file\"")
}

/// A reader of the log in `log_dir` that opens sealed records when it has
/// `keys`. Every consumer of the log reads through it, so none sees a sealed
/// payload.
pub fn log_reader(
    log_dir: &Path,
    keys: Option<&Arc<KeyRegistry>>,
) -> anyhow::Result<Arc<dyn LogReader>> {
    let reader = SequentialScanReader::local(log_dir)
        .with_context(|| format!("could not read the log in {}", log_dir.display()))?;
    Ok(match keys {
        Some(registry) => Arc::new(OpeningReader::new(reader, Arc::clone(registry))),
        None => Arc::new(reader),
    })
}

/// The log over `writer` and a reader of the same directory. `on_poisoned` is
/// called whenever a write meets a poisoned log. With a key registry the log
/// seals what it appends and its reader opens sealed records.
pub fn shared_log(
    writer: impl LogWriter + 'static,
    log_dir: &Path,
    bloom: BloomConfig,
    on_poisoned: Option<PoisonHook>,
    keys: Option<Arc<KeyRegistry>>,
) -> anyhow::Result<SharedLog> {
    let reader = log_reader(log_dir, keys.as_ref())?;
    let mut log = EventLog::new(Box::new(writer), reader, HashBloom::new(bloom));
    if let Some(registry) = keys {
        log = log.with_sealer(Sealer::new(registry));
    }
    if let Some(hook) = on_poisoned {
        log = log.with_poison_hook(hook);
    }
    Ok(Arc::new(tokio::sync::Mutex::new(log)))
}

/// Whether `dir` already is a log, which its manifest says.
pub fn holds_log(dir: &Path) -> bool {
    dir.join(MANIFEST_NAME).is_file()
}

/// Refuses a directory that holds something other than a log, before a writer
/// puts any file there. Only a missing or empty directory may become a new log;
/// the lock file a crashed first start left behind does not count as content.
pub fn ensure_usable_as_log(dir: &Path) -> anyhow::Result<()> {
    if holds_log(dir) {
        return Ok(());
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("could not read the log directory {}", dir.display()))
        }
    };
    for entry in entries {
        let entry =
            entry.with_context(|| format!("could not read the log directory {}", dir.display()))?;
        anyhow::ensure!(
            entry.file_name() == LOCK_NAME,
            "{} holds files but is not an event log, it has no {MANIFEST_NAME} file; point \
             [log] dir at an empty or new directory, or at the existing log",
            dir.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file_config(key_file: &Path) -> EncryptionConfig {
        EncryptionConfig {
            master_key: Some(MasterKeySource::File),
            key_file: Some(key_file.to_str().expect("utf8 path").to_owned()),
        }
    }

    #[test]
    fn a_plain_log_has_no_registry() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();

        // Act
        let keys = open_key_registry(dir.path(), &EncryptionConfig::default()).unwrap();

        // Assert
        assert!(keys.is_none());
        assert!(!dir.path().join(REGISTRY_FILE).exists());
    }

    #[test]
    fn a_first_use_makes_the_key_file_and_the_registry_and_a_second_use_reopens_them() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let key_file = dir.path().join("master.key");
        let config = file_config(&key_file);
        let log_dir = dir.path().join("log");
        std::fs::create_dir(&log_dir).unwrap();

        // Act
        let first = open_key_registry(&log_dir, &config).unwrap().unwrap();
        let key_id = first.new_event_key().unwrap();
        drop(first);
        let second = open_key_registry(&log_dir, &config).unwrap().unwrap();

        // Assert
        assert!(key_file.is_file());
        assert!(matches!(
            second.open_key(&key_id).unwrap(),
            liam_log::crypto::KeyLookup::Active(_)
        ));
    }

    #[test]
    fn a_log_with_a_registry_refuses_to_open_without_encryption_configured() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let key_file = dir.path().join("master.key");
        open_key_registry(dir.path(), &file_config(&key_file)).unwrap();

        // Act
        let result = open_key_registry(dir.path(), &EncryptionConfig::default());

        // Assert
        let error = result.err().expect("must refuse").to_string();
        assert!(error.contains("master_key"), "{error}");
    }

    #[test]
    fn a_missing_key_file_is_an_error_once_a_registry_exists() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let key_file = dir.path().join("master.key");
        open_key_registry(dir.path(), &file_config(&key_file)).unwrap();
        std::fs::remove_file(&key_file).unwrap();

        // Act
        let result = open_key_registry(dir.path(), &file_config(&key_file));

        // Assert
        assert!(result.is_err());
        assert!(
            !key_file.exists(),
            "no new key may be made for an existing registry"
        );
    }

    #[test]
    fn a_file_source_without_a_key_file_is_an_error() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let config = EncryptionConfig {
            master_key: Some(MasterKeySource::File),
            key_file: None,
        };

        // Act
        let result = open_key_registry(dir.path(), &config);

        // Assert
        assert!(result.is_err());
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn the_keychain_source_is_refused_off_macos() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let config = EncryptionConfig {
            master_key: Some(MasterKeySource::Keychain),
            key_file: None,
        };

        // Act
        let result = open_key_registry(dir.path(), &config);

        // Assert
        assert!(result.is_err());
    }
}
