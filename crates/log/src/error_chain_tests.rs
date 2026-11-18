// SPDX-License-Identifier: Apache-2.0
//! Each error that wraps a cause keeps it as the source, and its plain text
//! also names the cause, because most callers print the plain form.

use std::error::Error;
use std::io;
use std::path::PathBuf;

use crate::archive::ArchiveError;
use crate::compactor::CompactError;
use crate::crypto::CryptoError;
use crate::envelope::{OpenError, SealError};
use crate::event::EventError;
use crate::migrate::MigrateError;
use crate::reader::ReaderError;
use crate::wal::WalError;

const CAUSE: &str = "the underlying cause";

fn io_cause() -> io::Error {
    io::Error::other(CAUSE)
}

fn assert_cause_kept(error: &dyn Error) {
    let source = error.source().expect("the cause is the source");
    assert!(source.to_string().contains(CAUSE), "{source}");
    assert!(error.to_string().contains(CAUSE), "{error}");
}

#[test]
fn an_io_failure_on_append_prints_the_text_it_always_did() {
    // Arrange
    let error = WalError::Io(io::Error::other("No space left on device"));

    // Act
    let text = error.to_string();

    // Assert
    assert_eq!(text, "wal io failed: No space left on device");
}

#[test]
fn a_wal_io_failure_keeps_its_cause() {
    // Arrange
    let error = WalError::Io(io_cause());

    // Act / Assert
    assert_cause_kept(&error);
}

#[test]
fn a_wal_lock_failure_keeps_its_cause() {
    // Arrange
    let error = WalError::LockFailed {
        dir: PathBuf::from("log"),
        source: io_cause(),
    };

    // Act / Assert
    assert_cause_kept(&error);
}

#[test]
fn a_wal_decode_failure_keeps_the_event_error() {
    // Arrange
    let error = WalError::Decode {
        segment: PathBuf::from("0001.wal"),
        offset: 7,
        source: EventError::TrailingBytes { count: 3 },
    };

    // Act
    let text = error.to_string();

    // Assert
    assert!(error.source().is_some());
    assert!(text.contains("3 unexpected bytes"), "{text}");
}

#[test]
fn a_compaction_failure_keeps_the_wal_error_and_the_removal_cause() {
    // Arrange
    let wal = CompactError::Wal(WalError::Io(io_cause()));
    let removal = CompactError::Remove {
        segment: PathBuf::from("0001.wal"),
        source: io_cause(),
    };

    // Act
    let wal_text = wal.to_string();

    // Assert
    assert!(wal_text.contains(CAUSE), "{wal_text}");
    assert_cause_kept(&removal);
}

#[test]
fn a_migration_io_failure_keeps_its_cause() {
    // Arrange
    let error = MigrateError::Io(io_cause());

    // Act / Assert
    assert_cause_kept(&error);
}

#[test]
fn an_archive_io_failure_keeps_its_cause() {
    // Arrange
    let error = ArchiveError::Io(io_cause());

    // Act / Assert
    assert_cause_kept(&error);
}

#[test]
fn a_crypto_io_failure_keeps_its_cause() {
    // Arrange
    let error = CryptoError::Io(io_cause());

    // Act / Assert
    assert_cause_kept(&error);
}

#[test]
fn a_reader_failure_keeps_the_cause_through_every_wrapper() {
    // Arrange
    let io = ReaderError::Io(io_cause());
    let wal = ReaderError::Wal {
        sequence: 4,
        source: WalError::Io(io_cause()),
    };
    let archive = ReaderError::Archive(ArchiveError::Io(io_cause()));

    // Act
    let texts = [io.to_string(), wal.to_string(), archive.to_string()];

    // Assert
    assert_cause_kept(&io);
    for text in texts {
        assert!(text.contains(CAUSE), "{text}");
    }
}

#[test]
fn a_sealed_record_failure_keeps_the_codec_error() {
    // Arrange
    let decode_text = postcard::Error::DeserializeUnexpectedEnd.to_string();
    let encode_text = postcard::Error::SerializeBufferFull.to_string();
    let open = OpenError::Decode {
        event_id: "event-1".to_owned(),
        source: postcard::Error::DeserializeUnexpectedEnd,
    };
    let seal = SealError::Encode {
        event_id: "event-1".to_owned(),
        source: postcard::Error::SerializeBufferFull,
    };

    // Act
    let (open_text, seal_text) = (open.to_string(), seal.to_string());

    // Assert
    assert!(open.source().is_some());
    assert!(seal.source().is_some());
    assert!(open_text.contains(&decode_text), "{open_text}");
    assert!(seal_text.contains(&encode_text), "{seal_text}");
}
