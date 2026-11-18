// SPDX-License-Identifier: Apache-2.0
//! Each error that wraps a cause keeps it as the source, and its own message
//! stays short so `{:#}` prints the cause once.

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

/// The text `{:#}` prints for an `anyhow` wrapper: each cause once, joined by `: `.
fn chain(error: &dyn Error) -> String {
    let mut parts = vec![error.to_string()];
    let mut next = error.source();
    while let Some(cause) = next {
        parts.push(cause.to_string());
        next = cause.source();
    }
    parts.join(": ")
}

fn assert_cause_once(error: &dyn Error) {
    let source = error.source().expect("the cause is the source");
    assert!(source.to_string().contains(CAUSE), "{source}");
    assert!(!error.to_string().contains(CAUSE), "{error}");
    assert_eq!(chain(error).matches(CAUSE).count(), 1, "{}", chain(error));
}

#[test]
fn a_wal_io_failure_keeps_its_cause() {
    // Arrange
    let error = WalError::Io(io_cause());

    // Act / Assert
    assert_cause_once(&error);
}

#[test]
fn a_wal_lock_failure_keeps_its_cause() {
    // Arrange
    let error = WalError::LockFailed {
        dir: PathBuf::from("log"),
        source: io_cause(),
    };

    // Act / Assert
    assert_cause_once(&error);
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
    let text = chain(&error);

    // Assert
    assert!(error.source().is_some());
    assert_eq!(text.matches("3 unexpected bytes").count(), 1, "{text}");
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
    let wal_chain = chain(&wal);

    // Assert
    assert_eq!(wal_chain.matches(CAUSE).count(), 1, "{wal_chain}");
    assert_cause_once(&removal);
}

#[test]
fn a_migration_io_failure_keeps_its_cause() {
    // Arrange
    let error = MigrateError::Io(io_cause());

    // Act / Assert
    assert_cause_once(&error);
}

#[test]
fn an_archive_io_failure_keeps_its_cause() {
    // Arrange
    let error = ArchiveError::Io(io_cause());

    // Act / Assert
    assert_cause_once(&error);
}

#[test]
fn a_crypto_io_failure_keeps_its_cause() {
    // Arrange
    let error = CryptoError::Io(io_cause());

    // Act / Assert
    assert_cause_once(&error);
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
    let chains = [chain(&io), chain(&wal), chain(&archive)];

    // Assert
    assert_cause_once(&io);
    for text in chains {
        assert_eq!(text.matches(CAUSE).count(), 1, "{text}");
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
    let (open_chain, seal_chain) = (chain(&open), chain(&seal));

    // Assert
    assert!(open.source().is_some());
    assert!(seal.source().is_some());
    assert_eq!(open_chain.matches(&decode_text).count(), 1, "{open_chain}");
    assert_eq!(seal_chain.matches(&encode_text).count(), 1, "{seal_chain}");
}
