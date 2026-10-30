// SPDX-License-Identifier: Apache-2.0

use std::os::unix::fs::PermissionsExt;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use tempfile::TempDir;

use super::*;

const STUB_SERVER_SCRIPT: &str = "#!/bin/sh\nexit 0\n";
const STUB_ROOT_DIR: &str = "llama-test";

fn sha_of(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn asset<'a>(url: &'a str, sha256: &'a str, bytes: &[u8]) -> Asset<'a> {
    Asset {
        url,
        sha256,
        size_bytes: bytes.len() as u64,
    }
}

/// Serves fixed bytes and counts calls; `fail_after` cuts the body short.
struct FakeFetcher {
    body: Vec<u8>,
    calls: AtomicUsize,
    fail_after: Option<usize>,
}

impl FakeFetcher {
    fn serving(body: &[u8]) -> Self {
        Self {
            body: body.to_vec(),
            calls: AtomicUsize::new(0),
            fail_after: None,
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Fetcher for FakeFetcher {
    fn fetch(&self, _url: &str, _max: u64, sink: &mut dyn Write) -> anyhow::Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.fail_after {
            Some(cut) => {
                sink.write_all(&self.body[..cut])?;
                anyhow::bail!("connection lost")
            }
            None => Ok(sink.write_all(&self.body)?),
        }
    }
}

fn files_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

/// A gzip tarball with `llama-server` (mode 0755) and a library, under one
/// top-level directory like the real release.
fn server_archive(dir: &Path, with_server: bool) -> Vec<u8> {
    let root = dir.join("src").join(STUB_ROOT_DIR);
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("libggml.dylib"), b"lib").unwrap();
    if with_server {
        fs::write(root.join(SERVER_BINARY), STUB_SERVER_SCRIPT).unwrap();
        fs::set_permissions(root.join(SERVER_BINARY), fs::Permissions::from_mode(0o755)).unwrap();
    }
    let archive = dir.join("server.tar.gz");
    let status = Command::new("tar")
        .arg("-czf")
        .arg(&archive)
        .arg("-C")
        .arg(dir.join("src"))
        .arg(STUB_ROOT_DIR)
        .status()
        .unwrap();
    assert!(status.success());
    fs::read(archive).unwrap()
}

#[test]
fn model_with_a_correct_checksum_is_installed() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let body = b"gguf stub";
    let sha = sha_of(body);
    let fetcher = FakeFetcher::serving(body);
    let dest = temp.path().join("models").join("extraction.gguf");

    // Act
    let outcome = install_model(&fetcher, &asset("https://x/m", &sha, body), &dest).unwrap();

    // Assert
    assert_eq!(outcome, Installed::Downloaded);
    assert_eq!(fs::read(&dest).unwrap(), body);
    assert_eq!(files_in(dest.parent().unwrap()), ["extraction.gguf"]);
}

#[test]
fn model_with_a_wrong_checksum_aborts_and_leaves_no_file() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let body = b"gguf stub";
    let wrong = sha_of(b"something else");
    let fetcher = FakeFetcher::serving(body);
    let dest = temp.path().join("extraction.gguf");

    // Act
    let error = install_model(&fetcher, &asset("https://x/m", &wrong, body), &dest).unwrap_err();

    // Assert
    assert!(error.to_string().contains("sha256"), "{error}");
    assert!(
        files_in(temp.path()).is_empty(),
        "{:?}",
        files_in(temp.path())
    );
}

#[test]
fn model_of_the_wrong_size_aborts_and_leaves_no_file() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let body = b"gguf stub";
    let sha = sha_of(body);
    let fetcher = FakeFetcher::serving(body);
    let dest = temp.path().join("extraction.gguf");
    let short = Asset {
        url: "https://x/m",
        sha256: &sha,
        size_bytes: body.len() as u64 + 1,
    };

    // Act
    let error = install_model(&fetcher, &short, &dest).unwrap_err();

    // Assert
    assert!(error.to_string().contains("bytes"), "{error}");
    assert!(files_in(temp.path()).is_empty());
}

#[test]
fn model_download_cut_short_leaves_no_partial_file() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let body = b"gguf stub";
    let sha = sha_of(body);
    let mut fetcher = FakeFetcher::serving(body);
    fetcher.fail_after = Some(4);
    let dest = temp.path().join("extraction.gguf");

    // Act
    let result = install_model(&fetcher, &asset("https://x/m", &sha, body), &dest);

    // Assert
    assert!(result.is_err());
    assert!(files_in(temp.path()).is_empty());
}

#[test]
fn model_already_verified_is_not_downloaded_again() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let body = b"gguf stub";
    let sha = sha_of(body);
    let fetcher = FakeFetcher::serving(body);
    let dest = temp.path().join("extraction.gguf");
    let pinned = asset("https://x/m", &sha, body);
    install_model(&fetcher, &pinned, &dest).unwrap();

    // Act
    let outcome = install_model(&fetcher, &pinned, &dest).unwrap();

    // Assert
    assert_eq!(outcome, Installed::AlreadyPresent);
    assert_eq!(fetcher.calls(), 1);
}

#[test]
fn model_that_fails_its_hash_is_replaced() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let body = b"gguf stub";
    let sha = sha_of(body);
    let fetcher = FakeFetcher::serving(body);
    let dest = temp.path().join("extraction.gguf");
    fs::write(&dest, b"truncated").unwrap();

    // Act
    let outcome = install_model(&fetcher, &asset("https://x/m", &sha, body), &dest).unwrap();

    // Assert
    assert_eq!(outcome, Installed::Downloaded);
    assert_eq!(fs::read(&dest).unwrap(), body);
}

#[test]
fn server_with_a_correct_checksum_is_installed_and_executable() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let archive = server_archive(temp.path(), true);
    let sha = sha_of(&archive);
    let fetcher = FakeFetcher::serving(&archive);
    let dir = temp.path().join("bin").join("llama-test");

    // Act
    let outcome = install_server(&fetcher, &asset("https://x/s", &sha, &archive), &dir).unwrap();

    // Assert
    assert_eq!(outcome, Installed::Downloaded);
    assert!(is_executable_file(&dir.join(SERVER_BINARY)));
    assert!(dir.join("libggml.dylib").is_file());
    assert_eq!(files_in(dir.parent().unwrap()), ["llama-test"]);
}

#[test]
fn server_with_a_wrong_checksum_aborts_and_leaves_nothing() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let archive = server_archive(temp.path(), true);
    let wrong = sha_of(b"other");
    let fetcher = FakeFetcher::serving(&archive);
    let bin = temp.path().join("bin");
    let dir = bin.join("llama-test");

    // Act
    let result = install_server(&fetcher, &asset("https://x/s", &wrong, &archive), &dir);

    // Assert
    assert!(result.is_err());
    assert!(files_in(&bin).is_empty(), "{:?}", files_in(&bin));
}

#[test]
fn server_archive_without_the_binary_is_rejected_and_leaves_nothing() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let archive = server_archive(temp.path(), false);
    let sha = sha_of(&archive);
    let fetcher = FakeFetcher::serving(&archive);
    let bin = temp.path().join("bin");

    // Act
    let error = install_server(
        &fetcher,
        &asset("https://x/s", &sha, &archive),
        &bin.join("llama-test"),
    )
    .unwrap_err();

    // Assert
    assert!(error.to_string().contains(SERVER_BINARY), "{error}");
    assert!(files_in(&bin).is_empty());
}

#[test]
fn server_archive_with_an_absolute_entry_is_rejected() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let outside = temp.path().join("outside.txt");
    fs::write(&outside, b"x").unwrap();
    let archive_path = temp.path().join("evil.tar.gz");
    let status = Command::new("tar")
        .arg("-czPf")
        .arg(&archive_path)
        .arg(&outside)
        .status()
        .unwrap();
    assert!(status.success());
    let archive = fs::read(&archive_path).unwrap();
    let sha = sha_of(&archive);
    let fetcher = FakeFetcher::serving(&archive);
    let bin = temp.path().join("bin");

    // Act
    let error = install_server(
        &fetcher,
        &asset("https://x/s", &sha, &archive),
        &bin.join("llama-test"),
    )
    .unwrap_err();

    // Assert
    assert!(error.to_string().contains("unsafe entry"), "{error}");
    assert!(files_in(&bin).is_empty());
}

#[test]
fn server_already_installed_is_not_downloaded_again() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let archive = server_archive(temp.path(), true);
    let sha = sha_of(&archive);
    let fetcher = FakeFetcher::serving(&archive);
    let dir = temp.path().join("bin").join("llama-test");
    let pinned = asset("https://x/s", &sha, &archive);
    install_server(&fetcher, &pinned, &dir).unwrap();

    // Act
    let outcome = install_server(&fetcher, &pinned, &dir).unwrap();

    // Assert
    assert_eq!(outcome, Installed::AlreadyPresent);
    assert_eq!(fetcher.calls(), 1);
}

#[test]
fn platform_check_accepts_only_apple_silicon_macos() {
    // Arrange
    let cases = [
        ("macos", "aarch64", true),
        ("macos", "x86_64", false),
        ("linux", "aarch64", false),
        ("linux", "x86_64", false),
    ];

    for (os, arch, ok) in cases {
        // Act
        let result = check_platform(os, arch);

        // Assert
        assert_eq!(result.is_ok(), ok, "{os}/{arch}");
        if let Err(error) = result {
            assert!(
                error.to_string().contains(&format!("{os}/{arch}")),
                "{error}"
            );
        }
    }
}

fn is_lower_hex_sha256(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[test]
fn pinned_constants_are_well_formed() {
    for (name, pinned) in [("server", SERVER_ASSET), ("model", MODEL_ASSET)] {
        assert!(
            is_lower_hex_sha256(pinned.sha256),
            "{name} sha256: {}",
            pinned.sha256
        );
        assert!(
            is_well_formed_https_url(pinned.url),
            "{name} url: {}",
            pinned.url
        );
        assert!(pinned.size_bytes > 0, "{name} size");
    }
    assert!(SERVER_ASSET.url.contains(SERVER_TAG));
}

#[test]
fn url_shape_check_rejects_typos() {
    for bad in [
        "http://example.com/a",
        "https://",
        "https:///path",
        "https://exa mple.com/a",
        "https://user@example.com/a",
        "example.com/a",
    ] {
        assert!(!is_well_formed_https_url(bad), "{bad}");
    }
    assert!(is_well_formed_https_url("https://example.com/a?b=c"));
}
