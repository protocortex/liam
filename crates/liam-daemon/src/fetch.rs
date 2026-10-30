// SPDX-License-Identifier: Apache-2.0
//! Downloads, verifies and installs the files extraction needs: the pinned
//! `llama-server` release and the extraction GGUF.
//!
//! `liam fetch-models` and `install.sh` (through it) are the only callers. The
//! bytes come from a [`Fetcher`], so the checks here run in tests without a
//! network. Every file is written to a temporary name, checked against its
//! pinned size and sha256, and only then renamed into place: a failed check
//! leaves nothing at the final path.
//!
//! Licences of what is fetched, for the notices file: `llama.cpp` is MIT (its
//! `LICENSE` is kept inside the installed release directory) and
//! `Qwen3.5-4B` is Apache-2.0.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{ensure, Context};
use sha2::{Digest, Sha256};
use tracing::debug;

/// The `llama.cpp` release the sidecar is tested against. The server's flags,
/// environment variables and authentication behaviour were checked on this
/// line of the project, so a newer tag is a deliberate bump with a new hash.
pub const SERVER_TAG: &str = "b11860";

/// A file at a fixed URL with a known size and content hash.
#[derive(Debug, Clone, Copy)]
pub struct Asset<'a> {
    pub url: &'a str,
    /// Lowercase hex sha256 of the file.
    pub sha256: &'a str,
    /// Exact size. Also the download cap, so a hostile or broken server cannot
    /// fill the disk before the hash check runs.
    pub size_bytes: u64,
}

/// The macOS arm64 release archive: the server plus the shared libraries it
/// loads from its own directory.
pub const SERVER_ASSET: Asset<'static> = Asset {
    url: "https://github.com/ggml-org/llama.cpp/releases/download/b11860/llama-b11860-bin-macos-arm64.tar.gz",
    sha256: "204fe983df733653d1045d0dfb768f5604c18f8f2859ed804a561a54311a3a9d",
    size_bytes: 16_245_641,
};

/// Qwen3.5 4B at Q4_K_M, from a fixed commit of the unsloth repository so the
/// bytes cannot change under the hash.
pub const MODEL_ASSET: Asset<'static> = Asset {
    url: "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf",
    sha256: "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
    size_bytes: 2_740_937_888,
};

const SERVER_BINARY: &str = "llama-server";
const PARTIAL_SUFFIX: &str = ".partial";
const SUPPORTED_OS: &str = "macos";
const SUPPORTED_ARCH: &str = "aarch64";
const EXECUTABLE_BITS: u32 = 0o111;
const OWNER_ONLY_FILE_MODE: u32 = 0o600;
const INSTALLED_FILE_MODE: u32 = 0o644;
const READ_CHUNK_BYTES: usize = 1 << 16;
/// The release archive wraps everything in one directory named after the tag.
const ARCHIVE_STRIP_COMPONENTS: &str = "1";

/// Where the bytes of a URL come from.
pub trait Fetcher {
    /// Streams the body of `url` into `sink`. Fails once the body passes
    /// `max_bytes`, without reading further.
    fn fetch(&self, url: &str, max_bytes: u64, sink: &mut dyn Write) -> anyhow::Result<()>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Installed {
    Downloaded,
    /// A verified copy was already in place, so nothing was downloaded.
    AlreadyPresent,
}

/// Fails on any host the pinned server archive does not run on.
pub fn check_platform(os: &str, arch: &str) -> anyhow::Result<()> {
    ensure!(
        os == SUPPORTED_OS && arch == SUPPORTED_ARCH,
        "the extraction server is packaged for macOS on Apple Silicon only, found {os}/{arch}; \
         set extraction.llama_server_path to your own llama-server build and extraction.model_gguf \
         to a GGUF file instead"
    );
    Ok(())
}

/// Installs the GGUF at `dest`. A file already there with the pinned hash is
/// kept; anything else is replaced by a verified download.
pub fn install_model(
    fetcher: &dyn Fetcher,
    asset: &Asset<'_>,
    dest: &Path,
) -> anyhow::Result<Installed> {
    if hash_matches(dest, asset) {
        return Ok(Installed::AlreadyPresent);
    }
    let partial = partial_path(dest);
    download_verified(fetcher, asset, &partial)?;
    fs::rename(&partial, dest)
        .with_context(|| format!("moving the model into {}", dest.display()))?;
    debug!(path = %dest.display(), "fetch: model installed");
    Ok(Installed::Downloaded)
}

/// Installs the server release into `dir`, which holds `llama-server` and its
/// libraries. The directory appears only once the archive is verified and
/// unpacked, so its presence means a complete install.
pub fn install_server(
    fetcher: &dyn Fetcher,
    asset: &Asset<'_>,
    dir: &Path,
) -> anyhow::Result<Installed> {
    if is_executable_file(&dir.join(SERVER_BINARY)) {
        return Ok(Installed::AlreadyPresent);
    }
    let archive = partial_path(dir);
    let unpacked = unpacked_path(dir);
    let result = unpack_server(fetcher, asset, &archive, &unpacked).and_then(|()| {
        if dir.exists() {
            fs::remove_dir_all(dir).with_context(|| format!("replacing {}", dir.display()))?;
        }
        fs::rename(&unpacked, dir)
            .with_context(|| format!("moving the server into {}", dir.display()))
    });
    let _ = fs::remove_file(&archive);
    if result.is_err() {
        let _ = fs::remove_dir_all(&unpacked);
    }
    result?;
    debug!(path = %dir.display(), "fetch: server installed");
    Ok(Installed::Downloaded)
}

fn unpack_server(
    fetcher: &dyn Fetcher,
    asset: &Asset<'_>,
    archive: &Path,
    unpacked: &Path,
) -> anyhow::Result<()> {
    download_verified(fetcher, asset, archive)?;
    reject_unsafe_entries(archive)?;
    if unpacked.exists() {
        fs::remove_dir_all(unpacked)?;
    }
    fs::create_dir_all(unpacked)?;
    let status = Command::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(unpacked)
        .args(["--strip-components", ARCHIVE_STRIP_COMPONENTS])
        .status()
        .context("running tar to unpack the server")?;
    ensure!(
        status.success(),
        "tar could not unpack the server archive ({status})"
    );
    ensure!(
        is_executable_file(&unpacked.join(SERVER_BINARY)),
        "the server archive has no executable {SERVER_BINARY}"
    );
    Ok(())
}

/// The archive is hash-pinned, so this is a second line of defence: no entry
/// may name an absolute path or climb out of the target directory.
fn reject_unsafe_entries(archive: &Path) -> anyhow::Result<()> {
    let listing = Command::new("tar")
        .arg("-tzf")
        .arg(archive)
        .output()
        .context("running tar to list the server archive")?;
    ensure!(
        listing.status.success(),
        "tar could not read the server archive"
    );
    for name in String::from_utf8_lossy(&listing.stdout).lines() {
        let escapes = Path::new(name).is_absolute()
            || Path::new(name)
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir));
        ensure!(!escapes, "the server archive has an unsafe entry: {name}");
    }
    Ok(())
}

/// Downloads `asset` to `partial`, checks size and hash, and leaves the file
/// at `partial` only when both pass.
fn download_verified(
    fetcher: &dyn Fetcher,
    asset: &Asset<'_>,
    partial: &Path,
) -> anyhow::Result<()> {
    if let Some(parent) = partial.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let _ = fs::remove_file(partial);
    let result = write_and_check(fetcher, asset, partial);
    if result.is_err() {
        let _ = fs::remove_file(partial);
    }
    result
}

fn write_and_check(fetcher: &dyn Fetcher, asset: &Asset<'_>, partial: &Path) -> anyhow::Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(OWNER_ONLY_FILE_MODE)
        .open(partial)
        .with_context(|| format!("creating {}", partial.display()))?;
    let mut sink = DigestWriter::new(file);
    fetcher.fetch(asset.url, asset.size_bytes, &mut sink)?;
    let (file, size, sha256) = sink.finish();
    file.sync_all().context("flushing the download to disk")?;
    ensure!(
        size == asset.size_bytes,
        "the download is {size} bytes, expected {}",
        asset.size_bytes
    );
    ensure!(
        sha256 == asset.sha256,
        "the download failed its sha256 check: got {sha256}, expected {}",
        asset.sha256
    );
    fs::set_permissions(partial, fs::Permissions::from_mode(INSTALLED_FILE_MODE))?;
    Ok(())
}

fn hash_matches(path: &Path, asset: &Asset<'_>) -> bool {
    let Ok(mut file) = File::open(path) else {
        return false;
    };
    let mut hasher = Sha256::new();
    let mut chunk = vec![0; READ_CHUNK_BYTES];
    loop {
        match file.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => hasher.update(&chunk[..read]),
            Err(_) => return false,
        }
    }
    hex(&hasher.finalize()) == asset.sha256
}

fn is_executable_file(path: &Path) -> bool {
    fs::metadata(path)
        .map(|meta| meta.is_file() && meta.permissions().mode() & EXECUTABLE_BITS != 0)
        .unwrap_or(false)
}

fn partial_path(dest: &Path) -> PathBuf {
    sibling_with_suffix(dest, PARTIAL_SUFFIX)
}

fn unpacked_path(dest: &Path) -> PathBuf {
    sibling_with_suffix(dest, ".unpacked")
}

fn sibling_with_suffix(dest: &Path, suffix: &str) -> PathBuf {
    let mut name = dest.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Counts and hashes what passes through to the file.
struct DigestWriter {
    file: File,
    hasher: Sha256,
    written: u64,
}

impl DigestWriter {
    fn new(file: File) -> Self {
        Self {
            file,
            hasher: Sha256::new(),
            written: 0,
        }
    }

    fn finish(self) -> (File, u64, String) {
        (self.file, self.written, hex(&self.hasher.finalize()))
    }
}

impl Write for DigestWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.file.write(buf)?;
        self.hasher.update(&buf[..written]);
        self.written += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

/// A pinned URL must be plain HTTPS with a host, so a typo cannot ship.
pub fn is_well_formed_https_url(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("https://") else {
        return false;
    };
    let host = rest.split('/').next().unwrap_or_default();
    !host.is_empty()
        && !url.chars().any(|c| c.is_whitespace() || c.is_control())
        && !host.contains('@')
}

#[cfg(test)]
mod tests;
