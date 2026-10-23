// SPDX-License-Identifier: Apache-2.0
//! The local `llama-server` child process: spawn, wait until it is ready,
//! watch it, and stop it.
//!
//! The server listens on loopback TCP, which any local process can reach. The
//! guard therefore gives it a random key and alias through the environment
//! (argv is readable by other users), and sends the key only after the
//! unauthenticated model listing echoes that alias.
// Read by the guard that starts the server.
#![allow(dead_code)]

use std::fmt;
use std::io::{self, Read};
use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use crate::config::ExtractionConfig;

const API_KEY_ENV: &str = "LLAMA_API_KEY";
const ALIAS_ENV: &str = "LLAMA_ARG_ALIAS";
const HEALTH_PATH: &str = "/health";
const MODELS_PATH: &str = "/v1/models";
const HTTP_OK: u16 = 200;
const URANDOM: &str = "/dev/urandom";
const KEY_BYTES: usize = 32;
const ALIAS_BYTES: usize = 16;
const BYTES_PER_MB: u64 = 1024 * 1024;
/// Attempts to start on a fresh port when the child dies before it is ready,
/// which is what a lost port race looks like.
const MAX_START_ATTEMPTS: u32 = 3;
const READY_POLL_INTERVAL: Duration = Duration::from_millis(100);
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const PROBE_MAX_BYTES: usize = 64 * 1024;
const STOP_GRACE: Duration = Duration::from_secs(5);
const STOP_POLL_INTERVAL: Duration = Duration::from_millis(50);
const STDERR_CAP_BYTES: usize = 16 * 1024;
const STDERR_READ_CHUNK: usize = 4096;
const REDACTED: &str = "[redacted]";
const EXECUTABLE_BITS: u32 = 0o111;

/// Why the sidecar could not be started. Carries no event text or secret.
#[derive(Debug, PartialEq, Eq)]
pub enum SidecarError {
    /// A configured path is missing or unusable; names the config key.
    BadPath {
        key: &'static str,
        why: &'static str,
    },
    RamGate {
        free_mb: u64,
        budget_mb: u64,
    },
    /// No random bytes, no loopback address, or no free port.
    Setup(io::ErrorKind),
    Spawn(io::ErrorKind),
    /// The child ended before it was ready; `None` means a signal killed it.
    Exited(Option<i32>),
    StartTimeout,
}

impl fmt::Display for SidecarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadPath { key, why } => write!(f, "extraction.{key} {why}"),
            Self::RamGate { free_mb, budget_mb } => write!(
                f,
                "{free_mb} MB of RAM is free, below extraction.ram_budget_mb of {budget_mb} MB"
            ),
            Self::Setup(kind) => write!(f, "could not prepare the sidecar start: {kind}"),
            Self::Spawn(kind) => write!(f, "could not start the sidecar process: {kind}"),
            Self::Exited(Some(code)) => write!(f, "the sidecar exited with code {code} at start"),
            Self::Exited(None) => write!(f, "the sidecar was killed by a signal at start"),
            Self::StartTimeout => write!(f, "the sidecar was not ready in time"),
        }
    }
}

impl std::error::Error for SidecarError {}

/// Where one start attempt listens and the secrets the child is given.
pub struct LaunchParams<'a> {
    pub addr: SocketAddr,
    pub alias: &'a str,
    pub key: &'a str,
}

/// The server and model files, checked before anything is spawned.
struct Paths {
    server: PathBuf,
    model: PathBuf,
}

impl Paths {
    fn resolve(config: &ExtractionConfig) -> Result<Self, SidecarError> {
        let server = required(&config.llama_server_path, "llama_server_path")?;
        let model = required(&config.model_gguf, "model_gguf")?;
        let server_mode = std::fs::metadata(&server)
            .map_err(|_| bad_path("llama_server_path", "does not point to a file"))?
            .permissions()
            .mode();
        if server_mode & EXECUTABLE_BITS == 0 {
            return Err(bad_path("llama_server_path", "is not executable"));
        }
        if !model.is_file() {
            return Err(bad_path("model_gguf", "does not point to a file"));
        }
        Ok(Self { server, model })
    }
}

fn bad_path(key: &'static str, why: &'static str) -> SidecarError {
    SidecarError::BadPath { key, why }
}

fn required(value: &Option<String>, key: &'static str) -> Result<PathBuf, SidecarError> {
    match value.as_deref() {
        Some(path) if !path.is_empty() => Ok(PathBuf::from(path)),
        _ => Err(bad_path(key, "must be set when llm_enabled is true")),
    }
}

/// The only variables the child inherits: the daemon's own environment may
/// hold credentials that a model server has no use for.
const INHERITED_ENV: [&str; 9] = [
    "PATH",
    "HOME",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "XDG_CACHE_HOME",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
];

/// The real `llama-server` command line. No shell is involved, and the
/// secrets travel in the environment only.
fn server_command(config: &ExtractionConfig, paths: &Paths, launch: &LaunchParams) -> Command {
    let mut command = Command::new(&paths.server);
    command.env_clear();
    for name in INHERITED_ENV {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
        .arg("--host")
        .arg(launch.addr.ip().to_string())
        .arg("--port")
        .arg(launch.addr.port().to_string())
        .arg("--model")
        .arg(&paths.model)
        .arg("--ctx-size")
        .arg(config.ctx_tokens.to_string())
        .args(&config.extra_args)
        .env(API_KEY_ENV, launch.key)
        .env(ALIAS_ENV, launch.alias);
    command
}

/// The secret the client authenticates with; never printed.
struct Secret(String);

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(<redacted>)")
    }
}

/// The last bytes the child wrote to stderr.
#[derive(Debug, Default)]
struct StderrTail {
    bytes: Vec<u8>,
    dropped: bool,
}

impl StderrTail {
    fn push(&mut self, chunk: &[u8]) {
        self.bytes.extend_from_slice(chunk);
        if self.bytes.len() > STDERR_CAP_BYTES {
            let excess = self.bytes.len() - STDERR_CAP_BYTES;
            self.bytes.drain(..excess);
            self.dropped = true;
        }
    }

    /// Secrets removed. After a cut the front may hold only the tail of a
    /// secret, which no exact match could find: the first line is dropped, or
    /// with no line break the longest tail a secret could leave.
    fn redacted(&self, secrets: &[&str]) -> String {
        let start = if self.dropped {
            let longest = secrets.iter().map(|s| s.len()).max().unwrap_or(0);
            match self.bytes.iter().position(|&b| b == b'\n') {
                Some(line_end) => line_end + 1,
                None => longest.saturating_sub(1).min(self.bytes.len()),
            }
        } else {
            0
        };
        let mut text = String::from_utf8_lossy(&self.bytes[start..]).into_owned();
        for secret in secrets {
            text = text.replace(secret, REDACTED);
        }
        text
    }
}

fn random_hex(len: usize) -> Result<String, SidecarError> {
    let mut bytes = vec![0u8; len];
    std::fs::File::open(Path::new(URANDOM))
        .and_then(|mut file| file.read_exact(&mut bytes))
        .map_err(|e| SidecarError::Setup(e.kind()))?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests;
