// SPDX-License-Identifier: Apache-2.0
//! The local `llama-server` child process: spawn, wait until it is ready,
//! watch it, and stop it.
//!
//! The server listens on loopback TCP, which any local process can reach. The
//! guard therefore gives it a random key and alias through the environment
//! (argv is readable by other users), and sends the key only after the
//! unauthenticated model listing echoes that alias.

use std::fmt;
use std::io::{self, Read};
use std::net::{IpAddr, SocketAddr, TcpListener};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use serde::Deserialize;
use sysinfo::{MemoryRefreshKind, RefreshKind, System};
use tracing::{debug, info, warn};

use super::http::{self, Limits, Request};
use crate::config::ExtractionConfig;
use stale::{Pidfile, ProcessTable, SweepOutcome, SystemProcesses};

pub mod stale;

const API_KEY_ENV: &str = "LLAMA_API_KEY";
const ALIAS_ENV: &str = "LLAMA_ARG_ALIAS";
const HEALTH_PATH: &str = "/health";
const MODELS_PATH: &str = "/v1/models";
const HTTP_OK: u16 = 200;
const HTTP_UNAUTHORIZED: u16 = 401;
/// Not on the server's public list, so it answers 401 without the key.
const PROTECTED_PATH: &str = "/props";
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
#[cfg(test)]
const STOP_GRACE: Duration = Duration::from_secs(5);
#[cfg(test)]
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

/// Free memory, behind a trait so a test can set it.
pub trait FreeRam: Send {
    fn free_mb(&self) -> u64;
}

pub struct SystemRam;

impl FreeRam for SystemRam {
    fn free_mb(&self) -> u64 {
        let system = System::new_with_specifics(
            RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram()),
        );
        system.available_memory() / BYTES_PER_MB
    }
}

/// What a start needs from the outside world, so tests can set RAM and ports
/// and script the process table.
pub struct Deps {
    pub free_ram: Box<dyn FreeRam>,
    pub next_port: Box<dyn FnMut() -> io::Result<u16> + Send>,
    pub processes: Box<dyn ProcessTable>,
    /// Records the running child so a later daemon can clear it.
    pub pidfile: PathBuf,
    /// The executable a stale pidfile must name before its process is killed.
    pub server_exe: PathBuf,
}

impl Deps {
    pub fn system(config: &ExtractionConfig, pidfile: &Path, server_exe: &Path) -> Self {
        let (host, fixed_port) = (config.host.clone(), config.port);
        Self {
            processes: Box::new(SystemProcesses),
            pidfile: pidfile.to_path_buf(),
            server_exe: server_exe.to_path_buf(),
            free_ram: Box::new(SystemRam),
            next_port: Box::new(move || match fixed_port {
                0 => free_loopback_port(&host),
                port => Ok(port),
            }),
        }
    }
}

fn free_loopback_port(host: &str) -> io::Result<u16> {
    let ip: IpAddr = host
        .parse()
        .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    // The port is released before the child binds it; a lost race shows as an
    // early exit and is retried.
    Ok(TcpListener::bind((ip, 0))?.local_addr()?.port())
}

/// Where one start attempt listens and the secrets the child is given.
pub struct LaunchParams<'a> {
    pub addr: SocketAddr,
    pub alias: &'a str,
    /// Accepted by the server beside `key` and good for nothing else: it is
    /// the only secret a listener of unproven identity ever sees.
    pub probe_key: &'a str,
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
        .env(API_KEY_ENV, format!("{},{}", launch.probe_key, launch.key))
        .env(ALIAS_ENV, launch.alias);
    command
}

/// Where a started child is recorded and how its start time is read.
struct Recorder<'a> {
    pidfile: &'a Path,
    processes: &'a dyn ProcessTable,
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

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A cheap view of whether the child still runs, for code that does not own
/// the guard.
#[derive(Clone)]
pub struct AliveHandle(Arc<Mutex<Child>>);

impl AliveHandle {
    pub fn is_alive(&self) -> bool {
        matches!(lock(&self.0).try_wait(), Ok(None))
    }
}

/// Owns a running, ready sidecar. Dropping it kills and reaps the child.
#[derive(Debug)]
pub struct SidecarGuard {
    child: Arc<Mutex<Child>>,
    pid: u32,
    addr: SocketAddr,
    key: Secret,
    probe_key: Secret,
    alias: Secret,
    stderr: Arc<Mutex<StderrTail>>,
    pidfile: PathBuf,
}

impl fmt::Debug for AliveHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AliveHandle")
    }
}

impl SidecarGuard {
    /// Starts the configured `llama-server` and returns once it is ready.
    /// `pidfile` is where the running child is recorded.
    pub async fn start(config: &ExtractionConfig, pidfile: &Path) -> Result<Self, SidecarError> {
        let paths = Paths::resolve(config)?;
        Self::start_with(
            config,
            |launch| server_command(config, &paths, launch),
            Deps::system(config, pidfile, &paths.server),
        )
        .await
    }

    /// Clears a sidecar an earlier daemon left behind. Run once at daemon
    /// startup, because a killed daemon cannot run `Drop`.
    pub fn sweep_stale(config: &ExtractionConfig, pidfile: &Path) -> SweepOutcome {
        match &config.llama_server_path {
            Some(server) => stale::sweep(pidfile, Path::new(server), &SystemProcesses),
            None => SweepOutcome::NoPidfile,
        }
    }

    /// `start` with the command and the outside world injected.
    pub async fn start_with(
        config: &ExtractionConfig,
        command_for: impl Fn(&LaunchParams) -> Command,
        mut deps: Deps,
    ) -> Result<Self, SidecarError> {
        // Before the RAM gate: a stale server holds the memory the gate reads.
        let swept = stale::sweep(&deps.pidfile, &deps.server_exe, deps.processes.as_ref());
        debug!(?swept, "swept the previous sidecar");
        let free_mb = deps.free_ram.free_mb();
        if free_mb < config.ram_budget_mb {
            return Err(SidecarError::RamGate {
                free_mb,
                budget_mb: config.ram_budget_mb,
            });
        }
        let ip: IpAddr = config
            .host
            .parse()
            .map_err(|_| SidecarError::Setup(io::ErrorKind::InvalidInput))?;
        let deadline = Instant::now()
            .checked_add(Duration::from_secs(config.start_timeout_secs))
            .ok_or(SidecarError::Setup(io::ErrorKind::InvalidInput))?;
        let (key, alias) = (random_hex(KEY_BYTES)?, random_hex(ALIAS_BYTES)?);
        for attempt in 1..=MAX_START_ATTEMPTS {
            let port = (deps.next_port)().map_err(|e| SidecarError::Setup(e.kind()))?;
            // Fresh per attempt, so a listener that kept one learns nothing
            // about the next.
            let probe_key = random_hex(KEY_BYTES)?;
            let launch = LaunchParams {
                addr: SocketAddr::new(ip, port),
                alias: &alias,
                probe_key: &probe_key,
                key: &key,
            };
            let recorder = Recorder {
                pidfile: &deps.pidfile,
                processes: deps.processes.as_ref(),
            };
            match Self::launch(command_for(&launch), &launch, deadline, recorder).await {
                Err(SidecarError::Exited(code)) if attempt < MAX_START_ATTEMPTS => {
                    warn!(attempt, ?code, "sidecar exited at start, trying a new port");
                }
                result => return result,
            }
        }
        unreachable!("the last attempt always returns")
    }

    async fn launch(
        mut command: Command,
        launch: &LaunchParams<'_>,
        deadline: Instant,
        recorder: Recorder<'_>,
    ) -> Result<Self, SidecarError> {
        command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let mut child = command.spawn().map_err(|e| SidecarError::Spawn(e.kind()))?;
        let stderr = Arc::new(Mutex::new(StderrTail::default()));
        if let Some(pipe) = child.stderr.take() {
            capture_stderr(pipe, Arc::clone(&stderr));
        }
        // Built before the wait so every early return reaps the child.
        stale::claim(child.id());
        let guard = Self {
            pid: child.id(),
            child: Arc::new(Mutex::new(child)),
            addr: launch.addr,
            key: Secret(launch.key.to_string()),
            probe_key: Secret(launch.probe_key.to_string()),
            alias: Secret(launch.alias.to_string()),
            stderr,
            pidfile: recorder.pidfile.to_path_buf(),
        };
        // Written before the wait: a daemon killed mid-start still leaves a
        // record the next one can clear. A failure drops the guard, which
        // kills the child, since an unrecorded sidecar could never be swept.
        // Without a start time the record could never match, so it is refused.
        let start_time = recorder
            .processes
            .inspect(guard.pid)
            .ok_or(SidecarError::Setup(io::ErrorKind::NotFound))?
            .start_time;
        let record = Pidfile {
            pid: guard.pid,
            start_time,
            port: launch.addr.port(),
        };
        record
            .write(recorder.pidfile)
            .map_err(|e| SidecarError::Setup(e.kind()))?;
        match guard.wait_ready(deadline).await {
            Ok(()) => {
                info!(pid = guard.pid, port = guard.addr.port(), "sidecar ready");
                Ok(guard)
            }
            Err(error) => {
                warn!(%error, stderr = %guard.stderr_tail(), "sidecar did not become ready");
                Err(error)
            }
        }
    }

    async fn wait_ready(&self, deadline: Instant) -> Result<(), SidecarError> {
        loop {
            if let Some(status) = self.exit_status() {
                return Err(SidecarError::Exited(status.code()));
            }
            if self.answers_with_alias().await {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(SidecarError::StartTimeout);
            }
            tokio::time::sleep(READY_POLL_INTERVAL).await;
        }
    }

    /// Identity comes before trust. `/health` and the 401 check are
    /// anonymous. The alias is read with the probe key alone, so a listener
    /// that is not our child never sees the real key and cannot echo an alias
    /// it never received. A server that ignored the key variable fails the
    /// 401 check and is not trusted either.
    async fn answers_with_alias(&self) -> bool {
        if self.probe(HEALTH_PATH, None).await.map(|r| r.status) != Some(HTTP_OK) {
            return false;
        }
        let Some(models) = self.probe(MODELS_PATH, Some(&self.probe_key.0)).await else {
            return false;
        };
        let listed = models.status == HTTP_OK
            && serde_json::from_slice::<ModelList>(&models.body)
                .is_ok_and(|list| list.data.iter().any(|model| model.id == self.alias.0));
        if !listed {
            return false;
        }
        let guarded =
            self.probe(PROTECTED_PATH, None).await.map(|r| r.status) == Some(HTTP_UNAUTHORIZED);
        if !guarded {
            warn!("sidecar does not reject an unauthenticated request, so it is not trusted");
        }
        guarded
    }

    async fn probe(&self, path: &str, bearer: Option<&str>) -> Option<http::Response> {
        let request = Request {
            method: "GET",
            path,
            bearer,
            body: &[],
        };
        let limits = Limits {
            timeout: PROBE_TIMEOUT,
            max_body_bytes: PROBE_MAX_BYTES,
        };
        http::send(self.addr, &request, limits)
            .await
            .map_err(|error| debug!(path, %error, "sidecar probe failed"))
            .ok()
    }

    fn exit_status(&self) -> Option<ExitStatus> {
        lock(&self.child).try_wait().ok().flatten()
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    pub fn api_key(&self) -> &str {
        &self.key.0
    }

    #[cfg(test)]
    pub fn pid(&self) -> u32 {
        self.pid
    }

    #[cfg(test)]
    pub fn is_alive(&self) -> bool {
        self.alive_handle().is_alive()
    }

    pub fn alive_handle(&self) -> AliveHandle {
        AliveHandle(Arc::clone(&self.child))
    }

    /// What the child wrote to stderr last, with the key and alias removed.
    pub fn stderr_tail(&self) -> String {
        lock(&self.stderr).redacted(&[&self.key.0, &self.probe_key.0, &self.alias.0])
    }

    /// Asks the server to quit and waits up to a grace period; `Drop` then
    /// kills whatever is left. A run just drops the guard.
    #[cfg(test)]
    pub async fn stop(self) {
        {
            let mut child = lock(&self.child);
            // An unreaped child still owns its pid, so the signal cannot
            // reach a process that reused it.
            if matches!(child.try_wait(), Ok(None)) {
                // SAFETY: `kill` only sends a signal, and the child lock is
                // held so the pid cannot be reaped and recycled meanwhile.
                unsafe { libc::kill(self.pid as libc::pid_t, libc::SIGTERM) };
            }
        }
        let give_up = Instant::now() + STOP_GRACE;
        while self.is_alive() && Instant::now() < give_up {
            tokio::time::sleep(STOP_POLL_INTERVAL).await;
        }
    }
}

impl Drop for SidecarGuard {
    fn drop(&mut self) {
        let mut child = lock(&self.child);
        // Errors mean the child is already gone, which is the goal.
        let _ = child.kill();
        let _ = child.wait();
        stale::release(self.pid);
        // Another guard may have recorded its own child here; leave that.
        if Pidfile::recorded_pid(&self.pidfile) == Some(self.pid) {
            let _ = std::fs::remove_file(&self.pidfile);
        }
    }
}

#[derive(Deserialize)]
struct ModelList {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

/// Drains the pipe so the child never blocks on a full stderr.
fn capture_stderr(mut pipe: impl Read + Send + 'static, tail: Arc<Mutex<StderrTail>>) {
    std::thread::spawn(move || {
        let mut chunk = [0u8; STDERR_READ_CHUNK];
        while let Ok(read) = pipe.read(&mut chunk) {
            if read == 0 {
                break;
            }
            lock(&tail).push(&chunk[..read]);
        }
    });
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
