// SPDX-License-Identifier: Apache-2.0
//! A [`Fetcher`] that runs the system `curl`.
//!
//! A TLS client in the daemon's dependency tree would pull `rustls` and a
//! certificate bundle into every build for a download that happens once.
//! macOS ships `curl`, which is the only platform the fetch supports.

use std::io::{Read, Write};
use std::process::{Command, Stdio};

use anyhow::{bail, Context};

use super::Fetcher;

const MAX_REDIRECTS: &str = "5";
const CONNECT_TIMEOUT_SECS: &str = "30";
/// A transfer slower than this for `STALL_SECS` is abandoned, which bounds a
/// stalled download without capping how long a slow link may take.
const STALL_BYTES_PER_SEC: &str = "1024";
const STALL_SECS: &str = "60";
const HTTPS_ONLY: &str = "=https";
const READ_CHUNK_BYTES: usize = 1 << 16;
const STDERR_CAP_BYTES: u64 = 2048;

#[derive(Debug, Clone, Copy)]
pub struct CurlFetcher {
    protocols: &'static str,
}

impl CurlFetcher {
    /// Refuses any URL, and any redirect, that is not HTTPS.
    pub fn https_only() -> Self {
        Self {
            protocols: HTTPS_ONLY,
        }
    }

    /// Plain HTTP, so tests can talk to a loopback listener.
    #[cfg(test)]
    pub(super) fn plain_http() -> Self {
        Self { protocols: "=http" }
    }

    pub(super) fn command_args(&self, url: &str) -> Vec<String> {
        // `-q` must come first: it skips any curlrc that could widen the
        // protocol list or add credentials.
        [
            "-q",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--max-redirs",
            MAX_REDIRECTS,
            "--connect-timeout",
            CONNECT_TIMEOUT_SECS,
            "--speed-limit",
            STALL_BYTES_PER_SEC,
            "--speed-time",
            STALL_SECS,
            "--proto",
            self.protocols,
            "--proto-redir",
            self.protocols,
            "--url",
            url,
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }
}

impl Fetcher for CurlFetcher {
    fn fetch(&self, url: &str, max_bytes: u64, sink: &mut dyn Write) -> anyhow::Result<()> {
        let mut child = Command::new("curl")
            .args(self.command_args(url))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("starting curl; it is needed to download the extraction files")?;
        let mut body = child.stdout.take().context("curl has no output pipe")?;
        let copied = copy_capped(&mut body, sink, max_bytes);
        if copied.is_err() {
            let _ = child.kill();
        }
        let mut reason = String::new();
        if let Some(stderr) = child.stderr.take() {
            let _ = stderr.take(STDERR_CAP_BYTES).read_to_string(&mut reason);
        }
        let status = child.wait().context("waiting for curl")?;
        copied?;
        if !status.success() {
            bail!("curl failed ({status}): {}", reason.trim());
        }
        Ok(())
    }
}

/// Copies until the end of `body`, failing as soon as more than `max_bytes`
/// have arrived.
fn copy_capped(body: &mut impl Read, sink: &mut dyn Write, max_bytes: u64) -> anyhow::Result<()> {
    let mut total: u64 = 0;
    let mut chunk = vec![0; READ_CHUNK_BYTES];
    loop {
        let read = body.read(&mut chunk)?;
        if read == 0 {
            return Ok(());
        }
        total += read as u64;
        if total > max_bytes {
            bail!("the download is larger than the {max_bytes} bytes allowed");
        }
        sink.write_all(&chunk[..read])?;
    }
}
