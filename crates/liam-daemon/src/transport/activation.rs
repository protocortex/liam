// SPDX-License-Identifier: AGPL-3.0-only
//! launchd socket activation: where the listener comes from at startup.
//!
//! Two ways `liamd serve` can get a listening socket, resolved ONCE here:
//! launchd hands one over (`Activated`), or this process binds its own
//! (`Bound`). Activation is what makes the socket exist before the daemon
//! does, so a client connecting is what starts the daemon, and two daemons
//! can never race to bind the same path.
//!
//! macOS only, BY TARGET rather than by cargo feature.
//! `launch_activate_socket` is a libSystem call with no equivalent
//! elsewhere, so a Linux build must not even see `raunch`; the
//! `[target.'cfg(target_os = "macos")'.dependencies]` entry in `Cargo.toml`
//! is what guarantees that. The non-macOS arm below is a real function that
//! `resolve` really calls, not a stub behind an `allow(dead_code)`, so the
//! Linux build stays warning-clean on its own merits.
//!
//! Linux activation (systemd, via `listenfd`) is deliberately deferred with
//! the rest of Linux support.

use std::path::Path;

use tokio::net::UnixListener;

use crate::transport::socket;
use crate::transport::ListenerSource;

/// The name of the entry under `Sockets` in the launchd plist, and the name
/// passed to `launch_activate_socket`. These two MUST agree or activation
/// silently degrades to binding our own socket, which defeats the entire
/// point: launchd would hold a socket nobody serves while the daemon serves
/// one nobody connects to.
///
/// `packaging/dev.protocortex.liamd.plist` carries the other half, and
/// `tests::the_plist_socket_name_matches_the_activation_constant` asserts
/// they match by reading the shipped file, rather than trusting a comment.
pub const LAUNCHD_SOCKET_NAME: &str = "Listener";

/// Resolves where the listener comes from, once, at startup: an activated
/// socket if a supervisor handed one over, otherwise one bound at `path`.
///
/// Falling back rather than failing is deliberate. `cargo run`, the test
/// suite, and anyone running `liamd serve` by hand all have no supervisor,
/// and they must keep working exactly as they did before activation
/// existed.
///
/// Mode dispatch (WU-9) is what calls this from `main`; until then it is
/// unreachable outside tests, hence `#[allow(dead_code)]` rather than
/// actually unused, the same convention `socket::bind` uses.
#[allow(dead_code)]
pub async fn resolve(path: &Path) -> anyhow::Result<ListenerSource> {
    match activated_listener()? {
        Some(listener) => {
            tracing::info!(
                socket_name = LAUNCHD_SOCKET_NAME,
                "using the socket launchd activated; this process does not own the path \
                 and will neither chmod nor unlink it"
            );
            Ok(ListenerSource::Activated(listener))
        }
        None => socket::bind(path).await,
    }
}

/// Asks launchd for the socket named [`LAUNCHD_SOCKET_NAME`].
///
/// `Ok(None)` means "nobody activated anything, go bind your own", which is
/// the ordinary developer path, not an error.
#[cfg(target_os = "macos")]
fn activated_listener() -> anyhow::Result<Option<UnixListener>> {
    use std::os::unix::io::FromRawFd;

    let mut fds = match raunch::activate_socket(LAUNCHD_SOCKET_NAME) {
        Ok(fds) => fds,
        // ESRCH: no launchd job owns this process. The `cargo run` and
        // `cargo test` path, and the overwhelmingly common one.
        Err(raunch::Error::NotManaged) => return Ok(None),
        // ENOENT: launchd DOES manage us, but the plist has no socket under
        // this name. Almost always the two-names-out-of-sync bug, so say so
        // loudly, then fall back so the daemon still comes up.
        Err(raunch::Error::NotInPlist) => {
            tracing::warn!(
                socket_name = LAUNCHD_SOCKET_NAME,
                "launchd manages this process but its plist declares no socket under this \
                 name; binding our own instead. Check the key under `Sockets` in \
                 dev.protocortex.liamd.plist matches LAUNCHD_SOCKET_NAME"
            );
            return Ok(None);
        }
        // EALREADY: this ran twice. `resolve` is documented as once-at-startup,
        // so a second call is a bug worth failing on rather than papering over
        // by binding a second, competing socket.
        Err(error @ raunch::Error::AlreadyActivated) => {
            return Err(anyhow::anyhow!(
                "socket {LAUNCHD_SOCKET_NAME} was already activated: {error}. \
                 Listener resolution must happen exactly once at startup"
            ));
        }
        Err(error) => {
            return Err(anyhow::anyhow!(
                "failed to activate launchd socket {LAUNCHD_SOCKET_NAME}: {error}"
            ));
        }
    };

    if fds.is_empty() {
        tracing::warn!(
            socket_name = LAUNCHD_SOCKET_NAME,
            "launchd returned no file descriptors for this socket; binding our own instead"
        );
        return Ok(None);
    }

    // A `SockPathName` socket yields exactly one descriptor. If launchd ever
    // hands over more, serve the first and take OWNERSHIP of the rest so they
    // close on drop: leaking them would hold the extra sockets open for the
    // life of the daemon.
    if fds.len() > 1 {
        tracing::warn!(
            socket_name = LAUNCHD_SOCKET_NAME,
            count = fds.len(),
            "launchd returned more than one descriptor; serving the first and closing the rest"
        );
    }
    let primary = fds.remove(0);
    for extra in fds {
        // SAFETY: `activate_socket` transfers ownership of every descriptor
        // it returns (it frees the array, not the descriptors). Wrapping in a
        // std listener and dropping it is how that ownership gets released.
        drop(unsafe { std::os::unix::net::UnixListener::from_raw_fd(extra) });
    }

    // SAFETY: `primary` came from `launch_activate_socket`, so it is a valid,
    // already-bound, already-listening Unix socket whose ownership has just
    // been transferred to this process. Nothing else holds it.
    let std_listener = unsafe { std::os::unix::net::UnixListener::from_raw_fd(primary) };
    std_listener.set_nonblocking(true).map_err(|source| {
        anyhow::anyhow!("failed to set the activated socket non-blocking: {source}")
    })?;
    let listener = UnixListener::from_std(std_listener)
        .map_err(|source| anyhow::anyhow!("failed to adopt the activated socket: {source}"))?;

    Ok(Some(listener))
}

/// No socket activation off macOS yet, so there is never an activated
/// listener and `resolve` always binds. A real function rather than a
/// `todo!` or an `allow(dead_code)` stub: `resolve` calls it on every
/// platform, which is what keeps the Linux build free of dead-code warnings.
#[cfg(not(target_os = "macos"))]
fn activated_listener() -> anyhow::Result<Option<UnixListener>> {
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped launchd job. Read from the repo rather than embedded, so
    /// the assertions below fail if the real file drifts.
    const PLIST: &str = include_str!("../../../../packaging/dev.protocortex.liamd.plist");

    /// Pulls the text of the first `<key>NAME</key><string>VALUE</string>`
    /// pair whose key matches, ignoring whitespace between the tags. A tiny
    /// reader beats adding a plist parser dependency for two assertions.
    fn string_value_for_key(plist: &str, key: &str) -> Option<String> {
        let key_tag = format!("<key>{key}</key>");
        let after_key = plist.split_once(&key_tag)?.1;
        let open = after_key.find("<string>")? + "<string>".len();
        let close = after_key.find("</string>")?;
        Some(after_key[open..close].trim().to_string())
    }

    #[test]
    fn the_plist_socket_name_matches_the_activation_constant() {
        // Given the shipped plist
        // When its Sockets dictionary is read
        // Then the key it declares is the exact name passed to
        // `activate_socket`. These drifting apart is silent at runtime, so
        // it is pinned here instead.
        let sockets = PLIST
            .split_once("<key>Sockets</key>")
            .expect("the plist must declare a Sockets dictionary")
            .1;
        let declared = sockets
            .split_once("<key>")
            .expect("the Sockets dictionary must declare a socket")
            .1
            .split_once("</key>")
            .expect("the socket key must be closed")
            .0
            .trim()
            .to_string();
        assert_eq!(
            declared, LAUNCHD_SOCKET_NAME,
            "the plist's socket key and LAUNCHD_SOCKET_NAME must match, or activation \
             silently falls back to binding our own socket"
        );
    }

    #[test]
    fn the_plist_declares_no_keepalive_so_activation_stays_on_demand() {
        // Given the shipped plist
        // When it is checked for a KeepAlive key
        // Then there is none. Measured against a real launchd job:
        // KeepAlive, including the dict form with SuccessfulExit=false,
        // starts the job at bootstrap and overrides RunAtLoad=false, which
        // turns on-demand activation into an eager launch. Socket
        // activation already restarts the daemon after a crash on the next
        // connection, so the key buys nothing and costs the behaviour the
        // whole Work Unit exists for.
        assert!(
            !PLIST.contains("<key>KeepAlive</key>"),
            "the plist must not declare KeepAlive: it overrides RunAtLoad=false and \
             starts the daemon eagerly instead of on the first client connection"
        );
    }

    #[test]
    fn the_plist_starts_the_daemon_on_demand_rather_than_at_load() {
        // Given the shipped plist
        // Then RunAtLoad is false, so launchd waits for a client rather
        // than starting the daemon when the job is bootstrapped.
        let after = PLIST
            .split_once("<key>RunAtLoad</key>")
            .expect("the plist must set RunAtLoad")
            .1;
        assert!(
            after.trim_start().starts_with("<false/>"),
            "RunAtLoad must be false, or the socket's on-demand start is bypassed"
        );
    }

    #[test]
    fn the_plist_socket_path_matches_the_configured_default() {
        // Given the shipped plist and the built-in config default
        let configured = crate::config::Config::default().socket_path;
        let sock_path_name =
            string_value_for_key(PLIST, "SockPathName").expect("the plist must set SockPathName");

        // When both are reduced to the part that has to agree: launchd does
        // not expand `~`, and it does not expand environment variables
        // either, so the plist ships an installer-substituted placeholder
        // rather than a literal home directory.
        let configured_tail = configured
            .trim_start_matches('~')
            .trim_start_matches('/')
            .to_string();

        // Then the plist points at the same place the daemon would bind.
        assert!(
            sock_path_name.ends_with(&configured_tail),
            "plist SockPathName {sock_path_name} must end with the configured socket path \
             {configured_tail}, or launchd holds a socket at a different path than the one \
             clients and the Bound fallback use"
        );
    }

    #[tokio::test]
    async fn with_no_supervisor_the_source_resolves_to_bound() {
        // Given no launchd job managing the test process, which is the case
        // under `cargo test` on every platform.
        //
        // A short `/tmp` path, deliberately NOT a `tempfile` tempdir, for
        // the same reason `socket::tests::TestPath` avoids one: on macOS the
        // system temp directory is long enough that a socket bound inside it
        // can exceed `sun_path`'s 104-byte limit.
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path =
            std::path::PathBuf::from(format!("/tmp/liam-act-{}-{unique}", std::process::id()));

        // When the listener source resolves
        let source = resolve(&path).await.expect("resolve must succeed");

        // Then it bound its own socket rather than claiming an activated
        // one, and it behaves exactly as WU-6's bind did.
        assert!(
            matches!(source, ListenerSource::Bound { .. }),
            "with no supervisor the source must be Bound"
        );
        assert!(path.exists(), "the Bound path must exist after resolve");

        drop(source);
        let _ = std::fs::remove_file(&path);
    }
}
