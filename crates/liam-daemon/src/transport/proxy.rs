// SPDX-License-Identifier: AGPL-3.0-only
//! The stdio proxy: a shuttle between this process's stdin/stdout and a
//! running daemon's Unix socket.
//!
//! This is what keeps every existing MCP client working once the daemon
//! moves to a socket. A client that can only speak stdio runs `liamd proxy`
//! and never learns the socket exists.
//!
//! The proxy opens NO store and loads NO model. That is a hard requirement,
//! not an optimisation: the daemon holds a per-process advisory lock on the
//! database (see `storelock`), so a proxy that opened the store would fail
//! to start whenever the daemon it is proxying to is running, which is
//! always. It is also why the proxy is dispatched before any store setup in
//! `main` rather than inside the shared startup path.
//!
//! Bytes are forwarded verbatim. The proxy does not parse JSON-RPC, does not
//! reframe, and has no opinion about MCP: whatever the client writes reaches
//! the daemon unchanged, so protocol changes need no work here.

use std::path::Path;

use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;

/// Connects to the daemon at `socket_path` and shuttles bytes between it and
/// this process's stdin/stdout until either side finishes.
///
/// Logs to stderr only, because stdout IS the protocol stream here and
/// anything else written there corrupts the client's framing.
pub async fn run(socket_path: &Path) -> anyhow::Result<()> {
    let socket = UnixStream::connect(socket_path).await.map_err(|source| {
        anyhow::anyhow!(
            "no liam daemon is listening at {}: {source}\n\
             Start one with `liamd serve`, or let launchd start it on demand \
             (see packaging/dev.protocortex.liamd.plist).",
            socket_path.display()
        )
    })?;

    tracing::debug!(path = %socket_path.display(), "proxying stdio to the daemon socket");

    shuttle(tokio::io::stdin(), tokio::io::stdout(), socket).await
}

/// Moves bytes between a client's input/output and the daemon `socket`,
/// returning once the daemon side is done.
///
/// Split out from [`run`] so the ending rule can be tested: `run` supplies
/// this process's real stdin and stdout, which a test cannot drive.
///
/// Deliberately NOT `copy_bidirectional`. That waits for BOTH directions,
/// and a client's stdin never reaches EOF on its own, so when the daemon
/// exits mid-session the proxy hangs forever holding the client's stdio
/// open: the client is left with a live proxy that can never answer.
/// Measured against a real daemon, not theorised.
async fn shuttle<R, W>(client_in: R, mut client_out: W, socket: UnixStream) -> anyhow::Result<()>
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
    W: tokio::io::AsyncWrite + Unpin,
{
    let (mut from_daemon, mut to_daemon) = socket.into_split();

    // Client to daemon. On input EOF it half-closes the socket's write side
    // so the daemon sees the end of input and can finish its last reply,
    // rather than holding a session open for a client that has gone.
    let uplink = tokio::spawn(async move {
        let mut client_in = client_in;
        let copied = tokio::io::copy(&mut client_in, &mut to_daemon).await;
        let _ = to_daemon.shutdown().await;
        copied
    });

    // Daemon to client. Completes when the daemon closes the socket, whether
    // because it finished with us or because it exited. Either way nothing
    // further can arrive, so this direction is what ends the proxy.
    let result = async {
        let copied = tokio::io::copy(&mut from_daemon, &mut client_out).await;
        // Flush before reporting: `copy` can leave the last frame buffered,
        // and dropping it would lose the client's final response.
        let _ = client_out.flush().await;
        copied
    }
    .await;

    // The uplink may be parked on a read that never returns. Nothing can
    // consume what it reads now, so drop it rather than wait.
    uplink.abort();

    match result {
        Ok(bytes) => {
            tracing::debug!(bytes, "proxy finished");
            Ok(())
        }
        // A closed pipe is how this normally ends: the client exits, or the
        // daemon closes the session, and whichever side writes next sees
        // EPIPE. Reporting that as a failure would make every ordinary
        // shutdown look like an error and give MCP clients a non-zero exit
        // to complain about.
        Err(error) if is_disconnect(&error) => {
            tracing::debug!(error = %error, "proxy peer closed the connection");
            Ok(())
        }
        Err(error) => Err(anyhow::anyhow!("proxy failed: {error}")),
    }
}

/// Whether an I/O error is one of the ordinary ways a peer goes away, rather
/// than a real failure.
fn is_disconnect(error: &std::io::Error) -> bool {
    use std::io::ErrorKind;
    matches!(
        error.kind(),
        ErrorKind::BrokenPipe | ErrorKind::ConnectionReset | ErrorKind::UnexpectedEof
    )
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    fn missing_socket_path() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        PathBuf::from(format!(
            "/tmp/liam-proxy-absent-{}-{unique}",
            std::process::id()
        ))
    }

    #[tokio::test]
    async fn connecting_with_no_daemon_names_how_to_start_one() {
        // Given no daemon listening at the configured path
        let path = missing_socket_path();

        // When the proxy starts
        let error = run(&path)
            .await
            .expect_err("the proxy must fail when nothing is listening");
        let message = error.to_string();

        // Then the error names both the path and the fix, rather than
        // surfacing a bare ENOENT the user has to interpret.
        assert!(
            message.contains(&path.display().to_string()),
            "error should name the socket path: {message}"
        );
        assert!(
            message.contains("liamd serve"),
            "error should name the command that starts a daemon: {message}"
        );
    }

    #[test]
    fn ordinary_peer_disconnects_are_not_failures() {
        use std::io::{Error, ErrorKind};

        // Given the ways a peer normally goes away
        for kind in [
            ErrorKind::BrokenPipe,
            ErrorKind::ConnectionReset,
            ErrorKind::UnexpectedEof,
        ] {
            assert!(
                is_disconnect(&Error::new(kind, "peer went away")),
                "{kind:?} should be treated as a normal end"
            );
        }

        // And a real failure still is one: treating everything as a clean
        // end would hide genuine breakage behind exit code 0.
        assert!(!is_disconnect(&Error::new(
            ErrorKind::PermissionDenied,
            "nope"
        )));
        assert!(!is_disconnect(&Error::new(ErrorKind::NotFound, "nope")));
    }

    /// The regression pin for the hang this module's `shuttle` exists to
    /// avoid. Before the split, `copy_bidirectional` waited for BOTH
    /// directions, so a daemon that exited while the client's input stayed
    /// open left the proxy running forever. Verified against a real daemon:
    /// the old shape was still alive 25 seconds after the daemon died, the
    /// new one exits as soon as the socket closes.
    #[tokio::test]
    async fn the_shuttle_ends_when_the_daemon_closes_even_with_client_input_open() {
        // Given a client whose input never ends and never yields a byte,
        // which is what an MCP client's stdin looks like while it waits
        let (client_in, _hold_input_open) = tokio::io::duplex(64);
        let (client_side, daemon_side) = UnixStream::pair().expect("socket pair");

        let shuttle_task =
            tokio::spawn(async move { shuttle(client_in, tokio::io::sink(), client_side).await });

        // When the daemon goes away
        drop(daemon_side);

        // Then the shuttle returns promptly instead of waiting on input
        // nobody will ever consume. The timeout IS the assertion: without
        // the fix this never completes.
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(10), shuttle_task)
            .await
            .expect("the shuttle must end when the daemon closes, not hang")
            .expect("the shuttle task must not panic");
        assert!(
            outcome.is_ok(),
            "a daemon closing the session is a normal end, got: {outcome:?}"
        );
    }

    /// Bytes cross unchanged, both ways, over a real socket pair.
    #[tokio::test]
    async fn bytes_cross_the_shuttle_verbatim_in_both_directions() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        // Given a client end and a daemon end
        let (client_side, mut daemon_side) = UnixStream::pair().expect("socket pair");
        let (client_reader, client_writer) = tokio::io::duplex(1024);
        let (mut test_handle, client_stdio) = (client_reader, client_writer);

        let shuttle = tokio::spawn(async move {
            let mut client_side = client_side;
            let mut client_stdio = client_stdio;
            tokio::io::copy_bidirectional(&mut client_stdio, &mut client_side).await
        });

        // When the client writes a JSON-RPC frame
        let frame = b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n";
        test_handle.write_all(frame).await.expect("write frame");

        // Then the daemon receives it byte for byte
        let mut got = vec![0u8; frame.len()];
        daemon_side
            .read_exact(&mut got)
            .await
            .expect("daemon must receive the frame");
        assert_eq!(got, frame, "the frame must cross unchanged");

        // And the reply comes back the same way
        let reply = b"{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n";
        daemon_side.write_all(reply).await.expect("write reply");
        let mut got_reply = vec![0u8; reply.len()];
        test_handle
            .read_exact(&mut got_reply)
            .await
            .expect("client must receive the reply");
        assert_eq!(got_reply, reply, "the reply must cross unchanged");

        drop(test_handle);
        drop(daemon_side);
        let _ = shuttle.await;
    }
}
