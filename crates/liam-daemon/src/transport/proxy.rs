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

use tokio::net::UnixStream;

/// Connects to the daemon at `socket_path` and shuttles bytes between it and
/// this process's stdin/stdout until either side finishes.
///
/// Logs to stderr only, because stdout IS the protocol stream here and
/// anything else written there corrupts the client's framing.
pub async fn run(socket_path: &Path) -> anyhow::Result<()> {
    let mut socket = UnixStream::connect(socket_path).await.map_err(|source| {
        anyhow::anyhow!(
            "no liam daemon is listening at {}: {source}\n\
             Start one with `liamd serve`, or let launchd start it on demand \
             (see packaging/dev.protocortex.liamd.plist).",
            socket_path.display()
        )
    })?;

    tracing::debug!(path = %socket_path.display(), "proxying stdio to the daemon socket");

    // `stdin`/`stdout` rather than a duplex handle: they are separate
    // descriptors, and `copy_bidirectional` wants one read-write thing, so
    // they get joined here.
    let mut client = tokio::io::join(tokio::io::stdin(), tokio::io::stdout());

    match tokio::io::copy_bidirectional(&mut client, &mut socket).await {
        Ok((to_daemon, to_client)) => {
            tracing::debug!(to_daemon, to_client, "proxy finished");
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

    /// The shuttle itself, over a real socket pair. `run` reads this
    /// process's stdin, which a test cannot drive, so the copy is exercised
    /// directly against the same `copy_bidirectional` call.
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
