// SPDX-License-Identifier: AGPL-3.0-only
//! The Unix socket listener: path handling, binding, permissions, and the
//! accept loop that serves `MemoryServer` to however many clients connect.
//!
//! Shutdown, including unlinking the socket on the way out, is WU-6c's job;
//! nothing here hand-rolls it. Producer resolution is WU-7's; nothing here
//! looks at who is connecting, only that something did.

use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::path::Path;
use std::sync::Arc;

use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Semaphore;

use crate::mcp::MemoryServer;
use crate::transport::ListenerSource;

/// Owner-only file mode set on a freshly bound socket. The socket carries
/// full MCP tool access with no further authentication, so anyone who can
/// reach the file can act as any client; owner-only is what makes "reach
/// the file" mean "already this user."
const SOCKET_MODE: u32 = 0o600;

/// Smallest connection cap `accept_loop` will ever honour. A `Semaphore`
/// started with zero permits never issues one, so the loop's very first
/// permit acquire, which happens before it ever calls `accept()`, would
/// wait forever: a configured `max_connections` of 0 is floored to this
/// instead of taken literally, the same guard `MemoryServer::new` applies
/// to `max_concurrent_generations`.
const MIN_MAX_CONNECTIONS: usize = 1;

/// Ensures a parent directory exists, decides what to do about whatever is
/// already at `path` (see [`prepare_existing_path`] for the order that
/// matters), binds a fresh listener there, and locks its permissions down
/// to owner-only. Always returns [`ListenerSource::Bound`]: this process
/// bound the socket itself, so it is the one that owns the file.
///
/// Mode dispatch (WU-9) is what calls this from `main`; until then it is
/// unreachable outside tests, hence `#[allow(dead_code)]` rather than
/// actually unused.
#[allow(dead_code)]
pub async fn bind(path: &Path) -> anyhow::Result<ListenerSource> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent).await.map_err(|source| {
                anyhow::anyhow!(
                    "failed to create socket directory {}: {source}",
                    parent.display()
                )
            })?;
        }
    }

    prepare_existing_path(path).await?;

    let listener = UnixListener::bind(path).map_err(|source| {
        anyhow::anyhow!(
            "failed to bind socket at {}: {source} (check that the parent \
             directory is writable)",
            path.display()
        )
    })?;

    tokio::fs::set_permissions(path, std::fs::Permissions::from_mode(SOCKET_MODE))
        .await
        .map_err(|source| {
            anyhow::anyhow!(
                "bound socket at {} but failed to set owner-only permissions: {source}",
                path.display()
            )
        })?;

    Ok(ListenerSource::Bound {
        listener,
        path: path.to_path_buf(),
    })
}

/// Decides what to do about a path that may already exist, in the order
/// that keeps this safe: the file TYPE is checked before anything is ever
/// removed, never the other way around.
///
/// `socket_path` is operator-supplied (`liam.toml`), so a typo pointing it
/// at, say, a database file must refuse and leave that file alone rather
/// than deleting it because it happened to sit where a socket was
/// expected: unlinking a path just because something is there would be
/// destructive the moment that something is not actually ours. Only once
/// the path is confirmed to be a socket do we go on to ask whether
/// anything is listening on it, and only an unanswered (stale) socket is
/// ever unlinked; one that answers means a live daemon owns it, and this
/// process refuses to start rather than stealing its clients.
async fn prepare_existing_path(path: &Path) -> anyhow::Result<()> {
    // `symlink_metadata` (lstat), not `metadata` (stat): if `path` is a
    // symlink, its own file type is never "socket", so this refuses and
    // leaves it alone rather than following it into unlinking whatever it
    // points at.
    let metadata = match tokio::fs::symlink_metadata(path).await {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(anyhow::anyhow!(
                "failed to inspect existing path {}: {source}",
                path.display()
            ));
        }
    };

    if !metadata.file_type().is_socket() {
        anyhow::bail!(
            "refusing to start: {} already exists and is not a socket; remove it \
             yourself if that is intended, liamd will never delete it for you",
            path.display()
        );
    }

    if UnixStream::connect(path).await.is_ok() {
        anyhow::bail!(
            "refusing to start: another process is already listening on socket {}",
            path.display()
        );
    }

    // Confirmed a socket, and confirmed nothing answers it: a prior daemon
    // left this behind without cleaning up (dropping a `UnixListener` does
    // not unlink its path), so it is safe to replace.
    tokio::fs::remove_file(path).await.map_err(|source| {
        anyhow::anyhow!("failed to remove stale socket {}: {source}", path.display())
    })
}

/// Floors a configured connection cap so `accept_loop` can never wedge
/// itself waiting on a permit that can never be issued. See
/// [`MIN_MAX_CONNECTIONS`] for why 0 specifically is the dangerous value.
fn floor_max_connections(max_connections: usize) -> usize {
    if max_connections == 0 {
        tracing::warn!(
            "socket.max_connections was 0; clamping to {MIN_MAX_CONNECTIONS}, or the \
             listener would wait forever for a connection permit that could never be issued"
        );
        MIN_MAX_CONNECTIONS
    } else {
        max_connections
    }
}

/// Accepts connections forever, cloning `server` onto its own task per
/// connection so one client's error, or a clean disconnect, never touches
/// any other session. Only `listener.accept()` itself failing ends this
/// loop; a session's own error is caught and logged inside its task, never
/// propagated here.
///
/// `max_connections` bounds how many sessions may be open at once: without
/// it an unbounded accept loop can exhaust file descriptors, and each
/// session that ends up generating holds its own KV cache (measured around
/// 110MB), so this is a real resource bound, not a nicety. The permit is
/// acquired before `accept()` runs, so once the cap is reached further
/// clients queue in the kernel's own backlog instead of piling up as
/// accepted-but-unserved connections inside this process.
///
/// Mode dispatch (WU-9) is what calls this from `main`; until then it is
/// unreachable outside tests, hence `#[allow(dead_code)]` rather than
/// actually unused.
#[allow(dead_code)]
pub async fn accept_loop(
    source: ListenerSource,
    server: MemoryServer,
    max_connections: usize,
) -> anyhow::Result<()> {
    let permits = Arc::new(Semaphore::new(floor_max_connections(max_connections)));

    let ListenerSource::Bound { listener, path } = &source;
    tracing::info!(path = %path.display(), max_connections, "liamd socket listener started");

    loop {
        let permit = Arc::clone(&permits)
            .acquire_owned()
            .await
            .expect("the semaphore is never closed");
        let (stream, _addr) = listener.accept().await?;
        let server = server.clone();
        tokio::spawn(async move {
            let _permit = permit;
            if let Err(error) = handle_connection(stream, server).await {
                tracing::warn!(error = %error, "mcp session over the socket ended with an error");
            }
        });
    }
}

/// Runs one MCP session over an already-accepted connection. Errors (a
/// client that disconnects mid-handshake, a malformed first message, and
/// so on) are returned to the caller to log, never panicked on:
/// `accept_loop` spawns this per connection specifically so one session's
/// failure stays inside its own task and never reaches another.
async fn handle_connection(stream: UnixStream, server: MemoryServer) -> anyhow::Result<()> {
    use rmcp::ServiceExt;
    let running = server.serve(stream).await?;
    running.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use tokio::io::AsyncWriteExt;

    use super::*;

    /// Self-cleaning path under `/tmp`, short and unique per call.
    /// Deliberately NOT inside a `tempfile` tempdir: on macOS the system
    /// temp directory (`/var/folders/xx/.../T/`) is long enough that a
    /// socket bound inside it can exceed `sun_path`'s 104-byte limit and
    /// fail to bind with an opaque invalid-argument error that reads like
    /// a bug in the listener rather than a path-length problem. Unique per
    /// call so parallel test binaries, and leftovers from a crashed run,
    /// never collide.
    struct TestPath(PathBuf);

    impl TestPath {
        fn new() -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = PathBuf::from(format!("/tmp/liam-test-{}-{unique}", std::process::id()));
            Self(path)
        }
    }

    impl std::ops::Deref for TestPath {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestPath {
        fn drop(&mut self) {
            // Best-effort: whatever a test put here may already be gone
            // (replaced by a bind, or never created because the test
            // errored first), and a directory needs a recursive remove.
            if self.0.is_dir() {
                let _ = std::fs::remove_dir_all(&self.0);
            } else {
                let _ = std::fs::remove_file(&self.0);
            }
        }
    }

    /// Fresh in-memory `MemoryServer` with the mock embedder/reranker/llm,
    /// wired the same way `mcp`'s own tests do. Good enough for the socket
    /// layer's tests, which only need something `rmcp::ServiceExt::serve`
    /// can be called on, not real model output.
    async fn test_server() -> MemoryServer {
        let store = liam_store::DefaultGraph::open(":memory:", liam_store::GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        MemoryServer::new(
            Arc::new(store),
            Arc::new(liam_model::MockEmbedder::new(8)),
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
            30,
            false,
            8192,
            1,
        )
    }

    #[tokio::test]
    async fn binding_with_no_existing_file_sets_owner_only_permissions() {
        // Given no socket file
        let path = TestPath::new();

        // When serve starts
        let source = bind(&path).await.expect("bind must succeed");
        let ListenerSource::Bound {
            path: bound_path, ..
        } = &source;
        assert_eq!(bound_path, &*path);

        // Then it binds and the file mode is owner-only, read back rather
        // than assumed from the umask.
        let mode = std::fs::metadata(&*path)
            .expect("socket file must exist after bind")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            mode, 0o600,
            "socket mode should be owner-only, got {mode:o}"
        );
    }

    #[tokio::test]
    async fn a_stale_socket_with_no_listener_is_replaced() {
        // Given a stale socket file with no listener: bind once and drop
        // the listener without unlinking, which is exactly what a crashed
        // daemon leaves behind (dropping a `UnixListener` does not unlink
        // its path).
        let path = TestPath::new();
        let stale = UnixListener::bind(&*path).expect("bind stale listener");
        drop(stale);

        // When serve starts
        let source = bind(&path).await;

        // Then it replaces the file and binds: no error, and the fresh
        // listener genuinely accepts a connection.
        let source = source.expect("a stale socket must be replaced, not refused");
        drop(
            UnixStream::connect(&*path)
                .await
                .expect("the freshly bound socket must accept a connection"),
        );
        drop(source);
    }

    #[tokio::test]
    async fn a_live_socket_is_refused_and_the_original_survives() {
        // Given a LIVE listener
        let path = TestPath::new();
        let first = bind(&path).await.expect("first bind must succeed");

        // When a second serve starts
        let second = bind(&path).await;

        // Then it errors, naming the socket as already in use
        let message = second
            .err()
            .expect("a live socket must refuse a second bind")
            .to_string();
        assert!(
            message.contains(&path.display().to_string()),
            "message should name the socket path: {message}"
        );

        // And the original socket still exists and still works: survival
        // asserted directly, not inferred from the error alone.
        let metadata = std::fs::symlink_metadata(&*path).expect("original socket must survive");
        assert!(metadata.file_type().is_socket());
        drop(
            UnixStream::connect(&*path)
                .await
                .expect("the original socket must still accept connections"),
        );

        drop(first);
    }

    #[tokio::test]
    async fn a_regular_file_at_the_socket_path_is_refused_and_left_untouched() {
        // Given the path exists as a REGULAR FILE
        let path = TestPath::new();
        std::fs::write(&*path, b"not a socket, do not touch me").expect("write plain file");

        // When serve starts
        let result = bind(&path).await;

        // Then it errors and the file still exists afterwards, with its
        // contents intact.
        assert!(
            result.is_err(),
            "a regular file must never be treated as a socket"
        );
        let contents = std::fs::read(&*path).expect("file must survive the failed bind");
        assert_eq!(contents, b"not a socket, do not touch me");
    }

    #[tokio::test]
    async fn a_directory_at_the_socket_path_is_refused_and_not_removed() {
        // Given the path exists as a DIRECTORY
        let path = TestPath::new();
        std::fs::create_dir(&*path).expect("create directory");

        // When serve starts
        let result = bind(&path).await;

        // Then it errors without removing it
        assert!(
            result.is_err(),
            "a directory must never be treated as a socket"
        );
        assert!(path.is_dir(), "the directory must survive the failed bind");
    }

    #[tokio::test]
    async fn an_unwritable_parent_directory_names_the_path_in_the_error() {
        // Given an unwritable parent directory (a short path under /tmp,
        // not a `tempfile` tempdir, for the same sun_path reason `TestPath`
        // avoids one).
        let parent = TestPath::new();
        std::fs::create_dir_all(&*parent).expect("create parent dir");
        std::fs::set_permissions(&*parent, std::fs::Permissions::from_mode(0o500))
            .expect("remove write permission from parent");
        let path = parent.join("liamd.sock");

        // When serve starts
        let result = bind(&path).await;

        // Restore write permission before the temp dir's own cleanup runs.
        std::fs::set_permissions(&*parent, std::fs::Permissions::from_mode(0o700))
            .expect("restore write permission for cleanup");

        // Then the error names the path rather than surfacing a bare,
        // confusing ENOENT/EACCES from `bind` with no context.
        let message = result
            .err()
            .expect("binding under an unwritable parent must fail")
            .to_string();
        assert!(
            message.contains(&path.display().to_string()),
            "message should name the socket path: {message}"
        );
    }

    #[test]
    fn a_max_connections_of_zero_is_floored_to_one() {
        // Given a configured max_connections of 0
        // Then the accept loop's cap is floored, not taken literally,
        // since a semaphore with zero permits would wait forever.
        assert_eq!(floor_max_connections(0), MIN_MAX_CONNECTIONS);
    }

    #[test]
    fn a_positive_max_connections_is_honoured_unchanged() {
        assert_eq!(floor_max_connections(7), 7);
    }

    #[tokio::test]
    async fn handle_connection_errors_without_panicking_when_the_peer_drops_immediately() {
        // Arrange: a connected pair standing in for an accepted connection;
        // the peer end drops before any MCP handshake happens, exactly what
        // a client disconnecting immediately looks like from the server's
        // side.
        let (server_side, client_side) = UnixStream::pair().expect("socket pair");
        drop(client_side);
        let server = test_server().await;

        // Act
        let result = handle_connection(server_side, server).await;

        // Assert: the session ends with an error the accept loop can log,
        // not a panic that would take the whole listener down with it.
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn the_accept_loop_keeps_serving_other_clients_after_one_drops() {
        // Arrange: a real bound listener, served in the background.
        let path = TestPath::new();
        let source = bind(&path).await.expect("bind");
        let server = test_server().await;
        tokio::spawn(accept_loop(source, server, 4));

        // A client already connected before the drop, standing in for the
        // "other" session the disconnect must not disturb.
        let mut other = UnixStream::connect(&*path)
            .await
            .expect("the first client must connect");

        // When a second, connected client drops immediately
        let dropped = UnixStream::connect(&*path)
            .await
            .expect("the second client must connect");
        drop(dropped);

        // Then the daemon keeps serving others: the earlier connection is
        // still writable, and the listener still accepts fresh clients.
        other
            .write_all(b"\n")
            .await
            .expect("the other connection must still be alive after the drop");
        drop(
            UnixStream::connect(&*path)
                .await
                .expect("the listener must keep accepting after a dropped client"),
        );
    }
}
