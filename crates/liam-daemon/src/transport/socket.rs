// SPDX-License-Identifier: AGPL-3.0-only
//! The Unix socket listener: path handling, binding, permissions, and the
//! accept loop that serves `MemoryServer` to however many clients connect.
//!
//! Producer resolution itself (`producer::resolve`, a pure
//! client-name-to-id lookup) lives in `mcp::producer`; the wiring that calls
//! it once per connection, right after the MCP `initialize` handshake, is
//! here. See `handle_connection` for exactly where and why.
//!
//! Shutdown POLICY lives in `transport::shutdown`, not here: this module
//! decides when to stop accepting (the cancellation token in `accept_loop`)
//! and then hands off to `shutdown::drain` and
//! `shutdown::unlink_owned_socket` for the drain and the unlink. Keeping the
//! "is this socket ours to delete" decision in one place, behind
//! `ListenerSource::owned_path`, is what stops a future edit here from
//! unlinking a path launchd owns.

use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::config::ProducersConfig;
use crate::mcp::producer;
use crate::mcp::MemoryServer;
use crate::transport::shutdown;
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
/// Reached through [`crate::transport::activation::resolve`], which only
/// calls this when no supervisor handed a socket over. `resolve` in turn
/// waits on WU-9's mode dispatch to be called from `main`, so this is still
/// unreachable in the production build, hence the allow.
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

/// Accepts connections until `cancel` fires, cloning `server` onto its own
/// task per connection so one client's error, or a clean disconnect, never
/// touches any other session. Two things end this loop: cancellation, which
/// is the ordinary shutdown path, and `listener.accept()` itself failing,
/// which is not. A session's own error is caught and logged inside its
/// task, never propagated here. Either way the loop still drains and still
/// unlinks an owned socket on the way out.
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
///
/// `producers` is the `[producers]` table each connection's `initialize`
/// handshake gets resolved against (see `handle_connection`); an `Arc`
/// because it is shared, read-only, across however many connection tasks
/// are alive at once, the same reason `server`'s own `Arc` fields are Arcs.
///
/// `cancel` is what stops the loop. Cancelling it means "stop accepting",
/// never "kill the live sessions": those get the drain window in
/// [`shutdown::drain`] to finish on their own first.
///
/// `Ok(())` means the loop stopped because it was asked to, and that the
/// socket has been unlinked if this process owned it. It does NOT promise
/// every session finished: a drain that outruns `drain_deadline` aborts
/// what is left, logs a warning, and still returns `Ok(())`, because being
/// told to stop is not an error no matter how the sessions ended. Only a
/// failed `accept` comes back as `Err`.
#[allow(dead_code)]
pub async fn accept_loop(
    source: ListenerSource,
    server: MemoryServer,
    max_connections: usize,
    producers: Arc<ProducersConfig>,
    cancel: CancellationToken,
    drain_deadline: Duration,
) -> anyhow::Result<()> {
    let permits = Arc::new(Semaphore::new(floor_max_connections(max_connections)));
    let listener = source.listener();
    match source.owned_path() {
        Some(path) => {
            tracing::info!(path = %path.display(), max_connections, "liamd socket listener started")
        }
        None => tracing::info!(
            max_connections,
            "liamd listener started on an activated socket"
        ),
    }

    // Sessions live in a `JoinSet` rather than being detached, so shutdown
    // can wait for them and, past the deadline, genuinely abort them. A
    // detached `tokio::spawn` gives no handle to do either.
    let mut sessions: JoinSet<()> = JoinSet::new();
    let mut accept_error = None;

    loop {
        let permit = tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            permit = Arc::clone(&permits).acquire_owned() => {
                permit.expect("the semaphore is never closed")
            }
        };

        let accepted = tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            accepted = listener.accept() => accepted,
        };

        let (stream, _addr) = match accepted {
            Ok(accepted) => accepted,
            Err(error) => {
                // The listener itself failed, so this loop cannot continue.
                // Fall through to the drain rather than returning here: live
                // sessions still deserve their window, and an owned socket
                // still needs unlinking.
                accept_error = Some(error);
                break;
            }
        };

        let server = server.clone();
        let producers = Arc::clone(&producers);
        sessions.spawn(async move {
            let _permit = permit;
            if let Err(error) = handle_connection(stream, server, producers).await {
                tracing::warn!(error = %error, "mcp session over the socket ended with an error");
            }
        });

        // Reap sessions that already finished so the set does not grow for
        // the life of the daemon.
        while sessions.try_join_next().is_some() {}
    }

    // Whether the drain finished on its own or ran out of time is
    // deliberately not propagated: `drain` already logs the abort, and a
    // stuck client is not a reason to report the daemon's own shutdown as
    // failed. See this function's doc for what `Ok(())` does and does not
    // promise.
    let _drained = shutdown::drain(&mut sessions, drain_deadline).await;
    shutdown::unlink_owned_socket(&source);

    match accept_error {
        Some(error) => Err(anyhow::anyhow!(
            "socket listener stopped accepting: {error}"
        )),
        None => Ok(()),
    }
}

/// Runs one MCP session over an already-accepted connection. Errors (a
/// client that disconnects mid-handshake, a malformed first message, and
/// so on) are returned to the caller to log, never panicked on:
/// `accept_loop` spawns this per connection specifically so one session's
/// failure stays inside its own task and never reaches another.
///
/// This is where WU-8's wiring lives: `serve` does not return a
/// `RunningService` until the MCP `initialize` handshake is complete (rmcp
/// 3.1.2's `serve_server_with_ct_inner` sends the initialize response
/// before constructing it), so the connecting client's declared name is
/// available the instant `serve` resolves. `producer::resolve` turns that
/// name into a canonical producer id via `producers`, and
/// `MemoryServer::set_producer` stamps it on the very `MemoryServer`
/// instance `rmcp` is about to dispatch every tool call on for this
/// connection, through `RunningService::service`'s `&MemoryServer`.
///
/// RESIDUAL RACE, documented rather than hidden: the MCP protocol forbids a
/// client from sending a tool call before it receives the initialize
/// response, so this window is closed in practice, but a client that
/// pipelines a call immediately after receiving that response could in
/// principle reach `remember`/`recall` before `set_producer` below runs and
/// be stamped with the fallback producer instead. Nothing in the spec
/// requires that window to be closed, and closing it would mean taking a
/// lock on every tool call, on every connection, forever, to guard a
/// handshake that happens once. That trade is not made here.
async fn handle_connection(
    stream: UnixStream,
    server: MemoryServer,
    producers: Arc<ProducersConfig>,
) -> anyhow::Result<()> {
    use rmcp::ServiceExt;
    let running = server.serve(stream).await?;

    let client_name = running
        .peer_info()
        .map(|info| info.client_info.name.clone());
    running
        .service()
        .set_producer(producer::resolve(client_name.as_deref(), &producers));

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

    /// Same shape as `test_server`, but FILE-backed at `db_path`. WU-8's
    /// acceptance tests need this, not `:memory:`: they assert `producer`
    /// through a second, independent connection to the same database (see
    /// `node_producers` below), and each `:memory:` connection is its own
    /// private database, so a second connection would see an empty store.
    async fn test_server_on(db_path: &Path) -> MemoryServer {
        let store = liam_store::DefaultGraph::open(
            db_path.to_str().expect("temp db path is valid utf-8"),
            liam_store::GraphConfig::new(8),
        )
        .await
        .expect("open file-backed store");
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

    /// A `[producers]` table with no entries: every connection resolves to
    /// the fallback id. Good enough for tests that only need the accept loop
    /// to run, not to assert on a specific producer.
    fn default_producers() -> Arc<ProducersConfig> {
        Arc::new(ProducersConfig::default())
    }

    #[tokio::test]
    async fn binding_with_no_existing_file_sets_owner_only_permissions() {
        // Given no socket file
        let path = TestPath::new();

        // When serve starts
        let source = bind(&path).await.expect("bind must succeed");
        // `bind` always reports a socket this process owns, never an
        // activated one: activation is resolved before bind is ever reached.
        assert_eq!(
            source.owned_path(),
            Some(&*path),
            "bind must report the path it owns"
        );

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
        let result = handle_connection(server_side, server, default_producers()).await;

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
        tokio::spawn(accept_loop(
            source,
            server,
            4,
            default_producers(),
            CancellationToken::new(),
            shutdown::DEFAULT_DRAIN_DEADLINE,
        ));

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

    /// A `[producers]` table mapping two distinct declared MCP client names
    /// to two distinct, easily-recognised producer ids. Shared by both
    /// acceptance tests below so the mapping itself is asserted only once,
    /// here, by construction.
    fn two_client_producers() -> Arc<ProducersConfig> {
        Arc::new(ProducersConfig {
            unknown_id: "unknown".to_string(),
            clients: [
                ("alice-app".to_string(), "alice".to_string()),
                ("bob-app".to_string(), "bob".to_string()),
            ]
            .into_iter()
            .collect(),
        })
    }

    /// Declares a fixed name at `initialize` and nothing else. WU-8's whole
    /// point is that the daemon attributes writes to WHICHEVER client made
    /// them, so the acceptance test needs two clients that are genuinely
    /// distinguishable at the MCP protocol level, not just two connections.
    #[derive(Clone)]
    struct NamedClient {
        name: &'static str,
    }

    impl rmcp::ClientHandler for NamedClient {
        fn get_info(&self) -> rmcp::model::ClientInfo {
            rmcp::model::InitializeRequestParams::new(
                rmcp::model::ClientCapabilities::default(),
                rmcp::model::Implementation::new(self.name, "0.0.0"),
            )
        }
    }

    /// Connects a `NamedClient` to the daemon's socket and completes the MCP
    /// handshake, returning the running client session.
    async fn connect(
        socket_path: &Path,
        name: &'static str,
    ) -> rmcp::service::RunningService<rmcp::RoleClient, NamedClient> {
        use rmcp::ServiceExt;
        let stream = UnixStream::connect(socket_path)
            .await
            .expect("client connects to the daemon's socket");
        NamedClient { name }
            .serve(stream)
            .await
            .expect("client MCP handshake succeeds")
    }

    /// Calls `tool` with `arguments` (a JSON object) over `peer` and returns
    /// the tool's text content, the same shape `remember`/`recall` return.
    async fn call_tool(
        peer: &rmcp::service::Peer<rmcp::RoleClient>,
        tool: &str,
        arguments: serde_json::Value,
    ) -> String {
        let arguments = match arguments {
            serde_json::Value::Object(map) => map,
            other => panic!("tool arguments must be a JSON object, got {other:?}"),
        };
        let result = peer
            .call_tool(
                rmcp::model::CallToolRequestParams::new(tool.to_string()).with_arguments(arguments),
            )
            .await
            .unwrap_or_else(|error| panic!("{tool} call failed: {error}"));
        result
            .content
            .first()
            .and_then(|block| block.as_text())
            .map(|text| text.text.clone())
            .unwrap_or_else(|| panic!("{tool} result carried no text content"))
    }

    fn remember_args(label: &str, content: &str) -> serde_json::Value {
        serde_json::json!({
            "kind": "fact",
            "label": label,
            "content": content,
            "scope": null,
            "subject": null,
        })
    }

    fn recall_args(query: &str) -> serde_json::Value {
        serde_json::json!({
            "query": query,
            "kind": null,
            "scope": null,
            "k": 10,
        })
    }

    /// Reads back every node's `(label, producer)` through a FRESH
    /// connection to `db_path`. Mirrors `mcp`'s own
    /// `remember_stamps_the_servers_producer_on_the_written_node` test: the
    /// daemon's own connection sits behind the write lock, and `producer` is
    /// deliberately absent from `Hit`, so a raw query through a second
    /// connection is the only way to assert it from outside.
    async fn node_producers(db_path: &Path) -> std::collections::HashMap<String, String> {
        use liam_store::{Backend, DefaultBackend};
        let raw = DefaultBackend::open(db_path.to_str().expect("utf-8 db path"), 1)
            .await
            .expect("open a fresh connection to the same database file");
        raw.query("SELECT label, producer FROM nodes", &[])
            .await
            .expect("query nodes")
            .into_iter()
            .map(|row| {
                (
                    row.get_string(0).expect("label column"),
                    row.get_string(1).expect("producer column"),
                )
            })
            .collect()
    }

    /// Best-effort cleanup of a libSQL database file and its WAL sidecars
    /// (`-wal`, `-shm`), which a plain `TestPath` does not know about.
    fn remove_db_file(db_path: &Path) {
        let db_path_str = db_path.to_str().expect("utf-8 db path");
        let _ = std::fs::remove_file(db_path);
        let _ = std::fs::remove_file(format!("{db_path_str}-wal"));
        let _ = std::fs::remove_file(format!("{db_path_str}-shm"));
    }

    /// The milestone's acceptance test: two MCP clients, declaring different
    /// names, sharing one daemon over one socket. Both write, both read, and
    /// each node comes back stamped with the producer of whoever actually
    /// wrote it, not the connection that happens to read it back.
    #[tokio::test]
    async fn two_named_clients_share_one_daemon_and_each_write_is_attributed_correctly() {
        // Given a daemon serving one socket, backed by one file database,
        // and two MCP clients connected to it declaring different names.
        let socket_path = TestPath::new();
        let db_path = TestPath::new();
        let server = test_server_on(&db_path).await;
        let source = bind(&socket_path).await.expect("bind");
        tokio::spawn(accept_loop(
            source,
            server,
            4,
            two_client_producers(),
            CancellationToken::new(),
            shutdown::DEFAULT_DRAIN_DEADLINE,
        ));

        let alice = connect(&socket_path, "alice-app").await;
        let bob = connect(&socket_path, "bob-app").await;

        // When both write
        let alice_write = call_tool(
            alice.peer(),
            "remember",
            remember_args("alice-node", "alice wrote this memory"),
        )
        .await;
        assert!(
            alice_write.starts_with("remembered "),
            "alice's write failed: {alice_write}"
        );
        let bob_write = call_tool(
            bob.peer(),
            "remember",
            remember_args("bob-node", "bob wrote this memory"),
        )
        .await;
        assert!(
            bob_write.starts_with("remembered "),
            "bob's write failed: {bob_write}"
        );

        // And then both read
        let alice_view = call_tool(alice.peer(), "recall", recall_args("memory")).await;
        let bob_view = call_tool(bob.peer(), "recall", recall_args("memory")).await;

        // Then each sees both nodes, from either side of the connection.
        assert!(
            alice_view.contains("alice-node") && alice_view.contains("bob-node"),
            "alice should see both nodes, got: {alice_view}"
        );
        assert!(
            bob_view.contains("alice-node") && bob_view.contains("bob-node"),
            "bob should see both nodes, got: {bob_view}"
        );

        // And each node records the producer of whoever actually wrote it,
        // not "unknown" and not the other client's id. This is the claim
        // the whole milestone rests on.
        let producers = node_producers(&db_path).await;
        assert_eq!(
            producers.get("alice-node").map(String::as_str),
            Some("alice"),
            "alice-node should be attributed to alice, got: {producers:?}"
        );
        assert_eq!(
            producers.get("bob-node").map(String::as_str),
            Some("bob"),
            "bob-node should be attributed to bob, got: {producers:?}"
        );

        remove_db_file(&db_path);
    }

    /// Given both clients writing at the same time, when all writes
    /// complete, then none is lost: every label lands exactly once.
    #[tokio::test]
    async fn concurrent_writes_from_both_named_clients_are_not_lost() {
        const PER_CLIENT: usize = 5;

        // Arrange: the same one-daemon, two-client setup as above.
        let socket_path = TestPath::new();
        let db_path = TestPath::new();
        let server = test_server_on(&db_path).await;
        let source = bind(&socket_path).await.expect("bind");
        tokio::spawn(accept_loop(
            source,
            server,
            8,
            two_client_producers(),
            CancellationToken::new(),
            shutdown::DEFAULT_DRAIN_DEADLINE,
        ));

        let alice = connect(&socket_path, "alice-app").await;
        let bob = connect(&socket_path, "bob-app").await;
        let alice_peer = alice.peer().clone();
        let bob_peer = bob.peer().clone();

        // When both clients write concurrently
        let mut writes = Vec::new();
        for i in 0..PER_CLIENT {
            let peer = alice_peer.clone();
            writes.push(tokio::spawn(async move {
                call_tool(
                    &peer,
                    "remember",
                    remember_args(&format!("alice-{i}"), "alice wrote this, concurrently"),
                )
                .await
            }));
            let peer = bob_peer.clone();
            writes.push(tokio::spawn(async move {
                call_tool(
                    &peer,
                    "remember",
                    remember_args(&format!("bob-{i}"), "bob wrote this, concurrently"),
                )
                .await
            }));
        }
        for write in writes {
            let outcome = write.await.expect("write task must not panic");
            assert!(
                outcome.starts_with("remembered "),
                "a concurrent write failed: {outcome}"
            );
        }

        // Then none is lost: every one of the PER_CLIENT * 2 distinct labels
        // landed exactly once. A lost write, or two writes colliding onto
        // the same row, would both show up as fewer than PER_CLIENT * 2
        // entries here.
        let producers = node_producers(&db_path).await;
        assert_eq!(
            producers.len(),
            PER_CLIENT * 2,
            "expected {} nodes, found {}: {producers:?}",
            PER_CLIENT * 2,
            producers.len()
        );

        // And every one is attributed to the client that actually wrote it.
        // Counting rows alone would still pass if the two connections'
        // producers bled into each other under load, which is the one
        // failure concurrency can introduce here that serial attribution
        // testing cannot reach.
        for i in 0..PER_CLIENT {
            assert_eq!(
                producers.get(&format!("alice-{i}")).map(String::as_str),
                Some("alice"),
                "alice-{i} should be attributed to alice, got: {producers:?}"
            );
            assert_eq!(
                producers.get(&format!("bob-{i}")).map(String::as_str),
                Some("bob"),
                "bob-{i} should be attributed to bob, got: {producers:?}"
            );
        }

        remove_db_file(&db_path);
    }

    /// The shutdown path end to end, which neither `shutdown`'s unit tests
    /// nor the acceptance tests above reach: cancelling the token has to
    /// stop the accept loop, drain, unlink, and return.
    #[tokio::test]
    async fn cancelling_stops_the_accept_loop_and_unlinks_a_bound_socket() {
        // Given a running listener on a socket this process bound
        let path = TestPath::new();
        let source = bind(&path).await.expect("bind");
        let server = test_server().await;
        let cancel = CancellationToken::new();
        let loop_handle = tokio::spawn(accept_loop(
            source,
            server,
            4,
            default_producers(),
            cancel.clone(),
            Duration::from_secs(5),
        ));

        // and a client that has connected and then gone away, so the drain
        // has a finished session to reap rather than an empty set.
        drop(
            UnixStream::connect(&*path)
                .await
                .expect("a client must be able to connect while serving"),
        );
        assert!(path.exists(), "the socket must exist while serving");

        // When shutdown is triggered
        cancel.cancel();

        // Then the loop returns rather than running forever, reports a
        // clean stop, and takes the socket file with it. The timeout is the
        // assertion that matters most here: before cancellation existed
        // this loop had no exit at all.
        let outcome = tokio::time::timeout(Duration::from_secs(10), loop_handle)
            .await
            .expect("the accept loop must stop promptly after cancellation")
            .expect("the accept loop task must not panic");
        assert!(
            outcome.is_ok(),
            "a cancelled accept loop is a clean stop, got: {outcome:?}"
        );
        assert!(
            !path.exists(),
            "a Bound socket must be unlinked once the loop stops"
        );
    }

    /// The same shutdown, but on a listener standing in for one launchd
    /// handed over: the path must survive, because the supervisor owns it.
    #[tokio::test]
    async fn cancelling_leaves_an_activated_socket_in_place() {
        // Given a running listener on an activated socket
        let path = TestPath::new();
        let listener = UnixListener::bind(&*path).expect("bind");
        let server = test_server().await;
        let cancel = CancellationToken::new();
        let loop_handle = tokio::spawn(accept_loop(
            ListenerSource::Activated(listener),
            server,
            4,
            default_producers(),
            cancel.clone(),
            Duration::from_secs(5),
        ));

        // When shutdown is triggered
        cancel.cancel();

        // Then the loop stops cleanly and the socket file is still there:
        // removing it would leave launchd holding a descriptor for a name
        // that no longer exists, and on-demand start would stop working.
        let outcome = tokio::time::timeout(Duration::from_secs(10), loop_handle)
            .await
            .expect("the accept loop must stop promptly after cancellation")
            .expect("the accept loop task must not panic");
        assert!(outcome.is_ok(), "got: {outcome:?}");
        assert!(
            path.exists(),
            "an Activated socket must survive the accept loop stopping"
        );
    }
}
