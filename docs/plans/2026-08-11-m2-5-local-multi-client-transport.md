# M2.5 implementation plan: local multi-client transport

**Status:** Proposed | **Date:** 2026-08-11
**Design:** `docs/specs/2026-08-11-m2-5-local-multi-client-transport-design.md`
**Amendment:** `docs/specs/2026-08-11-multi-consumer-substrate-roadmap-amendment-design.md`
**Roadmap:** `docs/specs/2026-07-31-liam-gbrain-architecture-roadmap-design.md`
**Baseline:** `main` @ c0c1bb7 (M1, M2 landed; llama.cpp generation; rmcp 3; CI green)

Produced by `/scope --auto`. The five design decisions were settled in the
brainstorm and are not re-litigated here. Everything this run decided for itself
is in **Assumptions**.

## Architecture

The store gains real concurrency, then the daemon gains a second door.

**Module layout.** M2.5 adds six modules to `liam-daemon`, which would take it from
5 flat modules to 11, so they are created grouped rather than moved later:
`transport.rs` holds the `ListenerSource` enum and declares
`transport/{socket, activation, proxy, shutdown}`, while producer resolution lands
at `mcp/producer.rs` since its input is the MCP handshake. Both use the
`foo.rs` plus `foo/bar.rs` pattern already used by `llama.rs` and
`llama/template.rs`, so nothing existing moves and no file is renamed. `cli.rs` and
`storelock.rs` stay top level: neither is transport.
Deliberately NOT in scope: splitting `liam-store/src/graph.rs` (~680 lines of code,
the repo's only genuine implementation monolith). That is a pure move touching the
highest-risk correctness surface in the project, so it belongs in its own PR after
this milestone, not mixed into a feature.

`liam-store`'s libSQL backend currently holds one `Connection` and drops the
`Database` it came from (`crates/liam-store/src/backends/libsql.rs:88`), and no
journal pragma is set anywhere. So the first Segment retains the `Database`,
applies WAL and timeout pragmas per connection, and splits reads (a small pool)
from writes (one connection behind an async lock). Nothing above the `Backend`
trait changes shape.

`nodes` then gains a `producer` column. Because `crates/liam-store/src/schema.rs`
is all `CREATE TABLE IF NOT EXISTS`, an existing database will not pick the column
up from a schema edit, so a guarded `ALTER TABLE ADD COLUMN` runs at open.

The daemon keeps `MemoryServer` exactly as it is (it already derives `Clone` at
`crates/liam-daemon/src/mcp.rs:82`) and gains a Unix socket listener that clones
it per accepted connection over `rmcp`'s `transport-async-rw`. Producer identity
is resolved once per connection from the MCP `initialize` handshake and stamped on
every write from that connection.

Mode comes from the process argument, not config, so a single invocation is never
ambiguous about whether it owns the store:

- `liamd` with no argument: today's stdio server, unchanged.
- `liamd serve`: opens the store, listens on the socket, serves many clients.
- `liamd proxy`: opens nothing, shuttles stdin and stdout to the socket.

The proxy is a byte shuttle, not an MCP client. Both stdio MCP and the socket
carry newline-delimited JSON-RPC, so `tokio::io::copy_bidirectional` forwards
them transparently with no protocol awareness and no second parse.

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| crates/liam-store/src/backends/libsql.rs | modify | retain `Database`; per-connection pragmas; read pool + write lock; `:memory:` guard |
| crates/liam-store/src/backend.rs | modify | doc the read/write connection contract on the trait |
| crates/liam-store/Cargo.toml | modify | `tempfile` dev-dependency: WAL and pool tests need a file-backed database |
| crates/liam-store/src/migrate.rs | create | guarded, idempotent `ALTER TABLE ADD COLUMN` helper |
| crates/liam-store/src/lib.rs | modify | declare `mod migrate` |
| crates/liam-store/src/schema.rs | modify | `producer` column on `nodes` for fresh databases |
| crates/liam-store/src/types.rs | modify | `producer` on `NewNode` and the node row |
| crates/liam-store/src/graph.rs | modify | write `producer` on insert and carry it through supersede |
| crates/liam-daemon/Cargo.toml | modify | rmcp `transport-async-rw`; rmcp client in dev-deps |
| crates/liam-daemon/src/config.rs | modify | socket path, read pool size, `[producers]` table |
| crates/liam-daemon/src/mcp/producer.rs | create | resolve `clientInfo.name` to a canonical producer id |
| crates/liam-daemon/src/mcp.rs | modify | carry a producer on `MemoryServer`; stamp it on writes; `mod producer;` |
| crates/liam-daemon/src/transport.rs | create | module root: `ListenerSource` enum plus `mod socket/activation/proxy/shutdown` |
| crates/liam-daemon/src/transport/socket.rs | create | listener, stale-socket handling, permissions, accept loop |
| crates/liam-daemon/src/transport/activation.rs | create | launchd-supplied listener (macOS), Bound fallback |
| crates/liam-daemon/src/transport/shutdown.rs | create | signals, cancellation token, drain |
| crates/liam-daemon/src/transport/proxy.rs | create | stdin/stdout to socket byte shuttle |
| crates/liam-daemon/src/storelock.rs | create | exclusive advisory lock so only one process opens the store |
| crates/liam-daemon/src/cli.rs | create | clap derive: subcommands, `--version`, `--config` |
| crates/liam-daemon/src/main.rs | modify | thin dispatch; `mod transport;` `mod cli;` `mod storelock;` |
| packaging/dev.protocortex.liamd.plist | create | launchd job with the `Sockets` key for activation |
| liam.toml | modify | document the new keys with defaults |
| README.md | modify | document the three modes and the shared-daemon setup |

## Segments (suggested PRs)

Linear chain. WAL and the connection model come first because concurrent clients
on the default journal mode stall on each other, which would make the transport
look broken for a reason that has nothing to do with the transport.

| Seg | Title | Work Units | Requires | Concern | Est. lines |
|-----|-------|-----------|----------|---------|-----------|
| S1 | Store concurrency foundation | WU-0, WU-1, WU-2, WU-2b | none | WAL + read pool + write lock + one-writer lock | ~460 |
| S2 | Producer provenance and migration | WU-3, WU-4 | S1 | new column, safely, on existing data | ~380 |
| S3 | Unix socket transport | WU-5, WU-6, WU-7, WU-8 | S2 | the second door | ~470 |
| S3b | Supervision and lifecycle | WU-6b, WU-6c | S3 | launchd activation, signals, drain | ~280 |
| S4 | stdio proxy and mode dispatch | WU-9, WU-10 | S3b | keep existing configs working | ~210 |

## Deliverables (Work Units)

| WU | Title | Files | Requires | Segment | Parallel group | Done When |
|----|-------|-------|----------|---------|----------------|-----------|
| WU-0 | File-backed test harness (`tempfile` dev-dep) | crates/liam-store/Cargo.toml, crates/liam-store/src/graph.rs (tests) | none | S1 | none | a helper opens a graph on a temp file; existing `:memory:` tests untouched and green |
| WU-1 | Retain `Database`, apply pragmas per connection | crates/liam-store/src/backends/libsql.rs | WU-0 | S1 | none | `PRAGMA journal_mode` reports `wal` on a FILE database; every new connection gets `busy_timeout` |
| WU-2 | Read pool + write lock, with a `:memory:` guard | crates/liam-store/src/backends/libsql.rs, crates/liam-store/src/backend.rs | WU-1 | S1 | none | a read completes while a write is in flight; N concurrent writes all land with no `SQLITE_BUSY`; a `:memory:` path forces pool size 1 |
| WU-2b | Exclusive store lock for store-opening modes | crates/liam-daemon/src/main.rs, crates/liam-daemon/src/storelock.rs | WU-2 | S1 | none | a second store-opening `liamd` fails fast naming the holder; the proxy is unaffected |
| WU-3 | Guarded column migration helper | crates/liam-store/src/migrate.rs, crates/liam-store/src/lib.rs | WU-2 | S2 | none | running it twice is a no-op; two processes racing it both succeed |
| WU-4 | `producer` on nodes, end to end | crates/liam-store/src/schema.rs, crates/liam-store/src/types.rs, crates/liam-store/src/graph.rs | WU-3 | S2 | none | insert and supersede both record the producer; a pre-existing database gains the column with data intact |
| WU-5 | Config: socket path, pool size, producers table | crates/liam-daemon/src/config.rs, liam.toml | WU-4 | S3 | none | shipped liam.toml parses; defaults present; unknown keys still fail loudly |
| WU-6 | Socket listener and accept loop | crates/liam-daemon/src/transport/socket.rs, crates/liam-daemon/Cargo.toml, crates/liam-daemon/src/main.rs | WU-5 | S3 | none | `liamd serve` binds owner-only; a stale socket is replaced, a live one refused |
| WU-7 | Producer resolution per connection | crates/liam-daemon/src/mcp/producer.rs, crates/liam-daemon/src/mcp.rs | WU-5 | S3 | none | a known client maps to its id, an unknown one to `unknown` with a warning |
| WU-8 | Two-client integration test | crates/liam-daemon/src/transport/socket.rs (tests), crates/liam-daemon/Cargo.toml | WU-6, WU-7 | S3 | none | two clients on one socket both read and write; each node carries the right producer |
| WU-6b | launchd socket activation | crates/liam-daemon/src/transport/activation.rs, crates/liam-daemon/src/transport.rs, crates/liam-daemon/Cargo.toml, packaging/dev.protocortex.liamd.plist | WU-8 | S3b | none | an activated listener is used when launchd supplies one; the socket is never unlinked in that mode; Bound still works with no supervisor |
| WU-6c | Signals, cancellation, drain | crates/liam-daemon/src/transport/shutdown.rs, crates/liam-daemon/src/transport/socket.rs, crates/liam-daemon/Cargo.toml | WU-6b | S3b | none | SIGTERM stops accepting, cancels, drains within a deadline, then unlinks only if we own the socket |
| WU-9 | Mode dispatch and stdio proxy | crates/liam-daemon/src/transport/proxy.rs, crates/liam-daemon/src/cli.rs, crates/liam-daemon/src/main.rs, crates/liam-daemon/Cargo.toml | WU-6c | S4 | none | `liamd` unchanged; `liamd proxy` forwards; no daemon gives an actionable error |
| WU-10 | Document the modes | README.md, liam.toml | WU-9 | S4 | none | README explains the three modes and the shared-daemon setup |

## Parallel Groups

**None.** Every Work Unit either depends on its predecessor or shares a file with
it. WU-6 and WU-7 look independent (`socket.rs` versus `producer.rs` plus
`mcp.rs`) but WU-6's accept loop calls the API WU-7 introduces, so they run in
order. Marking them parallel would put two agents in the same wire-up.

## Per-Work-Unit Detail

### WU-0: File-backed test harness
- **Requires:** nothing
- **Files:** `crates/liam-store/Cargo.toml`, `crates/liam-store/src/graph.rs` (test module only)
- **Changes:** add `tempfile` as a dev-dependency and a test helper that opens a
  graph on a temp file path with an injected clock, alongside the existing
  `:memory:` helper. WHY this exists as its own Work Unit: every store test today
  uses `DefaultGraph::open_with_clock(":memory:", ...)` (`graph.rs:688` and four
  more), and `:memory:` cannot carry the rest of S1. WAL is a no-op on an
  in-memory database, so the journal-mode assertion would be vacuous, and each
  `:memory:` connection is a SEPARATE database, so a read pool would give every
  connection its own empty store. Without a file-backed harness the S1 tests
  cannot be written honestly.
  `tempfile` is already in the resolved tree (3.27.0, MIT OR Apache-2.0), so this
  adds no new supply chain, only a direct dev-dep edge.
- **Test scenarios:** Given the new helper, When a graph is opened and a node
  inserted, Then it reads back, and the temp file is cleaned up on drop.
- **Done When:**
  - [ ] a file-backed test helper exists next to the `:memory:` one
  - [ ] all existing store tests still pass unchanged
  - [ ] `tempfile` appears only under `[dev-dependencies]`

### WU-1: Retain `Database`, apply pragmas per connection
- **Requires:** WU-0
- **Files:** `crates/liam-store/src/backends/libsql.rs`
- **Changes:** keep the `Database` on the struct alongside the connection.
  Factor connection creation into one helper that applies
  `PRAGMA busy_timeout` and `PRAGMA synchronous=NORMAL` to every connection it
  hands out, and sets `PRAGMA journal_mode=WAL` once at open. WAL is persistent
  in the database file; `busy_timeout` is per connection, which is why it cannot
  be a one-time call at open.
- **Test scenarios:** Given a fresh store on a TEMP FILE, When opened, Then
  `PRAGMA journal_mode` returns `wal`. Given an opened store, When a second
  connection is created, Then its `busy_timeout` is the configured value. Given a
  `:memory:` store, When opened, Then `journal_mode` is NOT asserted to be wal,
  because WAL does not apply to in-memory databases; assert instead that opening
  still succeeds.
- **Done When:**
  - [ ] `Database` is retained and further connections can be opened
  - [ ] a test on a FILE database asserts `journal_mode` is `wal` by querying it back, not by asserting the call was made
  - [ ] a test asserts `busy_timeout` on a connection created after open
  - [ ] a `:memory:` store still opens cleanly, with no WAL assertion

### WU-2: Read pool + write lock, with a `:memory:` guard
- **Requires:** WU-1
- **Files:** `crates/liam-store/src/backends/libsql.rs`, `crates/liam-store/src/backend.rs`
- **Changes:** hold a small pool of read connections and one write connection
  behind a `tokio::sync::Mutex`. Route `query` to the pool and
  `execute`/`execute_batch`/`execute_atomic` to the write connection. Document on
  the `Backend` trait that writes serialize and reads do not, so the rusqlite
  backend implements the same contract when it is finished.
  **Guard `:memory:` explicitly.** Each connection to `:memory:` is its own
  private database, so a pool over it would hand out N empty stores and reads
  would silently miss every write. When the path is an in-memory database, force
  the pool to one connection and reuse the write connection for reads. This is a
  correctness guard, not an optimization: without it, setting
  `database_path = ":memory:"` produces a store that loses data with no error.
- **Test scenarios:** Given a write holding the write connection, When a read runs
  concurrently, Then the read completes without waiting for the write. Given N
  concurrent writes, When all complete, Then every row is present and no
  `SQLITE_BUSY` surfaced. Given a `:memory:` path, When a node is written and then
  read, Then the read sees it (proving the pool did not fan out to separate
  databases). Both concurrency tests run on a temp file, per WU-0.
- **Done When:**
  - [ ] reads and writes use separate connections on a file database
  - [ ] the read-during-write test asserts OVERLAP, not just completion: the write
        blocks on a rendezvous the test controls, and the read must complete while
        the write is still held. A test that only asserts "the read finished" also
        passes on a single shared connection, where the read merely queues behind
        the write, so it would not pin this behaviour at all
  - [ ] a concurrent-write test asserts every write landed
  - [ ] a `:memory:` write-then-read test passes, pinning the guard
  - [ ] `cargo test -p liam-store` green

### WU-2b: Exclusive store lock for store-opening modes
- **Requires:** WU-2
- **Files:** `crates/liam-daemon/src/storelock.rs` (create), `crates/liam-daemon/src/main.rs`
- **Changes:** before opening the store, take an exclusive advisory lock on a lock
  file beside the database (`<database_path>.lock`), holding it for the process
  lifetime. A second store-opening process fails immediately with an error naming
  the lock file and the likely cause (a daemon is already running, point clients at
  `liamd proxy`). The stdio proxy takes no lock, because it opens no store.
  **Why this is required, not optional:** the plan deliberately keeps plain
  `liamd` as a store-opening stdio server so existing MCP configs keep working
  (Assumption 8). That means a user who starts `liamd serve` while their agent
  still spawns plain `liamd` gets TWO processes writing one database. The
  in-process write mutex from WU-2 only serializes within a process, so without
  this lock the milestone's single-writer claim is false in the exact
  configuration the milestone targets. WAL plus `busy_timeout` would keep the data
  correct but reintroduce the cross-process write contention M2.5 exists to
  remove, silently.
  Use a real advisory file lock (`flock`-style, released by the OS if the process
  dies) rather than a PID file, so a crash does not leave a stale lock that needs
  manual cleanup.
- **Test scenarios:** Given no daemon, When a store-opening mode starts, Then it
  acquires the lock and runs. Given a held lock, When a second store-opening mode
  starts, Then it exits with an error naming the lock file and the proxy as the
  fix, and does NOT open the store. Given a held lock, When `liamd proxy` starts,
  Then it is unaffected. Given a process holding the lock that is killed, When a
  new one starts, Then it acquires the lock with no manual cleanup.
- **Done When:**
  - [ ] two store-opening processes cannot coexist, proven by a test
  - [ ] the error names the lock file and points at the proxy
  - [ ] the proxy path takes no lock
  - [ ] a killed holder leaves no stale lock

### WU-3: Guarded column migration helper
- **Requires:** WU-2b
- **Files:** `crates/liam-store/src/migrate.rs`, `crates/liam-store/src/lib.rs`
- **Changes:** a helper that adds a column only when absent: read
  `PRAGMA table_info(<table>)`, and if the column is missing run
  `ALTER TABLE ... ADD COLUMN`. Treat a duplicate-column error from a racing
  process as success, because two daemons can open the store at the same moment
  and both pass the check before either alters. Idempotency has to come from the
  error handling, not only from the pre-check.
- **Test scenarios:** Given a table without the column, When migrate runs, Then
  the column exists. Given the column already exists, When migrate runs again,
  Then it is a no-op and no error surfaces. Given two concurrent migrate calls,
  When both run, Then both return success and the column exists once.
- **Done When:**
  - [ ] the pre-check plus duplicate-column tolerance are both covered by tests
  - [ ] the concurrent case is tested, not assumed

### WU-4: `producer` on nodes, end to end
- **Requires:** WU-3
- **Files:** `crates/liam-store/src/schema.rs`, `crates/liam-store/src/types.rs`, `crates/liam-store/src/graph.rs`
- **Changes:** add `producer TEXT NOT NULL DEFAULT 'unknown'` to the `nodes` DDL
  for fresh databases, call the WU-3 helper at open so existing databases gain it,
  and add `producer` to `NewNode` and the node row. `insert` writes it;
  `upsert_by`/`supersede` carry it onto the new version so history attributes to
  whoever wrote each version. The default value is what makes the migration safe
  on rows that predate the column.
- **Test scenarios:** Given a node inserted with a producer, When read back, Then
  the producer round-trips. Given a live node from producer A, When producer B
  supersedes it by subject, Then the new version records B and the superseded
  version still records A. Given a database created before this change, When
  opened, Then the column exists and existing rows read as `unknown` with all
  other fields intact.
- **Done When:**
  - [ ] producer round-trips through insert and supersede
  - [ ] a test opens an old-schema database and asserts no data loss
  - [ ] existing store tests still pass unchanged

### WU-5: Config: socket path, pool size, producers table
- **Requires:** WU-4
- **Files:** `crates/liam-daemon/src/config.rs`, `liam.toml`
- **Changes:** add a socket path (default `~/.liam/liamd.sock`, matching the
  existing `~/.liam` convention used by `cache_dir`), a read pool size (default
  4), and a `[producers]` map from MCP client name to canonical producer id plus
  the fallback id for unknown clients. Follow the file's existing
  `#[serde(default, deny_unknown_fields)]` pattern so a typo still fails loudly.
- **Test scenarios:** Given the shipped liam.toml, When loaded, Then it parses and
  the new defaults are present. Given a config with a `[producers]` entry, When
  loaded, Then the mapping is available. Given an unknown key under the new
  section, When loaded, Then it fails.
- **Done When:**
  - [ ] `shipped_liam_toml_parses` extended with the new keys
  - [ ] tilde expansion for the socket path is tested

### WU-6: Socket listener and accept loop
- **Requires:** WU-5
- **Files:** `crates/liam-daemon/src/transport/socket.rs`, `crates/liam-daemon/Cargo.toml`, `crates/liam-daemon/src/main.rs`
- **Changes:** no rmcp feature change is needed. `transport-async-rw` is ALREADY
  enabled transitively: `cargo info rmcp` (2.2.0) shows
  `server = [transport-async-rw, ...]` and
  `transport-io = [transport-async-rw, tokio/io-std]`, and the daemon enables both.
  Verified 2026-08-12; an earlier draft of this plan wrongly said to add it.
  Create the parent
  directory (erroring clearly if it cannot be created or is unwritable). Handle an
  existing path in this order: if it is NOT a socket (a regular file or a
  directory), refuse and error, never unlink it, because unlinking a path that
  happens to be someone's data would be destructive; if it IS a socket, try to
  connect, and if something answers refuse to start rather than stealing another
  daemon's clients, otherwise unlink the stale socket and bind. Then bind, set the
  mode to owner-only, then accept in a loop,
  cloning `MemoryServer` per connection and spawning an `rmcp` session on the
  stream. Serve mode opens the store exactly as today.
  **Model the listener as a `ListenerSource` enum from the start**, with a
  `Bound { listener, path }` arm now and room for `Activated(listener)` in WU-6b.
  Only the `Bound` arm ever chmods or unlinks: when a supervisor owns the socket,
  touching the path breaks its restarts. Introducing the enum here rather than
  retrofitting it keeps WU-6b purely additive.
  **Cap concurrent connections** with a semaphore sized from config. M2's existing
  semaphore bounds model work, not accepts, so without a cap an unbounded accept
  loop can exhaust file descriptors, and each session that generates holds its own
  KV cache (measured at about 110MB).
  Shutdown, including when the socket is unlinked, is WU-6c's job; do not
  hand-roll it here.
- **Test scenarios:** Given no socket file, When serve starts, Then it binds and
  the file mode is owner-only. Given a stale socket file with no listener, When
  serve starts, Then it replaces the file and binds. Given a live listener, When a
  second serve starts, Then it errors and does NOT unlink the first socket. Given
  the path exists as a REGULAR FILE, When serve starts, Then it errors and the file
  still exists afterwards. Given the path exists as a DIRECTORY, When serve starts,
  Then it errors without removing it. Given an unwritable parent directory, When
  serve starts, Then the error names the path. Given a connected client that drops,
  When it disconnects, Then the daemon keeps serving others.
- **Done When:**
  - [ ] the live-socket case is proven not to unlink the existing socket
  - [ ] a regular file at the socket path is proven to survive a failed start
  - [ ] socket file permissions are asserted by reading the mode back, not assumed
  - [ ] a dropped client does not affect other sessions

### WU-7: Producer resolution per connection
- **Requires:** WU-5
- **Files:** `crates/liam-daemon/src/mcp/producer.rs`, `crates/liam-daemon/src/mcp.rs`
- **Changes:** a pure function mapping an optional client name to a canonical
  producer id via the config table, falling back to the configured unknown id and
  logging a warning once per connection. `MemoryServer` carries the resolved
  producer and passes it into `NewNode` on write. Keep the resolution pure and
  separate from rmcp so it is testable without a transport.
- **Test scenarios:** Given a client name in the table, When resolved, Then the
  canonical id. Given an unknown name, Then the fallback id. Given no client name
  at all, Then the fallback id. Given a name differing only by case, Then it
  resolves (normalize before lookup) or is documented as case-sensitive, tested
  either way.
- **Done When:**
  - [ ] resolution is a pure function with tests for known, unknown, and absent
  - [ ] a `MemoryServer` write carries the resolved producer, asserted through the store

### WU-8: Two-client integration test
- **Requires:** WU-6, WU-7
- **Files:** `crates/liam-daemon/src/transport/socket.rs` (tests), `crates/liam-daemon/Cargo.toml`
- **Changes:** add rmcp with client and `transport-async-rw` features as a
  dev-dependency. Start a daemon on a temp socket over a temp database, connect
  two clients declaring different names, have both write and both read, and assert
  each node carries the right producer and both clients see each other's writes.
  This is the milestone's acceptance test.
- **Test scenarios:** Given two connected clients, When both write and then both
  read, Then each sees both nodes and each node records the producer of whoever
  wrote it. Given both clients writing at once, When all writes complete, Then
  none is lost.
- **Done When:**
  - [ ] the test uses a temp socket and temp database, no fixed paths
  - [ ] it fails if the producer is not threaded through (verify by reverting WU-7 locally)
  - [ ] it runs in the normal `cargo test -p liam-daemon` run, needing no model

### WU-6b: launchd socket activation
- **Requires:** WU-8
- **Files:** `crates/liam-daemon/src/transport/activation.rs` (create), `crates/liam-daemon/src/transport.rs`, `crates/liam-daemon/Cargo.toml`, `packaging/dev.protocortex.liamd.plist` (create)
- **Changes:** add the `Activated(UnixListener)` arm to `ListenerSource` and resolve
  it once at startup: if launchd handed us a socket, use it; otherwise fall back to
  `Bound`. On macOS use `raunch` (1.0.1, MIT OR Apache-2.0, a safe wrapper over
  `launch_activate_socket`), behind `cfg(target_os = "macos")` and a
  target-specific dependency, following the pattern `liam-model` already uses for
  Metal. Ship a launchd plist under `packaging/`.
  **Why this earns its own Work Unit:** it removes an entire failure mode rather
  than adding a feature. launchd owns the socket, so it exists before the daemon
  does and launchd starts `liamd` on the first connection. That deletes the
  auto-start question the design left OPEN (whether `liamd proxy` should spawn a
  daemon), deletes the race between two daemons binding, and makes the
  no-daemon-running error a development-only path.
  In the plist, the key under `Sockets` must match the name passed to
  `raunch::activate_socket`, and it carries a `SockPathName` pointing at the same
  socket path config uses. Two names to keep in sync, so assert it in a test or a
  documented check rather than trusting the docs.
  **Keep the daemon resident. Do NOT configure an idle exit.** On-demand start is
  good; on-demand *stop* is not, because a cold start pays llama.cpp's Metal kernel
  compile, measured at roughly 10 seconds against a 0.81s warm load. Activation
  should decide when the daemon first starts, not how often it dies.
  The `Bound` path stays so `cargo run` and tests work with no supervisor.
  Linux (`listenfd` 1.0.2, Apache-2.0) is deliberately deferred with the rest of
  Linux support.
- **Test scenarios:** Given no activation environment, When the source resolves,
  Then it is `Bound` and behaves exactly as WU-6. Given a pre-bound listener passed
  in as if activated, When the source resolves, Then it is `Activated` and shutdown
  does NOT unlink the path. Given the plist, When parsed, Then its socket name
  matches the constant passed to `activate_socket` and its `SockPathName` matches
  the configured default path.
- **Done When:**
  - [ ] `ListenerSource` resolves Activated versus Bound once, at startup
  - [ ] the Activated arm never chmods and never unlinks
  - [ ] `Bound` still works with no supervisor, proven by the existing WU-6 tests
  - [ ] the plist socket name and path are checked against the code, not assumed
  - [ ] `raunch` is macOS-target-scoped, so a Linux build does not pull it

### WU-6c: Signals, cancellation, drain
- **Requires:** WU-6b
- **Files:** `crates/liam-daemon/src/transport/shutdown.rs` (create), `crates/liam-daemon/src/transport/socket.rs`, `crates/liam-daemon/Cargo.toml`
- **Changes:** the plan previously said only "unlink on graceful shutdown", which
  named neither what triggers shutdown nor what happens to in-flight sessions. With
  concurrent clients that is the whole problem, so make it explicit.
  Listen for SIGTERM and SIGINT via `tokio::signal::unix`. Hold a
  `CancellationToken` and a `TaskTracker` from `tokio-util`, passing a child token
  into each connection task. `tokio-util` is already in the shipped tree via rmcp's
  `transport-async-rw`, so this is a direct edge on an existing dependency, the same
  situation as clap.
  Shutdown order, and the order matters: stop accepting, cancel the token, await
  `tracker.wait()` under a deadline, abort whatever is left, then unlink the socket
  **only** in the `Bound` case.
  The drain deadline must be shorter than the supervisor's grace period, because
  launchd sends SIGTERM and then SIGKILL, and a drain that outlives the grace gets
  killed mid-write instead of finishing.
- **Test scenarios:** Given an idle daemon, When SIGTERM arrives, Then it exits 0
  and the `Bound` socket path is gone. Given an in-flight request, When SIGTERM
  arrives, Then the request completes and only then does the process exit. Given a
  request that outlives the drain deadline, When the deadline passes, Then it is
  aborted and the process still exits rather than hanging. Given an `Activated`
  listener, When shutting down, Then the socket path still exists afterwards.
  Given SIGTERM during shutdown (a second signal), Then it does not panic or
  double-unlink.
- **Done When:**
  - [ ] SIGTERM and SIGINT both trigger the same ordered shutdown
  - [ ] an in-flight request is allowed to finish within the deadline
  - [ ] exceeding the deadline aborts rather than hangs, proven by a test
  - [ ] the socket is unlinked only when we own it
  - [ ] the drain deadline is documented as needing to be under the supervisor grace

### WU-9: Mode dispatch and stdio proxy
- **Requires:** WU-6c
- **Files:** `crates/liam-daemon/src/transport/proxy.rs`, `crates/liam-daemon/src/main.rs`
- **Changes:** dispatch with clap's derive API: no subcommand is today's stdio
  server, `serve` is the socket daemon, `proxy` is the shuttle, plus `--version`,
  `--help`, and `--config` to override `LIAM_CONFIG`. Use clap rather than a
  hand-rolled match: every clap crate including `clap_derive` is already in the
  shipped tree (via `leiden-rs`), so it costs nothing new, and it gives unknown
  arguments a usage error with exit code 2 for free. That last part matters here
  because the argument decides whether this process opens the store, so a typo like
  `liamd serv` must never fall through to the store-opening default.
  Keep `Cli` in its own module with the mode as an enum, so `main` stays a thin
  dispatch and the mapping is unit-testable without spawning a process.
  The proxy connects to the socket and runs
  `tokio::io::copy_bidirectional` between it and stdin/stdout; if the socket is
  absent or refuses, exit with an error naming the command that starts a daemon.
  Log to stderr only, since stdout is the protocol stream.
- **Test scenarios:** Given no argument, When started, Then the stdio server path
  runs as before. Given `proxy` with a listening socket, When a JSON-RPC frame is
  written to stdin, Then the socket receives it byte for byte and the reply comes
  back on stdout. Given `proxy` with no daemon, Then it exits with an error naming
  the fix. Given an unknown argument such as `serv`, Then it exits 2 with a usage
  error and does NOT open the store. Given `--version`, Then it prints the crate
  version and exits 0. Given the reading end of the proxy's stdout closing
  mid-stream, When the shuttle writes, Then the `BrokenPipe` is treated as a normal
  end and the process exits 0 rather than reporting an error.
- **Done When:**
  - [ ] the default invocation is byte-for-byte unchanged in behaviour
  - [ ] the proxy round-trips a frame both ways
  - [ ] the no-daemon error names how to start one
  - [ ] a mistyped subcommand exits 2 without opening the store
  - [ ] `--version` works, since the packaging plan and bug reports need it
  - [ ] `BrokenPipe` on either direction of the shuttle exits 0, not an error
  - [ ] the proxy opens no store and loads no model, asserted by it working with no database configured

### WU-10: Document the modes
- **Requires:** WU-9
- **Files:** `README.md`, `liam.toml`
- **Changes:** document the three modes, the shared-daemon setup (one `liamd
  serve`, agents pointed at `liamd proxy`), the socket path and permissions, and
  the `[producers]` table. Note that WAL adds `-wal` and `-shm` files next to the
  database so anything copying the store by path knows.
- **Done When:**
  - [ ] a reader can set up two clients against one daemon from the README alone
  - [ ] the WAL sidecar files are mentioned

## Testing Strategy

Per `engineering-standards`: tests ship with the change, each test builds its own
data, mock only at real boundaries, TDD where logic is non-trivial.

- **Pure unit tests (no I/O):** producer resolution for known, unknown, and absent
  client names (WU-7); socket path tilde expansion (WU-5); argument-to-mode
  mapping (WU-9). These are the cheap regression pins.
- **Test databases must be FILE-backed for S1 (WU-0 exists for this).** Every
  store test today uses `:memory:` (`graph.rs:688` and four more). WAL does not
  apply to in-memory databases, so a journal-mode assertion there would be
  vacuous, and each `:memory:` connection is a separate database, so pool tests
  would pass or fail for the wrong reason. `:memory:` keeps its own test, for the
  guard that forces pool size 1.
- **Store integration tests against a temp libSQL file:** WAL is actually on,
  queried not assumed (WU-1); `busy_timeout` on a later connection (WU-1);
  read-during-write and concurrent writes, using a rendezvous rather than a sleep
  so they cannot pass by timing luck (WU-2); migration idempotency, including two
  concurrent callers (WU-3); producer round-trip through insert and supersede, and
  an old-schema database opened and migrated with data intact (WU-4).
- **Daemon integration tests over a real socket:** two clients with different
  identities writing and reading concurrently (WU-8); stale versus live socket
  handling, including the assertion that a live socket is never unlinked (WU-6);
  proxy round-trip and the no-daemon error (WU-9).
- **Socket paths in tests must be short AND unique.** On macOS `sun_path` is 104
  bytes, and the system temp directory is long (`/var/folders/xx/…/T/`), so a
  socket inside a `tempfile` tempdir can exceed the limit and fail with an opaque
  invalid-argument error that looks like a bug in the listener. Bind test sockets
  under a short unique path (for example `/tmp/liam-test-<pid>-<n>.sock`) and unlink
  on drop. Unique per test, because a fixed path collides when two test binaries
  run at once or when a crashed run leaves a socket behind. The default production
  path (`~/.liam/liamd.sock`) is comfortably short.
- **Regression pins kept:** every existing `liam-store` and `liam-daemon` test
  passes untouched. The default no-argument invocation keeps today's behaviour,
  which the mode-dispatch test asserts explicitly.
- **Mutation check on the acceptance test (WU-8):** confirm it fails when the
  producer is not threaded through, so it pins the behaviour rather than merely
  exercising it.
- **No model needed:** every test above runs on the mock embedder and identity
  reranker, so the suite stays fast and CI needs no weights.

## Risks

- **Write starvation.** A read pool plus a single write lock can let steady reads
  delay a write. The WU-2 test asserts a write completes while reads are in
  flight, not merely that both work alone.
- **Migration under a race.** Two daemons opening the store at once can both pass
  the column check. Tolerating the duplicate-column error is the mitigation, and
  WU-3 tests it.
- **Socket theft.** Unlinking a socket that a live daemon owns would silently
  steal its clients. WU-6 connects first and refuses rather than unlinking, and
  the test asserts the live socket survives.
- **Producer identity is client-controlled.** Any local client can claim any name.
  Accepted: the socket's permissions already mean every connection is the owner.
  M3.5 is where provenance gets teeth.
- **Two store-opening processes.** Backwards compatibility means plain `liamd`
  still opens the store, so a user running `liamd serve` alongside an agent that
  spawns plain `liamd` would have two writers. WU-2b's exclusive lock is what makes
  this prevented rather than merely discouraged. Without it the single-writer claim
  is false in exactly the configuration this milestone targets.
- **WAL on a network filesystem.** WAL needs shared memory, and it misbehaves on
  NFS and some network mounts. Documented, not coded around: the store is a local
  file by design.
- **Rejected simpler alternative: skip the read pool.** WAL plus a single shared
  connection is less code, but reads would still serialize behind the one
  connection, so "concurrent clients" would be true at the transport and false at
  the store. Rejected because it would make the milestone's headline claim
  misleading.
- **Rejected simpler alternative: keep stdio as a second full server.** Additive
  and zero-migration, but two processes would write one SQLite file, which is the
  contention M2.5 exists to remove.

## Assumptions (`--auto` self-answered)

Each was the answer this run would have recommended in the interview.

1. **Mode from argv, not config.** `liamd` / `liamd serve` / `liamd proxy`. Config
   supplies parameters, argv selects the door. A config-selected mode makes a
   single invocation ambiguous about whether it owns the store, and a `both` mode
   would have every stdio spawn race to bind the socket.
2. **Adopt clap with the derive feature** (REVERSED 2026-08-12; the original
   assumption said no clap, on the grounds that three modes did not earn a new
   dependency). That premise was false. `cargo tree -e no-dev -p liam-daemon` shows
   `clap 4.6.1`, `clap_builder`, `clap_lex`, AND `clap_derive` already in the
   SHIPPED tree, pulled by `leiden-rs` through `liam-store`, with `syn` present too.
   So the derive API adds zero new crates and zero marginal compile.
   Three further reasons it is not scope creep: a CLI is already on the roadmap
   (`status-and-roadmap` lists "add a CLI and Unix-socket IPC" as pending), the mode
   argument decides whether the process OPENS THE STORE so a silently mis-parsed
   argument is a correctness hazard rather than a cosmetic one, and `--version` is
   needed by the packaging plan and by anyone identifying a running build.
   **Keep the surface small**: three subcommands plus `--version`, `--help`, and
   `--config` to override `LIAM_CONFIG`. A flag per config key would be the real
   speculative work, and that is still out.
3. **The proxy is a byte shuttle, not an MCP client.** `copy_bidirectional` over
   newline-delimited JSON-RPC is transparent, avoids a second parse, and keeps
   the production build free of rmcp client features.
4. **rmcp client is a dev-dependency only**, for the WU-8 acceptance test.
5. **Defaults:** read pool 4, `busy_timeout` 5000ms, `synchronous=NORMAL`, socket
   at `~/.liam/liamd.sock`. NORMAL is the standard companion to WAL; FULL costs a
   sync per commit for durability this workload does not need.
6. **`producer` on nodes only, not edges.** Edge assertion is not yet on the MCP
   surface; that is M2.6.
7. **`producer` defaults to `'unknown'`** so the migration is safe for rows that
   predate it.
8. **Default invocation stays the stdio server**, so no existing config changes
   behaviour when this lands.
9. **OPEN: whether `liamd proxy` should auto-start a daemon.** Leading option is
   no, only an actionable error, because auto-start risks two daemons racing to
   bind and hides lifecycle from the operator. Flagged in the design's risks and
   worth a decision before WU-9.

## Confidence + open items

- Confidence: HIGH. Every file and line referenced was read during the
  brainstorm, the three premise-breaking facts (WAL absent, `Database` dropped, no
  migration mechanism) were verified directly, `MemoryServer` is already `Clone`,
  and `rmcp` 2.2.0 ships the `transport-async-rw` feature this depends on.
- Open items (verify downstream):
  - Whether `rmcp`'s `serve` over `transport-async-rw` needs any per-session
    configuration beyond the stream. Verify in WU-6 against the crate source
    before writing the accept loop.
  - Whether libSQL's local builder honours `PRAGMA journal_mode=WAL` on the first
    connection, or needs it before any other statement. Verify in WU-1 by querying
    the pragma back, which is already the Done When.
  - `liamd proxy` auto-start, per assumption 9.
