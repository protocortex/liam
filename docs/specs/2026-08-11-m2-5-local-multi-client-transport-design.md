# M2.5: local multi-client transport

**Date:** 2026-08-11
**Status:** Approved design, pending implementation plan
**Implements:** M2.5 from `2026-08-11-multi-consumer-substrate-roadmap-amendment-design.md`
**Roadmap:** `2026-07-31-liam-gbrain-architecture-roadmap-design.md` (M2.5 is the local slice carved out of that roadmap's M6)
**Baseline:** `main` at the llama.cpp migration (M1 and M2 landed: `Llm` trait, llama.cpp provider, `ask` synthesis, bounded generation)

## Purpose

One `liamd` must serve several local clients at once. Today the daemon speaks
stdio only (`crates/liam-daemon/src/main.rs`, `rmcp::transport::stdio()`), and
stdio MCP is one client per process, so the coding agent and ai-notetaker cannot
share a store. The amendment names this the blocker that gates M2.6, M3, and
M3.5.

M2.5 delivers the transport and the write discipline underneath it. It does not
widen the tool surface (M2.6) and does not change scope semantics (M3.5).

## What the code actually requires (verified, 2026-08-11)

Three things the amendment assumed are not true yet. They are in scope because
the transport is not safe without them.

1. **WAL is not enabled.** No `journal_mode`, `busy_timeout`, or `synchronous`
   pragma exists anywhere in `liam-store`. The amendment says "readers during
   write (WAL)" as if it were current behaviour. On SQLite's default rollback
   journal a write blocks readers, so concurrent clients would stall on each
   other. M2.5 turns WAL on deliberately.
2. **There is one shared connection, and the `Database` handle is discarded.**
   `crates/liam-store/src/backends/libsql.rs:88` returns `Self { conn }`, keeping
   a single `Connection` and dropping the `Database` it came from. Even with WAL
   on, reads cannot run in parallel until the backend retains `db` and can open
   more connections.
3. **There is no migration mechanism.** `crates/liam-store/src/schema.rs` is all
   `CREATE TABLE IF NOT EXISTS`, which skips an existing table wholesale. Adding
   a `producer` column to a database that already exists needs an explicit
   `ALTER TABLE ADD COLUMN` step, not a schema edit.

Two things are already in our favour: `MemoryServer` derives `Clone`
(`crates/liam-daemon/src/mcp.rs:82`), so it can be handed to one `serve()` per
connection; and `rmcp` 2.2.0 ships `transport-async-rw`, which is generic over
`AsyncRead` plus `AsyncWrite`, so a `UnixStream` needs no custom transport.

## Decisions

**1. Unix domain socket carrying raw JSON-RPC.** `liamd` listens on a UDS at a
configured path and runs one `rmcp` session per accepted connection, all sharing
one `Arc<DefaultGraph>`.
Why: access control is filesystem permissions (owner-only, 0600), so a local
socket needs no token, no port, and cannot be reached off the machine. Loopback
HTTP would expose the store to every local process and browser page, which means
shipping auth now, and auth is M6 work by design. Streamable HTTP over a UDS
stays available as a later upgrade if a client needs MCP session resumability;
it buys nothing yet and costs hyper wiring we do not have.

**2. WAL, plus an explicit write lock, plus a read pool.** Set
`journal_mode=WAL`, `busy_timeout`, and `synchronous=NORMAL` at open. Reads take
a connection from a small pool. Writes serialize behind one application-level
lock over one write connection.
Why: WAL is what lets readers proceed during a write. The lock makes the
single-writer rule something our code asserts and a test can prove, rather than
something we hope SQLite's busy handling resolves under contention. A writer
actor task would also serialize by construction, but it turns the store API into
message passing, which is more redesign than this milestone needs.

**3. Producer identity from the MCP handshake, normalized through config.** The
`clientInfo.name` a client already sends at `initialize` maps through a
`[producers]` table in `liam.toml` to a canonical producer id, held for the life
of the connection and stamped on every node written through it. An unrecognized
client is accepted and recorded as `unknown`, with a warning.
Why: no protocol invention and no per-call argument a client can forget. A
forgotten per-call field is precisely the silent-provenance-loss failure the
amendment already flags for `valid_from`. The string is client-controlled, which
does not matter here: the socket's permissions already mean any connection is
the owner. Rejecting unknown clients would turn a new consumer's first run into
a hard failure for no safety gain on a single-user machine.

**4. stdio stays, as a proxy to the socket.** Invoked in stdio mode, `liamd`
forwards JSON-RPC to the daemon's socket instead of opening the store.
Why: every existing MCP config keeps working untouched, while exactly one
process ever holds the database file. Leaving stdio as a second full server is
the trap: the agent on stdio and ai-notetaker on the socket would be two
processes writing one SQLite file, which is the contention this milestone exists
to remove. Dropping stdio outright would break the current agent setup for no
reason.

**5. Conflicting writes keep last-writer-wins; the producer is recorded.**
`upsert_by` semantics do not change. Two producers writing the same
`subject` in the same `scope` still supersede each other, but every version now
records who wrote it.
Why: M3.5 is the milestone that isolates domains by scope, and inventing
conflict semantics here means designing namespacing twice. Recording the
producer makes the collision visible and queryable instead of silently resolved,
which is what a later design needs as input. Adding the producer to the
supersede key would stop the stomping today at the cost of two live nodes for
one real subject, pre-empting M3.5's identity work.

## Design

### Transport

- Config gains a socket path (default under the existing data directory) and an
  optional listen mode: socket, stdio, or both.
- On start: create the parent directory, remove a stale socket file only after
  confirming nothing answers on it, bind, then set the mode to owner-only.
  Removing a live socket would silently steal another daemon's clients.
- Accept loop: per connection, clone `MemoryServer`, resolve the producer from
  the handshake, hand the stream to `rmcp`'s async-rw transport, spawn the
  session. A dead client drops its session and nothing else.
- Shutdown unlinks the socket. A crash leaves the file behind, which the
  stale-socket check on next start handles.

### Write discipline

- `Backend::open` retains the `Database` handle so further connections can be
  opened.
- One write connection behind an async lock. One read pool sized from config,
  small by default.
- Pragmas applied to every connection as it is created, not once at open, since
  `busy_timeout` is per connection.
- The existing generation semaphore from M2 already bounds concurrent model
  work, so multiple clients calling `ask` at once inherit that bound rather than
  needing a new one.

### Producer provenance

- `nodes` gains a `producer` column, written on insert and carried through
  supersede so history attributes correctly.
- Applied to existing databases by an `ALTER TABLE ADD COLUMN` guarded by a
  column check, because `CREATE TABLE IF NOT EXISTS` will not do it.
- Not yet exposed as a filter on read. That is M3.5, where provenance becomes
  first-class.

### stdio proxy

- Stdio mode connects to the socket and shuttles bytes both ways, with a clear
  error if no daemon is listening, naming the command to start one.
- It opens no store and loads no model, so it stays cheap to spawn per agent.

## Acceptance

M2.5 is done when two clients are connected to one `liamd` at the same time,
each writing and reading, with writes serialized and reads not blocked; when
every node records the producer that wrote it; when an existing database gains
the column without losing data; and when the current stdio agent config still
works unchanged, through the proxy.

## Non-goals

- No HTTP, no TCP port, no OAuth, no multi-tenancy. Those stay at M6.
- No widening of `remember`, `recall`, or `ask`. That is M2.6.
- No scope hierarchy, no multi-scope recall, no cross-scope `same_as`, and no
  producer filter on read. That is M3.5.
- No Windows support. AF_UNIX on Windows is partial, and the target is one
  developer on macOS.
- No change to RRF or decay maths.
- The rusqlite backend stays stubbed. The ai-notetaker storage spike's finding
  that brute-force cosine beats a libSQL vector index at this scale is a note
  for later, not work here.

## Open risks

- **Write starvation under a chatty reader.** A read pool plus a single write
  lock can let steady reads delay a write. Needs a test that a write completes
  while reads are in flight, not just that both work alone.
- **The migration runs once, in the wrong place.** Adding a column on open means
  every process that opens the store might attempt it. The guard has to be
  idempotent and safe when two processes start at once.
- **Producer identity is only as good as the handshake.** A client that omits
  `clientInfo` lands in `unknown`, and if two do, their data is indistinguishable
  by producer. The config table is the mitigation, and M3.5 is where provenance
  gets teeth.
- **The proxy hides daemon lifecycle.** An agent spawning the stdio proxy will
  fail if no daemon runs. Whether the proxy should start one, or only explain how,
  is a plan-level decision. Auto-starting risks two daemons racing to bind.
- **WAL changes the file set.** `-wal` and `-shm` files appear beside the
  database. Anything that copies or backs up the store by path needs to know.

## Next step

Scope M2.5 into an implementation plan in LIAM's convention
(`docs/plans/`), with the WAL and connection-model work as its own
unit ahead of the transport, since the transport is unsafe without it.
