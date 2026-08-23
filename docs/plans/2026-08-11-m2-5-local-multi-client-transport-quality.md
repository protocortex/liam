# Quality gate: M2.5 local multi-client transport

**Plan:** `docs/plans/2026-08-11-m2-5-local-multi-client-transport.md`
**Date:** 2026-08-12 | **Base:** main @ 96a1369 | **Result:** PASS after 2 iterations

## Summary

| Phase | Verdict | How it was run |
|-------|---------|----------------|
| 1 Fact-Check | PASS | run directly (agent hung, see Provenance) |
| 2 Adversarial | PASS after fixes | run directly (agent hung); 1 blocking + 3 medium findings, all folded in |
| 3 Test Review | PASS after fixes | run directly (agent hung); 2 findings, both folded in |

## Provenance of these results (read this before trusting the table)

All three gate agents (`fact-checker`, `critic`, `test-reviewer`) were dispatched as
the gate requires. **All three hung without returning**, on the same
SendMessage/mailbox path that already failed in this repo during the llama.cpp
migration gate (see that milestone's quality report, which records the identical
failure for `fact-checker`). Each was given a direct follow-up request and then
stopped.

Rather than record a vacuous PASS, each phase was then performed directly against
the repository. That is sound for Phase 1, whose claims are deterministic file and
line checks, and it is honest but weaker for Phases 2 and 3, where an independent
adversary would be worth more than self-review. Treat the Phase 2 and 3 verdicts as
self-review, not as independent confirmation. The findings below are real and were
acted on; the risk is in what self-review did not think to look for.

## Phase 1: Fact-Check (direct)

Confirmed against the code:

| Claim | Result |
|---|---|
| 12 existing paths in the file table | all present |
| 5 paths marked `create` | all genuinely absent |
| No `journal_mode`/`busy_timeout`/`synchronous` pragma in `liam-store` | confirmed, 0 matches |
| `backends/libsql.rs:88` drops the `Database` | confirmed, returns `Self { conn }` |
| `schema.rs` uses `CREATE TABLE IF NOT EXISTS nodes`, no migration mechanism | confirmed, 0 matches for ALTER/migrate/user_version |
| `MemoryServer` derives `Clone` (`mcp.rs:82`) | confirmed |
| Daemon rmcp features are `["server","macros","transport-io"]` | confirmed (line 24) |
| No CLI, no clap, no argv parsing in the daemon | confirmed, 0 matches |
| Serves via `rmcp::transport::stdio()` | confirmed |
| rmcp 2.2.0 has `transport-async-rw`, generic over `AsyncRead + AsyncWrite` | confirmed (`async_rw.rs:40`) |
| rmcp has a `client` feature for the WU-8 dev-dep | confirmed |
| `NewNode` fields, and no `producer` today | confirmed |
| `graph.rs` has insert / upsert_by / supersede / link / query | confirmed with signatures |
| WU graph acyclic, 12 WUs | PASS, topological order computed |
| Every WU in exactly one Segment, no forward cross-Segment dependency | PASS |
| Segment budgets under the 500 soft limit | PASS (460 / 380 / 470 / 190) |

**FAIL found in iteration 1, fixed in iteration 2:** every store test opens
`:memory:` (`graph.rs:688` and four more) and `liam-store` has no `tempfile`
dev-dependency. WAL is a no-op on an in-memory database, so WU-1's journal-mode
assertion would have been vacuous, and each `:memory:` connection is a separate
database, so WU-2's pool tests would have passed or failed for the wrong reason.
Added **WU-0** (file-backed test harness) and a `:memory:` guard in WU-2.

Confidence: HIGH for the path and API claims (all read directly).

## Phase 2: Adversarial review (direct)

**Blocking, fixed: two store-opening processes can still write one database.**
Assumption 8 keeps plain `liamd` as a store-opening stdio server so existing MCP
configs keep working. A user who starts `liamd serve` while their agent still
spawns plain `liamd` therefore gets two writers, and WU-2's in-process mutex only
serializes within a process. The milestone's headline single-writer claim would have
been false in precisely the configuration it targets. Added **WU-2b**: an exclusive
advisory lock on a lock file beside the database, held by any store-opening mode,
so the second process fails fast and points at the proxy. Advisory `flock`-style
rather than a PID file, so a crash leaves no stale lock.

**Medium, fixed: the socket path could be unlinked when it is not a socket.** WU-6
originally said "try to connect, and if nothing answers, unlink and bind". If the
path were a regular file or a directory, that would delete something that is not
ours. WU-6 now checks the file type first and refuses, with tests asserting a
regular file survives a failed start.

**Medium, fixed: unwritable socket parent directory** had no defined behaviour. Now
an explicit error naming the path, with a test.

**Medium, fixed: WAL on a network filesystem** misbehaves (it needs shared memory).
Recorded as a documented risk rather than code, since the store is a local file by
design.

**Challenged and kept:** dropping the read pool would be less code, but reads would
still serialize on one connection, making "concurrent clients" true at the
transport and false at the store. Already recorded as a rejected alternative.
Nothing else in the plan is droppable: the migration is required by existing
databases, the proxy is what keeps current configs working, and producer identity is
load-bearing for design decisions 3 and 5.

**No M2.6 or M3.5 creep found.** The plan adds no tool-surface fields and no scope
or provenance semantics on read. `producer` is written and stored but deliberately
not exposed as a read filter.

## Phase 3: Test review (direct)

**Fixed: the read-during-write test would have passed against broken code.** As
written it asserted only that the read completed. On a single shared connection the
read merely queues behind the write and still completes, so the test would pass
with no pool at all. The Done When now requires asserting overlap: the write blocks
on a test-controlled rendezvous and the read must complete while the write is still
held.

**Fixed: socket paths in tests could break or collide.** On macOS `sun_path` is 104
bytes and the system temp directory is long (`/var/folders/xx/…/T/`), so a socket
inside a `tempfile` tempdir can exceed the limit and fail with an opaque
invalid-argument error that reads like a listener bug. Tests now bind under a short
unique path and unlink on drop, which also prevents collisions between concurrent
test binaries and leftovers from a crashed run.

**Confirmed sound:** mocking only at the `Llm`/embedder boundary while using a real
store and a real socket matches this repo's convention and the standards. The WU-8
mutation check (revert WU-7 locally and confirm the acceptance test fails) is a
valid way to prove the pin. The suite stays model-free, so CI needs no weights.

**WARN, not blocking:** the "socket file mode is owner-only" assertion could pass by
umask coincidence rather than an explicit chmod. Reading the mode back is already
required, which is the best available check short of testing under a different
umask.

## Result

```
Fact-Check:        PASS (16/16 checks, 1 FAIL fixed in iteration 2)
Adversarial:       PASS (1 blocking + 3 medium fixed; self-review, not independent)
Test Review:       PASS (2 findings fixed, 1 WARN)
```

Net effect of the gate: 2 new Work Units (WU-0, WU-2b), one of which closes a hole
that would have made the milestone's central claim untrue, plus 5 sharpened test
requirements. The plan grew from 10 to 12 Work Units and S1 from ~270 to ~460
estimated lines.
