# Segment S4 blueprint: the GC tick and the clusters tool

Completes [ADR-0002](../adr/0002-cluster-recompute-cadence.md). Written 2026-08-24 against
`main` at `61ab841`, after S3 landed.

## What is already done, so S4 does not redo it

- `Graph::communities()` is the lazy-read seam: it checks the fingerprint and recomputes on a
  mismatch. WU-10's store-side mechanism landed with WU-9.
- `edges_with_fingerprint()` takes the edge set and its fingerprint in one windowed statement
  (Amendment 5). `build_seed`, `cold_start_due` and `detect(.., seed)` all exist and are tested.
- ADR-0003 landed, so deleting a node removes its edges and cluster row through the database.

**There is still no production caller.** S4 adds both of them, which is the condition
Amendment 4 said must trigger its decision.

## Decisions

### Share one `Graph`, resolving Amendment 4

The record left this open between sharing a connection and serving stale with a marker.
Sharing wins, and the reason is a fact about the code rather than a preference.

`gc` is not one long write. It issues five independent `execute` calls plus
`vector_sweep_orphans`, and `LibsqlBackend::execute` takes the write mutex and drops it when
the statement returns. Verified by reading it: the guard is a local in `execute`, not held
across the sweep. So sharing does not queue client writes behind a whole sweep. The longest
single hold a write can wait on is one statement, `PRAGMA incremental_vacuum`.

Reads are untouched either way, because `query` goes through the read pool and never takes the
write mutex. ADR-0002's "a call that finds nothing stale never waits" survives sharing exactly.

Not sharing does not avoid contention, it relocates it into an error: two write connections on
one file contend on SQLite's lock with a 5000 ms cap and then fail. That already applies to
`remember` and `relate` during a sweep. Sharing removes the class for every writer.

The marker loses on a harder point. `Error::Backend` carries the driver's message as an opaque
string, so telling a busy error from a real one means substring-matching text this crate does
not own. A marker that fires on every error serves stale data on a genuine failure, which is
what ADR-0002 exists to prevent. That is the same class of inference ADR-0001 Amendment 4 was
burned by.

The `Arc<DefaultGraph>` is already in scope and unmoved before the `spawn_gc` call, so the
change is passing it in and deleting the second `open`.

**Two comments become false and must be corrected in the same change**, not left to rot: the
one on `spawn_gc` and the one in `storelock.rs`, both of which claim GC "never contends with
requests". It always contended, at the file lock. The storelock invariant itself is unaffected.

### The tick calls a new `refresh_communities`, not either existing function

`recompute_communities` is unconditional and would run Leiden every six hours forever, which
contradicts the record's own "an idle store does no clustering work at all".
`communities()` has the right rule but then reads the whole assignment, which the tick throws
away. Extract the check into `refresh_communities() -> Result<bool>` and have `communities()`
delegate to it. The record requires the invariant live in one method both callers use, and this
satisfies that literally.

`communities()` keeps its signature. Its many existing test call sites become the net that
catches the two seams drifting apart.

### The community integer never leaves the store

Enforced by a type rather than by discipline. `community_groups()` returns
`Vec<Vec<ClusterMember>>`, and `ClusterMember` carries id, kind and label with **no community
field**, so the daemon cannot print an integer even by accident. Groups come back largest-first
with a deterministic tie-break, so truncation is meaningful and stable.

The presentation read filters on node liveness. `node_community` legitimately names superseded
nodes, because clustering filters edge liveness and never node liveness, and ADR-0001 records
that asymmetry deliberately. Filter at presentation only, never in the recompute.

### The output is bounded by a token budget derived from the configured context

A fixed group count is the wrong knob. The operator already declares how much text this machine
can handle, through `llm.context_tokens`, and that number tracks available memory. A store that
fits comfortably on a large machine should not be truncated to the same ten groups a small one
gets.

So `clusters` renders groups largest-first while a **token budget** allows, then stops and says
what it withheld. The budget is one tenth of `llm.context_tokens`. Counting uses
`Llm::count_tokens` with the same `estimate_tokens` fallback `ask` already uses, so it measures
the real thing rather than a proxy for it.

Measured with `o200k_base` over 200 nodes in 20 groups:

| `llm.context_tokens` | budget | actual output | groups shown |
|---|---|---|---|
| 2,048 | 204 | 174 | 2 of 20 |
| 4,096 | 409 | 409 | 5 of 20 |
| **8,192 (default)** | **819** | **786** | **10 of 20** |
| 16,384 | 1,638 | 1,431 | 20 of 20 |
| 131,072 | 13,107 | 1,431 | 20 of 20 |

Two properties matter. At the default context the rule yields ten groups, which is exactly what
hand-tuning arrived at, so nothing regresses at today's setting. And the budget is a ceiling
rather than a target: past 16k there is nothing left to withhold, so a large machine gets the
whole answer instead of padded output.

For comparison, the shapes that were rejected, at the default 8192:

| shape | tokens | share of context |
|---|---|---|
| every node, recall-shaped | 4,488 | 54.8% |
| every node, handle only | 2,481 | 30.3% |
| group sizes only | 67 | 0.8% |

Handle-only is half the cost and useless: nothing consumes a handle except `relate`, and a
model cannot choose what to relate without a label. Sizes-only answers nothing.

`k` and `members` stay as optional client arguments that NARROW the result, clamped `0` up to
`1` like the existing `ask` clamp. They cannot widen past the budget, which is the hard ceiling.
An earlier fixed proposal of 50 x 20 was rejected on measurement: it permits 4,285 tokens, only
4% better than no cap at all.

**One honest caveat.** `llm.context_tokens` sizes the LOCAL model's prompt, while this output
travels to the MCP client, whose context this daemon knows nothing about. It is used here as
the operator's declared "how much text is reasonable on this machine", which correlates with
memory and needs no new configuration. If the two ever need to differ, that is a dedicated
setting rather than a different formula.

**No scope or kind filter.** The partition is computed over the whole graph, so a filtered
presentation invites the reader to believe the grouping was computed within that filter. That
is a lie no caveat can carry. Per-scope clustering is its own decision.

Member lines are byte-identical to `recall`'s per-hit prefix, `[{kind} {handle}] {label}`,
minus the content line, so a handle read out of `clusters` feeds straight into `relate`. The
header states the totals and what was withheld, which is the client's only signal to raise `k`.

### Failure behaviour

The tool returns `clusters failed: {e}`, matching every other handler. No stale fallback: that
is the record's requirement, and sharing one `Graph` removes the reason anyone wanted one.

The tick logs and continues, matching `sweep`. That is not serving stale, because the tick
serves nobody and the next read re-runs the same check. The refresh runs whether or not the
sweep succeeded, because `gc` is a sequence of independent statements rather than a
transaction, so a partial sweep still changed the edge set.

## Defect found before coding: the check path is still two reads

Amendment 5 closed the read window on the recompute path. The **check** path still has it:
`communities()` reads `cluster_state`, then reads the live fingerprint, as two statements on
pooled connections that never serialize with writes.

| order | a `relate` lands between the reads | result |
|---|---|---|
| state, then live (what the code does) | live sees the edge, stored state does not | mismatch, recompute. Safe. |
| live, then state | live measured before the edge, state describes the pre-edge assignment | fingerprints **match**, an assignment predating a live edge is served |

That is Amendment 5's failure on the sibling path. The current order is safe by luck, nothing
documents it as load-bearing, and no test pins it. Amendment 5 explicitly rejected an ordering
rule as the fix because it relies on every future reader knowing why.

Same fix, one statement, with the stored state as scalar subqueries alongside the aggregate.
**Confirmed through libSQL, not only `sqlite3`:** an empty store returns exactly one row
`[Int(0), Null, Null, Null, Null, Null]`, and a populated one returns all six values. The four
state columns are `NOT NULL` in the schema, so NULL there means "no prior run" unambiguously
and Defect 2 cannot return through a `get_i64` reading NULL as 0.

Add it as a new method rather than changing `read_cluster_state` or `edge_fingerprint`, both of
which have many existing callers. Record as **ADR-0002 Amendment 6**.

## Work units

**WU-11a, the shared refresh seam (store).** `staleness_snapshot()` as the one statement above,
reusing the existing live-semantic-edges clause so the two halves cannot drift.
`refresh_communities() -> Result<bool>`. `communities()` delegates. Tests: the combined read is
one statement (via the existing interposing backend, asserting the injection fired); a missing
state row reads as no prior run; an edgeless store with a prior assignment still refreshes;
refresh reports work only when there is work. Every existing `communities()` test passes
untouched.

**WU-11b, the tick (daemon).** `spawn_gc` takes the shared `Arc`, loses its own `open`, and
becomes infallible. A `maintenance_tick` runs sweep then refresh, used by both the
run-on-start branch and the loop. Correct the two false comments. Tests: the tick refreshes
after the sweep; the refresh runs after the sweep and not before (a swept edge must leave the
fingerprint settled); an idle store reports no work.

**WU-12a, `community_groups()` (store).** The liveness-filtered join, grouped and ordered.
Tests: a superseded node appears in no group; an all-superseded group is dropped; ordering is
largest-first and deterministic; members carry what recall would show; the call refreshes first.

**WU-12b, the `clusters` tool (daemon).** A pure `render_clusters(groups, budget, count)` free
function taking the budget and a counting closure, so the format and the budgeting are testable
with no store and no model. The tool passes `llm.context_tokens / 10` and the same
count-with-fallback closure `ask` uses. Optional `k` and `members` narrow within the budget.
Module doc updated to five tools.

Tests: format pinned; a rendered handle resolves through `relate`; the output never exceeds the
budget; a smaller budget shows fewer groups and a larger one shows more; truncation is announced
with the count withheld; a budget too small for even one group still returns the header rather
than an empty string; empty store says so; the tool is registered with the argument names
clients send; no group identifier is ever emitted.

## Traps

- `gc` is not atomic, and `execute` takes the write mutex per statement. Do not reason about
  sharing as if the sweep held one lock.
- `execute_atomic` **does** hold the write mutex across its whole transaction, and a recompute
  builds one statement per assigned node. That is the longest write hold in the system, and
  sharing makes it visible to client writes for the first time. Log it; do not batch it here,
  because the record requires measurement first.
- `ORDER BY id` on the edge read must not be removed. `PRAGMA incremental_vacuum` runs inside
  `gc` immediately before the refresh, and it is armed by default, so this is the one place the
  unordered-scan argument is load-bearing.
- `node_community` legitimately names superseded nodes. Filter at presentation, never in the
  recompute.
- `run_on_start` defaults off and the loop drops its first tick, so a short-lived daemon never
  refreshes. That is why the lazy read exists; do not "fix" it by clustering at boot.
- Two concurrent reads can both recompute. Deliberate, wasteful not corrupting. Do not add a
  lock.

## PR split

Three, landed in order rather than stacked, since this repo is squash-only with a workflow that
auto-closes superseded children.

1. `feat(store)`: WU-11a plus Amendment 6. Store-only, no production caller yet.
2. `feat(daemon)`: WU-11b plus Amendment 4 resolved in place. The first production call site,
   and where the sharing argument gets reviewed on its own.
3. `feat`: WU-12a and WU-12b together, plus the changelog. Landing the store half alone would
   ship code whose shape a reviewer cannot judge.
