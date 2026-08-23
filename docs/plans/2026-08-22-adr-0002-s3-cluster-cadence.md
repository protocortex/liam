# Segment S3 blueprint: cluster fingerprint, warm start, recompute rewrite

Executes [ADR-0002](../adr/0002-cluster-recompute-cadence.md) work units WU-7, WU-8 and
WU-9. Written 2026-08-22, after S2 merged as PR #56 (`main` at `dc5ae6d`, 219 tests green).

The S2 blueprint was lost because it was written under `docs/superpowers/`, which a global
gitignore rule (`~/.gitignore.global`) silently excludes. Nothing under that directory has ever
been committed, including the M1, M2.5 and architecture-roadmap design docs. Plans live here
instead, under a path that is actually tracked.

## What S2 already landed, so S3 does not redo it

- `Graph::relate` and `Graph::resolve_handle` exist.
- `recompute_communities` filters `type != 'supersedes'` and orders by edge id.
- Edge building lives in a pure `build_cluster_input(rows) -> (labels, edges)` that collapses
  unordered pairs on `(min_idx, max_idx, type)`. Reuse it unchanged. It is pinned by five
  tests and by a mutation run.
- `cluster_state` exists in the schema with its four columns, and is never read or written.
- `recompute_communities` still has **no production caller**. That stays true after S3.

## Verified against leiden-rs 0.8.1 source

Line references below are into the `leiden-rs` 0.8.1 sources as vendored by cargo, at the
version pinned in `crates/liam-store/Cargo.toml`.

- Warm start is `Leiden::run_with_initial_partition(&graph, Partition)` (`src/leiden.rs:478`).
  Both it and `Partition::from_membership` (`src/partition.rs:29`) take their argument **by
  value**, so the seed should be an owned `Vec<usize>`, not a slice.
- The carried-forward claim about disjoint community ids is **correct**. `from_membership`
  itself only stores the vector, but `renumber` (`partition.rs:84-103`) maps each distinct raw
  integer to one contiguous id, so a new node handed an id an existing community already holds
  is seeded into that community. Fresh ids must be `max(seed so far) + 1 + n`.
- That formula is load-bearing, not stylistic. `renumber` allocates
  `vec![usize::MAX; max_comm + 1]` (`partition.rs:91`). A hash or a large constant offset turns
  a handful of elements into an allocation sized by the id value.
- A short seed is a typed `LeidenError::InvalidPartition` (`leiden.rs:490-498`), not a panic.
  There is no way to say "this node has no seed": the partition is a dense vector of length
  `node_count`, so `build_seed` must emit exactly one id per label.
- The `num_communities() > node_count` check at `leiden.rs:501-509` is unreachable, because
  the renumber at `leiden.rs:500` runs first. Do not write clamping code for it.

## Defects in ADR-0002, found before writing code

### Defect 1: the fingerprint and the edge set are two reads, and nothing orders them

ADR-0002 argues the guarantee as read time versus commit time, and mandates two separate
statements: the fingerprint `SELECT COUNT(*), MAX(tx_from)` and the graph `SELECT src, dst,
type`. Reads do not serialize with writes (`backend.rs:16-29`), so a `relate` can land between
them, and which failure you get depends on an order the record never fixes.

| order | a `relate` lands between the reads | result |
|---|---|---|
| fingerprint, then edges | graph has the edge, stored fingerprint does not | fingerprint is behind, next check mismatches, redundant recompute. Safe. |
| edges, then fingerprint | stored fingerprint has the edge, graph does not | fingerprint is ahead, next check **matches**, serves an assignment that predates a live edge |

The second row is exactly the failure ADR-0002 exists to prevent, reintroduced through
statement order. Reading edges first is the natural way to write it.

**Fix: one statement, not an ordering rule.** SQLite window functions carry both:

```sql
SELECT src, dst, type, COUNT(*) OVER (), MAX(tx_from) OVER ()
  FROM edges WHERE tx_to = ?1 AND type != ?2 ORDER BY id
```

Confirmed working through the real libSQL backend, not just `sqlite3`. This removes the window
rather than making its failure safe, and an empty edge set returns zero rows, which maps to
fingerprint `(0, 0)` with no null handling on this path at all.

The standalone `SELECT COUNT(*), MAX(tx_from)` is still needed for the **cheap check**, where
the whole point is answering "did anything change" without loading the edge set. Amendment 3's
`Value::Null` rule applies there and only there.

### Defect 2: a missing `cluster_state` row is not a zero fingerprint

`read_cluster_state().unwrap_or_default()` is wrong and there is already a test fixture that
breaks on it (`graph.rs:1618-1684`): a database with a populated `node_community`, no
`cluster_state`, and no edges. Live fingerprint is `(0, 0)`. Default the stored one to `(0, 0)`
and they match, so a pre-ADR-0002 assignment is served forever. `schema.rs:80-83` already
describes the required behaviour in prose.

**Fix:** return `Option<ClusterState>`; `None` forces a recompute unconditionally.

### Defect 3: the ADR does not say what a failed warm run falls back to

`cluster.rs` promises singleton degradation on any failure. On the warm path that would write
an all-singleton assignment **plus a matching fingerprint**, so the bad answer sticks until the
next edge write.

**Fix:** a failed warm run retries cold. Only a failed cold run degrades to singletons.

### Defect 4: a warm run must carry `last_cold_start_at` forward

The write shape is `DELETE` + `INSERT` and the column is `NOT NULL`, so a warm run has to write
something. Writing `now` recreates Amendment 1's defect at the write site instead of the read
site, with the same symptom: the rule never fires again. A warm run is only reachable when a
prior row exists (Defect 2), so there is always a value to carry.

### Defect 5: "no assignment exists at all" is ambiguous

`cluster_state` absent, or `node_community` empty? They differ for a store whose edges were all
swept. Numerically it does not matter, since a seed of all-fresh ids renumbers to the identity,
which is what a cold run builds anyway. It matters for bookkeeping, because only a cold run
advances `last_cold_start_at`.

**Fix:** cold when `cluster_state` is absent **or** the stored assignment is empty. Calling a
singleton seed "warm" is a lie in the logs.

### Unmet requirement: logging

ADR-0002 requires both paths to log node and edge counts and elapsed time. Today
`recompute_communities` logs nodes and communities only.

## Work units

### WU-7, the fingerprint seam (parallel with WU-8)

`graph.rs`, plus a re-export if the types land in `types.rs`.

- `Fingerprint { edge_count: i64, max_tx_from: Millis }`, `PartialEq`.
- `ClusterState { fingerprint, computed_at, last_cold_start_at }`.
- `edge_fingerprint()` for the cheap check. Read column 1 by matching `Value::Null` to 0. No
  `COALESCE` (Amendment 3).
- `read_cluster_state() -> Result<Option<ClusterState>>`, all four columns in one statement.
- A private helper returning the `DELETE` + `INSERT` statement pair, so WU-9 appends both to
  the same `execute_atomic` list.

Tests, each naming the guard it kills:

| test | kills |
|---|---|
| `the_fingerprint_of_a_store_with_no_edges_is_zero` | `row.get_i64(1)?` instead of the `Value::Null` arm (Amendment 3) |
| `the_fingerprint_ignores_supersedes_edges` | dropping `type != ?2` |
| `the_fingerprint_falls_when_gc_sweeps_an_edge` | dropping `COUNT(*)`, the deletion half a maximum cannot see |
| `the_fingerprint_ignores_a_closed_edge` | dropping `tx_to = ?1` |
| `a_store_that_has_never_clustered_reads_as_no_prior_run` | `unwrap_or_default()` (Defect 2) |

### WU-8, warm-start detection and seed construction (parallel with WU-7)

`graph.rs` free functions beside `build_cluster_input`, plus `cluster.rs`.

- `cold_start_due(last_cold_start_at, now)`, strictly greater than `Millis::days(1)`. ADR-0002
  records the ~30-hour consequence deliberately; do not tighten it.
- `build_seed(labels, stored) -> Vec<usize>`, pure. Iterate **labels** in index order, never the
  stored rows. Reuse a stored community or take the next fresh id.
- `detect(node_count, edges, seed: Option<Vec<usize>>)`. Keep `LeidenConfig { seed: Some(42) }`
  identical on both paths. Warm failure retries cold (Defect 3).
- Extend the version-check comment in `Cargo.toml` with `leiden.rs:500` and `partition.rs:91`.

| test | kills |
|---|---|
| `a_cold_start_is_due_only_strictly_after_twenty_four_hours` | `>` becoming `>=` |
| `a_new_node_takes_a_community_id_above_every_id_in_the_seed` | fresh ids starting at 0, or `max` without `+1` |
| `no_fresh_id_ever_collides_with_a_stored_community` | any gap-filling scheme, using a stored set with a hole |
| `a_stored_node_that_no_longer_appears_drops_out_of_the_seed` | iterating `stored` instead of `labels` |
| `the_seed_is_exactly_as_long_as_the_label_list` | any early `continue` |
| `warm_starting_from_singletons_matches_a_cold_run` | a different `LeidenConfig` on the warm path |
| `a_wrong_length_seed_falls_back_to_cold_not_singletons` | the singleton fallback on the warm path (Defect 3). Pick a graph where cold is not singletons, or it proves nothing |

### WU-9, the recompute rewrite (after both)

- One statement for edges plus fingerprint (Defect 1). The captured value is what gets written,
  never a fresh query at commit.
- Cold when `cluster_state` is absent, or the stored assignment is empty, or the 24-hour rule
  fires.
- Write the assignment, the captured fingerprint, `computed_at = now`, and
  `last_cold_start_at = if cold { now } else { carried }` (Defect 4), in one `execute_atomic`.
- Log nodes, edges, communities, elapsed, and whether the run was warm.
- `communities()` becomes the seam both future callers use: check, recompute on mismatch, then
  return. Rename the raw read to `stored_communities` and make it `pub(crate)`, so the only
  public path to assignments is the checked one. `recompute_communities` keeps `Result<usize>`;
  `mcp.rs` calls it.

| test | kills |
|---|---|
| `a_second_call_with_nothing_changed_does_not_recompute` | deleting the comparison |
| `an_edge_asserted_after_the_last_run_forces_a_recompute` | a hardwired match, or dropping either fingerprint half |
| `a_warm_run_does_not_advance_last_cold_start_at` | writing `now` into it (Defect 4) |
| `a_cold_run_advances_both_timestamps` | never advancing it |
| `a_warm_run_in_between_does_not_postpone_the_daily_cold_run` | reading `computed_at` instead (Amendment 1's original defect) |
| `a_prior_assignment_with_no_cluster_state_is_recomputed_cold` | `unwrap_or_default()` (Defect 2) |
| `two_recomputes_leave_exactly_one_cluster_state_row` | dropping the `DELETE`, which the schema cannot catch |
| `the_edges_query_carries_its_own_fingerprint` | splitting it back into two reads (Defect 1) |

## Community ids are not durable

Deterministic within a run, unstable across runs. `run_core` renumbers its output by first
appearance in dense-index order, and a `gc` delete removes rows from the middle of the ordered
edge set, shifting every later index and renumbering groups whose membership never changed.
Warm starting adds no stability: the seed is renumbered on entry and the output on exit.

Persisting them in `node_community` is fine, since the store replaces them wholesale. The warm
seed reading them back is legitimate too, because it needs the equivalence classes and not the
identities. A **client** must never store one, and the future `clusters` tool must present
"these nodes group together in this response". Tests assert grouping, never a literal id.

## Amendment 4 stays deferred

The hazard needs two concurrent recompute paths on two connections. S3 ships zero call sites:
`spawn_gc` only sweeps, and there is no `clusters` tool. Both of Amendment 4's options are
changes to callers, not to the seam, so neither is expressible here. Decide it in the PR that
adds the **second** call site. Keep errors propagating out of `communities()` so that PR can
wrap a fallback without touching the store, and do not add a staleness-marker type now.

## PR split

1. `feat(store): add the cluster fingerprint seam and warm-start seed` (WU-7 + WU-8). Purely
   additive, unreachable from any caller, so the existing tests cannot regress.
2. `feat(store): recompute clusters against the fingerprint, warm-started` (WU-9). The only
   behaviour change, and the only one carrying the Defect 1 argument and the `communities()`
   rename.

Both descriptions must say what S3 does not do: the seam has no production caller. ADR-0002 is
not satisfied until S4 wires the GC tick and the `clusters` tool.

**ADR-0002 Amendment 5 lands with PR 2**, recording Defect 1 and the one-statement fix, and
folding in Defects 2 to 5. Amendments 1 and 3 set the precedent: defects found while
blueprinting, before code, confirmed against source.

## Open, not resolved here

A backwards clock makes `cold_start_due` false forever, since `Millis` is a signed `i64` and
`FixedClock::set` can move backwards. ADR-0002 does not address it. Noted rather than guarded,
because inventing a guard here would be scope the record never asked for.
