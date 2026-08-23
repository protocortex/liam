# Always-available clusters

**Date:** 2026-07-23
**Status:** Approved design, pending implementation plan

## Problem

Graph community detection (the Leiden-based "cluster" feature) is compiled in
by default but is only conditionally available and never used at runtime:

- It sits behind the `cluster` Cargo feature (`liam-store`), gated at four
  sites, with `leiden-rs` as an optional dependency. A build without the
  feature has no clustering at all.
- Even when compiled, `Graph::recompute_communities()` and
  `Graph::communities()` are dead code: nothing in `liamd` calls them, and the
  MCP surface exposes no way to produce or query community assignments.

The goal: clustering is **always available** (no compile-time gate) and
**always live** (produced by the daemon and queryable by agents).

## Goals

- Remove the `cluster` compile-time feature so clustering exists in every build.
- Produce community assignments automatically on the daemon's maintenance cycle.
- Expose clusters to agents over MCP, both as a dedicated tool and inline on
  recall results.

## Non-goals (YAGNI)

- Incremental / delta recomputation (`changes_since`-driven). Full recompute only.
- A separate cluster scheduling interval or config block.
- Cluster-aware reranking or RRF changes.
- Scope/kind filtering on the new tool (may come later).

## Design

### 1. Remove the compile-time gate

`liam-store/Cargo.toml`:
- Delete the `cluster = ["dep:leiden-rs"]` feature.
- Promote `leiden-rs` from an optional dependency to a hard, unconditional
  dependency.
- Remove `cluster` from the `default` feature set (it is no longer a feature).

Delete every clustering `cfg` guard so the code compiles unconditionally:
- `liam-store/src/lib.rs:14-15` — `#[cfg(feature = "cluster")] pub mod cluster;`
- `liam-store/src/graph.rs:450` — `#[cfg(feature = "cluster")] impl<B: Backend> Graph<B>`
- `liam-store/src/graph.rs:515` — `#[cfg(feature = "cluster")] fn intern(...)`
- `liam-store/src/schema.rs:64` — `if cfg!(feature = "cluster")` guard around the
  `node_community` table + `node_community_by_community` index (now always created).

**Dependency decision:** `leiden-rs` is a hard dependency in every build,
including mock/test paths. No feature fallback. The existing graceful
degradation in `cluster::detect()` (returns singletons if the graph builder or
Leiden run fails) remains the only runtime safety net.

**Pre-flight check (blocker if it fails):** confirm `leiden-rs` introduces no
transitive dependency that conflicts with the pinned candle/fastembed
constraints (see memory: dependency-constraints). If a conflict exists, stop and
surface it before proceeding.

### 2. Recompute on the GC cycle

In `liam-daemon/src/main.rs`, the `spawn_gc` background task already owns a
dedicated store connection and ticks on `gc.interval_hours` (default 6h).

- After each `sweep(&store, &policy)`, call `store.recompute_communities()` on
  the same connection. Do this both on the `run_on_start` path and on every
  interval tick.
- Order: recompute **after** the sweep, so communities are computed over the
  post-GC graph (deleted nodes/edges are already gone).
- Failure posture: non-fatal, matching `sweep`. On error, `tracing::warn!` and
  continue the loop. On success, log the community count.
- No new config field; clustering shares the GC maintenance tick by design.

### 3. MCP surface

Both a dedicated tool and recall enrichment (`liam-daemon/src/mcp.rs`).

**New `communities` tool** (read-only):
- Description: returns current graph community assignments.
- No arguments this round.
- Calls `store.communities()` and formats the `(node_id, community)` list in the
  same style as `recall` output.

**Recall enrichment:**
- After `recall` ranks its hits, load the `node_id → community` map once (single
  query) and tag each returned hit with its community id.
- A freshly-remembered node has no assignment until the next recompute cycle.
  Render such hits with an explicit sentinel (e.g. `community: —` /
  `community: unassigned`) rather than omitting the field, so the agent can
  distinguish "not yet clustered" from a real community.

## Testing

- **Store:** existing cluster unit tests now compile and run unconditionally;
  remove any `feature = "cluster"` gate from their test attributes.
- **Daemon integration:** insert several connected nodes, run one GC/recompute
  cycle (via `run_on_start` or by invoking the recompute path directly), then
  assert:
  - the `communities` tool returns non-empty assignments, and
  - `recall` hits carry a community id, with the unassigned sentinel for a node
    inserted after the last recompute.

## Affected files

- `crates/liam-store/Cargo.toml`
- `crates/liam-store/src/lib.rs`
- `crates/liam-store/src/graph.rs`
- `crates/liam-store/src/schema.rs`
- `crates/liam-daemon/src/main.rs`
- `crates/liam-daemon/src/mcp.rs`
- Tests in the above crates.

## Open risks

- `leiden-rs` transitive footprint vs. candle/fastembed pins (pre-flight check
  above).
- Full recompute cost grows with live-edge count; acceptable at the 6h cadence
  for now, revisit if the graph grows large enough to make a 6h tick expensive.
