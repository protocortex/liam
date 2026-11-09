# Status and roadmap

[Back to the docs index](../README.md)

## Status

LIAM is pre-1.0. The last release is v0.1.1 (macOS on Apple Silicon); `main`
carries more since. It re-extracts the memory core of the retired v0 (the
archived `liam-archive` repo, a larger code-intelligence engine), rebuilt
smaller with RRF fusion and a full bitemporal model.

What is built:

- Storage and retrieval: the bitemporal graph, hybrid search with a cold arm,
  entity pages, scope hierarchy, producer-scoped collisions and `same_as`
  links.
- The event log, with rebuild, snapshots and restore, and consolidation runs
  that extract facts and link near-duplicates.
- Summary trees with strict and infer reading modes, and utility-based
  tiering.
- Deletion and log archive, with metrics and alerts, and a producer trust cap
  on usage.
- Hosted providers, off unless `[hosted]` is set: pinned models, a default-deny
  privacy gate, batch submit and collect, cost metrics, and cancel on delete.
  The provider request shapes are unchecked against live accounts.

What is checked: the workspace builds, `cargo test --workspace` is green, the
`local` build compiles against fastembed v5, and the community-detection
wrapper in `cluster.rs` is covered by tests against `leiden-rs` 0.8.1. The
real-model tests are gated and run on demand, not in the default test run.

## Roadmap

- Check the hosted adapters against live accounts and record the live R10
  results.
- Hosted deployment: HTTP transport, OAuth and per-tenant scoping. Not
  planned yet.
- Agent Client Protocol support. Not planned yet.
