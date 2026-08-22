# LIAM as a multi-consumer local substrate: roadmap amendment

**Date:** 2026-08-11
**Status:** Approved design (roadmap-level), pending per-milestone plans
**Amends:** `2026-07-31-liam-gbrain-architecture-roadmap-design.md`
**Baseline:** `v0.1.0` plus M1 and M2 in progress (Llm trait, llama.cpp provider, `ask` synthesis are wired per git history)

## Purpose

The gbrain roadmap designed LIAM around one consumer: an AI coding agent storing code-intelligence memory (`symbol`, `episode`, `decision`, `fact`). A second consumer now exists: ai-notetaker (`igorjs/ai-notetaker`), a local-first meeting notetaker that will use LIAM as its knowledge and retrieval layer. Both are owned by the same person and will run on the same machine at the same time.

This amendment records what serving a second, structured consumer requires, and reorders the milestones so the shared-brain scenario works. It does not replace the gbrain roadmap; it adds to it and pulls one milestone forward.

## The driving scenario

One developer, one machine. LIAM's daemon is running, connected to their codebase memory through the coding agent. The same developer runs ai-notetaker for business and technical meetings. During a meeting a technical decision comes up, and they want the codebase's related decisions surfaced live. They want business facts (people, orgs, commitments) and technical facts (symbols, code decisions) in one brain, recallable together, without the two domains colliding.

This decomposes into two requirements the current design does not meet:

1. **One store, two live producers, no clash.** The coding agent and ai-notetaker both write to and read from one LIAM instance concurrently, and their data stays separable.
2. **Cross-domain recall on the go.** A meeting-time query reaches codebase memory, and meeting entities link to codebase entities without merging into them.

## What the current design cannot do (grounded in the code)

- **Transport is stdio only** (`crates/liam-daemon/src/main.rs`, `rmcp::transport::stdio()`). stdio MCP is one client per process. Two independent apps cannot share one `liamd`. This is the blocker.
- **The MCP surface is lossy.** `remember` accepts only `kind, label, content, scope, subject` (`crates/liam-daemon/src/mcp.rs`). The store's `NewNode` also carries `attributes` (a JSON bag), `valid_from` (backdating), `confidence`, and the graph supports typed edges via `link` and a reserved `MENTIONS` relation (`crates/liam-store/src/types.rs`). None of `attributes`, `valid_from`, `confidence`, edge assertion, or the read-side `as_of`/`half_life` cross the MCP boundary. A structured consumer loses its structure at the wire.
- **Ingestion assumes raw text.** The roadmap's M3 is "text/markdown in, LIAM's LLM extracts." ai-notetaker already extracts typed facts with a larger model (Gemma 4 E4B, validated at zero hallucinated commitments). Re-extracting from raw transcript with LIAM's smaller default model would be worse and redundant.
- **Scope is a single flat string with a one-or-all filter** (`crates/liam-store/src/graph.rs`, `AND n.scope = ?`). It cannot isolate two domains on write yet span them on read.
- **No producer provenance.** With two writers there is no field recording which app wrote a node, so business and technical facts cannot be told apart except by convention.

## Additional foundational decisions (extend the gbrain roadmap's approved list)

4. **Multi-consumer by default, local first.** One `liamd` serves several local clients concurrently over a local socket, as the single writer. This is the local slice of M6, pulled forward and separated from the cloud (HTTP, OAuth, multi-tenant) parts, which stay at M6.
5. **Non-lossy structured ingest.** The write path accepts the store's full node and edge model (`attributes`, `valid_from`, `confidence`, typed edges, external-source refs) and a batch form that lands a whole episode atomically. A consumer that has already extracted structure pushes it directly; text-extraction ingest (M3) remains for consumers that have not.
6. **Namespacing isolates writes and spans reads.** Scope gains a hierarchy or multi-scope query so two domains under a shared parent do not collide on `subject`-supersede yet are recallable together. Every node records its producer.
7. **Cross-scope identity is explicit, not automatic.** The same person or concept across domains links through an explicit `same_as`/alias edge or a resolve step, never a blind `subject` merge across scopes. This bridges a meeting decision to the code that implements it without collapsing one into the other.

## Amended milestone roadmap

The gbrain milestones (M1 foundation, M2 entity pages plus synthesis, M3 ingestion, M4 enrichment, M5 self-maintenance, M6 deployment) stand. This amendment inserts and reorders:

- **M1, M2 continue as planned.** Entity dimension, provenance edges, entity pages, and synthesized `ask` all serve ai-notetaker directly. No change.
- **New M2.5: Local multi-client transport.** A local socket transport for `liamd` accepting concurrent clients, single writer, readers-during-write (WAL). Producer identity on the connection. This is the scenario's blocker and gates everything below it. It is the local, no-OAuth slice carved out of M6.
- **New M2.6: Non-lossy ingest surface.** Widen `remember` to carry `attributes`, `valid_from`, `confidence`, and edge assertion; widen `recall`/`ask` to carry `as_of` and `half_life`. Add a batch/structured ingest that takes pre-extracted nodes plus edges plus external-source refs in one atomic episode. Every future consumer benefits.
- **M3 amended: two ingest doors.** Keep the text-extraction path, but the structured path from M2.6 is the one ai-notetaker uses. Dedup and provenance writes apply to both.
- **M3.5: Namespacing and cross-scope identity.** Hierarchical or multi-scope recall, producer provenance as a first-class field, and explicit cross-scope `same_as` linking. Depends on M2.5 and M2.6.
- **M4, M5, M6 unchanged**, except M6 loses the local-transport slice already delivered in M2.5 and keeps only cloud (HTTP, OAuth, multi-tenant).

Dependency order: M1 to M2 (in progress) to M2.5 (transport) to M2.6 (surface) to M3 (ingest, both doors) to M3.5 (namespacing) to M4+.

## ai-notetaker boundary (recorded here so both repos agree)

ai-notetaker is an MCP client of the shared local `liamd`, alongside the coding agent. It keeps its own store for immutable capture data (audio, utterances, transcripts, summaries, provenance) and writes distilled facts to LIAM through the structured ingest path. It never runs its own embedder or vector search; LIAM owns memory and retrieval.

Library-linking `liam-store` into ai-notetaker is legally fine (common ownership), but the shared service is preferred because the coding agent needs the same store concurrently, and one writer over a socket avoids the write contention of two processes on one file.

## Repo hygiene (not blocking)

The root `LICENSE` is AGPL while every crate's `Cargo.toml` declares `MIT OR Apache-2.0`. These contradict each other for outside users and should be reconciled to one intended licence. It does not affect the owner's own cross-use of LIAM in ai-notetaker.

## Non-goals

- No cloud, HTTP, OAuth, or multi-tenant work before M6. M2.5 is local socket only.
- No fixed entity ontology; entity type stays open, per the gbrain roadmap.
- No change to RRF or decay math beyond exposing existing knobs (`as_of`, `half_life`) over the wire.
- ai-notetaker does not hand raw transcripts to LIAM for extraction; it pushes structured facts.

## Open risks

- **Concurrent write correctness.** One writer over a socket is the model; the transport must serialize writes cleanly and let readers proceed under WAL. Getting the single-writer discipline wrong risks corruption or stalls.
- **Cross-scope over-linking.** Explicit `same_as` is safer than automatic merge, but a wrong alias still bridges two things that should stay apart. Needs a review or confidence gate.
- **Backdating and history.** `valid_from` on ingest is what makes "as of last month" correct for backfilled meetings. If a consumer omits it, history silently collapses to ingest time. The surface should make valid time hard to forget.
- **rusqlite backend is stubbed** (`crates/liam-store/src/backends/rusqlite.rs`, `todo!()`). The ai-notetaker storage spike found plain SQLite brute-force beats a libSQL vector index at this scale, but acting on that inside LIAM waits on finishing this backend. Near term LIAM stays libSQL; the finding is a note, not an action.

## Next step

Per-milestone brainstorm to scope to implement, in LIAM's convention, starting with **M2.5 (local multi-client transport)**, the blocker. This amendment is the reference those milestone plans cite, alongside the gbrain roadmap.
