# LIAM

Layered Intelligent Agent Memory. Local-first and bitemporal, for AI agents. It
runs as one process, stores to a single file, and speaks the Model Context
Protocol (MCP) so an agent reads and writes long-term memory.

Agents forget between sessions, and a plain vector database loses history and
contradiction. LIAM tracks when a fact was true and when it was learned, so you
can ask what you believed last Tuesday. See [Why LIAM](docs/concepts/why.md).

## What you get

- Hybrid retrieval: full-text, vector and graph signals, fused and reranked.
- As-of recall over both timelines.
- Contradiction handling on write, with version history kept.
- Labelled provenance: recorded, extracted or inferred, with the events behind it.
- A model-free core: embedding and reranking live in a separate crate.

## Quickstart

On Apple Silicon macOS:

```sh
brew tap protocortex/tap
brew install liam
liam fetch-models
```

Then point your MCP client at the proxy, which talks to the shared daemon:

```json
{ "command": "liamd", "args": ["proxy"] }
```

See [Install](docs/getting-started/install.md) and
[Running LIAM](docs/getting-started/running.md) for the rest.

## Documentation

The full index is in [docs/README.md](docs/README.md).

| Section | Pages |
|---|---|
| [Getting started](docs/getting-started/install.md) | [Install](docs/getting-started/install.md), [Running LIAM](docs/getting-started/running.md) |
| Reference | [Tools](docs/reference/tools.md), [Configuration](docs/reference/configuration.md), [Feature flags](docs/reference/feature-flags.md) |
| Concepts | [The event log](docs/concepts/event-log.md), [The embedder fingerprint](docs/concepts/embedder-fingerprint.md), [Reading modes](docs/concepts/reading-modes.md), [Summaries](docs/concepts/summaries.md), [Tiering](docs/concepts/tiering.md), [Extraction](docs/concepts/extraction.md), [Consolidation](docs/concepts/consolidation.md), [Hosted providers](docs/concepts/hosted-providers.md) |
| Operations | [Garbage collection](docs/operations/garbage-collection.md), [Deleting a subject](docs/operations/deleting-a-subject.md), [Metrics and alerts](docs/operations/metrics-and-alerts.md) |
| Development | [Workspace](docs/development/workspace.md), [Build and test](docs/development/build-and-test.md), [Evaluating retrieval](docs/development/evaluating-retrieval.md), [Status and roadmap](docs/development/status-and-roadmap.md) |
| Design history | [ADRs](docs/adr/), [Specs](docs/specs/), [Plans](docs/plans/) |

## Status

LIAM is pre-1.0. The last release is v0.1.1 (macOS on Apple Silicon); `main`
carries more since. See [Status and roadmap](docs/development/status-and-roadmap.md).

## License

Apache-2.0. See [LICENSE](LICENSE).

Dependencies are permissive too: everything in the tree is MIT, Apache-2.0,
BSD, ISC, Zlib or similar, apart from MPL-2.0, which is weak, file-level copyleft.
`cargo deny check licenses` enforces that with an allowlist, so a new GPL, LGPL, or AGPL
dependency fails CI rather than arriving unnoticed.
