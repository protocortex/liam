# Workspace

[Back to the docs index](../README.md)

Four crates, each usable on its own.

| Crate | Role |
|---|---|
| `liam-store` | The core. Bitemporal graph, hybrid retrieval, GC, clustering. Backend-generic over libSQL or SQLite. No ML. |
| `liam-log` | The append-only event log: write-ahead segments, Parquet compaction, encryption and archive. The store rebuilds from it. |
| `liam-model` | In-process embedding and reranking via `fastembed`, behind the `local` feature. Mock and identity defaults keep dev builds light. |
| `liam-daemon` | The MCP server (binary `liamd`). Wires store and model, serves `remember` and `recall` over stdio, runs GC in the background. |

```
agent ──MCP──▶ liamd ──embed/rerank──▶ liam-model
                 │
                 └── store/retrieve ──▶ liam-store ──▶ libSQL/SQLite file
```

The store is the reusable part. The daemon is one consumer; a `liam` CLI or a web
graph view would be others.
