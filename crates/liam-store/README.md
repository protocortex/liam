# liam-store

A bitemporal graph store with hybrid retrieval. Nothing is overwritten:
superseded facts stay readable as history. It stores nodes and typed edges,
tracks two time axes, and retrieves by fusing full-text, vector, and graph
signals. Community detection over the edge graph is included.

It is generic over its storage engine and deliberately opinionated. It owns
structure, time, and retrieval; it does not own your domain model or your
embedding model.

## Backends

One backend ships today.

- `backend-libsql` (default): native vector search via `F32_BLOB` and
  `vector_distance_cos`.

The seam is small by design, and stays even with a single implementation.
Full-text, graph, and CRUD SQL run through `Backend::execute`/`query`. Only
vector storage and search are engine-specific, so those are their own trait
methods and a backend owns its dialect and physical layout: libSQL keeps
embeddings in a `node_vectors` table and prefilters against the live set.

A rusqlite backend was scaffolded here and removed once it was clear nothing
built it. The `Backend` trait stays: it is what keeps a second engine cheap
to add, and dropping it would mean rewriting `Graph<B: Backend>` throughout
`graph.rs`.

## SQLite-file compatibility

The database is a standard SQLite file. The `sqlite3` CLI and any SQLite
binding can open a file this crate writes; only the vector functions are
engine-specific. So "portable, inspectable with SQLite tooling" is
true out of the box, with no extra work.

## Data model

```
nodes(id, kind, label, content, attributes JSON, valid_from, valid_until, tx_from, tx_to)
edges(id, src, dst, type, attributes JSON, tx_from, tx_to)
```

Embeddings are not a node column; they live in backend-owned vector storage.
`kind` and `type` are opaque strings the library filters on but never interprets.
`attributes` is a JSON bag for anything the library does not model. Open
intervals use a `FOREVER` sentinel, so every currency check is positive.

## Opinions, stated

- Supersession, not overwrite: `supersede(old, new)` closes the old row and links
  it with a reserved `supersedes` edge.
- The library never embeds; you pass vectors in.
- Exact vector scan, no ANN, at the scale this targets: filter first, score second.
- RRF fuses full-text and vector candidates, then a one-hop graph expansion.
- Communities included (Leiden), in every build. There is no feature to turn it off.
- An injectable clock, so temporal behaviour is testable at a chosen instant.

## Surface

```rust
let graph = DefaultGraph::open("graph.db", GraphConfig::new(768)).await?;

let id = graph.insert(NewNode::now("decision", "Use libSQL", "single file")).await?;
let next = graph.supersede(&id, NewNode::now("decision", "Use libSQL v2", "...")).await?;
graph.link(NewEdge::new(&next, &id, "references")).await?;

let hits = graph.query(&Query::text("libSQL").with_k(8)).await?;
graph.gc(&RetentionPolicy::keep("episode", Millis::days(30))).await?;

graph.recompute_communities().await?;
```

## Features

```
default        = ["backend-libsql"]
backend-libsql = ["dep:libsql", "dep:tokio"]
```

## Status

The abstraction, the libSQL backend, and the shared graph logic are implemented
and compile. The acceptance gate is `cargo test --features backend-libsql`,
which is green.

Verification boundaries, isolated on purpose:
- libSQL parameter binding (`params_from_iter`) and row accessors
  (`column_count`, `get_value`) in `backends/libsql.rs`.
- `vector_distance_cos` plus `bm25` running in one connection.
- The `leiden-rs` membership accessor in `cluster.rs`.
- The name: confirm `liam-store` is free on crates.io before publishing.
