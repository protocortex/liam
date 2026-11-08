# Why

[Back to the docs index](../README.md)

Agents forget between sessions. The usual fix is a vector database, but that
loses two things a memory needs: history and contradiction. LIAM keeps both.

Every fact tracks two timelines. Valid time is when the fact is true in the
world. Transaction time is when the store learned it. Nothing is overwritten, so
you ask "what did I believe last Tuesday" and get the answer as of then. When a
new fact contradicts an old one, the old version stays readable and the store
links them.

Retrieval blends three signals instead of relying on vectors alone: full-text
match, vector similarity, and one-hop graph expansion, fused into one ranked
list and reranked for precision.

## What you get

- Hybrid retrieval. BM25 full-text, cosine vector search, and graph expansion,
  fused with reciprocal rank fusion (RRF), then reranked by a cross-encoder.
- As-of recall. Point-in-time queries over both timelines, so history is a
  first-class read, not an audit log you dig through.
- Contradiction handling on write. `upsert_by` supersedes the prior fact with
  the same subject and records a `supersedes` edge between them.
- Ranking that fades. Confidence and a per-query half-life down-weight old or
  low-trust facts; graph-expanded neighbours rank below direct matches.
- Housekeeping built in. A change cursor for incremental work, community
  detection (Leiden), and retention GC by kind.
- Labelled provenance. Every item says whether it was recorded, extracted or
  inferred and which log events it rests on, and strict reads never return
  model-written text.
- A model-free core. The store takes vectors as input. Embedding and reranking
  live in a separate crate, so you swap models without touching storage.
