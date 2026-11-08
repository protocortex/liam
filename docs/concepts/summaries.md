# Summaries

[Back to the docs index](../README.md)

Summaries are model-written overviews of clusters of recorded memory, built level
by level into a tree: level 1 summarizes clusters of memories, and level 2 and 3
summarize clusters of the level below. They are off by default and need a real
`llm` provider, not the mock. A `summarize` step in each consolidation run writes
the missing or out of date ones for every scope, and closes the ones whose
cluster is gone. A rolled back run leaves none of its summaries behind.

A summary is kept only if its content words are mostly found in the items it
summarizes (`summary.min_grounded_share`). Each one stores the log events behind
it, which are the events of its children, so it can be traced down to the
records.

Keys, all under `[summary]` in `liam.toml`:

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | `false` makes the whole step a no-op. |
| `provider` | `"local"` | Who writes summaries: `local`, or `anthropic`, `openai` or `gemini` (these need `[hosted]`). `ask` and entity synthesis stay local. |
| `knn` | `8` | Nearest neighbours considered for each item. |
| `min_similarity` | `0.5` | Least cosine similarity at which two items are linked. |
| `time_window_hours` | `720` | Items further apart than this are not linked. |
| `min_cluster_size` | `3` | Fewest members of a cluster that gets a summary. |
| `max_cluster_size` | `30` | A larger cluster is split. |
| `max_depth` | `3` | Levels of summaries above the nodes, at most 3. |
| `recluster_change_ratio` | `0.2` | Share of a level that may change before it is clustered again. |
| `recluster_interval_days` | `7` | Age of the last full build after which a level is clustered again. |
| `min_attach_similarity` | `0.6` | Least similarity to a cluster's centroid for a new item to join it. |
| `max_unattached` | `20` | New items that may wait unattached before a level is clustered again. |
| `max_summaries_per_run` | `20` | Most summaries one run generates. A capped level is finished by the next run. |
| `run_deadline_secs` | `600` | Bounds the generation of one run. |
| `min_grounded_share` | `0.5` | Least share of a summary's content words its items must hold. |
| `max_summary_failures` | `3` | Failures of one child set after which it is skipped. |
| `min_summary_similarity` | `0.5` | Least similarity of a question to a summary for `ask` to use it. |
| `dissolve_grace_runs` | `3` | Runs a summary whose cluster is gone waits for a replacement before it is closed. |
| `dissolve_overlap` | `0.5` | Share of its items a stored summary must hold to replace it. |
| `max_basis_ids_rendered` | `8` | Event ids a provenance line lists before it counts the rest. |

## Reading summaries

`ask` in infer mode adds up to three summaries to its evidence: the live ones in
the question's scope (that scope and its descendants) that are at least
`summary.min_summary_similarity` close to it. Strict `ask` never reads them.

`summaries` lists them, ranked and cut by `query` when one is given, and takes
`level`, `scope` and `limit` (default 20, at most 100). With `id` it shows one
summary, open or closed, then each child as it stood when the summary was
written, with the events behind it. A child that no longer exists is named.

Time travel reads live summaries only. `ask` with `as_of` uses a summary that
was stored by then and does not cover a later time, and a summary closed after
`as_of` is not eligible, because closed summaries are not read.

## What the exit criteria prove

Two checks run in the test suite over a store written through a real log, and a
real summarize pass:

- Every live summary traces to real events. Its events are exactly those of its
  children, at least one is a real log event rather than a backfill placeholder,
  and each is an event the log holds.
- No inferred content appears in a recorded field of any tool's output. A strict
  output holds no inferred text or id, apart from the labelled `timeline` page,
  and every item shown carries the derivation the store holds for it.

These prove the links hold. They do not prove a summary is faithful to what it
summarizes: the grounding gate is a lexical check on words and nothing stronger,
so a summary can still misstate its sources. Follow the `provenance:` line to
the events before relying on one.
