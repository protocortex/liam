# Reading modes and provenance

[Back to the docs index](../README.md)

Every item a read tool returns says where it came from. There are three
derivations:

- `recorded`: text a client wrote through `remember`.
- `extracted`: rows the extractor took out of recorded content.
- `inferred`: text a model composed, which is entity pages and summaries.

Reads come in two modes. `strict` returns what was recorded or extracted and
nothing a model wrote. `infer` may add inferred content, always labelled.

| Tool | Default | `infer` |
|---|---|---|
| `recall` | `strict` | Refused. |
| `clusters` | `strict` | Refused. |
| `timeline` | `strict` | Refused. The entity page heading the output is the one inferred item, and it is labelled so. |
| `ask` | `infer` | The default. Evidence may include entity pages and summaries. `strict` returns the recorded evidence and makes no model call. |
| `summaries` | not applicable | Lists summaries, which are always inferred. |

Entity pages are labelled `inferred` and left out of every strict read, both
from the results and from the ranking.

Each item ends with a `provenance:` line:

```text
provenance: derivation: recorded; events: 01a1..., 01a2...
provenance: derivation: inferred; basis: 01a1..., 01a2...; summary: 2
provenance: derivation: inferred; basis: unknown
```

`events` are the log events that wrote the item. `basis` are the events an
inferred item rests on, `basis: unknown` marks an entity page written before
LIAM tracked that, and `summary: N` gives a summary's level. A list longer than
`summary.max_basis_ids_rendered` is cut with `(+N more)`, and an event minted by
the backfill of an older store is marked `(synthetic)`.
