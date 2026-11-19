# Reading modes and provenance

[Back to the docs index](../README.md)

Every item a read tool returns says where it came from. There are three
derivations:

- `recorded`: text a client wrote through `remember`.
- `extracted`: rows the extractor took out of recorded content.
- `inferred`: text that is not a client's own record: entity pages, which are
  cut from recorded memories without a model, and summaries, which a model wrote.

Reads come in two modes. `strict` returns what was recorded or extracted and
never text a model wrote; the one inferred item it shows is an entity page,
built from stored sentences. `infer` may add summaries, always labelled.

| Tool | Default | `infer` |
|---|---|---|
| `recall` | `strict` | Refused. |
| `clusters` | `strict` | Refused. |
| `timeline` | `strict` | Refused. The entity page heading the output is the one inferred item, and it is labelled so. |
| `summaries` | not applicable | Lists summaries, which are always inferred. |

Entity pages are labelled `inferred` and left out of every strict ranking:
`recall` and `clusters` never return one. `timeline` shows one.

## Entity pages

An entity page is read through `timeline`, as the item heading its output. It
is built without a model. For one entity, LIAM takes the memories that mention
it, drops each that is nearly the same as a newer one (cosine similarity of 0.92
or more), keeps the two newest and the two most central of the rest, and lists
them oldest first, one line each, led by its date:

```text
Maria Costa (person)
2026-04-07: Maria Costa moved to the Platform team in April.
2026-05-12: Maria Costa works on the Kubernetes upgrade for the Platform team.
2026-10-06: Maria Costa reports to Dev Patel and works from Porto.
2026-10-27: Maria Costa took over the on-call rota for the Platform team.
```

A page holds at most 600 characters. Every line is a sentence a client
recorded, so a page cannot state a fact no memory states, and never contains
text a model wrote. It does not merge or resolve: where a value changed, the
page shows both lines and the dates say which is newer. A page of a scoped
entity quotes only memories inside that scope. A page is rebuilt when a new
mention makes its entity stale, on `remember` or on the next maintenance tick.
A page an earlier version of LIAM had a model write is not shown: `timeline`
heads its output with no text until the tick replaces it.

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
