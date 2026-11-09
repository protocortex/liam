# The embedder fingerprint

[Back to the docs index](../README.md)

A store's vectors only make sense next to others from the same embedder. Beside
the database, `liamd` keeps a record of what wrote them: `<database>.embedder.json`
(for `liam.db`, that is `liam.db.embedder.json`). It holds four fields and a
file `version`:

| Field | Source |
|---|---|
| `provider` | `embedder.provider` |
| `model` | `embedder.model` |
| `dims` | `embedding_dims` |
| `prompt_scheme` | A number built into the binary: the version of the convention that turns text into model input. Scheme 2 wraps Qwen3 search queries as `Instruct: <task>\nQuery:<query>`, the format on the model card, and leaves stored text plain; BGE-M3 and the mock embedder get no wrapper, and each small ONNX model gets the prefix on its model card. Scheme 1 used a `query: ` prefix instead. |

The Qwen3 query text is `Instruct: Given a web search query, retrieve relevant
passages that answer the query` and a newline, `Query:` and the question.
Documents are embedded as they are. `liam eval retrieval` scores the same on
this format as on the old `query: ` prefix (recall@10 0.975, nDCG@10 0.836, MRR
0.817 either way), because the golden set is too easy to tell them apart. The
format was adopted for matching what the model was trained on, and the scheme
bump makes every store written under scheme 1 go through `liam rebuild` once.

Every start of `liamd` and `liamd serve` compares the record with the
configured values before the store opens. `liamd proxy` opens no store and
skips the check.

- **No record, and no database or an empty one:** a fresh install. The record
  is written after the store opens, so a start that fails leaves nothing
  behind.
- **No record, and a database that holds data:** a store from before
  fingerprints. It is taken to hold the configured provider, model and
  dimensions under prompt scheme 0, so it fails the check (the binary is on
  scheme 2). `liam rebuild` clears this and needs an event log. A store that
  predates the log has none, so `liamd` copies its rows into a new log before
  it runs the check. The start is still refused, but the log now exists: run
  `liam rebuild --database PATH` and start again.
- **Record matches:** the start goes on.
- **Record differs in any field:** `liamd` exits before opening the store. The
  error names the recorded and the configured values and the rebuild command.
- **Record cannot be read, parsed, or has another `version`:** `liamd` exits
  with a message distinct from the mismatch one. The file is never overwritten,
  because that could hide a change of embedder.

The default `embedder.model` is `Qwen/Qwen3-Embedding-0.6B-GGUF` (Q8_0 on llama.cpp). A store built
with the earlier default, `Qwen/Qwen3-Embedding-0.6B` (candle), is refused at start. Run `liam rebuild`,
or set `embedder.model = "Qwen/Qwen3-Embedding-0.6B"` to keep the old model. Both spellings of the
GGUF id (`...-GGUF` and `...-GGUF:Q8_0`) record as one model.

To change `embedder.provider`, `embedder.model` or `embedding_dims`, stop
`liamd`, edit `liam.toml`, and run:

```sh
liam rebuild --database PATH
```

It re-embeds every live node with the configured embedder, resizes the vector
table, and records the new fingerprint, but only when no embedding failed. Do
not delete the `.embedder.json` file to get past the error: the old vectors
would stay in `node_vectors`, mixed with the new ones. Deleting the file is
safe only after the vectors are cleared or re-embedded. The other way out is to
restore the previous `liam.toml` settings.
