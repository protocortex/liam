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
| `prompt_scheme` | A number built into the binary: the version of the convention that turns text into model input. Scheme 2 is the current one: search queries get the `query: ` prefix of the arctic model card and stored text stays plain. Scheme 1 gave the same prefix to the model that was the default then, and scheme 2 first wrapped that model's queries as `Instruct: <task>\nQuery:<query>`. Any store written under an older scheme goes through `liam rebuild` once. |

Every start of `liamd` and `liamd serve` compares the record with the
configured values before the store opens. `liamd proxy` opens no store and
skips the check.

- **No record, and no database or an empty one:** a fresh install. The record
  is written after the store opens, so a start that fails leaves nothing
  behind.
- **No record, and a database that holds data:** a store from before
  fingerprints. It is taken to hold the vectors of the default from that time,
  `Qwen/Qwen3-Embedding-0.6B` at 768 dimensions, under prompt scheme 0. That
  model no longer exists in this build, so it fails the check under the current
  binary (it is on scheme 2). `liam rebuild` clears this and needs an event log. A store that
  predates the log has none, so `liamd` copies its rows into a new log before
  it runs the check. The start is still refused, but the log now exists: run
  `liam rebuild --database PATH` and start again.
- **Record matches:** the start goes on.
- **Record differs in any field:** `liamd` exits before opening the store. The
  error names the recorded and the configured values and the rebuild command.
- **Record cannot be read, parsed, or has another `version`:** `liamd` exits
  with a message distinct from the mismatch one. The file is never overwritten,
  because that could hide a change of embedder.

`embedder.model` is `Snowflake/snowflake-arctic-embed-m-v2.0` (int8 ONNX on the CPU),
the only embedder. A store built with an earlier one, such as
`Qwen/Qwen3-Embedding-0.6B-GGUF` (Q8_0 on llama.cpp) or `Qwen/Qwen3-Embedding-0.6B`
(candle), is refused at start. Run `liam rebuild` to re-embed it with arctic.
The id is recorded as the model library spells it, whatever the letter case in
`liam.toml`.

To change `embedder.provider` or `embedding_dims`, stop
`liamd`, edit `liam.toml`, and run:

```sh
liam rebuild --database PATH
```

It re-embeds every live node with the configured embedder, resizes the vector
table, and records the new fingerprint, but only when no embedding failed. Do
not delete the `.embedder.json` file to get past the error: the old vectors
would stay in `node_vectors`, mixed with the new ones. Deleting the file is
safe only after the vectors are cleared or re-embedded. For a change of
`embedder.provider` or `embedding_dims`, the other way out is to restore the
previous `liam.toml` settings.
