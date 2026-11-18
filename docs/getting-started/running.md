# Run

[Back to the docs index](../README.md)

```sh
cargo run -p liam-daemon         # builds and runs the `liamd` binary
```

`liamd` has three modes. Running it with no subcommand is the original one, so
existing MCP client configs keep working unchanged.

| Command | What it does | Opens the store |
|---|---|---|
| `liamd` | Serves MCP over this process's stdio. One client, one process. | yes |
| `liamd serve` | The shared daemon: serves many clients over a Unix socket. | yes |
| `liamd proxy` | Forwards stdio to a running daemon's socket. | no |

`--config PATH` overrides `LIAM_CONFIG` for any of them.

## One daemon, many agents

Running `liamd` per client gives each one its own process, and only one of them
can hold the database: the others fail to start on the store lock. Use the daemon
instead when more than one agent needs the same memory.

Start it once:

```sh
liamd serve
```

Then point every MCP client at the proxy rather than at `liamd` directly:

```json
{ "command": "liamd", "args": ["proxy"] }
```

Each client keeps its own identity through the proxy, because the proxy forwards
bytes untouched and the daemon reads the name from the client's own MCP
handshake. Map those names to stable producer ids in `[producers.clients]`, and
every fact records who wrote it.

On macOS, launchd can own the socket and start the daemon on the first
connection, so nothing has to run `liamd serve` by hand. See
`packaging/dev.protocortex.liamd.plist` for the job and its install steps.

## The socket

`socket_path` defaults to `~/.liam/liamd.sock` and is created owner-only (0600).
It carries full tool access with no further authentication, so anyone who can
reach the file can act as any client; owner-only is what makes "can reach it"
mean "is already you". `liamd serve` refuses to start if another daemon is
already listening, and replaces the socket only when it is stale.

## Where LIAM keeps its files

The state lives under `~/.liam`: the database, the event log, the socket and the
model cache. Memory is stored in plain text unless `[encryption]` is on, so
`liamd` creates its state directories as owner-only (0700) and the files it
creates as owner-only (0600), whatever the umask says. At start it tightens an
existing `~/.liam` that is wider, and the database files too. If you point a path
at a directory of your own, `liamd` never changes that directory's mode: it logs a
warning when the directory is readable by others, and you can fix it with `chmod 700`.
The model cache holds public files and keeps the default mode.

If `liam.toml` is not found, `liamd` runs with mock models and logs one warning
that names the path it looked for. Pass `--config <path>` or set `LIAM_CONFIG`.

The tools a client sees, with their fields, are listed in
[Tools](../reference/tools.md).

The shipped `liam.toml` uses the local embedder (`Snowflake/snowflake-arctic-embed-m-v2.0` as an int8 ONNX model on the CPU, cut to
384 dimensions; Qwen3-Embedding-0.6B on llama.cpp stays selectable through `embedder.model`). With
`provider = "mock"` the embedder is a hash-derived placeholder, so retrieval
leans on full-text and graph signals; that is for tests and dev.

## Try it with a real embedder

From your clone of this repo, build, switch the config to the `local`
provider if it is not already, then start the daemon as described under
[One daemon, many agents](#one-daemon-many-agents):

```sh
cargo build -p liam-daemon
```

Set `embedder.provider = "local"` in `liam.toml`; the first start downloads the model, or run `liam fetch-models` first (see the [Configuration table](../reference/configuration.md)
for the rest of the keys). Run `liamd` as shown above, then call
`recall` from your MCP client. The vector channel now carries real semantic
similarity instead of the mock embedder's hash-derived placeholder, so
results ranked mainly by meaning (not just shared words) start showing up
alongside the full-text and graph signals `recall` always blends in.
