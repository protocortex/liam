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

The tools a client sees, with their fields, are listed in
[Tools](../reference/tools.md).

The shipped `liam.toml` uses the local embedder and reranker. With
`provider = "mock"` the embedder is a hash-derived placeholder, so retrieval
leans on full-text and graph signals; that is for tests and dev.

## Try it with a real embedder

From your clone of this repo, build, switch the config to the `local`
provider if it is not already, then start the daemon as described under
[One daemon, many agents](#one-daemon-many-agents):

```sh
cargo build -p liam-daemon
```

Set `embedder.provider = "local"` in `liam.toml` (see the [Configuration table](../reference/configuration.md)
for the rest of the keys). Run `liamd` as shown above, then call
`recall` from your MCP client. The vector channel now carries real semantic
similarity instead of the mock embedder's hash-derived placeholder, so
results ranked mainly by meaning (not just shared words) start showing up
alongside the full-text and graph signals `recall` always blends in.
