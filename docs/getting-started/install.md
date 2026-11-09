# Install

[Back to the docs index](../README.md)

## Install with Homebrew

On Apple Silicon macOS:

```sh
brew tap protocortex/tap
brew install liam              # the latest tagged release
brew install --HEAD liam       # builds main from source with cargo; slow
```

The formula does not download models. Its caveats say how to fetch them with
`liam fetch-models` and how to point an MCP client at `liamd proxy`. The tap
lives in [protocortex/homebrew-tap](https://github.com/protocortex/homebrew-tap).

## Packaging

Models run inside the binary, so shipping means getting weights onto the
machine before the first query. Releases take the fetch-on-install route: the
tarball carries `liamd`, `liam`, a config, and `install.sh`, and weighs a few
megabytes.

```sh
tar -xzf liam-vX.Y.Z-aarch64-apple-darwin-netinstall.tar.gz -C liam
cd liam && ./install.sh
```

`install.sh` copies both binaries to `~/.local/bin`, writes `~/.liam/liam.toml`
if you do not already have one, runs `liam fetch-models`, and registers the
launchd agent. Use `--prefix`, `--skip-models`, or `--no-launchd` to change
that. Running it again is safe.

`liam fetch-models` downloads **and loads** each model. Loading is the point: a
truncated file downloads happily and only fails when something tries to use it,
which would otherwise be a user's first `recall`, long after the install.

With the default config the embedder is the Q8_0 GGUF of Qwen3-Embedding-0.6B (about 0.6 GB, checked
against a pinned sha256). The candle safetensors (about 2.4 GB) are downloaded only when
`embedder.model = "Qwen/Qwen3-Embedding-0.6B"`.

Both models land under `embedder.cache_dir`, each in its own `models--Org--Name`
subdirectory. That makes a bundle-in-release option possible (ship the weights
in the tarball, pointed at by that same directory), though fetch-on-install is
still the route the current release workflow takes.
