# Feature flags

[Back to the docs index](../README.md)

libSQL storage, Leiden community detection and the hosted-provider code are
always built; there is no feature to turn them off. Hosted calls stay off at
runtime until `[hosted]` is configured.

`liam-daemon`:

- `local`: load fastembed in-process (Qwen3 embedder, cross-encoder reranker).
  Off by default, which keeps the dev build small.
- `llama`: llama.cpp text generation, which is what `ask` needs. Metal comes
  from the macOS target, not from this flag.
- `cuda`: NVIDIA support. Needs the CUDA toolkit at build time, so it is
  opt-in; Apple GPU support is automatic and needs nothing.

A release build sets `local,llama`. Without them the daemon refuses to start
against a config naming those providers, rather than falling back to mocks
that would return random vectors and invented answers.
