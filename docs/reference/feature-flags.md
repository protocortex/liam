# Feature flags

[Back to the docs index](../README.md)

`liam-store`:

- `backend-libsql` (default): libSQL with native vector search (`F32_BLOB`,
  `vector_distance_cos`).
Leiden community detection is unconditional; there is no feature to disable it.

`liam-daemon`:

- `local`: load fastembed in-process (Qwen3 embedder, cross-encoder reranker).
  Off by default, which keeps the dev build small.
- `llama`: llama.cpp text generation, which is what `ask` needs. Metal comes
  from the macOS target, not from this flag.
- `hosted`: calls to Anthropic, OpenAI and Gemini over HTTPS (see
  [Hosted providers](../../README.md#hosted-providers)). Off by default, which keeps TLS out of the default build.
- `cuda`: NVIDIA support. Needs the CUDA toolkit at build time, so it is
  opt-in; Apple GPU support is automatic and needs nothing.

A release build sets `local,llama`. Without them the daemon refuses to start
against a config naming those providers, rather than falling back to mocks
that would return random vectors and invented answers.
