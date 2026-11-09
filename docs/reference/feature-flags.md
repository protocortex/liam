# Feature flags

[Back to the docs index](../README.md)

LIAM has few build flags, because every part of the product is needed. The
local embedder and reranker (fastembed), llama.cpp generation, libSQL storage,
Leiden community detection and the hosted-provider code are part of every
build. Hosted calls stay off at runtime until `[hosted]` is configured, and the
privacy gate and host allowlist decide what may leave the machine, not the
build. See [Hosted providers](../concepts/hosted-providers.md).

The flags that remain:

- `cuda` (`liam-daemon`, `liam-model`): NVIDIA support. It needs the CUDA
  toolkit and `nvcc` at build time, which many machines lack, so it is opt-in.
  Apple GPU support is automatic and needs nothing.
- `test-support` (`liam-store`, `liam-log`): constructors and fixtures that
  tests in other crates use. They stay out of a release build so production code
  cannot mint a recorded value it did not read from the store or the log.

The mock embedder, reranker and LLM are not build flags. They are runtime
choices (`provider = "mock"`) that tests and dev use.
