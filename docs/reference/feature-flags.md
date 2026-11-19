# Feature flags

[Back to the docs index](../README.md)

LIAM has few build flags, because every part of the product is needed. The
local embedder (arctic-embed-m-v2.0 on the ONNX runtime, through fastembed), libSQL storage,
Leiden community detection and the hosted-provider code are part of every
build. Hosted calls stay off at runtime until `[hosted]` is configured, and the
privacy gate and host allowlist decide what may leave the machine, not the
build. See [Hosted providers](../concepts/hosted-providers.md).

The one flag that remains:

- `test-support` (`liam-store`, `liam-log`): constructors and fixtures that
  tests in other crates use. They stay out of a release build so production code
  cannot mint a recorded value it did not read from the store or the log.

The mock embedder is not a build flag. It is a runtime choice
(`provider = "mock"`) that tests and dev use.
