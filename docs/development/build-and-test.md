# Build and test

[Back to the docs index](../README.md)

Rust stable, edition 2021.

```sh
cargo build --workspace
cargo test  --workspace          # libSQL backend, in-memory
```

Every build includes the in-process model stack (fastembed with the ONNX
runtime). Tests use the mock embedder (`provider = "mock"`), so they
download no model. The tests that load real
models are `#[ignore]`d and run only when you ask for them.
