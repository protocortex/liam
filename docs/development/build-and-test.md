# Build and test

[Back to the docs index](../README.md)

Rust stable, edition 2021.

```sh
cargo build --workspace          # mock embedder, no ML deps
cargo test  --workspace          # libSQL backend, in-memory
```

The base build stays light. The in-process model stack (fastembed, candle, ONNX
runtime) pulls a large dependency tree and is opt-in:

```sh
cargo build -p liam-daemon --features local
```
