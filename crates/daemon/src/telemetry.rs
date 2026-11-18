// SPDX-License-Identifier: Apache-2.0
//! Structured logging to STDERR. Never stdout: the MCP stdio transport carries
//! JSON-RPC there, and any stray write corrupts the protocol.

use tracing_subscriber::{fmt, EnvFilter};

/// The tracing target llama.cpp's own log lines arrive under.
const LLAMA_CPP_TARGET: &str = "llama-cpp-2";

/// The filter used when `RUST_LOG` is not set: `default_filter`, with
/// llama.cpp held to warnings so its model-loading and shader-compile lines
/// stay out of the output. Set `RUST_LOG=info,llama-cpp-2=debug` to see them.
fn default_env_filter(default_filter: &str) -> EnvFilter {
    EnvFilter::new(format!("{default_filter},{LLAMA_CPP_TARGET}=warn"))
}

pub fn init(default_filter: &str) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| default_env_filter(default_filter));
    fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(true)
        .compact()
        .init();
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Clone, Default)]
    struct Buffer(Arc<Mutex<Vec<u8>>>);

    impl io::Write for Buffer {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> fmt::MakeWriter<'a> for Buffer {
        type Writer = Buffer;

        fn make_writer(&'a self) -> Buffer {
            self.clone()
        }
    }

    #[test]
    fn llama_cpp_info_lines_are_hidden_by_default_and_its_warnings_are_not() {
        // Arrange
        let buffer = Buffer::default();
        let subscriber = fmt()
            .with_env_filter(default_env_filter("info"))
            .with_writer(buffer.clone())
            .finish();

        // Act
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(target: "llama-cpp-2", "tensor loading");
            tracing::warn!(target: "llama-cpp-2", "metal fell back");
            tracing::info!("liam line");
        });

        // Assert
        let text = String::from_utf8(buffer.0.lock().unwrap().clone()).unwrap();
        assert!(!text.contains("tensor loading"), "{text}");
        assert!(text.contains("metal fell back"), "{text}");
        assert!(text.contains("liam line"), "{text}");
    }
}
