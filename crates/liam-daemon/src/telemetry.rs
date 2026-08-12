// SPDX-License-Identifier: AGPL-3.0-only
//! Structured logging to STDERR. Never stdout: the MCP stdio transport carries
//! JSON-RPC there, and any stray write corrupts the protocol.

use tracing_subscriber::{fmt, EnvFilter};

pub fn init(default_filter: &str) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));
    fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(true)
        .compact()
        .init();
}
