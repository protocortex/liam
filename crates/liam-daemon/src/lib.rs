// SPDX-License-Identifier: Apache-2.0
//! The internals shared by the two binaries this crate ships.
//!
//! `liamd` is the daemon that serves memory over MCP. `liam` is the
//! user-facing command line tool. They run at different times, but they have
//! to agree exactly on two things: how `liam.toml` is read, and where models
//! are stored. So those two modules live here rather than inside either
//! binary.
//!
//! That agreement is not a style preference. If the CLI resolved
//! `embedder.cache_dir` even slightly differently from the daemon, `liam
//! fetch-models` would download several gigabytes to one directory and the
//! daemon would then re-download them to another, with nothing in either
//! process able to notice. The unexpanded-tilde bug did exactly that once
//! already, from a single missed call site.
//!
//! The store lock is shared for a third reason: `liam rebuild` rewrites the
//! projection, so it must take the same lock the daemon holds for as long as
//! it runs. The stderr logging setup is shared so `liam` reports failures the
//! way the daemon does.
//!
//! Everything else the daemon owns (the MCP surface, the socket transport)
//! stays private to the `liamd` binary, because the CLI has no business
//! reaching into it.

pub mod config;
pub mod event_log;
pub mod models;
pub mod storelock;
pub mod telemetry;
