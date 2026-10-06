// SPDX-License-Identifier: Apache-2.0

//! Append-only event log for LIAM.
//!
//! The store keeps derived state (graph, facts, summaries); this crate keeps
//! the immutable record those are rebuilt from, so a write-ahead log and its
//! compacted Parquet segments outlive any change to the derived schema.

pub mod event;
