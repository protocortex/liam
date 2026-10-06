// SPDX-License-Identifier: Apache-2.0

//! Append-only event log for LIAM.
//!
//! The store keeps derived state (graph, facts, summaries); this crate keeps
//! the immutable record those are rebuilt from, so a write-ahead log and its
//! compacted Parquet segments outlive any change to the derived schema.

#[cfg(test)]
mod tests {
    #[test]
    fn crate_links() {
        assert_eq!(env!("CARGO_PKG_NAME"), "liam-log");
    }
}
