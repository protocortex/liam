// SPDX-License-Identifier: Apache-2.0
//! The consolidation job: a snapshot, a set of maintenance steps and a probe
//! gate that commits or rolls back. Named "consolidation" to stay apart from
//! the event log's unrelated WAL-to-Parquet compactor.

pub mod dedup;
pub mod dedup_cursor;
pub mod determinism;
pub mod hooks;
pub mod maintenance;
pub mod manifest;
pub mod probe_set;
pub mod probes;
pub mod run;
pub mod settings;
pub mod shared_store;
