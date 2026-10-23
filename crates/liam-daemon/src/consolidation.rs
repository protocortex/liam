// SPDX-License-Identifier: Apache-2.0
//! The consolidation job: a snapshot, a set of maintenance steps and a probe
//! gate that commits or rolls back. Named "consolidation" to stay apart from
//! the event log's unrelated WAL-to-Parquet compactor.

pub mod manifest;
pub mod probes;
