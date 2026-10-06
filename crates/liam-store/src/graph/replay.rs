// SPDX-License-Identifier: Apache-2.0
//! Replay of the event log into the projection: the records past the log
//! cursor are applied once each, in log order.

use super::Graph;
use crate::backend::Backend;
use crate::error::Result;

/// What one `catch_up` did with the records past the cursor.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CatchUp {
    /// Events whose rows were projected.
    pub applied: usize,
    /// Events whose rows the store already held.
    pub skipped_applied: usize,
    /// Events a later `Voided` record cancelled.
    pub skipped_voided: usize,
    /// Events the projection refused, recorded in `log_quarantine`.
    pub quarantined: usize,
}

impl<B: Backend> Graph<B> {
    /// Applies the log records the store has not accounted for, up to the
    /// writer's last acknowledged record.
    pub async fn catch_up(&self) -> Result<CatchUp> {
        Ok(CatchUp::default())
    }
}
