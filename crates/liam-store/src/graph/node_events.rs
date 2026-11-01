// SPDX-License-Identifier: Apache-2.0
//! Reading back the log events a node was written from.

use super::Graph;
use crate::backend::Backend;
use crate::error::Result;
use crate::ids::NodeId;
use crate::types::{EventRef, EventRole};

impl<B: Backend> Graph<B> {
    /// The events listed for `node_id` in `role`, ordered by event id. A node
    /// written before events were listed has none.
    pub async fn node_event_ids(&self, node_id: &NodeId, role: EventRole) -> Result<Vec<EventRef>> {
        let rows = self
            .backend
            .query(
                "SELECT event_id, synthetic FROM node_events
                 WHERE node_id = ?1 AND role = ?2 ORDER BY event_id",
                &[node_id.as_str().into(), role.as_str().into()],
            )
            .await?;
        rows.iter()
            .map(|row| {
                Ok(EventRef {
                    event_id: row.get_string(0)?,
                    synthetic: row.get_i64(1)? != 0,
                })
            })
            .collect()
    }
}
