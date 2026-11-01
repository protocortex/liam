// SPDX-License-Identifier: Apache-2.0
//! Reading back the log events a node was written from.

use std::collections::HashMap;

use super::Graph;
use crate::backend::Backend;
use crate::error::Result;
use crate::ids::NodeId;
use crate::types::{EventRef, EventRole};
use crate::value::Value;

/// The events listed for one node, by role.
#[derive(Default)]
pub(super) struct ListedEvents {
    pub(super) origin: Vec<EventRef>,
    pub(super) basis: Vec<EventRef>,
}

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

    /// The events listed for each of `ids`, in one query. A node with none is
    /// absent from the result.
    pub(super) async fn listed_events(
        &self,
        ids: &[&str],
    ) -> Result<HashMap<String, ListedEvents>> {
        let placeholders = (1..=ids.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let params: Vec<Value> = ids.iter().map(|id| (*id).into()).collect();
        let rows = self
            .backend
            .query(
                &format!(
                    "SELECT node_id, event_id, role, synthetic FROM node_events
                     WHERE node_id IN ({placeholders}) ORDER BY node_id, event_id"
                ),
                &params,
            )
            .await?;
        let mut listed: HashMap<String, ListedEvents> = HashMap::new();
        for row in &rows {
            let event = EventRef {
                event_id: row.get_string(1)?,
                synthetic: row.get_i64(3)? != 0,
            };
            let entry = listed.entry(row.get_string(0)?).or_default();
            if row.get_string(2)? == EventRole::Origin.as_str() {
                entry.origin.push(event);
            } else {
                entry.basis.push(event);
            }
        }
        Ok(listed)
    }
}
