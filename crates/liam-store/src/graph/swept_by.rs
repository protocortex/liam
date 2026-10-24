// SPDX-License-Identifier: Apache-2.0
//! Which nodes the log records as swept by one write source.

use std::collections::HashSet;

use futures_util::StreamExt;
use liam_log::event::{LogPayload, TombstoneTable};

use super::logged_write::HeldLog;
use super::Graph;
use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::ids::NodeId;

impl<B: Backend> Graph<B> {
    /// The ids of the nodes whose tombstones carry `source`, as the log holds
    /// them now. A consolidation run asks this of its own source to tell the
    /// nodes its sweep removed on purpose from nodes that vanished otherwise.
    ///
    /// # Errors
    ///
    /// Fails without an event log, on a poisoned log, and when the log cannot
    /// be read.
    pub async fn nodes_swept_by(&self, source: &str) -> Result<HashSet<NodeId>> {
        let log = self.log.as_ref().ok_or(Error::NoEventLog)?;
        let (reader, head) = {
            let held = HeldLog::acquire(log).await?;
            (held.reader()?, held.head()?)
        };
        let mut swept = HashSet::new();
        let Some(head) = head else {
            return Ok(swept);
        };
        let mut records = reader.scan_through(None, head);
        while let Some(record) = records.next().await {
            let event = record?.event;
            if event.source != source {
                continue;
            }
            if let LogPayload::Tombstone(targets) = event.payload {
                swept.extend(
                    targets
                        .into_iter()
                        .filter(|target| target.table == TombstoneTable::Nodes)
                        .map(|target| NodeId::from_raw(&target.id)),
                );
            }
        }
        Ok(swept)
    }
}
