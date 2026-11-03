// SPDX-License-Identifier: Apache-2.0
//! Checking, before a vector is dropped, that the log still holds what the
//! node says: a demoted node's row keeps its text, but the log is where it
//! could be rebuilt from.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};

use futures_util::StreamExt;
use liam_log::event::{LogEvent, LogPayload, NodeRow, RowEffect};
use liam_log::reader::LogStream;

use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::graph::logged_write::HeldLog;
use crate::graph::replay::{scan_ahead, Ahead};
use crate::graph::Graph;
use crate::ids::NodeId;
use crate::types::{OriginClaim, OriginVerdict};

impl<B: Backend> Graph<B> {
    /// Judges each claim against the log, which is read once. A claim is
    /// verified when one of its origin events is in the log, in force, and
    /// writes the label and content the claim gives. An event that duplicates
    /// an earlier one is judged by that earlier event. A second read happens
    /// only to find the earlier events of duplicates.
    ///
    /// # Errors
    ///
    /// Fails without an event log, on a poisoned log, and when the log cannot
    /// be read.
    pub async fn verify_origins(
        &self,
        claims: &[OriginClaim],
    ) -> Result<HashMap<NodeId, OriginVerdict>> {
        let log = self.log.as_ref().ok_or(Error::NoEventLog)?;
        let (reader, head) = {
            let held = HeldLog::acquire(log).await?;
            (held.reader()?, held.head()?)
        };
        let Some(head) = head else {
            return Ok(judge_all(claims, &HashMap::new(), &Ahead::default()));
        };
        let wanted: HashSet<String> = claims
            .iter()
            .flat_map(|claim| claim.origin_event_ids.iter().cloned())
            .collect();
        let found = Arc::new(Mutex::new(HashMap::<String, LogEvent>::new()));
        let tap = Arc::clone(&found);
        let stream: LogStream = Box::pin(reader.scan_through(None, head).inspect(move |record| {
            if let Ok(record) = record {
                if wanted.contains(&record.event.event_id) {
                    tap.lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .insert(record.event.event_id.clone(), record.event.clone());
                }
            }
        }));
        let ahead = scan_ahead(stream).await?;
        let mut events = std::mem::take(&mut *found.lock().unwrap_or_else(PoisonError::into_inner));
        let mut firsts: HashSet<String> = events
            .values()
            .filter_map(|event| match &event.payload {
                LogPayload::DuplicateOf { first_event_id } => Some(first_event_id.clone()),
                _ => None,
            })
            .filter(|first| !events.contains_key(first))
            .collect();
        if !firsts.is_empty() {
            let mut records = reader.scan_through(None, head);
            while let Some(record) = records.next().await {
                let event = record?.event;
                if firsts.remove(&event.event_id) {
                    events.insert(event.event_id.clone(), event);
                    if firsts.is_empty() {
                        break;
                    }
                }
            }
        }
        Ok(judge_all(claims, &events, &ahead))
    }
}

fn judge_all(
    claims: &[OriginClaim],
    events: &HashMap<String, LogEvent>,
    ahead: &Ahead,
) -> HashMap<NodeId, OriginVerdict> {
    claims
        .iter()
        .map(|claim| (claim.node_id.clone(), judge(claim, events, ahead)))
        .collect()
}

/// The best verdict among the claim's origin events.
fn judge(claim: &OriginClaim, events: &HashMap<String, LogEvent>, ahead: &Ahead) -> OriginVerdict {
    claim
        .origin_event_ids
        .iter()
        .map(|id| judge_event(claim, id, events, ahead))
        .max_by_key(|verdict| strength(*verdict))
        .unwrap_or(OriginVerdict::OriginMissing)
}

fn strength(verdict: OriginVerdict) -> u8 {
    match verdict {
        OriginVerdict::OriginMissing => 0,
        OriginVerdict::Excluded => 1,
        OriginVerdict::Mismatch => 2,
        OriginVerdict::Verified => 3,
    }
}

fn judge_event(
    claim: &OriginClaim,
    event_id: &str,
    events: &HashMap<String, LogEvent>,
    ahead: &Ahead,
) -> OriginVerdict {
    let Some(event) = events.get(event_id) else {
        return OriginVerdict::OriginMissing;
    };
    let excluded = |event: &LogEvent| {
        ahead.voided.contains(&event.event_id)
            || ahead.rolled_back.skips(event)
            || ahead.removed.nodes.contains(claim.node_id.as_str())
            || ahead.shredded_nodes.contains(claim.node_id.as_str())
    };
    if excluded(event) {
        return OriginVerdict::Excluded;
    }
    let written = match &event.payload {
        LogPayload::DuplicateOf { first_event_id } => {
            let Some(first) = events.get(first_event_id) else {
                return OriginVerdict::OriginMissing;
            };
            if excluded(first) {
                return OriginVerdict::Excluded;
            }
            &first.payload
        }
        payload => payload,
    };
    match written_row(written, &claim.node_id) {
        Some(row) if row.label == claim.label && row.content == claim.content => {
            OriginVerdict::Verified
        }
        _ => OriginVerdict::Mismatch,
    }
}

/// The node row a payload writes for `node_id`. A single node write is the
/// row whatever its id, since the first of two duplicates names another node;
/// a batch is searched by id.
fn written_row<'a>(payload: &'a LogPayload, node_id: &NodeId) -> Option<&'a NodeRow> {
    match payload {
        LogPayload::NodeWrite(row) => Some(row),
        LogPayload::EpisodeBatch(effects) => effects.iter().find_map(|effect| match effect {
            RowEffect::Node(row) if row.id == node_id.as_str() => Some(row),
            _ => None,
        }),
        _ => None,
    }
}
