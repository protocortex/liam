// SPDX-License-Identifier: Apache-2.0
//! The `feedback` tool: a producer's verdict on a result `recall` or `ask`
//! gave it, which the utility score reads as how often the item was right.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use liam_store::NodeId;
use schemars::JsonSchema;
use serde::Deserialize;

use super::MemoryServer;
use crate::tiering::usage::BoundedMap;

/// How far back the per-producer rate limit looks.
const THROTTLE_WINDOW: Duration = Duration::from_secs(3600);

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FeedbackArgs {
    /// The handle `recall` or `ask` printed for the result, as shown.
    pub result_id: String,
    /// True when the result was right and useful, false when it was wrong.
    pub correct: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum FeedbackError {
    /// Tiering is off, so verdicts have no use.
    Disabled,
    /// Empty, or with a character no handle has.
    InvalidHandle,
    /// No result with that handle was returned to this connection.
    NotIssued(String),
    /// Two results this connection saw share the handle.
    Ambiguous {
        handle: String,
        candidates: Vec<String>,
    },
    /// The result has been removed from memory.
    Deleted(String),
    RateLimited {
        producer: String,
        retry_after: Duration,
    },
    Store(String),
}

impl fmt::Display for FeedbackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => write!(f, "tiering is not enabled, so verdicts are not kept"),
            Self::InvalidHandle => write!(
                f,
                "the handle must be letters and digits, as recall and ask print it"
            ),
            Self::NotIssued(handle) => write!(
                f,
                "no result with handle {handle} was returned on this connection"
            ),
            Self::Ambiguous { handle, candidates } => write!(
                f,
                "handle {handle} matches more than one result: {}",
                candidates.join(", ")
            ),
            Self::Deleted(handle) => {
                write!(f, "the memory behind handle {handle} no longer exists")
            }
            Self::RateLimited {
                producer,
                retry_after,
            } => write!(
                f,
                "{producer} has used its verdicts for this hour; try again in {} seconds",
                retry_after.as_secs().max(1)
            ),
            Self::Store(reason) => write!(f, "{reason}"),
        }
    }
}

/// Caps how many verdicts one producer gives in an hour. Clients that do not
/// name themselves share one producer, and so one budget.
pub(super) struct Throttle {
    per_hour: u32,
    given: HashMap<String, VecDeque<Instant>>,
}

impl Throttle {
    pub(super) fn new(per_hour: u32) -> Self {
        Self {
            per_hour,
            given: HashMap::new(),
        }
    }

    /// Counts a verdict by `producer` at `now`, or says how long until one
    /// more is allowed.
    pub(super) fn admit(&mut self, producer: &str, now: Instant) -> Result<(), Duration> {
        let recent = self.given.entry(producer.to_string()).or_default();
        while let Some(&oldest) = recent.front() {
            if now.saturating_duration_since(oldest) < THROTTLE_WINDOW {
                break;
            }
            recent.pop_front();
        }
        if recent.len() >= self.per_hour as usize {
            let oldest = recent.front().copied().unwrap_or(now);
            return Err(THROTTLE_WINDOW.saturating_sub(now.saturating_duration_since(oldest)));
        }
        recent.push_back(now);
        Ok(())
    }
}

/// The results this connection was shown, by full id, up to a bound. Cloning
/// gives the new connection an empty set, because a handle is only good to the
/// connection it was issued to.
pub(crate) struct IssuedHandles {
    capacity: usize,
    issued: Mutex<BoundedMap<NodeId, ()>>,
}

impl IssuedHandles {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            issued: Mutex::new(BoundedMap::new(capacity)),
        }
    }

    pub(crate) fn issue<'a>(&self, nodes: impl IntoIterator<Item = &'a NodeId>) {
        let mut issued = self.issued.lock().unwrap_or_else(PoisonError::into_inner);
        for node in nodes {
            issued.insert(node.clone(), ());
        }
    }

    /// Drops the handles of `nodes` and returns how many were held.
    pub(crate) fn forget(&self, nodes: &HashSet<String>) -> usize {
        let mut issued = self.issued.lock().unwrap_or_else(PoisonError::into_inner);
        let held = issued_among(&issued, nodes);
        for node in &held {
            issued.remove(node);
        }
        held.len()
    }

    /// The ids among `nodes` that were issued, sorted.
    pub(crate) fn holds(&self, nodes: &HashSet<String>) -> Vec<String> {
        let issued = self.issued.lock().unwrap_or_else(PoisonError::into_inner);
        let mut held: Vec<String> = issued_among(&issued, nodes)
            .iter()
            .map(|node| node.as_str().to_owned())
            .collect();
        held.sort();
        held
    }

    /// The one issued node whose id starts with `handle`, in any letter case.
    pub(super) fn resolve(&self, handle: &str) -> Result<NodeId, FeedbackError> {
        let sent = handle.trim();
        if sent.is_empty() || !sent.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err(FeedbackError::InvalidHandle);
        }
        let wanted = sent.to_ascii_uppercase();
        let issued = self.issued.lock().unwrap_or_else(PoisonError::into_inner);
        let mut matching: Vec<NodeId> = issued
            .iter()
            .filter(|(node, _)| node.as_str().to_ascii_uppercase().starts_with(&wanted))
            .map(|(node, _)| node.clone())
            .collect();
        matching.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        match matching.len() {
            0 => Err(FeedbackError::NotIssued(sent.to_string())),
            1 => Ok(matching.remove(0)),
            _ => Err(FeedbackError::Ambiguous {
                handle: sent.to_string(),
                candidates: matching.iter().map(|n| n.as_str().to_string()).collect(),
            }),
        }
    }
}

fn issued_among(issued: &BoundedMap<NodeId, ()>, nodes: &HashSet<String>) -> Vec<NodeId> {
    issued
        .iter()
        .filter(|(node, _)| nodes.contains(node.as_str()))
        .map(|(node, _)| node.clone())
        .collect()
}

impl Clone for IssuedHandles {
    fn clone(&self) -> Self {
        Self::new(self.capacity)
    }
}

/// What the tool needs when tiering is on: the budget every connection shares
/// and the handles this one was given.
#[derive(Clone)]
pub(super) struct Feedback {
    throttle: Arc<Mutex<Throttle>>,
    issued: IssuedHandles,
}

impl Feedback {
    pub(super) fn new(per_hour: u32, max_issued: usize) -> Self {
        Self {
            throttle: Arc::new(Mutex::new(Throttle::new(per_hour))),
            issued: IssuedHandles::new(max_issued),
        }
    }

    pub(super) fn issue<'a>(&self, nodes: impl IntoIterator<Item = &'a NodeId>) {
        self.issued.issue(nodes);
    }
}

pub(super) async fn give(server: &MemoryServer, args: FeedbackArgs) -> String {
    match record(server, args).await {
        Ok(()) => "feedback recorded".to_string(),
        Err(error) => format!("feedback failed: {error}"),
    }
}

async fn record(server: &MemoryServer, args: FeedbackArgs) -> Result<(), FeedbackError> {
    let feedback = server.feedback.as_ref().ok_or(FeedbackError::Disabled)?;
    let node = feedback.issued.resolve(&args.result_id)?;
    let producer = server.producer();
    feedback
        .throttle
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .admit(&producer, Instant::now())
        .map_err(|retry_after| FeedbackError::RateLimited {
            producer: producer.clone(),
            retry_after,
        })?;
    let store = server
        .store()
        .await
        .map_err(|error| FeedbackError::Store(error.to_string()))?;
    let lineage = store
        .lineage_of(&node)
        .await
        .map_err(|error| FeedbackError::Store(error.to_string()))?
        .ok_or_else(|| FeedbackError::Deleted(args.result_id.trim().to_string()))?;
    store
        .put_feedback_logged(&lineage, &producer, args.correct)
        .await
        .map_err(|error| FeedbackError::Store(error.to_string()))
}

#[cfg(test)]
mod tests;
