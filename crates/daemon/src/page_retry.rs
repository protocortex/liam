// SPDX-License-Identifier: Apache-2.0
//! Bounds how often the maintenance tick retries an entity page build that
//! keeps failing, so one broken entity cannot take a slot on every tick.

use std::collections::HashMap;
use std::sync::Mutex;

use liam_store::{Fingerprint, Millis, NodeId};

/// Failed builds of one entity after which it is given up on for a while.
const MAX_ATTEMPTS: u32 = 3;

/// The wait after the first failure. Each further failure doubles it.
const FIRST_RETRY_DELAY_MS: i64 = 10 * 60 * 1000;

/// How long an entity that used up its attempts is left alone.
const GIVE_UP_PAUSE_MS: i64 = 6 * 60 * 60 * 1000;

/// What a recorded failure means for the log.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Failure {
    /// The first failure since the entity last built or changed.
    First,
    /// A later failure, with attempts left.
    Again,
    /// The last attempt failed; the entity is paused for `GIVE_UP_PAUSE_MS`.
    GivenUp,
}

struct Record {
    failures: u32,
    last_failed_at: Millis,
    subject: String,
    scope: Option<String>,
    /// The mentions the failed build saw. A different set means new input, so
    /// the count starts again.
    seen: Fingerprint,
}

impl Record {
    fn retry_at(&self) -> i64 {
        let wait = if self.failures >= MAX_ATTEMPTS {
            GIVE_UP_PAUSE_MS
        } else {
            FIRST_RETRY_DELAY_MS << (self.failures - 1)
        };
        self.last_failed_at.0 + wait
    }
}

/// The failed builds the daemon remembers, by entity row.
#[derive(Default)]
pub(crate) struct PageRetries {
    records: Mutex<HashMap<NodeId, Record>>,
}

impl PageRetries {
    fn records(&self) -> std::sync::MutexGuard<'_, HashMap<NodeId, Record>> {
        self.records.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The entities whose next attempt is not due yet.
    pub(crate) fn paused(&self, now: Millis) -> Vec<NodeId> {
        self.records()
            .iter()
            .filter(|(_, record)| record.retry_at() > now.0)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// The entities with a failure on record, with what is needed to ask
    /// whether their mentions have changed since.
    pub(crate) fn recorded(&self) -> Vec<(NodeId, String, Option<String>, Fingerprint)> {
        self.records()
            .iter()
            .map(|(id, r)| (id.clone(), r.subject.clone(), r.scope.clone(), r.seen))
            .collect()
    }

    pub(crate) fn record_failure(
        &self,
        id: &NodeId,
        subject: &str,
        scope: Option<&str>,
        seen: Fingerprint,
        now: Millis,
    ) -> Failure {
        let mut records = self.records();
        let record = records.entry(id.clone()).or_insert_with(|| Record {
            failures: 0,
            last_failed_at: now,
            subject: subject.to_string(),
            scope: scope.map(str::to_string),
            seen,
        });
        record.failures += 1;
        record.last_failed_at = now;
        match record.failures {
            1 => Failure::First,
            n if n >= MAX_ATTEMPTS => Failure::GivenUp,
            _ => Failure::Again,
        }
    }

    pub(crate) fn clear(&self, id: &NodeId) {
        self.records().remove(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint(edges: i64) -> Fingerprint {
        Fingerprint {
            edge_count: edges,
            max_tx_from: Millis(1),
        }
    }

    fn fail(retries: &PageRetries, id: &NodeId, at: i64) -> Failure {
        retries.record_failure(id, "ada", None, fingerprint(1), Millis(at))
    }

    #[test]
    fn the_wait_doubles_after_each_failure_and_the_last_one_gives_up() {
        // Arrange
        let retries = PageRetries::default();
        let id = NodeId::from_raw("e1");

        // Act
        let outcomes = [
            fail(&retries, &id, 0),
            fail(&retries, &id, 0),
            fail(&retries, &id, 0),
        ];

        // Assert
        assert_eq!(outcomes, [Failure::First, Failure::Again, Failure::GivenUp]);
        assert_eq!(
            retries.paused(Millis(GIVE_UP_PAUSE_MS - 1)),
            std::slice::from_ref(&id)
        );
        assert!(retries.paused(Millis(GIVE_UP_PAUSE_MS)).is_empty());
    }

    #[test]
    fn a_failed_entity_is_paused_until_its_delay_passes() {
        // Arrange
        let retries = PageRetries::default();
        let id = NodeId::from_raw("e1");
        fail(&retries, &id, 100);

        // Act / Assert
        assert_eq!(retries.paused(Millis(100 + FIRST_RETRY_DELAY_MS - 1)), [id]);
        assert!(retries
            .paused(Millis(100 + FIRST_RETRY_DELAY_MS))
            .is_empty());
    }

    #[test]
    fn clearing_an_entity_forgets_its_failures() {
        // Arrange
        let retries = PageRetries::default();
        let id = NodeId::from_raw("e1");
        fail(&retries, &id, 0);

        // Act
        retries.clear(&id);

        // Assert
        assert!(retries.paused(Millis(0)).is_empty());
        assert!(retries.recorded().is_empty());
        assert_eq!(fail(&retries, &id, 0), Failure::First);
    }
}
