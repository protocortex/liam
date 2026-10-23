// SPDX-License-Identifier: Apache-2.0
//! The runs a `RunRolledBack` record cancels. A replay skips every write whose
//! source is such a run, and lets the writes that came after still apply.

use std::collections::{HashMap, HashSet};

use liam_log::event::{FactRowEffect, LogEvent, LogPayload};

/// The source prefix of the writes a consolidation run makes; the run id
/// follows it.
pub const RUN_SOURCE_PREFIX: &str = "consolidation:";

/// A fact write seen while scanning, kept as small as the reconciliation needs.
pub(super) struct SeenFactWrite {
    pub event_id: String,
    pub source: String,
    pub closed: Vec<String>,
    pub inserted: Vec<(String, String)>,
}

impl SeenFactWrite {
    pub(super) fn of(event: &LogEvent) -> Option<Self> {
        let LogPayload::FactWrite {
            closed_version_ids,
            inserted,
            ..
        } = &event.payload
        else {
            return None;
        };
        Some(Self {
            event_id: event.event_id.clone(),
            source: event.source.clone(),
            closed: closed_version_ids.clone(),
            inserted: inserted
                .iter()
                .map(|row| (row.id.clone(), row.lineage_id.clone()))
                .collect(),
        })
    }
}

#[derive(Default)]
pub(super) struct RolledBack {
    runs: HashSet<String>,
    /// Fact versions that skipped writes would have inserted, with the
    /// versions each of those writes closed.
    missing: HashMap<String, Vec<String>>,
    /// Lineages that skipped writes would have created.
    lineages: HashSet<String>,
}

impl RolledBack {
    /// `voided` holds the cancelled events; a cancelled write or marker counts
    /// for nothing.
    pub(super) fn new(
        runs: HashSet<String>,
        fact_writes: &[SeenFactWrite],
        voided: &HashSet<String>,
    ) -> Self {
        let mut rolled_back = Self {
            runs,
            ..Self::default()
        };
        for write in fact_writes {
            if voided.contains(&write.event_id) || !rolled_back.owns(&write.source) {
                continue;
            }
            for (id, lineage) in &write.inserted {
                rolled_back.missing.insert(id.clone(), write.closed.clone());
                // A version is the first of its lineage when it names itself.
                if id == lineage {
                    rolled_back.lineages.insert(lineage.clone());
                }
            }
        }
        rolled_back
    }

    fn owns(&self, source: &str) -> bool {
        source
            .strip_prefix(RUN_SOURCE_PREFIX)
            .is_some_and(|run_id| self.runs.contains(run_id))
    }

    /// Whether the event is a write of a rolled-back run. The source must match
    /// the whole run id, so rolling back run 1 leaves run 12 alone.
    pub(super) fn skips(&self, event: &LogEvent) -> bool {
        self.owns(&event.source)
    }

    /// A write that built on a skipped write's rows applies without them: it
    /// does not close a version that never existed, and closes the versions the
    /// skipped write would have closed, so one belief stays open for the key. It
    /// drops the later versions and source links of lineages the skipped writes
    /// created.
    pub(super) fn reconcile(&mut self, payload: LogPayload) -> LogPayload {
        let LogPayload::FactWrite {
            at,
            closed_version_ids,
            inserted,
            source_links,
            suppressions,
        } = payload
        else {
            return payload;
        };
        if self.missing.is_empty() {
            return LogPayload::FactWrite {
                at,
                closed_version_ids,
                inserted,
                source_links,
                suppressions,
            };
        }
        let closed_version_ids = self.surviving_closes(closed_version_ids);
        let (dropped, inserted): (Vec<FactRowEffect>, Vec<FactRowEffect>) = inserted
            .into_iter()
            .partition(|row| self.lineages.contains(&row.lineage_id));
        for row in dropped {
            self.missing.insert(row.id, Vec::new());
        }
        let source_links = source_links
            .into_iter()
            .filter(|(lineage, _)| !self.lineages.contains(lineage))
            .collect();
        LogPayload::FactWrite {
            at,
            closed_version_ids,
            inserted,
            source_links,
            suppressions,
        }
    }

    /// The versions to close once the missing ones are replaced by what their
    /// writes had closed, followed through any chain of skipped writes.
    fn surviving_closes(&self, closes: Vec<String>) -> Vec<String> {
        let mut pending = closes;
        let mut closed: Vec<String> = Vec::new();
        while let Some(id) = pending.pop() {
            match self.missing.get(&id) {
                Some(restored) => pending.extend(restored.iter().cloned()),
                None if !closed.contains(&id) => closed.push(id),
                None => {}
            }
        }
        closed.reverse();
        closed
    }
}
