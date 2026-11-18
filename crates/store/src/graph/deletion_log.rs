// SPDX-License-Identifier: Apache-2.0
//! The records a delete job leaves in the log, and the key groups a subject
//! owns. A record is appended once however often a resumed job asks for it.

use futures_util::StreamExt;
use liam_log::event::LogPayload;

use super::logged_plan::{event, STORE_SOURCE, STORE_TRUST};
use super::logged_write::HeldLog;
use super::{scope_within, Graph};
use crate::backend::Backend;
use crate::error::Result;
use crate::value::Value;

/// What a delete job writes into the log. A replay reads these to treat the
/// rows of a deleted event as shredded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeletionRecord {
    /// The job committed to destroying `key_ids` and the rows of `event_ids`.
    Recorded {
        job_id: String,
        event_ids: Vec<String>,
        key_ids: Vec<String>,
    },
    KeyDestroyed {
        key_id: String,
    },
    /// The job was cancelled before it destroyed anything.
    Cancelled {
        job_id: String,
    },
}

impl DeletionRecord {
    fn payload(&self, at: i64) -> LogPayload {
        match self {
            Self::Recorded {
                job_id,
                event_ids,
                key_ids,
            } => LogPayload::DeletionRecorded {
                job_id: job_id.clone(),
                event_ids: event_ids.clone(),
                key_ids: key_ids.clone(),
                at,
            },
            Self::KeyDestroyed { key_id } => LogPayload::KeyDestroyed {
                key_id: key_id.clone(),
                at,
            },
            Self::Cancelled { job_id } => LogPayload::DeletionCancelled {
                job_id: job_id.clone(),
            },
        }
    }

    /// Whether `payload` is this record, whenever it was written.
    fn is_written_as(&self, payload: &LogPayload) -> bool {
        match (self, payload) {
            (Self::Recorded { job_id, .. }, LogPayload::DeletionRecorded { job_id: other, .. })
            | (Self::Cancelled { job_id }, LogPayload::DeletionCancelled { job_id: other }) => {
                job_id == other
            }
            (Self::KeyDestroyed { key_id }, LogPayload::KeyDestroyed { key_id: other, .. }) => {
                key_id == other
            }
            _ => false,
        }
    }
}

/// Selects the key groups of a subject; `{scope}` is filled by `scope_filter`.
/// The subject is matched the way the key registry normalises it (trimmed,
/// lowercased).
const SUBJECT_KEYS_SQL: &str = "SELECT DISTINCT nk.key_id FROM node_keys nk
     JOIN nodes n ON n.id = nk.node_id
     WHERE LOWER(TRIM(n.subject)) = ?1 AND {scope}";

/// The scope a subject lives in: exactly `scope` (none for an unscoped one), or
/// with `include_children` that scope and every one below it.
fn scope_filter(scope: Option<&str>, include_children: bool) -> (String, Vec<Value>) {
    match (scope, include_children) {
        (None, false) => ("COALESCE(n.scope, '') = ''".to_owned(), Vec::new()),
        (None, true) => ("1 = 1".to_owned(), Vec::new()),
        (Some(scope), false) => (
            "COALESCE(n.scope, '') = ?2".to_owned(),
            vec![Value::from(scope)],
        ),
        (Some(scope), true) => (
            scope_within("COALESCE(n.scope, '')", 2),
            vec![Value::from(scope)],
        ),
    }
}

impl<B: Backend> Graph<B> {
    /// Appends `record` unless the log already holds it, and says whether it
    /// did. The check and the append run under the log lock, so a job resumed
    /// after a crash between the append and its own bookkeeping adds nothing.
    pub async fn record_deletion(&self, record: &DeletionRecord) -> Result<bool> {
        let Some(log) = &self.log else {
            return Ok(false);
        };
        let mut held = HeldLog::acquire_for_write(log).await?;
        if let Some(head) = held.head()? {
            let mut records = held.reader()?.scan_through(None, head);
            while let Some(logged) = records.next().await {
                if record.is_written_as(&logged?.event.payload) {
                    return Ok(false);
                }
            }
        }
        let now = self.clock.now();
        let logged = event(
            [0; 32],
            (STORE_SOURCE, STORE_TRUST),
            now.0,
            now,
            record.payload(now.0),
        );
        let tx = self.backend.begin().await?;
        self.append_and_commit(tx, &mut held, logged, &[]).await?;
        Ok(true)
    }

    /// The key groups that sealed the nodes of `subject` in `scope`, every
    /// version of it. A subject that never had a node has none, and no group is
    /// made for it.
    pub async fn key_groups_of_subject(
        &self,
        subject: &str,
        scope: Option<&str>,
        include_children: bool,
    ) -> Result<Vec<String>> {
        let (filter, scope_params) = scope_filter(scope, include_children);
        let sql = SUBJECT_KEYS_SQL.replace("{scope}", &filter);
        let mut params = vec![Value::from(subject.trim().to_lowercase().as_str())];
        params.extend(scope_params);
        let rows = self.backend.query(&sql, &params).await?;
        rows.iter().map(|row| row.get_string(0)).collect()
    }
}
