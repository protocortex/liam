// SPDX-License-Identifier: Apache-2.0
//! The `delete` tool. It only reports what a delete would destroy: the job
//! runs from `liamd delete`, with the daemon stopped, after a person has seen
//! these counts and confirmed them.

use liam_store::validate_scope;
use schemars::JsonSchema;
use serde::Deserialize;

use super::MemoryServer;
use crate::deletion::{DeleteError, DeleteTarget, ScanError};

const NOTHING_MATCHES: &str = "nothing in memory matches the target";
const NEEDS_ONE_TARGET: &str = "give either a subject or an event_id, not both and not neither";
const NO_PREVIEW: &str = "the log has no key registry, so nothing in it can be deleted by key; \
                          configure [encryption] first";

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteArgs {
    /// The subject whose memories to count, matched without regard to case.
    pub subject: Option<String>,
    /// Only the subject's memories in this scope. Absent matches the subject
    /// without a scope.
    pub scope: Option<String>,
    /// Also count the subject in every scope below `scope`.
    pub include_children: Option<bool>,
    /// Counts one logged event instead of a subject.
    pub event_id: Option<String>,
}

impl DeleteArgs {
    fn into_target(self) -> Result<DeleteTarget, String> {
        match (self.subject, self.event_id) {
            (Some(subject), None) if !subject.trim().is_empty() => Ok(DeleteTarget::Subject {
                subject,
                scope: validate_scope(&self.scope).map_err(|error| error.to_string())?,
                include_children: self.include_children.unwrap_or(false),
            }),
            (None, Some(event_id)) if !event_id.trim().is_empty() => {
                Ok(DeleteTarget::Event { event_id })
            }
            _ => Err(NEEDS_ONE_TARGET.to_owned()),
        }
    }
}

pub(super) async fn read(server: &MemoryServer, args: DeleteArgs) -> String {
    if let Err(refusal) = server.require_admin("delete") {
        return refusal.to_string();
    }
    let target = match args.into_target() {
        Ok(target) => target,
        Err(reason) => return format!("delete failed: {reason}"),
    };
    let Some(preview) = &server.delete_preview else {
        return format!("delete failed: {NO_PREVIEW}");
    };
    match preview.read(&server.shared_store(), &target).await {
        Ok(found) => format!(
            "a delete would destroy {found}\nnothing was changed; run `liamd delete` with the \
             daemon stopped to delete"
        ),
        Err(DeleteError::Refused(ScanError::NothingToDelete)) => NOTHING_MATCHES.to_owned(),
        Err(error) => format!("delete failed: {error}"),
    }
}
