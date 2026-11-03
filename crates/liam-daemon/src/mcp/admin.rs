// SPDX-License-Identifier: Apache-2.0
//! The guard the `delete` and `consolidate` tools share. A producer name is a
//! claim a client makes, so this stops honest mistakes and nothing more.

use std::fmt;

use super::MemoryServer;

/// A producer outside `[admin].producers` called an admin tool.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct AdminRefusal {
    tool: &'static str,
    producer: String,
}

impl fmt::Display for AdminRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} refused: producer \"{}\" is not listed in [admin].producers",
            self.tool, self.producer
        )
    }
}

impl MemoryServer {
    /// Lets the call through only for a producer `[admin]` lists. The unknown
    /// fallback producer is refused like any other name that is not listed.
    pub(super) fn require_admin(&self, tool: &'static str) -> Result<(), AdminRefusal> {
        let producer = self.producer();
        if self.admin.allows(&producer) {
            return Ok(());
        }
        tracing::warn!(tool, %producer, "an admin tool was refused");
        Err(AdminRefusal { tool, producer })
    }
}
