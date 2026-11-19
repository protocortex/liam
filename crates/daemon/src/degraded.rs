// SPDX-License-Identifier: Apache-2.0
//! Reasons a recall reply is degraded rather than complete.

use schemars::JsonSchema;
use serde::Serialize;

/// Why a reply is less than the full result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Degraded {
    EmbedFailed,
}

impl Degraded {
    #[cfg(test)]
    pub const ALL: [Degraded; 1] = [Degraded::EmbedFailed];

    /// The snake_case name used in structured output and the marker line.
    pub const fn as_str(self) -> &'static str {
        match self {
            Degraded::EmbedFailed => "embed_failed",
        }
    }

    /// A short phrase telling the reader what they are getting instead.
    pub const fn hint(self) -> &'static str {
        match self {
            Degraded::EmbedFailed => "searched by keyword only",
        }
    }
}
