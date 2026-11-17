// SPDX-License-Identifier: Apache-2.0
//! Reasons a recall or ask reply is degraded rather than complete.

use schemars::JsonSchema;
use serde::Serialize;

/// Why a reply is less than the full, synthesised answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Degraded {
    EmbedFailed,
    SynthesisUnavailable,
}

impl Degraded {
    #[cfg(test)]
    pub const ALL: [Degraded; 2] = [Degraded::EmbedFailed, Degraded::SynthesisUnavailable];

    /// The snake_case name used in structured output and the marker line.
    pub const fn as_str(self) -> &'static str {
        match self {
            Degraded::EmbedFailed => "embed_failed",
            Degraded::SynthesisUnavailable => "synthesis_unavailable",
        }
    }

    /// A short phrase telling the reader what they are getting instead.
    pub const fn hint(self) -> &'static str {
        match self {
            Degraded::EmbedFailed => "searched by keyword only",
            Degraded::SynthesisUnavailable => "showing the retrieved evidence",
        }
    }
}
