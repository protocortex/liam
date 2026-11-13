// SPDX-License-Identifier: Apache-2.0
//! Reasons a recall or ask reply is degraded rather than fully ranked.

use schemars::JsonSchema;
use serde::Serialize;

/// Why a reply is less than the full ranked, synthesised answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Degraded {
    EmbedFailed,
    RerankSkipped,
    RerankFailed,
    RerankTimeout,
    SynthesisUnavailable,
}

impl Degraded {
    #[cfg(test)]
    pub const ALL: [Degraded; 5] = [
        Degraded::EmbedFailed,
        Degraded::RerankSkipped,
        Degraded::RerankFailed,
        Degraded::RerankTimeout,
        Degraded::SynthesisUnavailable,
    ];

    /// The snake_case name used in structured output and the marker line.
    pub const fn as_str(self) -> &'static str {
        match self {
            Degraded::EmbedFailed => "embed_failed",
            Degraded::RerankSkipped => "rerank_skipped",
            Degraded::RerankFailed => "rerank_failed",
            Degraded::RerankTimeout => "rerank_timeout",
            Degraded::SynthesisUnavailable => "synthesis_unavailable",
        }
    }

    /// A short phrase telling the reader what they are getting instead.
    pub const fn hint(self) -> &'static str {
        match self {
            Degraded::EmbedFailed => "searched by keyword only",
            Degraded::RerankSkipped | Degraded::RerankFailed | Degraded::RerankTimeout => {
                "hits are in store order"
            }
            Degraded::SynthesisUnavailable => "showing the retrieved evidence",
        }
    }
}
