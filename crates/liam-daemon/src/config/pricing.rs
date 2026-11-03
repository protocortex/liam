// SPDX-License-Identifier: Apache-2.0
//! Who answers a model call and what its tokens cost. These sit with the
//! config because `[extraction]`, `[summary]` and `[hosted.prices]` name them.

use serde::Deserialize;
#[cfg(test)]
use serde::Serialize;

/// Who answered a model call. The names are the ones `[hosted.models]` uses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
pub enum Provider {
    #[default]
    #[serde(rename = "local")]
    Local,
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "openai")]
    OpenAi,
    #[serde(rename = "gemini")]
    Gemini,
}

impl Provider {
    /// The provider's name in `[hosted.models]`, or `local`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
            Self::Gemini => "gemini",
        }
    }
}

/// The tokens a provider reports for one call. Cached input is part of
/// `input_tokens` and is billed at its own rate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
}

/// What a model's tokens cost, in millionths of a dollar per million tokens.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(default, deny_unknown_fields)]
pub struct Price {
    pub input_micro_per_mtok: u64,
    pub output_micro_per_mtok: u64,
    /// Input the provider served from its prompt cache.
    pub cached_input_micro_per_mtok: u64,
    pub batch_input_micro_per_mtok: u64,
    pub batch_output_micro_per_mtok: u64,
}

const TOKENS_PER_MTOK: u128 = 1_000_000;

impl Price {
    /// The cost of `tokens` in millionths of a dollar. Cached input tokens are
    /// part of the input count and are charged at the cached rate instead of
    /// the input rate. A batch call uses the batch rates for the rest.
    pub fn cost_micro(&self, tokens: &TokenUsage, batch: bool) -> u64 {
        let (input_rate, output_rate) = if batch {
            (
                self.batch_input_micro_per_mtok,
                self.batch_output_micro_per_mtok,
            )
        } else {
            (self.input_micro_per_mtok, self.output_micro_per_mtok)
        };
        let cached = tokens.cached_input_tokens.min(tokens.input_tokens);
        let fresh = tokens.input_tokens - cached;
        let micro = u128::from(fresh) * u128::from(input_rate)
            + u128::from(cached) * u128::from(self.cached_input_micro_per_mtok)
            + u128::from(tokens.output_tokens) * u128::from(output_rate);
        u64::try_from(micro / TOKENS_PER_MTOK).unwrap_or(u64::MAX)
    }
}
