// SPDX-License-Identifier: Apache-2.0
//! Pinned model ids for hosted providers. A pin is a dated id: an alias such as
//! `claude-sonnet-4-5` or `gpt-4o` moves to a new model without notice, and a
//! cached answer and a recorded F1 would then describe a model that is gone.
//!
//! The id shapes below are the ones the providers' documentation showed when
//! this was written, and have not been checked against a live account.

use std::fmt;

/// The providers a pin can name.
pub const PIN_PROVIDERS: [&str; 3] = ["anthropic", "openai", "gemini"];

/// The HTTP status a provider answers with for a model that no longer exists.
const NOT_FOUND: u16 = 404;

/// A model id fixed to one dated version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPin {
    pub provider: String,
    pub model_id: String,
}

/// Why a pin was refused or did not hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinError {
    /// The id has no date or version, so it can move.
    Alias {
        provider: String,
        model_id: String,
    },
    UnknownProvider {
        provider: String,
    },
    /// The provider answered with a different model than the pin.
    Mismatch {
        pinned: String,
        reported: String,
    },
    /// The provider no longer has the pinned model. There is no fallback.
    Retired {
        provider: String,
        model_id: String,
    },
}

impl fmt::Display for PinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alias { provider, model_id } => write!(
                f,
                "hosted.models.{provider} = {model_id:?} is an alias; pin a dated model id"
            ),
            Self::UnknownProvider { provider } => write!(
                f,
                "hosted.models.{provider} is not a known provider; use one of {}",
                PIN_PROVIDERS.join(", ")
            ),
            Self::Mismatch { pinned, reported } => write!(
                f,
                "the provider answered with model {reported} but the pin is {pinned}"
            ),
            Self::Retired { provider, model_id } => write!(
                f,
                "{provider} no longer has the pinned model {model_id}; update hosted.models"
            ),
        }
    }
}

impl std::error::Error for PinError {}

impl ModelPin {
    /// A pin for `provider`, refused when the id is an alias.
    pub fn new(provider: &str, model_id: &str) -> Result<Self, PinError> {
        if !PIN_PROVIDERS.contains(&provider) {
            return Err(PinError::UnknownProvider {
                provider: provider.to_owned(),
            });
        }
        if !has_dated_shape(provider, model_id) {
            return Err(PinError::Alias {
                provider: provider.to_owned(),
                model_id: model_id.to_owned(),
            });
        }
        Ok(Self {
            provider: provider.to_owned(),
            model_id: model_id.to_owned(),
        })
    }

    /// The model a response says it came from must be the pinned one.
    pub fn check(&self, reported: &str) -> Result<(), PinError> {
        if reported == self.model_id {
            Ok(())
        } else {
            Err(PinError::Mismatch {
                pinned: self.model_id.clone(),
                reported: reported.to_owned(),
            })
        }
    }

    /// A not-found answer for the pinned id means it was retired.
    pub fn check_status(&self, status: u16) -> Result<(), PinError> {
        if status == NOT_FOUND {
            Err(PinError::Retired {
                provider: self.provider.clone(),
                model_id: self.model_id.clone(),
            })
        } else {
            Ok(())
        }
    }
}

/// `-YYYYMMDD` (Anthropic), `-YYYY-MM-DD` (OpenAI) or `-NNN` (Gemini's versioned
/// form) at the end of the id.
fn has_dated_shape(provider: &str, model_id: &str) -> bool {
    let digits = |text: &str, expected: &[usize]| {
        let groups: Vec<&str> = text.split('-').collect();
        groups.len() == expected.len()
            && groups.iter().zip(expected).all(|(group, len)| {
                group.len() == *len && group.chars().all(|c| c.is_ascii_digit())
            })
    };
    let tail = |groups: usize| -> Option<String> {
        let parts: Vec<&str> = model_id.split('-').collect();
        (parts.len() > groups).then(|| parts[parts.len() - groups..].join("-"))
    };
    match provider {
        "anthropic" => tail(1).is_some_and(|date| digits(&date, &[8])),
        "openai" => tail(3).is_some_and(|date| digits(&date, &[4, 2, 2])),
        "gemini" => tail(1).is_some_and(|version| digits(&version, &[3])),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dated_id_is_pinned_for_each_provider() {
        for (provider, id) in [
            ("anthropic", "claude-sonnet-4-5-20250929"),
            ("openai", "gpt-4o-2024-08-06"),
            ("gemini", "gemini-2.0-flash-001"),
        ] {
            // Act
            let pin = ModelPin::new(provider, id);

            // Assert
            assert_eq!(pin.unwrap().model_id, id, "{provider}");
        }
    }

    #[test]
    fn an_alias_is_refused_for_each_provider() {
        for (provider, id) in [
            ("anthropic", "claude-sonnet-4-5"),
            ("anthropic", "claude-3-5-sonnet-latest"),
            ("anthropic", "claude-sonnet-4-5-2025"),
            ("openai", "gpt-4o"),
            ("openai", "gpt-4o-latest"),
            ("openai", "gpt-4o-2024-08"),
            ("gemini", "gemini-2.0-flash"),
            ("gemini", "gemini-pro-latest"),
            ("gemini", "gemini-2.0-flash-1"),
            ("anthropic", ""),
        ] {
            // Act
            let pin = ModelPin::new(provider, id);

            // Assert
            assert!(
                matches!(pin, Err(PinError::Alias { .. })),
                "{provider} {id}"
            );
        }
    }

    #[test]
    fn an_unknown_provider_is_refused() {
        // Act
        let pin = ModelPin::new("mistral", "model-20250101");

        // Assert
        assert!(matches!(pin, Err(PinError::UnknownProvider { .. })));
    }

    #[test]
    fn a_response_from_another_model_is_a_mismatch() {
        // Arrange
        let pin = ModelPin::new("openai", "gpt-4o-2024-08-06").unwrap();

        // Act
        let same = pin.check("gpt-4o-2024-08-06");
        let other = pin.check("gpt-4o-2024-11-20");

        // Assert
        assert!(same.is_ok());
        assert_eq!(
            other,
            Err(PinError::Mismatch {
                pinned: "gpt-4o-2024-08-06".into(),
                reported: "gpt-4o-2024-11-20".into()
            })
        );
    }

    #[test]
    fn a_not_found_answer_is_a_retired_model_and_other_statuses_are_not() {
        // Arrange
        let pin = ModelPin::new("anthropic", "claude-sonnet-4-5-20250929").unwrap();

        // Act and Assert
        assert!(matches!(
            pin.check_status(404),
            Err(PinError::Retired { .. })
        ));
        for status in [200, 400, 429, 500] {
            assert!(pin.check_status(status).is_ok(), "{status}");
        }
    }
}
