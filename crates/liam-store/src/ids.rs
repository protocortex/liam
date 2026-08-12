// SPDX-License-Identifier: AGPL-3.0-only
//! Opaque identifiers and millisecond time.

use serde::{Deserialize, Serialize};

/// Milliseconds since the Unix epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Millis(pub i64);

/// Sentinel end of an open interval (2100-01-01 UTC).
pub const FOREVER: Millis = Millis(4_102_444_800_000);

impl Millis {
    pub fn now() -> Self {
        let ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        Millis(ms)
    }

    /// A duration of `days` expressed in milliseconds, for retention windows.
    pub fn days(days: i64) -> Self {
        Millis(days * 86_400_000)
    }
}

macro_rules! branded_id {
    ($name:ident) => {
        #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub struct $name(String);

        impl $name {
            pub fn new() -> Self {
                Self(ulid::Ulid::gen().to_string())
            }
            pub fn from_raw(raw: impl Into<String>) -> Self {
                Self(raw.into())
            }
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

branded_id!(NodeId);
branded_id!(EdgeId);
