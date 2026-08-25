// SPDX-License-Identifier: Apache-2.0
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

/// Characters of a node id that `recall` renders and `relate` resolves back.
///
/// 13, not git's 7, because a ULID is not uniform random at the front: its
/// first 10 characters are a 48-bit millisecond timestamp, so prefixes cluster
/// by write time instead of spreading. Measured, 7 characters are shared by
/// every node written within 32.8 seconds and 10 by every node written within
/// the same millisecond. 13 keeps all 10 timestamp characters and adds 3 from
/// the random half, which drops the collision rate for 8 writes landing in one
/// millisecond from 100% at 10 characters to 0.08%, for one extra token.
/// ADR-0001 Amendment 3 carries the full measurement.
pub const HANDLE_LEN: usize = 13;

impl NodeId {
    /// The client-facing handle for this id.
    ///
    /// Counts characters rather than slicing bytes: `from_raw` validates
    /// nothing, so a non-ASCII id must not panic here.
    pub fn handle(&self) -> &str {
        match self.0.char_indices().nth(HANDLE_LEN) {
            Some((byte, _)) => &self.0[..byte],
            None => &self.0,
        }
    }
}
