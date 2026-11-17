// SPDX-License-Identifier: Apache-2.0
//! The source stages and tool calls are timed against. It is kept apart from
//! the wall `Clock` that stamps samples: wall time is whole milliseconds and
//! can step, which would blur sub-millisecond stages or make a span negative.

use std::time::{Duration, Instant};

/// Time since a fixed origin, never going backwards.
pub trait Monotonic: Send + Sync {
    fn elapsed(&self) -> Duration;
}

/// Reads `std::time::Instant` against the moment it was made.
pub struct InstantSource {
    origin: Instant,
}

impl InstantSource {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Monotonic for InstantSource {
    fn elapsed(&self) -> Duration {
        self.origin.elapsed()
    }
}

/// Advances by a fixed step on every read, so each timed span lasts a known,
/// nonzero time.
#[cfg(test)]
pub struct TickingMonotonic {
    step: Duration,
    reads: std::sync::atomic::AtomicU32,
}

#[cfg(test)]
impl TickingMonotonic {
    pub fn new(step: Duration) -> Self {
        Self {
            step,
            reads: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

#[cfg(test)]
impl Monotonic for TickingMonotonic {
    fn elapsed(&self) -> Duration {
        let read = self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.step * read
    }
}
