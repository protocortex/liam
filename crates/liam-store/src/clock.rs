// SPDX-License-Identifier: AGPL-3.0-only
//! Time is a dependency, not an ambient call, so supersession and validity
//! windows are testable at a chosen instant.

use crate::ids::Millis;
use std::sync::Mutex;

pub trait Clock: Send + Sync {
    fn now(&self) -> Millis;
}

/// Production clock: reads the wall clock.
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Millis {
        Millis::now()
    }
}

/// Test clock: a fixed instant a test can advance between calls.
pub struct FixedClock(Mutex<Millis>);

impl FixedClock {
    pub fn new(at: Millis) -> Self {
        Self(Mutex::new(at))
    }
    pub fn set(&self, at: Millis) {
        *self.0.lock().expect("clock poisoned") = at;
    }
}

impl Clock for FixedClock {
    fn now(&self) -> Millis {
        *self.0.lock().expect("clock poisoned")
    }
}
