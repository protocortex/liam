// SPDX-License-Identifier: Apache-2.0
//! Time is a dependency, not an ambient call, so supersession and validity
//! windows are testable at a chosen instant.

use crate::ids::Millis;
use std::sync::Mutex;

use crate::sync::lock;

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
        *lock(&self.0) = at;
    }
}

impl Clock for FixedClock {
    fn now(&self) -> Millis {
        *lock(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_poisoned_clock_still_reads_and_moves() {
        // Arrange
        let clock = FixedClock::new(Millis(1));
        std::thread::scope(|scope| {
            let _ = scope
                .spawn(|| {
                    let _guard = clock.0.lock().unwrap();
                    panic!("poison the clock");
                })
                .join();
        });
        assert!(clock.0.is_poisoned());

        // Act
        clock.set(Millis(7));

        // Assert
        assert_eq!(clock.now(), Millis(7));
    }
}
