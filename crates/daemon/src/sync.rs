// SPDX-License-Identifier: Apache-2.0
//! Locking that survives a panic in another thread.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Locks `mutex`, recovering the data if a panicking holder poisoned it.
///
/// For state that stays valid after a panic. Use it where a second panic
/// would be worse than reading the last written value, such as in a `Drop`.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_poisoned_lock_is_recovered_with_its_last_value() {
        // Arrange
        let mutex = Mutex::new(1);
        std::thread::scope(|scope| {
            let _ = scope
                .spawn(|| {
                    let mut guard = mutex.lock().unwrap();
                    *guard = 2;
                    panic!("poison the mutex");
                })
                .join();
        });
        assert!(mutex.is_poisoned());

        // Act
        let mut guard = lock(&mutex);
        *guard += 1;

        // Assert
        assert_eq!(*guard, 3);
    }
}
