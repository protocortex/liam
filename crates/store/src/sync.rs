// SPDX-License-Identifier: Apache-2.0
//! Locking that survives a panic in another thread.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Locks `mutex`, recovering the data if a panicking holder poisoned it.
///
/// For state that stays valid after a panic, so a second panic never
/// follows the first.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
