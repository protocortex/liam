// SPDX-License-Identifier: Apache-2.0
//! The tuning window before any call, and a cache that cannot be written.

use super::*;

const MODEL: &str = "model-a";
const BACKEND: &str = "llama.cpp/cpu";
const CONTEXT_TOKENS: usize = 8192;
const SLOTS: usize = 3;

#[test]
fn an_empty_window_has_zero_averages_instead_of_a_division_by_zero() {
    // Arrange
    let window = RollingWindow::new();

    // Act / Assert
    assert_eq!(window.average_latency(), Duration::ZERO);
    assert_eq!(window.average_queue_wait(), Duration::ZERO);
}

#[test]
fn a_cache_directory_that_is_a_file_costs_a_repeat_benchmark_and_nothing_else() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let blocker = dir.path().join("blocker");
    std::fs::write(&blocker, b"").unwrap();
    let cache_dir = blocker.to_str().unwrap();

    // Act
    save_cache(cache_dir, MODEL, BACKEND, SLOTS, CONTEXT_TOKENS);
    let loaded = load_cached(cache_dir, MODEL, BACKEND, CONTEXT_TOKENS);

    // Assert
    assert_eq!(loaded, None);
}

#[test]
fn a_saved_value_is_only_reused_under_the_same_model_backend_and_context() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir.path().to_str().unwrap();
    save_cache(cache_dir, MODEL, BACKEND, SLOTS, CONTEXT_TOKENS);

    // Act / Assert
    assert_eq!(
        load_cached(cache_dir, MODEL, BACKEND, CONTEXT_TOKENS),
        Some(SLOTS)
    );
    assert_eq!(
        load_cached(cache_dir, "model-b", BACKEND, CONTEXT_TOKENS),
        None
    );
    assert_eq!(
        load_cached(cache_dir, MODEL, "llama.cpp/metal", CONTEXT_TOKENS),
        None
    );
    assert_eq!(
        load_cached(cache_dir, MODEL, BACKEND, CONTEXT_TOKENS * 2),
        None
    );
}
