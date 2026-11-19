// SPDX-License-Identifier: Apache-2.0
//! The size rules of a `recall` call: how many hits it returns and how many
//! candidates the store fuses to find them.

/// Default and upper bound on the hits `recall` returns. WHY the cap: an
/// unbounded `k` would make one call arbitrarily slow.
const DEFAULT_RECALL_K: usize = 8;
const MAX_RECALL_K: usize = 32;

/// Fewest candidates the store fuses for `recall`, however small `k` is.
const CANDIDATE_POOL_FLOOR: usize = 16;

/// Resolve the caller's `k` into `1..=MAX_RECALL_K`, defaulting to
/// `DEFAULT_RECALL_K`. `k = 0` would return nothing for a query the store can
/// answer.
pub fn clamp_recall_k(k: Option<usize>) -> usize {
    k.unwrap_or(DEFAULT_RECALL_K).clamp(1, MAX_RECALL_K)
}

/// Candidates the store fuses before the cut to `k`: twice `k`, within the
/// floor and `MAX_RECALL_K`. The fused order depends on the pool size, so
/// narrowing it changes which hits come first.
pub(crate) fn candidate_pool(k: usize) -> usize {
    (k * 2).clamp(CANDIDATE_POOL_FLOOR, MAX_RECALL_K)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pool_is_twice_k_within_the_floor_and_the_recall_maximum() {
        // Arrange: k and its pool
        let cases = [(1, 16), (8, 16), (9, 18), (16, 32), (20, 32), (32, 32)];

        for (k, expected) in cases {
            // Act
            let pool = candidate_pool(k);

            // Assert
            assert_eq!(pool, expected, "pool for k={k}");
        }
    }

    #[test]
    fn clamp_recall_k_defaults_and_bounds_the_hit_count() {
        // Arrange / Act / Assert
        assert_eq!(clamp_recall_k(None), DEFAULT_RECALL_K, "absent k defaults");
        assert_eq!(clamp_recall_k(Some(4)), 4, "in-range k passes through");
        assert_eq!(
            clamp_recall_k(Some(MAX_RECALL_K)),
            MAX_RECALL_K,
            "the cap itself is allowed"
        );
        assert_eq!(
            clamp_recall_k(Some(10_000)),
            MAX_RECALL_K,
            "oversized k is capped"
        );
        assert_eq!(clamp_recall_k(Some(0)), 1, "k=0 returns one hit, not none");
    }
}
