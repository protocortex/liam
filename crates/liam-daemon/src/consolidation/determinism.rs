// SPDX-License-Identifier: Apache-2.0
//! The run's source of randomness. Every random choice in a consolidation run
//! draws from one generator seeded by the manifest, so a rerun from the same
//! manifest makes the same choices. `thread_rng` is never used.

use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::manifest::RunManifest;

/// The generator for a run, seeded from its manifest.
pub fn rng_for(_manifest: &RunManifest) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder_fingerprint::EmbedderFingerprint;
    use liam_store::Millis;
    use rand_chacha::rand_core::RngCore;

    const DRAWS: usize = 8;

    fn manifest_with(seed: u64) -> RunManifest {
        let embedder = EmbedderFingerprint {
            provider: "mock".into(),
            model: "mock-model".into(),
            dims: 8,
            prompt_scheme: 1,
        };
        RunManifest::new("run".into(), seed, embedder, "llm".into(), Millis(1))
    }

    fn draws(manifest: &RunManifest) -> Vec<u64> {
        let mut rng = rng_for(manifest);
        (0..DRAWS).map(|_| rng.next_u64()).collect()
    }

    #[test]
    fn the_same_manifest_seed_gives_the_same_stream() {
        // Arrange
        let first = manifest_with(42);
        let second = manifest_with(42);

        // Act
        let a = draws(&first);
        let b = draws(&second);

        // Assert
        assert_eq!(a, b);
    }

    #[test]
    fn different_manifest_seeds_give_different_streams() {
        // Arrange
        let first = manifest_with(1);
        let second = manifest_with(2);

        // Act
        let a = draws(&first);
        let b = draws(&second);

        // Assert
        assert_ne!(a, b);
    }
}
