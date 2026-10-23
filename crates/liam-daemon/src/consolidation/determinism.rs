// SPDX-License-Identifier: Apache-2.0
//! The run's source of randomness. Every random choice the job itself makes
//! draws from one generator seeded by the manifest, so a rerun from the same
//! manifest makes the same choices. Ids the store mints for nodes, edges and
//! snapshots stay random, so runs match by content, not by id.

use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::manifest::RunManifest;

/// The generator for a run, seeded from its manifest.
pub fn rng_for(manifest: &RunManifest) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(manifest.seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder_fingerprint::EmbedderFingerprint;
    use liam_store::Millis;
    use rand_chacha::rand_core::RngCore;

    const DRAWS: usize = 8;
    /// Pinned so a changed algorithm, seeding rule or crate release fails here
    /// instead of silently changing every run's choices.
    const PINNED_SEED_42: [u64; DRAWS] = [
        12_578_764_544_318_200_737,
        17_529_487_244_874_322_312,
        7_886_285_670_807_131_020,
        11_572_758_976_476_374_866,
        5_323_617_429_756_461_744,
        2_766_252_901_828_231_838,
        5_682_345_367_224_914_708,
        14_828_835_203_913_492_612,
    ];

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
    fn seed_42_gives_the_pinned_first_draws() {
        // Arrange
        let manifest = manifest_with(42);

        // Act
        let first = draws(&manifest);

        // Assert
        assert_eq!(first, PINNED_SEED_42);
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
