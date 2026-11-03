// SPDX-License-Identifier: Apache-2.0
//! The compact forms of an embedding and the in-memory index over the binary
//! one. Both forms are pure functions of the vector, so a rebuild reproduces
//! them from the embeddings.

use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, PoisonError, RwLock};

use crate::error::{Error, Result};
use crate::ids::NodeId;

/// Bits one packed word of a binary signature holds.
pub const BITS_PER_WORD: usize = u64::BITS as usize;

/// The largest magnitude the int8 form stores.
const I8_LIMIT: f32 = i8::MAX as f32;

/// The scale of a vector with no magnitude, which keeps the division defined.
const UNIT_SCALE: f32 = 1.0;

/// The version of the scheme that turns a vector into its signature. A scheme
/// change is a new version and a rebuild.
pub const QUANT_VERSION: i64 = 1;

/// The binary and int8 forms of one embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct Quantized {
    /// One bit per dimension, set when the component is positive, packed into
    /// words from the lowest bit. Bits past the last dimension are zero.
    pub bin: Vec<u64>,
    /// Symmetric scalar quantization: each component divided by `scale`.
    pub i8: Vec<i8>,
    pub scale: f32,
}

/// Quantizes `vector`. A non-finite component is refused, because it has no
/// sign or magnitude to keep.
pub fn quantize(vector: &[f32]) -> Result<Quantized> {
    if let Some(index) = vector.iter().position(|c| !c.is_finite()) {
        return Err(Error::NonFiniteEmbedding { index });
    }
    let largest = vector
        .iter()
        .fold(0.0_f32, |largest, c| largest.max(c.abs()));
    let scale = largest / I8_LIMIT;
    // A vector of zeros, or one so small the division underflows, has no usable
    // scale; its components all round to zero under the unit scale.
    let scale = if scale.is_normal() { scale } else { UNIT_SCALE };
    let i8 = vector
        .iter()
        .map(|c| (c / scale).round().clamp(-I8_LIMIT, I8_LIMIT) as i8)
        .collect();
    let mut bin = vec![0_u64; vector.len().div_ceil(BITS_PER_WORD)];
    for (index, _) in vector.iter().enumerate().filter(|(_, c)| **c > 0.0) {
        bin[index / BITS_PER_WORD] |= 1 << (index % BITS_PER_WORD);
    }
    Ok(Quantized { bin, i8, scale })
}

/// The vector the int8 form stands for, within `scale / 2` of the original per
/// component.
pub fn dequantize(quantized: &Quantized) -> Vec<f32> {
    quantized
        .i8
        .iter()
        .map(|c| f32::from(*c) * quantized.scale)
        .collect()
}

const WORD_BYTES: usize = std::mem::size_of::<u64>();

/// Little-endian bytes of a signature, the stored form.
pub fn words_to_bytes(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// The signature `words_to_bytes` wrote, or `None` when the length is not a
/// whole number of words.
pub fn bytes_to_words(bytes: &[u8]) -> Option<Vec<u64>> {
    let (words, rest) = bytes.as_chunks::<WORD_BYTES>();
    rest.is_empty()
        .then(|| words.iter().map(|word| u64::from_le_bytes(*word)).collect())
}

/// The number of bits in which two signatures of one width differ.
pub fn hamming(left: &[u64], right: &[u64]) -> u32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a ^ b).count_ones())
        .sum()
}

struct Entry {
    cold: bool,
    bin: Vec<u64>,
}

/// The binary signature of every live node that has one, shared by every handle
/// to a store. It only mirrors `node_quant`, so a reload from the table is
/// always a repair.
#[derive(Clone, Default)]
pub struct QuantIndex {
    entries: Arc<RwLock<HashMap<String, Entry>>>,
}

/// An entry as `snapshot` reports it: id, cold flag, signature.
pub type IndexedEntry = (String, bool, Vec<u64>);

impl QuantIndex {
    fn read<T>(&self, use_entries: impl FnOnce(&HashMap<String, Entry>) -> T) -> T {
        let entries = self.entries.read().unwrap_or_else(PoisonError::into_inner);
        use_entries(&entries)
    }

    fn write<T>(&self, use_entries: impl FnOnce(&mut HashMap<String, Entry>) -> T) -> T {
        let mut entries = self.entries.write().unwrap_or_else(PoisonError::into_inner);
        use_entries(&mut entries)
    }

    pub fn insert(&self, id: &str, cold: bool, bin: Vec<u64>) {
        self.write(|entries| entries.insert(id.to_string(), Entry { cold, bin }));
    }

    pub fn remove<'a>(&self, ids: impl IntoIterator<Item = &'a str>) {
        self.write(|entries| ids.into_iter().for_each(|id| drop(entries.remove(id))));
    }

    /// Marks the entries of `ids` cold or hot. An id the index does not hold is
    /// skipped.
    pub fn set_cold<'a>(&self, ids: impl IntoIterator<Item = &'a str>, cold: bool) {
        self.write(|entries| {
            for id in ids {
                if let Some(entry) = entries.get_mut(id) {
                    entry.cold = cold;
                }
            }
        });
    }

    /// Replaces every entry at once, so a reader sees the old index or the new.
    pub fn replace_all(&self, replacement: impl IntoIterator<Item = IndexedEntry>) {
        let replacement = replacement
            .into_iter()
            .map(|(id, cold, bin)| (id, Entry { cold, bin }))
            .collect();
        self.write(|entries| *entries = replacement);
    }

    pub fn len(&self) -> usize {
        self.read(HashMap::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn cold_count(&self) -> usize {
        self.read(|entries| entries.values().filter(|entry| entry.cold).count())
    }

    /// Every entry, ordered by id.
    pub fn snapshot(&self) -> Vec<IndexedEntry> {
        let mut all: Vec<IndexedEntry> = self.read(|entries| {
            entries
                .iter()
                .map(|(id, entry)| (id.clone(), entry.cold, entry.bin.clone()))
                .collect()
        });
        all.sort_by(|a, b| a.0.cmp(&b.0));
        all
    }

    /// The `k` entries nearest `query` by Hamming distance, nearest first and
    /// equal distances by id, each with its distance. With `only_cold` the
    /// hot entries are not candidates. An entry of another width than `query`
    /// is an error, because its distance would mean nothing.
    pub fn hamming_top(
        &self,
        query: &[u64],
        k: usize,
        only_cold: bool,
    ) -> Result<Vec<(NodeId, u32)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        self.read(|entries| {
            // A max-heap on (distance, id) keeps the k smallest seen so far.
            let mut nearest: BinaryHeap<(u32, &str)> = BinaryHeap::with_capacity(k + 1);
            for (id, entry) in entries.iter().filter(|(_, entry)| entry.cold || !only_cold) {
                if entry.bin.len() != query.len() {
                    return Err(Error::SignatureWidth {
                        expected: query.len(),
                        got: entry.bin.len(),
                    });
                }
                nearest.push((hamming(query, &entry.bin), id.as_str()));
                if nearest.len() > k {
                    nearest.pop();
                }
            }
            Ok(nearest
                .into_sorted_vec()
                .into_iter()
                .map(|(distance, id)| (NodeId::from_raw(id), distance))
                .collect())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_is_deterministic_and_dequantizes_within_half_a_step() {
        // Arrange
        let vector = [0.9_f32, -0.31, 0.0, 0.5, -1.2, 0.07];

        // Act
        let first = quantize(&vector).unwrap();
        let second = quantize(&vector).unwrap();
        let restored = dequantize(&first);

        // Assert
        assert_eq!(first, second);
        for (original, back) in vector.iter().zip(&restored) {
            assert!(
                (original - back).abs() <= first.scale / 2.0 * (1.0 + 1e-5),
                "{original} came back as {back} at scale {}",
                first.scale
            );
        }
    }

    #[test]
    fn quantize_refuses_a_component_without_a_sign_or_magnitude() {
        // Arrange
        let cases = [(f32::NAN, 1), (f32::INFINITY, 0), (f32::NEG_INFINITY, 2)];

        for (bad, index) in cases {
            let mut vector = vec![1.0, 2.0, 3.0];
            vector[index] = bad;

            // Act
            let result = quantize(&vector);

            // Assert
            assert!(
                matches!(result, Err(Error::NonFiniteEmbedding { index: at }) if at == index),
                "{bad}: {result:?}"
            );
        }
    }

    #[test]
    fn quantize_keeps_a_finite_scale_for_a_huge_component() {
        // Arrange
        let vector = [f32::MAX, -f32::MAX, 1.0];

        // Act
        let quantized = quantize(&vector).unwrap();

        // Assert
        assert!(quantized.scale.is_finite() && quantized.scale > 0.0);
        assert_eq!(quantized.i8, [127, -127, 0]);
        assert_eq!(quantized.bin, [0b101]);
    }

    #[test]
    fn quantize_packs_odd_widths_and_gives_both_zeros_bit_zero() {
        // Arrange
        let mut sixty_five = vec![-1.0_f32; 65];
        sixty_five[0] = 1.0;
        sixty_five[63] = 1.0;
        sixty_five[64] = 1.0;
        let cases: [(&[f32], &[u64]); 4] = [
            (&[], &[]),
            (&[0.0, -0.0], &[0]),
            (&[2.0], &[1]),
            (&sixty_five, &[1 | 1 << 63, 1]),
        ];

        for (vector, expected) in cases {
            // Act
            let quantized = quantize(vector).unwrap();

            // Assert
            assert_eq!(quantized.bin, expected, "{vector:?}");
        }
    }

    #[test]
    fn a_vector_with_no_magnitude_has_the_unit_scale_and_dequantizes_to_zeros() {
        // Arrange
        let cases: [&[f32]; 3] = [&[], &[0.0, -0.0, 0.0], &[1e-45, -1e-45]];

        for vector in cases {
            // Act
            let quantized = quantize(vector).unwrap();

            // Assert
            assert_eq!(quantized.scale, UNIT_SCALE, "{vector:?}");
            assert!(quantized.i8.iter().all(|c| *c == 0));
            assert!(dequantize(&quantized).iter().all(|c| *c == 0.0));
        }
    }

    #[test]
    fn signature_bytes_round_trip_and_a_ragged_length_is_refused() {
        // Arrange
        let words = [0_u64, u64::MAX, 0x0123_4567_89ab_cdef];

        // Act
        let bytes = words_to_bytes(&words);

        // Assert
        assert_eq!(bytes_to_words(&bytes).as_deref(), Some(&words[..]));
        assert_eq!(bytes_to_words(&bytes[1..]), None);
    }

    fn index_of(entries: &[(&str, bool, u64)]) -> QuantIndex {
        let index = QuantIndex::default();
        for (id, cold, word) in entries {
            index.insert(id, *cold, vec![*word]);
        }
        index
    }

    fn named(found: Vec<(NodeId, u32)>) -> Vec<(String, u32)> {
        found
            .into_iter()
            .map(|(id, distance)| (id.as_str().to_string(), distance))
            .collect()
    }

    #[test]
    fn hamming_top_edges_return_the_right_sizes() {
        // Arrange
        let empty = QuantIndex::default();
        let index = index_of(&[("a", true, 0b1), ("b", false, 0b11), ("c", true, 0b111)]);
        let all_hot = index_of(&[("a", false, 1)]);

        // Act
        let sizes = [
            empty.hamming_top(&[0], 3, false).unwrap().len(),
            index.hamming_top(&[0], 0, false).unwrap().len(),
            index.hamming_top(&[0], 10, false).unwrap().len(),
            index.hamming_top(&[0], 10, true).unwrap().len(),
            index.hamming_top(&[0], 2, false).unwrap().len(),
            all_hot.hamming_top(&[0], 5, true).unwrap().len(),
        ];

        // Assert
        assert_eq!(sizes, [0, 0, 3, 2, 2, 0]);
    }

    #[test]
    fn hamming_top_orders_by_distance_then_id() {
        // Arrange
        let index = index_of(&[
            ("far", true, 0b1111),
            ("b-tie", true, 0b0011),
            ("a-tie", true, 0b1100),
            ("near", true, 0b0001),
        ]);

        // Act
        let found = index.hamming_top(&[0b0000], 4, false).unwrap();

        // Assert
        assert_eq!(
            named(found),
            [
                ("near".to_string(), 1),
                ("a-tie".to_string(), 2),
                ("b-tie".to_string(), 2),
                ("far".to_string(), 4),
            ]
        );
    }

    #[test]
    fn hamming_top_refuses_a_signature_of_another_width() {
        // Arrange
        let index = index_of(&[("a", true, 1)]);

        // Act
        let result = index.hamming_top(&[0, 0], 1, false);

        // Assert
        assert!(matches!(
            result,
            Err(Error::SignatureWidth {
                expected: 2,
                got: 1
            })
        ));
    }
}
