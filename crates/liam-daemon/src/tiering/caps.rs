// SPDX-License-Identifier: Apache-2.0
//! Bounding how much one producer can move the hits of the items it reads. A
//! client that recalls the same item over and over, or reads thousands of
//! items, gets no more weight than the caps allow, so it cannot keep an item in
//! the hot tier alone.

use std::collections::BTreeMap;

use liam_store::{LineageKey, ProducerUsage, UsageSnapshot};

use super::utility::TierParams;

/// The hits of each lineage by day after both caps, from the rows of every
/// producer. Each (producer, item, day) value is clamped to `item_cap` first.
/// Then a producer whose total over items for a day is above `producer_cap`
/// loses the excess from its largest contributions first, a tie going to the
/// lower lineage key. The result depends on the rows alone.
pub fn capped_hits(
    rows: &[ProducerUsage],
    item_cap: i64,
    producer_cap: i64,
) -> BTreeMap<LineageKey, Vec<(i64, i64)>> {
    let mut by_producer_day: BTreeMap<(&str, i64), Vec<(&LineageKey, i64)>> = BTreeMap::new();
    for row in rows {
        by_producer_day
            .entry((row.producer.as_str(), row.day))
            .or_default()
            .push((&row.lineage_key, row.hits.min(item_cap).max(0)));
    }
    let mut by_lineage: BTreeMap<LineageKey, BTreeMap<i64, i64>> = BTreeMap::new();
    for ((_, day), mut contributions) in by_producer_day {
        let total: i64 = contributions.iter().map(|(_, hits)| hits).sum();
        let mut excess = (total - producer_cap).max(0);
        contributions.sort_by(|(left_key, left), (right_key, right)| {
            right.cmp(left).then_with(|| left_key.cmp(right_key))
        });
        for (lineage, hits) in contributions {
            let cut = hits.min(excess);
            excess -= cut;
            *by_lineage
                .entry(lineage.clone())
                .or_default()
                .entry(day)
                .or_default() += hits - cut;
        }
    }
    by_lineage
        .into_iter()
        .map(|(lineage, days)| {
            let days = days.into_iter().filter(|(_, hits)| *hits > 0).collect();
            (lineage, days)
        })
        .collect()
}

/// Replaces the hits of every item in `snapshot` with the capped ones.
pub fn apply_caps(snapshot: &mut UsageSnapshot, params: &TierParams) {
    let mut capped = capped_hits(
        &snapshot.producer_usage,
        i64::from(params.hits_per_producer_item_day_cap),
        i64::from(params.hits_per_producer_day_cap),
    );
    for item in &mut snapshot.items {
        item.hits_by_day = capped.remove(&item.lineage_key).unwrap_or_default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(lineage: &str, producer: &str, day: i64, hits: i64) -> ProducerUsage {
        ProducerUsage {
            lineage_key: LineageKey::from_raw(lineage),
            producer: producer.to_owned(),
            day,
            hits,
        }
    }

    fn hits_of(capped: &BTreeMap<LineageKey, Vec<(i64, i64)>>, lineage: &str) -> Vec<(i64, i64)> {
        capped
            .get(&LineageKey::from_raw(lineage))
            .cloned()
            .unwrap_or_default()
    }

    #[test]
    fn one_producer_hammering_an_item_is_clamped_to_the_item_cap() {
        // Arrange
        let rows = [row("a", "p1", 5, 1_000)];

        // Act
        let capped = capped_hits(&rows, 10, 1_000);

        // Assert
        assert_eq!(hits_of(&capped, "a"), [(5, 10)]);
    }

    #[test]
    fn a_value_at_the_cap_minus_one_the_cap_and_above_it_comes_out_at_most_the_cap() {
        for (hits, expected) in [(9, 9), (10, 10), (11, 10)] {
            // Act
            let capped = capped_hits(&[row("a", "p1", 5, hits)], 10, 1_000);

            // Assert
            assert_eq!(hits_of(&capped, "a"), [(5, expected)], "{hits}");
        }
    }

    #[test]
    fn a_zero_cap_counts_nothing() {
        // Act
        let capped = capped_hits(&[row("a", "p1", 5, 7)], 0, 100);

        // Assert
        assert!(hits_of(&capped, "a").is_empty());
    }

    #[test]
    fn a_day_total_over_the_producer_cap_loses_the_excess_from_the_largest_first_by_lineage_key() {
        // Arrange: contributions 9, 9, 7, 5 against a cap of 20, so 10 goes
        let rows = [
            row("d", "p1", 5, 5),
            row("c", "p1", 5, 7),
            row("b", "p1", 5, 9),
            row("a", "p1", 5, 9),
        ];

        // Act
        let capped = capped_hits(&rows, 10, 20);

        // Assert: "a" gives up all 9 (lower key first among equals), "b" 1
        let kept: Vec<_> = ["a", "b", "c", "d"]
            .iter()
            .map(|lineage| hits_of(&capped, lineage))
            .collect();
        assert_eq!(kept, [vec![], vec![(5, 8)], vec![(5, 7)], vec![(5, 5)]]);
    }

    #[test]
    fn two_producers_at_the_cap_on_one_item_are_both_counted() {
        // Arrange
        let rows = [row("a", "p1", 5, 10), row("a", "p2", 5, 10)];

        // Act
        let capped = capped_hits(&rows, 10, 10);

        // Assert
        assert_eq!(hits_of(&capped, "a"), [(5, 20)]);
    }

    #[test]
    fn splitting_one_claim_over_many_items_gains_no_more_than_the_producer_cap() {
        // Arrange
        let rows: Vec<_> = (0..50)
            .map(|index| row(&format!("item-{index:02}"), "p1", 5, 10))
            .collect();

        // Act
        let capped = capped_hits(&rows, 10, 100);

        // Assert
        let total: i64 = capped.values().flatten().map(|(_, hits)| hits).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn the_same_rows_give_the_same_answer_in_any_order() {
        // Arrange
        let rows = vec![
            row("b", "p1", 5, 9),
            row("a", "p1", 5, 9),
            row("c", "p2", 6, 4),
        ];
        let mut reversed = rows.clone();
        reversed.reverse();

        // Act
        let first = capped_hits(&rows, 10, 12);
        let second = capped_hits(&reversed, 10, 12);

        // Assert
        assert_eq!(first, second);
    }

    #[test]
    fn the_legacy_producer_counts_as_one_producer() {
        // Arrange
        let rows = [row("a", "", 5, 10), row("b", "", 5, 10)];

        // Act
        let capped = capped_hits(&rows, 10, 12);

        // Assert
        let total: i64 = capped.values().flatten().map(|(_, hits)| hits).sum();
        assert_eq!(total, 12);
    }

    #[test]
    fn the_snapshot_items_take_the_capped_hits_and_an_item_without_rows_has_none() {
        // Arrange
        let mut snapshot = UsageSnapshot {
            now: liam_store::Millis(0),
            producer_usage: vec![row("a", "p1", 5, 1_000), row("a", "p2", 5, 3)],
            observation_since: None,
            items: vec![item("a", &[(5, 1_003)]), item("b", &[(4, 9)])],
        };
        let params = super::super::utility::fixtures::params_with([1.0, 0.0, 0.0, 0.0], 0.5);

        // Act
        apply_caps(&mut snapshot, &params);

        // Assert
        let hits: Vec<_> = snapshot
            .items
            .iter()
            .map(|i| i.hits_by_day.clone())
            .collect();
        let cap = i64::from(params.hits_per_producer_item_day_cap);
        assert_eq!(hits, [vec![(5, cap + 3)], vec![]]);
    }

    fn item(lineage: &str, hits: &[(i64, i64)]) -> liam_store::ItemView {
        liam_store::ItemView {
            lineage_key: LineageKey::from_raw(lineage),
            hits_by_day: hits.to_vec(),
            ..super::super::utility::fixtures::item(lineage)
        }
    }

    #[test]
    fn rows_a_second_before_and_at_midnight_fall_on_different_days() {
        // Arrange
        let before = liam_store::Millis(86_400_000 - 1_000).day_number();
        let at = liam_store::Millis(86_400_000).day_number();

        // Act
        let capped = capped_hits(&[row("a", "p1", before, 1), row("a", "p1", at, 1)], 10, 10);

        // Assert
        assert_eq!(hits_of(&capped, "a"), [(before, 1), (at, 1)]);
    }
}
