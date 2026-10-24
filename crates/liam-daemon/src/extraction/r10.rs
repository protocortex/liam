// SPDX-License-Identifier: Apache-2.0
//! R10 scoring: how close a set of extracted facts is to the labelled truth.
//! Facts compare as normalized `(subject, predicate, object, polarity)`
//! tuples, so a statement that should have been `ended` but came out
//! `asserted` is both a miss and a false positive.

use std::collections::{BTreeMap, BTreeSet};

use super::response::Polarity;

mod corpus;
#[cfg(test)]
mod tiers;

/// A fact reduced to what the comparison cares about.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Tuple {
    subject: String,
    predicate: String,
    object: String,
    ended: bool,
}

/// Lower-cases, trims and collapses runs of whitespace, so casing and spacing
/// never decide a match.
pub(crate) fn normalize_tuple(
    subject: &str,
    predicate: &str,
    object: &str,
    polarity: Polarity,
) -> Tuple {
    Tuple {
        subject: normalize(subject),
        predicate: normalize(predicate),
        object: normalize(object),
        ended: polarity == Polarity::Ended,
    }
}

fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Counts and ratios of one comparison. An empty side is vacuously perfect:
/// no predictions make no wrong ones, and nothing expected leaves nothing
/// missed. That keeps every ratio defined without dividing by zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Score {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
}

/// Scores `predicted` against `expected`, each counted as a set.
pub(crate) fn score(predicted: &[Tuple], expected: &[Tuple]) -> Score {
    let predicted: BTreeSet<&Tuple> = predicted.iter().collect();
    let expected: BTreeSet<&Tuple> = expected.iter().collect();
    let tp = predicted.intersection(&expected).count();
    let fp = predicted.len() - tp;
    let fn_ = expected.len() - tp;
    let precision = ratio(tp, predicted.len());
    let recall = ratio(tp, expected.len());
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    Score {
        precision,
        recall,
        f1,
        tp,
        fp,
        fn_,
    }
}

fn ratio(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        1.0
    } else {
        part as f64 / whole as f64
    }
}

/// One score per predicate named on either side.
pub(crate) fn score_by_predicate(
    predicted: &[Tuple],
    expected: &[Tuple],
) -> BTreeMap<String, Score> {
    let predicates: BTreeSet<&str> = predicted
        .iter()
        .chain(expected)
        .map(|tuple| tuple.predicate.as_str())
        .collect();
    let of = |tuples: &[Tuple], predicate: &str| -> Vec<Tuple> {
        tuples
            .iter()
            .filter(|tuple| tuple.predicate == predicate)
            .cloned()
            .collect()
    };
    predicates
        .into_iter()
        .map(|predicate| {
            let scored = score(&of(predicted, predicate), &of(expected, predicate));
            (predicate.to_string(), scored)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asserted(subject: &str, predicate: &str, object: &str) -> Tuple {
        normalize_tuple(subject, predicate, object, Polarity::Asserted)
    }

    fn works_at(subject: &str, object: &str) -> Tuple {
        asserted(subject, "works_at", object)
    }

    #[test]
    fn identical_sets_score_one() {
        // Arrange
        let facts = [works_at("ana", "acme"), works_at("bo", "globex")];

        // Act
        let result = score(&facts, &facts);

        // Assert
        assert_eq!(
            (result.precision, result.recall, result.f1),
            (1.0, 1.0, 1.0)
        );
        assert_eq!((result.tp, result.fp, result.fn_), (2, 0, 0));
    }

    #[test]
    fn a_partial_overlap_scores_by_hand() {
        // Arrange: 1 hit, 1 wrong, 3 expected, so p = 1/2 and r = 1/3.
        let predicted = [works_at("ana", "acme"), works_at("bo", "initech")];
        let expected = [
            works_at("ana", "acme"),
            works_at("bo", "globex"),
            works_at("cy", "umbrella"),
        ];

        // Act
        let result = score(&predicted, &expected);

        // Assert
        assert_eq!((result.tp, result.fp, result.fn_), (1, 1, 2));
        assert_eq!(result.precision, 0.5);
        assert_eq!(result.recall, 1.0 / 3.0);
        assert!((result.f1 - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn disjoint_sets_score_zero() {
        // Act
        let result = score(&[works_at("ana", "acme")], &[works_at("bo", "globex")]);

        // Assert
        assert_eq!(
            (result.precision, result.recall, result.f1),
            (0.0, 0.0, 0.0)
        );
    }

    #[test]
    fn a_repeated_prediction_counts_once() {
        // Arrange
        let predicted = [works_at("ana", "acme"), works_at("ana", "acme")];

        // Act
        let result = score(&predicted, &[works_at("ana", "acme")]);

        // Assert
        assert_eq!((result.tp, result.fp, result.fn_), (1, 0, 0));
        assert_eq!(result.f1, 1.0);
    }

    #[test]
    fn case_and_spacing_do_not_change_a_match() {
        // Arrange
        let predicted = [normalize_tuple(
            "  Ana   Lima ",
            "Works_At",
            "ACME\tCorp",
            Polarity::Asserted,
        )];
        let expected = [works_at("ana lima", "acme corp")];

        // Act
        let result = score(&predicted, &expected);

        // Assert
        assert_eq!(result.f1, 1.0);
    }

    #[test]
    fn a_wrong_polarity_is_a_false_positive_and_a_miss() {
        // Arrange
        let predicted = [works_at("ana", "acme")];
        let expected = [normalize_tuple("ana", "works_at", "acme", Polarity::Ended)];

        // Act
        let result = score(&predicted, &expected);

        // Assert
        assert_eq!((result.tp, result.fp, result.fn_), (0, 1, 1));
        assert_eq!(result.f1, 0.0);
    }

    #[test]
    fn two_empty_sets_score_one_without_dividing_by_zero() {
        // Act
        let result = score(&[], &[]);

        // Assert
        assert_eq!(
            (result.precision, result.recall, result.f1),
            (1.0, 1.0, 1.0)
        );
    }

    #[test]
    fn predicting_nothing_for_something_scores_recall_zero() {
        // Act
        let result = score(&[], &[works_at("ana", "acme")]);

        // Assert
        assert_eq!(result.recall, 0.0);
        assert_eq!(result.f1, 0.0);
        assert_eq!((result.tp, result.fp, result.fn_), (0, 0, 1));
    }

    #[test]
    fn predicting_something_for_nothing_scores_precision_zero() {
        // Act
        let result = score(&[works_at("ana", "acme")], &[]);

        // Assert
        assert_eq!(result.precision, 0.0);
        assert_eq!(result.f1, 0.0);
        assert_eq!((result.tp, result.fp, result.fn_), (0, 1, 0));
    }

    #[test]
    fn the_breakdown_scores_each_predicate_on_its_own() {
        // Arrange: works_at is perfect, lives_in has one hit and one miss.
        let predicted = [works_at("ana", "acme"), asserted("ana", "lives_in", "rome")];
        let expected = [
            works_at("ana", "acme"),
            asserted("ana", "lives_in", "rome"),
            asserted("bo", "lives_in", "oslo"),
        ];

        // Act
        let by_predicate = score_by_predicate(&predicted, &expected);

        // Assert
        assert_eq!(by_predicate["works_at"].f1, 1.0);
        let lives_in = by_predicate["lives_in"];
        assert_eq!((lives_in.tp, lives_in.fp, lives_in.fn_), (1, 0, 1));
        assert_eq!(lives_in.recall, 0.5);
        assert_eq!(by_predicate.len(), 2);
    }

    #[test]
    fn a_predicate_seen_on_one_side_only_still_gets_a_score() {
        // Act
        let by_predicate = score_by_predicate(&[asserted("ana", "knows", "bo")], &[]);

        // Assert
        assert_eq!(by_predicate["knows"].precision, 0.0);
    }
}
