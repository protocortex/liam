// SPDX-License-Identifier: Apache-2.0
//! The per-language table printed after a multilingual run.

use super::*;

const BASE_SCORE: f64 = 0.8;
const MEASURED_SCORE: f64 = 0.75;

fn uniform(score: f64) -> BTreeMap<Language, f64> {
    Language::ALL.into_iter().map(|l| (l, score)).collect()
}

#[test]
fn every_language_has_its_own_two_letter_code() {
    // Act
    let codes: Vec<&str> = Language::ALL
        .iter()
        .map(|language| language.code())
        .collect();

    // Assert
    assert_eq!(codes, ["en", "pt", "es", "zh", "fr", "de", "ja", "ar"]);
}

#[test]
fn the_table_has_one_row_per_language_marking_which_ones_gate() {
    // Arrange
    let baseline = uniform(BASE_SCORE);
    let measured = uniform(MEASURED_SCORE);

    // Act
    let table = render_table(&baseline, &measured);

    // Assert
    let lines: Vec<&str> = table.lines().collect();
    assert_eq!(lines[0], "lang  ndcg@10  baseline  delta   role");
    assert_eq!(lines.len(), 1 + Language::ALL.len());
    assert_eq!(lines[1], "en    0.750    0.800     -0.050  gates");
    assert_eq!(lines[4], "zh    0.750    0.800     -0.050  reported");
}

#[test]
fn a_language_missing_from_either_side_has_no_row() {
    // Arrange
    let mut baseline = uniform(BASE_SCORE);
    let mut measured = uniform(MEASURED_SCORE);
    baseline.remove(&Language::Fr);
    measured.remove(&Language::Ja);

    // Act
    let table = render_table(&baseline, &measured);

    // Assert
    assert!(!table.contains("\nfr "), "{table}");
    assert!(!table.contains("\nja "), "{table}");
    assert_eq!(table.lines().count(), 1 + Language::ALL.len() - 2);
}
