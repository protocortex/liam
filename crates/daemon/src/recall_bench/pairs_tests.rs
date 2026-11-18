// SPDX-License-Identifier: Apache-2.0
//! The pair file a benchmark run reads: one `query<TAB>relevant fact` per line.

use super::*;

fn pair_file(text: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pairs.tsv");
    std::fs::write(&path, text).unwrap();
    (dir, path)
}

#[test]
fn blank_lines_are_skipped_and_each_side_is_trimmed() {
    // Arrange
    let (_dir, path) =
        pair_file("  where do I live \t I live in Lisbon \n\n   \nfavourite tea\tgreen tea\n");

    // Act
    let pairs = read_pairs(&path).unwrap();

    // Assert
    let read: Vec<(&str, &str)> = pairs
        .iter()
        .map(|pair| (pair.query.as_str(), pair.fact.as_str()))
        .collect();
    assert_eq!(
        read,
        vec![
            ("where do I live", "I live in Lisbon"),
            ("favourite tea", "green tea")
        ]
    );
}

#[test]
fn a_line_without_a_tab_is_refused_with_its_line_number() {
    // Arrange
    let (_dir, path) = pair_file("good query\tgood fact\n\nno tab here\n");

    // Act
    let error = read_pairs(&path).err().expect("the file is refused");

    // Assert
    assert_eq!(
        error.to_string(),
        "pair file line 3 needs `query<TAB>relevant fact`"
    );
}

#[test]
fn a_file_with_only_blank_lines_has_no_pairs() {
    // Arrange
    let (_dir, path) = pair_file("\n  \n");

    // Act
    let error = read_pairs(&path).err().expect("the file is refused");

    // Assert
    assert_eq!(
        error.to_string(),
        format!("the pair file {} has no pairs", path.display())
    );
}

#[test]
fn a_file_that_does_not_exist_names_its_path() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing.tsv");

    // Act
    let error = read_pairs(&path).err().expect("the file is refused");

    // Assert
    assert_eq!(
        error.to_string(),
        format!("reading the pair file {}", path.display())
    );
}
