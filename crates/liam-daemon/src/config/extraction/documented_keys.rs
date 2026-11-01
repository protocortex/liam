// SPDX-License-Identifier: Apache-2.0
//! Every `[extraction]` key must be documented in the shipped `liam.toml` and in
//! the README, and the notices must name the server the extraction fetches.

use super::ExtractionConfig;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

fn read(file: &str) -> String {
    std::fs::read_to_string(format!("{ROOT}/{file}")).unwrap_or_else(|e| panic!("read {file}: {e}"))
}

/// The key names of the default config, each with whether it holds a table. A
/// key whose default is `None` still appears, as `null`.
fn default_keys() -> Vec<(String, bool)> {
    let value = serde_json::to_value(ExtractionConfig::default()).expect("serialize defaults");
    value
        .as_object()
        .expect("config serializes as an object")
        .iter()
        .map(|(key, value)| (key.clone(), value.is_object()))
        .collect()
}

/// The lines of the table that starts at the exact `header` line, up to the
/// next line that opens another table.
fn toml_table<'a>(toml: &'a str, header: &str) -> Vec<&'a str> {
    toml.lines()
        .skip_while(|line| line.trim() != header)
        .skip(1)
        .take_while(|line| !line.trim_start().starts_with('['))
        .collect()
}

/// A line of the form `key =`, with an optional leading `#`.
fn defines_key(line: &str, key: &str) -> bool {
    line.trim_start()
        .trim_start_matches('#')
        .trim_start()
        .strip_prefix(key)
        .is_some_and(|rest| rest.trim_start().starts_with('='))
}

/// The README text from the `## Extraction` heading to the next `##` heading.
fn readme_extraction_section(readme: &str) -> String {
    readme
        .lines()
        .skip_while(|line| *line != "## Extraction")
        .skip(1)
        .take_while(|line| !line.starts_with("## "))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn every_extraction_key_is_in_the_shipped_liam_toml() {
    // Arrange
    let toml = read("liam.toml");
    let table = toml_table(&toml, "[extraction]");
    assert!(!table.is_empty(), "liam.toml has no [extraction] table");

    for (key, is_table) in default_keys() {
        // Act
        let documented = if is_table {
            toml.lines()
                .any(|line| line.trim() == format!("[extraction.{key}]"))
        } else {
            table.iter().any(|line| defines_key(line, &key))
        };

        // Assert
        assert!(documented, "liam.toml does not document extraction.{key}");
    }
}

#[test]
fn every_extraction_key_is_in_the_readme_extraction_section() {
    // Arrange
    let section = readme_extraction_section(&read("README.md"));
    assert!(!section.is_empty(), "README has no ## Extraction section");

    for (key, _) in default_keys() {
        // Act
        let documented = section.contains(&format!("`{key}`"));

        // Assert
        assert!(documented, "README does not document extraction.{key}");
    }
}

#[test]
fn the_notices_name_llama_server() {
    // Arrange
    let notices = read("THIRD-PARTY-NOTICES.md");

    // Act
    let named = notices.contains("llama-server");

    // Assert
    assert!(
        named,
        "THIRD-PARTY-NOTICES.md does not mention llama-server"
    );
}
