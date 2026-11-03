// SPDX-License-Identifier: Apache-2.0
//! The shipped `liam.toml` and the README name every key of the sections that
//! guard deletion, archive and metrics, so a key added in code cannot ship
//! undocumented.

use serde_json::Value;

use super::{AdminConfig, ArchiveConfig, DeletionConfig, EncryptionConfig, MetricsConfig};

/// Each section with its default config as JSON. A key whose default is
/// `None` stays visible as `null`.
fn sections() -> Vec<(&'static str, Value)> {
    fn json<T: serde::Serialize>(value: T) -> Value {
        serde_json::to_value(value).unwrap()
    }
    vec![
        ("encryption", json(EncryptionConfig::default())),
        ("deletion", json(DeletionConfig::default())),
        ("admin", json(AdminConfig::default())),
        ("archive", json(ArchiveConfig::default())),
        ("metrics", json(MetricsConfig::default())),
    ]
}

fn keys_of(value: &Value) -> Vec<String> {
    value
        .as_object()
        .expect("a section serialises as a table")
        .keys()
        .cloned()
        .collect()
}

fn read_root(file: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(file);
    std::fs::read_to_string(path).unwrap()
}

/// The lines of the `[section]` table, up to the next table.
fn toml_table<'a>(text: &'a str, section: &str) -> Vec<&'a str> {
    let header = format!("[{section}]");
    text.lines()
        .skip_while(|line| line.trim_end() != header)
        .skip(1)
        .take_while(|line| !line.starts_with('['))
        .collect()
}

/// Whether `line` assigns `key`, commented out or not.
fn assigns(line: &str, key: &str) -> bool {
    let line = line.trim_start().trim_start_matches('#').trim_start();
    line.strip_prefix(key)
        .is_some_and(|rest| rest.trim_start().starts_with('='))
}

fn missing_from_toml(text: &str) -> Vec<String> {
    let mut missing = Vec::new();
    for (section, value) in sections() {
        let table = toml_table(text, section);
        for key in keys_of(&value) {
            if !table.iter().any(|line| assigns(line, &key)) {
                missing.push(format!("{section}.{key}"));
            }
        }
    }
    missing
}

fn missing_from_readme(text: &str) -> Vec<String> {
    let mut missing = Vec::new();
    for (section, value) in sections() {
        for key in keys_of(&value) {
            let row = format!("| `{section}.{key}` |");
            if !text.lines().any(|line| line.starts_with(&row)) {
                missing.push(format!("{section}.{key}"));
            }
        }
    }
    missing
}

#[test]
fn liam_toml_documents_every_key_of_the_new_sections() {
    // Arrange
    let text = read_root("liam.toml");

    // Act
    let missing = missing_from_toml(&text);

    // Assert
    assert!(missing.is_empty(), "liam.toml must document {missing:?}");
}

#[test]
fn the_readme_configuration_table_documents_every_key_of_the_new_sections() {
    // Arrange
    let text = read_root("README.md");

    // Act
    let missing = missing_from_readme(&text);

    // Assert
    assert!(missing.is_empty(), "README.md must document {missing:?}");
}

#[test]
fn a_file_with_a_key_removed_fails_the_check() {
    // Arrange
    let toml: String = read_root("liam.toml")
        .lines()
        .filter(|line| !assigns(line, "deadline_days"))
        .map(|line| format!("{line}\n"))
        .collect();
    let readme: String = read_root("README.md")
        .lines()
        .filter(|line| !line.starts_with("| `archive.after_days` |"))
        .map(|line| format!("{line}\n"))
        .collect();

    // Act
    let (from_toml, from_readme) = (missing_from_toml(&toml), missing_from_readme(&readme));

    // Assert
    assert_eq!(from_toml, vec!["deletion.deadline_days".to_string()]);
    assert_eq!(from_readme, vec!["archive.after_days".to_string()]);
}

#[test]
fn the_notices_name_each_dependency_the_core_changes_added() {
    // Arrange
    let text = read_root("THIRD-PARTY-NOTICES.md");
    let added = [
        "aes-gcm",
        "hmac",
        "sha2",
        "postcard",
        "parquet",
        "zstd",
        "zstd-sys",
        "fastbloom",
        "sysinfo",
        "object_store",
    ];

    // Act
    let missing: Vec<&str> = added
        .into_iter()
        .filter(|name| !text.contains(name))
        .collect();

    // Assert
    assert!(missing.is_empty(), "the notices must name {missing:?}");
}
