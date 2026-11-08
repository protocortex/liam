// SPDX-License-Identifier: Apache-2.0
//! The shipped `liam.toml` and the configuration page name every key of the sections that
//! guard deletion, archive, metrics and hosted calls, so a key added in code
//! cannot ship undocumented.

use serde_json::Value;

use super::{
    AdminConfig, ArchiveConfig, DeletionConfig, EncryptionConfig, HostedConfig, MetricsConfig,
};

/// Each section with its default config as JSON, and each table nested in one
/// as a section of its own (`hosted.privacy`). A key whose default is `None`
/// stays visible as `null`.
fn sections() -> Vec<(String, Value)> {
    fn json<T: serde::Serialize>(value: T) -> Value {
        serde_json::to_value(value).unwrap()
    }
    fn with_tables(name: &str, value: Value, found: &mut Vec<(String, Value)>) {
        let tables: Vec<(String, Value)> = value
            .as_object()
            .expect("a section serialises as a table")
            .iter()
            .filter(|(_, inner)| inner.is_object())
            .map(|(key, inner)| (format!("{name}.{key}"), inner.clone()))
            .collect();
        found.push((name.to_owned(), value));
        for (table, inner) in tables {
            with_tables(&table, inner, found);
        }
    }
    let mut found = Vec::new();
    for (name, value) in [
        ("encryption", json(EncryptionConfig::default())),
        ("deletion", json(DeletionConfig::default())),
        ("admin", json(AdminConfig::default())),
        ("archive", json(ArchiveConfig::default())),
        ("metrics", json(MetricsConfig::default())),
        ("hosted", json(HostedConfig::default())),
    ] {
        with_tables(name, value, &mut found);
    }
    found
}

/// The keys a section assigns itself. The tables nested in it are sections of
/// their own.
fn keys_of(value: &Value) -> Vec<String> {
    value
        .as_object()
        .expect("a section serialises as a table")
        .iter()
        .filter(|(_, inner)| !inner.is_object())
        .map(|(key, _)| key.clone())
        .collect()
}

/// The documentation pages the config tests read, relative to the repository root.
pub(super) const CONFIGURATION_PAGE: &str = "docs/reference/configuration.md";
pub(super) const TIERING_PAGE: &str = "docs/concepts/tiering.md";
pub(super) const SUMMARIES_PAGE: &str = "docs/concepts/summaries.md";

/// A page's title heading. Everything after it belongs to the page.
pub(super) const PAGE_TITLE_MARK: &str = "# ";

pub(super) fn read_root(file: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(file);
    std::fs::read_to_string(path).unwrap()
}

/// The lines of a docs page that follow its `# Title` heading, up to the next
/// heading of the same level. Empty when the page does not carry that title.
pub(super) fn page_body(text: &str, title_heading: &str) -> Vec<String> {
    text.lines()
        .skip_while(|line| *line != title_heading)
        .skip(1)
        .take_while(|line| !line.starts_with(PAGE_TITLE_MARK))
        .map(str::to_owned)
        .collect()
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
        if !text
            .lines()
            .any(|line| line.trim_end() == format!("[{section}]"))
        {
            missing.push(format!("[{section}]"));
        }
        let table = toml_table(text, &section);
        for key in keys_of(&value) {
            if !table.iter().any(|line| assigns(line, &key)) {
                missing.push(format!("{section}.{key}"));
            }
        }
    }
    missing
}

fn missing_from_page(text: &str) -> Vec<String> {
    let mut missing = Vec::new();
    for (section, value) in sections() {
        let mut named: Vec<String> = keys_of(&value)
            .into_iter()
            .map(|key| format!("{section}.{key}"))
            .collect();
        // A nested table has a row of its own, besides its keys.
        if section.contains('.') {
            named.push(section.clone());
        }
        for name in named {
            let row = format!("| `{name}` |");
            if !text.lines().any(|line| line.starts_with(&row)) {
                missing.push(name);
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
fn the_configuration_page_documents_every_key_of_the_new_sections() {
    // Arrange
    let text = read_root(CONFIGURATION_PAGE);

    // Act
    let missing = missing_from_page(&text);

    // Assert
    assert!(
        missing.is_empty(),
        "{CONFIGURATION_PAGE} must document {missing:?}"
    );
}

#[test]
fn a_file_with_a_key_removed_fails_the_check() {
    // Arrange
    let toml: String = read_root("liam.toml")
        .lines()
        .filter(|line| !assigns(line, "deadline_days"))
        .map(|line| format!("{line}\n"))
        .collect();
    let page: String = read_root(CONFIGURATION_PAGE)
        .lines()
        .filter(|line| !line.starts_with("| `archive.after_days` |"))
        .map(|line| format!("{line}\n"))
        .collect();

    // Act
    let (from_toml, from_page) = (missing_from_toml(&toml), missing_from_page(&page));

    // Assert
    assert_eq!(from_toml, vec!["deletion.deadline_days".to_string()]);
    assert_eq!(from_page, vec!["archive.after_days".to_string()]);
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

#[test]
fn the_hosted_tables_and_keys_are_all_named() {
    // Arrange
    let names: Vec<String> = sections().into_iter().map(|(name, _)| name).collect();

    // Assert
    for table in [
        "hosted",
        "hosted.privacy",
        "hosted.models",
        "hosted.prices",
        "hosted.prices.models",
        "hosted.key_files",
    ] {
        assert!(
            names.iter().any(|name| name == table),
            "{table} in {names:?}"
        );
    }
}

#[test]
fn a_hosted_key_missing_from_either_file_fails_the_check() {
    // Arrange
    let toml: String = read_root("liam.toml")
        .lines()
        .filter(|line| !assigns(line, "allowed_hosts"))
        .map(|line| format!("{line}\n"))
        .collect();
    let page: String = read_root(CONFIGURATION_PAGE)
        .lines()
        .filter(|line| !line.starts_with("| `hosted.privacy.allowed_scopes` |"))
        .map(|line| format!("{line}\n"))
        .collect();

    // Act
    let (from_toml, from_page) = (missing_from_toml(&toml), missing_from_page(&page));

    // Assert
    assert_eq!(from_toml, vec!["hosted.allowed_hosts".to_string()]);
    assert_eq!(from_page, vec!["hosted.privacy.allowed_scopes".to_string()]);
}

#[test]
fn the_notices_name_each_crate_the_hosted_feature_adds() {
    // Arrange
    let text = read_root("THIRD-PARTY-NOTICES.md");
    let added = [
        "ureq",
        "ureq-proto",
        "rustls",
        "rustls-webpki",
        "rustls-pki-types",
        "webpki-roots",
        "ring",
        "untrusted",
        "subtle",
    ];

    // Act
    let missing: Vec<&str> = added
        .into_iter()
        .filter(|name| !text.contains(name))
        .collect();

    // Assert
    assert!(missing.is_empty(), "the notices must name {missing:?}");
}
