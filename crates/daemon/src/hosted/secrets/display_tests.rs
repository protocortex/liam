// SPDX-License-Identifier: Apache-2.0
//! What an operator reads when a provider key cannot be loaded.

use super::*;

const KEY_PATH: &str = "/keys/provider.key";
const SERVICE: &str = "liam-anthropic";
const MODE: u32 = 0o644;

#[test]
fn each_key_failure_names_the_path_or_service_and_the_reason() {
    // Arrange
    let path = PathBuf::from(KEY_PATH);
    let cases = [
        (
            SecretError::Unreadable {
                path: path.clone(),
                reason: "permission denied".into(),
            },
            vec![KEY_PATH, "permission denied"],
        ),
        (
            SecretError::EmptyFile { path: path.clone() },
            vec![KEY_PATH],
        ),
        (
            SecretError::NotInKeychain {
                service: SERVICE.into(),
            },
            vec![SERVICE],
        ),
        (
            SecretError::Keychain {
                service: SERVICE.into(),
                reason: "locked".into(),
            },
            vec![SERVICE, "locked"],
        ),
        (
            SecretError::LooseMode { path, mode: MODE },
            vec![KEY_PATH, "644", "chmod 600"],
        ),
    ];

    for (error, carried) in cases {
        // Act
        let text = error.to_string();

        // Assert
        for part in carried {
            assert!(text.contains(part), "{text:?} should carry {part:?}");
        }
    }
}
