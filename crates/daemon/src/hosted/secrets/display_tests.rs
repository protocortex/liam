// SPDX-License-Identifier: Apache-2.0
//! What an operator reads when a provider key cannot be loaded.

use super::*;

#[test]
fn each_key_failure_says_what_to_fix() {
    // Arrange
    let path = PathBuf::from("/keys/provider.key");
    let cases = [
        (
            SecretError::Unreadable {
                path: path.clone(),
                reason: "permission denied".into(),
            },
            "key file /keys/provider.key could not be read: permission denied",
        ),
        (
            SecretError::EmptyFile { path: path.clone() },
            "the key file /keys/provider.key is empty",
        ),
        (
            SecretError::NotInKeychain {
                service: "liam-anthropic".into(),
            },
            "the Keychain has no key for service liam-anthropic",
        ),
        (
            SecretError::Keychain {
                service: "liam-anthropic".into(),
                reason: "locked".into(),
            },
            "the Keychain could not be read for service liam-anthropic: locked",
        ),
    ];

    for (error, expected) in cases {
        // Act
        let text = error.to_string();

        // Assert
        assert_eq!(text, expected);
    }
}
