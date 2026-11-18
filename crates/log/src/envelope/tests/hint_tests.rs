// SPDX-License-Identifier: Apache-2.0
//! A key hint the sealer cannot honour is refused with the reason, never
//! replaced by another key.

use super::*;

const UNKNOWN_KEY: &str = "no-such-key";

fn edge_event_with_hint(key_id: &str) -> (LogEvent, KeyHints) {
    let batch = event(
        "e1",
        LogPayload::EpisodeBatch(vec![
            RowEffect::Node(node("b1", Some("bob"))),
            RowEffect::Edge(edge("hinted", "elsewhere", "b1")),
        ]),
    );
    let hints = KeyHints {
        edge_sources: [("elsewhere".to_owned(), key_id.to_owned())].into(),
        ..KeyHints::default()
    };
    (batch, hints)
}

#[test]
fn a_hint_naming_a_key_the_registry_does_not_hold_is_a_missing_key() {
    // Arrange
    let fixture = fixture();
    let (batch, hints) = edge_event_with_hint(UNKNOWN_KEY);

    // Act
    let result = fixture.sealer.seal(&batch, &hints);

    // Assert
    assert!(
        matches!(&result, Err(SealError::Crypto(CryptoError::KeyMissing { key_id })) if key_id == UNKNOWN_KEY),
        "{result:?}"
    );
}

#[test]
fn a_hint_naming_a_destroyed_key_is_a_locked_group() {
    // Arrange
    let fixture = fixture();
    let destroyed = fixture
        .registry
        .key_for_subject(Some(SCOPE), "carol")
        .unwrap();
    fixture.registry.destroy(&destroyed).unwrap();
    let (batch, hints) = edge_event_with_hint(&destroyed);

    // Act
    let result = fixture.sealer.seal(&batch, &hints);

    // Assert
    assert!(
        matches!(&result, Err(SealError::Crypto(CryptoError::KeyGroupLocked { key_id })) if *key_id == destroyed),
        "{result:?}"
    );
}
