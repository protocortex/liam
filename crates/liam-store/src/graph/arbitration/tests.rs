// SPDX-License-Identifier: Apache-2.0
//! Every arbitration rule, read off a state and a claim with no store.

use super::*;
use crate::ids::{BEGINNING, FOREVER};
use crate::types::{Cardinality, ExtractorName, MatchMode};
use crate::Error;

const AT: Millis = Millis(10_000);
const NEW_EVENT: &str = "e-new";

fn version(id: &str, object: &str) -> Fact {
    Fact {
        id: FactId::from_raw(id),
        lineage_id: FactId::from_raw(id),
        subject: "alice".into(),
        predicate: "lives_in".into(),
        object: object.into(),
        object_key: object_key(object),
        scope: None,
        evidence_at: Millis(100),
        valid_from: Millis(0),
        valid_until: FOREVER,
        tx_from: Millis(0),
        tx_to: FOREVER,
        closed_by: None,
        confidence: 1.0,
        trust_score: 0.5,
        extractor_id: ExtractorName::Rules,
        extractor_version: "v1".into(),
        source_event_ids: vec![format!("e-{id}")],
    }
}

fn new_fact(object: &str) -> NewFact {
    NewFact {
        subject: "alice".into(),
        predicate: "lives_in".into(),
        object: object.into(),
        scope: None,
        valid_from: Millis(1000),
        valid_until: FOREVER,
        evidence_at: Millis(200),
        confidence: 1.0,
        trust_score: 0.5,
        extractor: ExtractorName::Rules,
        extractor_version: "v1".into(),
        source_event_ids: vec![NEW_EVENT.into()],
    }
}

fn claim(object: &str) -> Claim {
    Claim::new(new_fact(object), String::new()).unwrap()
}

fn rules(cardinality: Cardinality, match_mode: MatchMode) -> Rules {
    Rules {
        spec: PredicateSpec {
            cardinality,
            match_mode,
            ..PredicateSpec::default()
        },
        caps: ArbitrationCaps::default(),
        at: AT,
    }
}

fn single() -> Rules {
    rules(Cardinality::One, MatchMode::Exact)
}

fn state(versions: Vec<Fact>) -> KeyState {
    KeyState {
        versions,
        suppressions: vec![],
    }
}

fn mint() -> impl FnMut() -> FactId {
    let mut next = 0;
    move || {
        next += 1;
        FactId::from_raw(format!("m{next}"))
    }
}

fn assert_with(state: &KeyState, claim: &Claim, rules: &Rules) -> Decision<ArbitrationOutcome> {
    decide_assert(state, claim, rules, &mut mint()).unwrap()
}

fn ids(raw: &[&str]) -> Vec<FactId> {
    raw.iter().map(|id| FactId::from_raw(*id)).collect()
}

fn inserted<'a>(plan: &'a Plan, id: &str) -> &'a Fact {
    plan.insert
        .iter()
        .find(|fact| fact.id.as_str() == id)
        .unwrap_or_else(|| panic!("no inserted version {id}"))
}

fn corroborated(lineage: &str, changed: bool) -> ArbitrationOutcome {
    ArbitrationOutcome::Corroborated {
        lineage_id: FactId::from_raw(lineage),
        changed,
    }
}

fn linked(lineage: &str) -> Vec<(FactId, Vec<String>)> {
    vec![(FactId::from_raw(lineage), vec![NEW_EVENT.to_string()])]
}

#[test]
fn an_object_key_is_trimmed_collapsed_and_case_folded() {
    assert_eq!(object_key("  Sydney,   AUSTRALIA "), "sydney, australia");
}

#[test]
fn a_claim_with_nothing_live_starts_a_new_lineage() {
    // Arrange
    let claim = claim("Sydney");

    // Act
    let decision = assert_with(&state(vec![]), &claim, &single());

    // Assert
    let id = FactId::from_raw("m1");
    assert_eq!(
        decision.outcome,
        ArbitrationOutcome::Inserted { id: id.clone() }
    );
    let row = inserted(&decision.plan, "m1");
    assert_eq!(row.lineage_id, id);
    assert_eq!((row.tx_from, row.tx_to), (AT, FOREVER));
    assert_eq!(decision.plan.link, vec![(id, vec![NEW_EVENT.to_string()])]);
    assert!(decision.plan.close.is_empty());
}

#[test]
fn a_many_valued_predicate_keeps_every_object_and_closes_none() {
    // Arrange
    let live = state(vec![version("a", "Sydney")]);

    // Act
    let decision = assert_with(
        &live,
        &claim("Melbourne"),
        &rules(Cardinality::Many, MatchMode::Exact),
    );

    // Assert
    assert!(matches!(
        decision.outcome,
        ArbitrationOutcome::Inserted { .. }
    ));
    assert!(decision.plan.close.is_empty());
}

#[test]
fn a_many_valued_predicate_corroborates_the_same_object() {
    // Arrange
    let live = state(vec![version("a", "Sydney")]);

    // Act
    let decision = assert_with(
        &live,
        &claim("SYDNEY"),
        &rules(Cardinality::Many, MatchMode::Exact),
    );

    // Assert
    assert_eq!(decision.outcome, corroborated("a", true));
    assert!(decision.plan.insert.is_empty());
    assert_eq!(decision.plan.link, linked("a"));
}

#[test]
fn containment_agrees_for_a_contains_predicate_and_keeps_the_stored_text() {
    for object in ["Sydney, Australia", "SYDNEY", "sydney"] {
        // Arrange
        let live = state(vec![version("a", "Sydney")]);

        // Act
        let decision = assert_with(
            &live,
            &claim(object),
            &rules(Cardinality::One, MatchMode::Contains),
        );

        // Assert
        assert_eq!(decision.outcome, corroborated("a", true), "{object}");
        assert!(decision.plan.insert.is_empty(), "{object}");
        assert!(decision.plan.close.is_empty(), "{object}");
    }
}

#[test]
fn an_exact_predicate_never_merges_objects_that_only_contain_each_other() {
    for (stored, claimed) in [("Engineer", "Senior Engineer"), ("active", "not active")] {
        // Arrange
        let live = state(vec![version("a", stored)]);

        // Act
        let decision = assert_with(&live, &claim(claimed), &single());

        // Assert
        assert!(
            matches!(decision.outcome, ArbitrationOutcome::Replaced { .. }),
            "{stored} then {claimed}"
        );
    }
}

#[test]
fn an_ambiguous_containment_match_counts_as_a_different_object() {
    // Arrange
    let live = state(vec![
        version("a", "Sydney, Australia"),
        version("b", "Melbourne, Australia"),
    ]);

    // Act
    let decision = assert_with(
        &live,
        &claim("Australia"),
        &rules(Cardinality::One, MatchMode::Contains),
    );

    // Assert
    assert_eq!(
        decision.outcome,
        ArbitrationOutcome::Replaced {
            closed: ids(&["a", "b"]),
            inserted: FactId::from_raw("m1")
        }
    );
}

#[test]
fn a_closed_version_does_not_count_toward_containment_ambiguity() {
    // Arrange
    let mut closed = version("b", "Melbourne, Australia");
    closed.valid_until = Millis(500);
    let live = state(vec![version("a", "Sydney, Australia"), closed]);

    // Act
    let decision = assert_with(
        &live,
        &claim("Australia"),
        &rules(Cardinality::One, MatchMode::Contains),
    );

    // Assert
    assert_eq!(decision.outcome, corroborated("a", true));
}

#[test]
fn corroborating_from_a_new_event_at_or_below_the_stored_trust_only_links_it() {
    for trust in [0.3, 0.5] {
        // Arrange
        let live = state(vec![version("a", "Sydney")]);
        let mut claim = claim("Sydney");
        claim.fact.trust_score = trust;

        // Act
        let decision = assert_with(&live, &claim, &single());

        // Assert
        assert_eq!(
            decision.plan,
            Plan {
                link: linked("a"),
                ..Plan::default()
            },
            "trust {trust}"
        );
    }
}

#[test]
fn corroborating_with_higher_trust_replaces_the_version_and_stays_live() {
    // Arrange
    let live = state(vec![version("a", "Sydney")]);
    let mut claim = claim("Sydney");
    claim.fact.trust_score = 0.9;

    // Act
    let decision = assert_with(&live, &claim, &single());

    // Assert
    assert_eq!(decision.plan.close, ids(&["a"]));
    let clone = inserted(&decision.plan, "m1");
    assert_eq!(clone.lineage_id, FactId::from_raw("a"));
    assert_eq!(clone.trust_score, 0.9);
    assert_eq!((clone.valid_from, clone.valid_until), (Millis(0), FOREVER));
    assert_eq!(clone.tx_from, AT);
}

#[test]
fn a_higher_trust_corroboration_closes_a_lower_trust_contradiction() {
    // Arrange
    let mut contradiction = version("b", "Melbourne");
    contradiction.trust_score = 0.3;
    let live = state(vec![version("a", "Sydney"), contradiction]);
    let mut claim = claim("Sydney");
    claim.fact.trust_score = 0.9;

    // Act
    let decision = assert_with(&live, &claim, &single());

    // Assert
    assert_eq!(decision.plan.close, ids(&["a", "b"]));
    let closed = inserted(&decision.plan, "m2");
    assert_eq!(closed.lineage_id, FactId::from_raw("b"));
    assert_eq!(closed.closed_by, Some(FactId::from_raw("m1")));
    assert_eq!(closed.valid_until, Millis(0));
}

#[test]
fn a_higher_trust_corroboration_leaves_a_contradiction_on_newer_evidence() {
    // Arrange
    let mut contradiction = version("b", "Melbourne");
    contradiction.trust_score = 0.3;
    contradiction.evidence_at = Millis(300);
    let live = state(vec![version("a", "Sydney"), contradiction]);
    let mut claim = claim("Sydney");
    claim.fact.trust_score = 0.9;

    // Act
    let decision = assert_with(&live, &claim, &single());

    // Assert
    assert_eq!(decision.plan.close, ids(&["a"]));
}

#[test]
fn repeating_an_assert_whose_events_are_all_linked_changes_nothing() {
    // Arrange
    let mut stored = version("a", "Sydney");
    stored.source_event_ids = vec![NEW_EVENT.to_string()];

    // Act
    let decision = assert_with(&state(vec![stored]), &claim("Sydney"), &single());

    // Assert
    assert_eq!(decision.outcome, corroborated("a", false));
    assert_eq!(decision.plan, Plan::default());
}

#[test]
fn a_differing_claim_is_decided_by_trust_with_ties_to_the_newer_fact() {
    for (live_trust, claim_trust, competes) in
        [(0.9, 0.5, true), (0.5, 0.9, false), (0.5, 0.5, false)]
    {
        // Arrange
        let mut stored = version("a", "Sydney");
        stored.trust_score = live_trust;
        let mut claim = claim("Melbourne");
        claim.fact.trust_score = claim_trust;

        // Act
        let decision = assert_with(&state(vec![stored]), &claim, &single());

        // Assert
        let label = format!("live {live_trust}, claim {claim_trust}");
        if competes {
            assert_eq!(
                decision.outcome,
                ArbitrationOutcome::Competing {
                    id: FactId::from_raw("m1")
                },
                "{label}"
            );
            assert!(decision.plan.close.is_empty(), "{label}");
        } else {
            assert!(
                matches!(decision.outcome, ArbitrationOutcome::Replaced { .. }),
                "{label}"
            );
        }
    }
}

#[test]
fn a_higher_trust_claim_replaces_every_competing_live_version() {
    // Arrange
    let mut weaker = version("b", "Perth");
    weaker.trust_score = 0.3;
    weaker.valid_from = Millis(2000);
    let live = state(vec![version("a", "Sydney"), weaker]);
    let mut claim = claim("Melbourne");
    claim.fact.trust_score = 0.8;

    // Act
    let decision = assert_with(&live, &claim, &single());

    // Assert
    assert_eq!(
        decision.outcome,
        ArbitrationOutcome::Replaced {
            closed: ids(&["a", "b"]),
            inserted: FactId::from_raw("m1")
        }
    );
    assert_eq!(inserted(&decision.plan, "m2").valid_until, Millis(1000));
    let by_b = inserted(&decision.plan, "m3");
    assert_eq!(by_b.valid_until, Millis(2000));
    assert_eq!(by_b.closed_by, Some(FactId::from_raw("m1")));
}

#[test]
fn a_replacement_never_widens_the_validity_it_closes() {
    // Arrange
    let mut ending = version("a", "Sydney");
    ending.valid_from = Millis(1000);
    ending.valid_until = Millis(3000);
    let mut claim = claim("Melbourne");
    claim.fact.valid_from = Millis(5000);

    // Act
    let decision = assert_with(&state(vec![ending]), &claim, &single());

    // Assert
    assert_eq!(inserted(&decision.plan, "m2").valid_until, Millis(3000));
}

#[test]
fn a_replacement_with_the_same_valid_from_closes_to_an_empty_interval() {
    // Arrange
    let mut claim = claim("Melbourne");
    claim.fact.valid_from = Millis(0);

    // Act
    let decision = assert_with(&state(vec![version("a", "Sydney")]), &claim, &single());

    // Assert
    let closed = inserted(&decision.plan, "m2");
    assert_eq!(
        (closed.valid_from, closed.valid_until),
        (Millis(0), Millis(0))
    );
}

fn competing_set(trusts: [f64; 5]) -> Vec<Fact> {
    let mut primary = version("p", "Sydney");
    primary.trust_score = 0.9;
    let rivals = trusts.iter().enumerate().map(|(i, trust)| {
        let mut rival = version(&format!("c{i}"), &format!("Rival {i}"));
        rival.trust_score = *trust;
        rival.tx_from = Millis(10 * (i as i64 + 1));
        rival
    });
    std::iter::once(primary).chain(rivals).collect()
}

#[test]
fn past_the_cap_the_weakest_oldest_competing_lineage_is_retracted() {
    for (trusts, claim_trust, evicted) in [
        ([0.5; 5], 0.5, "c0"),
        ([0.6, 0.3, 0.6, 0.6, 0.6], 0.5, "c1"),
    ] {
        // Arrange
        let mut claim = claim("Newcomer");
        claim.fact.trust_score = claim_trust;

        // Act
        let decision = assert_with(&state(competing_set(trusts)), &claim, &single());

        // Assert
        assert_eq!(decision.plan.close, ids(&[evicted]), "{trusts:?}");
        assert_eq!(decision.plan.suppress.len(), 1, "{trusts:?}");
        assert_eq!(decision.plan.suppress[0].retracted_at, AT);
        assert!(
            matches!(decision.outcome, ArbitrationOutcome::Competing { .. }),
            "{trusts:?}"
        );
    }
}

#[test]
fn a_newcomer_that_is_itself_the_weakest_is_suppressed_instead_of_stored() {
    // Arrange
    let mut claim = claim("Newcomer");
    claim.fact.trust_score = 0.2;

    // Act
    let decision = assert_with(&state(competing_set([0.6; 5])), &claim, &single());

    // Assert
    assert_eq!(decision.outcome, ArbitrationOutcome::Suppressed);
    assert!(decision.plan.insert.is_empty());
    assert_eq!(decision.plan.suppress[0].object_key, "newcomer");
    assert_eq!(decision.plan.suppress[0].trust_score, 0.2);
}

#[test]
fn at_the_cap_nothing_is_evicted() {
    // Arrange
    let mut claim = claim("Newcomer");
    claim.fact.trust_score = 0.4;
    let mut competing = competing_set([0.6; 5]);
    competing.pop();

    // Act
    let decision = assert_with(&state(competing), &claim, &single());

    // Assert
    assert!(decision.plan.close.is_empty());
    assert!(decision.plan.suppress.is_empty());
}

#[test]
fn older_evidence_becomes_history_ending_where_the_live_fact_starts() {
    for (valid_until, expected_until) in [(FOREVER, Millis(2000)), (Millis(1500), Millis(1500))] {
        // Arrange
        let mut live = version("a", "Sydney");
        live.valid_from = Millis(2000);
        let mut claim = claim("Melbourne");
        claim.fact.evidence_at = Millis(50);
        claim.fact.valid_from = Millis(1000);
        claim.fact.valid_until = valid_until;

        // Act
        let decision = assert_with(&state(vec![live]), &claim, &single());

        // Assert
        assert_eq!(
            decision.outcome,
            ArbitrationOutcome::Historical {
                id: FactId::from_raw("m1")
            }
        );
        assert_eq!(inserted(&decision.plan, "m1").valid_until, expected_until);
        assert!(decision.plan.close.is_empty());
    }
}

#[test]
fn older_evidence_that_starts_after_the_live_fact_is_rejected() {
    // Arrange
    let mut live = version("a", "Sydney");
    live.valid_from = Millis(2000);
    let mut claim = claim("Melbourne");
    claim.fact.evidence_at = Millis(50);
    claim.fact.valid_from = Millis(3000);

    // Act
    let result = decide_assert(&state(vec![live]), &claim, &single(), &mut mint());

    // Assert
    assert!(matches!(
        result,
        Err(Error::InvalidFactInterval {
            valid_from: 3000,
            valid_until: 2000
        })
    ));
}

#[test]
fn newer_evidence_with_an_early_valid_from_still_replaces() {
    // Arrange
    let mut live = version("a", "Sydney");
    live.valid_from = Millis(2000);
    let mut claim = claim("Melbourne");
    claim.fact.valid_from = Millis(1000);

    // Act
    let decision = assert_with(&state(vec![live]), &claim, &single());

    // Assert
    assert!(matches!(
        decision.outcome,
        ArbitrationOutcome::Replaced { .. }
    ));
    assert_eq!(inserted(&decision.plan, "m2").valid_until, Millis(2000));
}

#[test]
fn equal_evidence_goes_to_the_trust_rules() {
    // Arrange
    let mut claim = claim("Melbourne");
    claim.fact.evidence_at = Millis(100);

    // Act
    let decision = assert_with(&state(vec![version("a", "Sydney")]), &claim, &single());

    // Assert
    assert!(matches!(
        decision.outcome,
        ArbitrationOutcome::Replaced { .. }
    ));
}

#[test]
fn a_version_ending_in_the_future_is_still_arbitrated() {
    // Arrange
    let mut ending = version("a", "Sydney");
    ending.valid_from = Millis(1000);
    ending.valid_until = Millis(5000);
    let mut claim = claim("Melbourne");
    claim.fact.evidence_at = Millis(50);
    claim.fact.valid_from = Millis(500);

    // Act
    let decision = assert_with(&state(vec![ending]), &claim, &single());

    // Assert
    assert!(matches!(
        decision.outcome,
        ArbitrationOutcome::Historical { .. }
    ));
}

#[test]
fn older_evidence_is_compared_with_the_newest_evidence_among_closed_and_open_versions() {
    // Arrange
    let mut closed = version("a", "Sydney");
    closed.valid_until = Millis(500);
    closed.evidence_at = Millis(300);
    let mut open = version("b", "Melbourne");
    open.valid_from = Millis(500);
    open.evidence_at = Millis(300);
    let mut claim = claim("Perth");
    claim.fact.evidence_at = Millis(250);
    claim.fact.valid_from = Millis(100);

    // Act
    let decision = assert_with(&state(vec![closed, open]), &claim, &single());

    // Assert
    assert_eq!(
        decision.outcome,
        ArbitrationOutcome::Historical {
            id: FactId::from_raw("m1")
        }
    );
    assert_eq!(inserted(&decision.plan, "m1").valid_until, Millis(500));
}

#[test]
fn a_value_that_returns_after_another_replaced_it_becomes_live_again() {
    // Arrange
    let mut closed_sydney = version("a", "Sydney");
    closed_sydney.valid_until = Millis(150);
    closed_sydney.evidence_at = Millis(200);
    closed_sydney.closed_by = Some(FactId::from_raw("b"));
    let mut melbourne = version("b", "Melbourne");
    melbourne.valid_from = Millis(150);
    melbourne.evidence_at = Millis(200);
    let mut claim = claim("Sydney");
    claim.fact.evidence_at = Millis(300);
    claim.fact.valid_from = Millis(700);

    // Act
    let decision = assert_with(&state(vec![closed_sydney, melbourne]), &claim, &single());

    // Assert
    assert_eq!(
        decision.outcome,
        ArbitrationOutcome::Replaced {
            closed: ids(&["b"]),
            inserted: FactId::from_raw("m1")
        }
    );
    assert_eq!(
        inserted(&decision.plan, "m1").lineage_id,
        FactId::from_raw("m1")
    );
}

#[test]
fn restating_a_replaced_value_from_a_new_event_corroborates_the_closed_version() {
    for (claim_evidence, claim_from) in [(300, 0), (150, 700)] {
        // Arrange
        let mut closed = version("a", "Sydney");
        closed.valid_until = Millis(150);
        closed.evidence_at = Millis(200);
        let mut melbourne = version("b", "Melbourne");
        melbourne.valid_from = Millis(150);
        melbourne.evidence_at = Millis(200);
        let mut claim = claim("Sydney");
        claim.fact.evidence_at = Millis(claim_evidence);
        claim.fact.valid_from = Millis(claim_from);

        // Act
        let decision = assert_with(&state(vec![closed, melbourne]), &claim, &single());

        // Assert
        let label = format!("evidence {claim_evidence}, from {claim_from}");
        assert_eq!(decision.outcome, corroborated("a", true), "{label}");
        assert!(decision.plan.insert.is_empty(), "{label}");
        assert!(decision.plan.close.is_empty(), "{label}");
    }
}

#[test]
fn replaying_the_events_of_a_replaced_value_changes_nothing() {
    // Arrange
    let mut closed = version("a", "Sydney");
    closed.valid_until = Millis(500);
    let mut claim = claim("Sydney");
    claim.fact.source_event_ids = vec!["e-a".into()];

    // Act
    let decision = assert_with(&state(vec![closed]), &claim, &single());

    // Assert
    assert_eq!(decision.plan, Plan::default());
}

fn suppression(object: &str, trust: f64) -> Suppression {
    Suppression {
        subject: "alice".into(),
        predicate: "lives_in".into(),
        scope: String::new(),
        object_key: object_key(object),
        trust_score: trust,
        retracted_at: Millis(100),
    }
}

#[test]
fn a_suppression_blocks_old_evidence_unless_it_is_more_trusted() {
    for (evidence, trust, blocked) in [
        (100, 0.5, true),
        (90, 0.5, true),
        (101, 0.5, false),
        (90, 0.6, false),
    ] {
        // Arrange
        let state = KeyState {
            versions: vec![],
            suppressions: vec![suppression("Sydney", 0.5)],
        };
        let mut claim = claim("Sydney");
        claim.fact.evidence_at = Millis(evidence);
        claim.fact.trust_score = trust;

        // Act
        let decision = assert_with(&state, &claim, &single());

        // Assert
        let label = format!("evidence {evidence}, trust {trust}");
        assert_eq!(
            decision.outcome == ArbitrationOutcome::Suppressed,
            blocked,
            "{label}"
        );
        if blocked {
            assert_eq!(decision.plan, Plan::default(), "{label}");
        }
    }
}

#[test]
fn a_suppression_follows_the_predicates_match_mode() {
    // Arrange
    let state = KeyState {
        versions: vec![],
        suppressions: vec![suppression("Sydney", 0.5)],
    };
    let mut claim = claim("Sydney, Australia");
    claim.fact.evidence_at = Millis(100);

    // Act
    let contains = assert_with(
        &state,
        &claim,
        &rules(Cardinality::One, MatchMode::Contains),
    );
    let exact = assert_with(&state, &claim, &single());

    // Assert
    assert_eq!(contains.outcome, ArbitrationOutcome::Suppressed);
    assert!(matches!(exact.outcome, ArbitrationOutcome::Inserted { .. }));
}

#[test]
fn an_assertion_is_validated_before_any_rule_runs() {
    // Arrange
    let invalid = [
        (
            "empty interval",
            NewFact {
                valid_until: Millis(1000),
                ..new_fact("x")
            },
        ),
        (
            "inverted interval",
            NewFact {
                valid_until: Millis(5),
                ..new_fact("x")
            },
        ),
        (
            "nan trust",
            NewFact {
                trust_score: f64::NAN,
                ..new_fact("x")
            },
        ),
        (
            "negative trust",
            NewFact {
                trust_score: -0.1,
                ..new_fact("x")
            },
        ),
        (
            "trust above one",
            NewFact {
                trust_score: 1.5,
                ..new_fact("x")
            },
        ),
        (
            "infinite trust",
            NewFact {
                trust_score: f64::INFINITY,
                ..new_fact("x")
            },
        ),
        (
            "no source",
            NewFact {
                source_event_ids: vec![],
                ..new_fact("x")
            },
        ),
    ];

    for (label, fact) in invalid {
        // Act
        let result = Claim::new(fact, String::new());

        // Assert
        assert!(result.is_err(), "{label}");
    }
}

#[test]
fn an_assertion_with_only_an_end_may_start_at_the_beginning() {
    // Arrange
    let fact = NewFact {
        valid_from: BEGINNING,
        valid_until: Millis(100),
        ..new_fact("x")
    };

    // Act
    let result = Claim::new(fact, String::new());

    // Assert
    assert!(result.is_ok());
}

#[test]
fn the_error_names_what_was_wrong_with_the_assertion() {
    // Arrange
    let no_source = NewFact {
        source_event_ids: vec![],
        ..new_fact("x")
    };
    let bad_trust = NewFact {
        trust_score: 1.5,
        ..new_fact("x")
    };

    // Act
    let missing = Claim::new(no_source, String::new()).unwrap_err();
    let trust = Claim::new(bad_trust, String::new()).unwrap_err();

    // Assert
    assert!(matches!(missing, Error::FactWithoutSource));
    assert!(matches!(trust, Error::InvalidFactTrust(t) if t == 1.5));
}
