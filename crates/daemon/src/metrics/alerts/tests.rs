// SPDX-License-Identifier: Apache-2.0

use super::*;

const AT: Millis = Millis(10 * 3_600_000);
const HOUR: i64 = 3_600_000;
const METRIC: MetricId = MetricId::ProbeRecallAt10;

fn delta(pct: f64, min_samples: u32) -> RuleSet {
    RuleSet::from([(METRIC, vec![Rule::Delta { pct, min_samples }])])
}

fn observed(value: f64, samples: u32) -> Vec<Observation> {
    vec![Observation {
        metric: METRIC,
        value,
        samples,
    }]
}

fn previous(metric: MetricId, value: f64, at: i64) -> AlertState {
    let mut state = AlertState::default();
    state.values.insert(metric, Previous { value, at });
    state
}

#[test]
fn a_delta_rule_fires_at_its_percent_and_above_and_not_below() {
    // Arrange
    let state = previous(METRIC, 100.0, 0);
    let rules = delta(20.0, 1);

    for (now, fires) in [(119.9, false), (120.0, true), (120.1, true), (80.0, true)] {
        // Act
        let alerts = evaluate(&state, &observed(now, 1), &rules, AT);

        // Assert
        assert_eq!(!alerts.is_empty(), fires, "{now}");
    }
}

#[test]
fn a_previous_value_of_zero_raises_nothing_and_divides_by_nothing() {
    // Act
    let alerts = evaluate(
        &previous(METRIC, 0.0, 0),
        &observed(50.0, 9),
        &delta(20.0, 1),
        AT,
    );

    // Assert
    assert!(alerts.is_empty());
}

#[test]
fn too_few_samples_raise_no_delta_alert() {
    // Act
    let few = evaluate(
        &previous(METRIC, 100.0, 0),
        &observed(200.0, 4),
        &delta(20.0, 5),
        AT,
    );
    let enough = evaluate(
        &previous(METRIC, 100.0, 0),
        &observed(200.0, 5),
        &delta(20.0, 5),
        AT,
    );

    // Assert
    assert!(few.is_empty());
    assert_eq!(enough.len(), 1);
}

#[test]
fn an_absolute_rule_fires_at_its_ceiling_whatever_the_previous_value() {
    // Arrange
    let metric = MetricId::DeletionBacklogAgeDays;
    let rules = RuleSet::from([(metric, vec![Rule::Absolute { ceiling: 30.0 }])]);
    let at = |value| {
        vec![Observation {
            metric,
            value,
            samples: 1,
        }]
    };

    // Act
    let below = evaluate(&AlertState::default(), &at(29.9), &rules, AT);
    let on_first_run = evaluate(&AlertState::default(), &at(30.0), &rules, AT);
    let with_history = evaluate(&previous(metric, 40.0, 0), &at(40.0), &rules, AT);

    // Assert
    assert!(below.is_empty());
    assert_eq!(on_first_run.len(), 1);
    assert_eq!(with_history.len(), 1);
}

#[test]
fn a_rate_rule_fires_at_its_rate_over_the_clock_and_not_below_it() {
    // Arrange
    let metric = MetricId::ModelCalls;
    let rules = RuleSet::from([(
        metric,
        vec![Rule::Rate {
            per_hour: 10.0,
            min_samples: 1,
        }],
    )]);
    let start = AT.0 - 2 * HOUR;
    let state = previous(metric, 100.0, start);
    let at = |value| {
        vec![Observation {
            metric,
            value,
            samples: 1,
        }]
    };

    // Act: 20 more calls in two hours is exactly the rate
    let on_the_edge = evaluate(&state, &at(120.0), &rules, AT);
    let under = evaluate(&state, &at(119.0), &rules, AT);
    let no_time_passed = evaluate(&state, &at(500.0), &rules, Millis(start));

    // Assert
    assert_eq!(on_the_edge.len(), 1);
    assert!(under.is_empty());
    assert!(no_time_passed.is_empty());
}

#[test]
fn a_value_that_is_not_finite_never_alerts() {
    // Arrange
    let rules = RuleSet::from([(METRIC, vec![Rule::Absolute { ceiling: 0.0 }])]);

    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        // Act
        let alerts = evaluate(&AlertState::default(), &observed(value, 9), &rules, AT);

        // Assert
        assert!(alerts.is_empty(), "{value}");
    }
}

#[test]
fn a_first_run_a_missing_file_and_a_corrupt_file_give_a_fresh_state_and_no_alert() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing.json");
    let corrupt = dir.path().join("corrupt.json");
    std::fs::write(&corrupt, "{ not json").unwrap();
    let rules = delta(20.0, 1);

    // Act
    let from_missing = AlertState::load(&missing);
    let from_corrupt = AlertState::load(&corrupt);
    let alerts = evaluate(&from_corrupt.state, &observed(500.0, 9), &rules, AT);

    // Assert
    assert_eq!((from_missing.warnings, from_corrupt.warnings), (0, 1));
    assert!(from_missing.state.values.is_empty() && from_corrupt.state.values.is_empty());
    assert!(alerts.is_empty());
}

#[test]
fn a_run_saves_what_the_next_compares_with_and_a_restore_failure_saves_nothing() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let path = state_path_for(&dir.path().join("liam.db"));
    let rules = delta(20.0, 1);

    // Act
    let first = evaluate_run(&path, &observed(100.0, 1), &rules, AT, false).unwrap();
    let saved = std::fs::read(&path).unwrap();
    let failed = evaluate_run(&path, &observed(1.0, 1), &rules, AT, true).unwrap();
    let second = evaluate_run(&path, &observed(150.0, 1), &rules, AT, false).unwrap();

    // Assert
    assert!(first.is_empty());
    assert!(failed.is_empty());
    assert_eq!(second.len(), 1, "compared with 100, not the failed run's 1");
    assert_ne!(std::fs::read(&path).unwrap(), saved);
}
