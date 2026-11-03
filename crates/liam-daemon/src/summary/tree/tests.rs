// SPDX-License-Identifier: Apache-2.0
//! The tree builder: which items link and group, which summaries a level
//! keeps, regenerates or dissolves, and when it is clustered again.

use super::*;

const DIMS: usize = 4;
const SCOPE: &str = "work/liam";
const LEVEL: u8 = 1;
const NOW: Millis = Millis(1_000_000);
const INTERVAL: Millis = Millis(10_000);
const WINDOW: Millis = Millis(1000);

fn params() -> TreeParams {
    TreeParams {
        knn: 8,
        min_similarity: 0.5,
        time_window: WINDOW,
        min_cluster_size: 3,
        max_cluster_size: 12,
        max_depth: 3,
        recluster_change_ratio: 0.1,
        recluster_interval: INTERVAL,
        min_attach_similarity: 0.5,
        max_unattached: 5,
    }
}

fn axis(index: usize) -> Vec<f32> {
    let mut vector = vec![0.0; DIMS];
    vector[index] = 1.0;
    vector
}

fn point(id: &str, vector: Vec<f32>, time: i64) -> Point {
    Point {
        id: id.to_string(),
        vector,
        time: Millis(time),
    }
}

/// `count` items on one axis, named `g<axis>-<n>` and a millisecond apart.
fn group(axis_index: usize, count: usize) -> Vec<Point> {
    (0..count)
        .map(|n| point(&format!("g{axis_index}-{n:02}"), axis(axis_index), n as i64))
        .collect()
}

fn groups(counts: &[usize]) -> Vec<Point> {
    counts
        .iter()
        .enumerate()
        .flat_map(|(axis_index, count)| group(axis_index, *count))
        .collect()
}

fn request<'a>(
    points: &'a [Point],
    previous: &'a [PreviousCluster],
    state: Option<&'a TreeState>,
    now: Millis,
) -> LevelRequest<'a> {
    LevelRequest {
        scope: SCOPE,
        level: LEVEL,
        points,
        previous,
        state,
        now,
    }
}

fn plan(points: &[Point]) -> ChangePlan {
    plan_level(&request(points, &[], None, NOW), &params(), &detect_cold).unwrap()
}

fn clusters_of(plan: &ChangePlan) -> Vec<&PlannedCluster> {
    let mut all: Vec<&PlannedCluster> = plan.keep.iter().collect();
    all.extend(plan.regenerate.iter().map(|r| &r.cluster));
    all
}

fn as_sets(plan: &ChangePlan) -> BTreeSet<Vec<String>> {
    clusters_of(plan)
        .into_iter()
        .map(|cluster| cluster.members.clone())
        .collect()
}

fn ids(points: &[Point]) -> Vec<String> {
    let mut ids: Vec<String> = points.iter().map(|p| p.id.clone()).collect();
    ids.sort_unstable();
    ids
}

/// The live summaries a plan would leave behind, named `s0`, `s1`, ...
fn stored(plan: &ChangePlan) -> Vec<PreviousCluster> {
    clusters_of(plan)
        .into_iter()
        .enumerate()
        .map(|(n, cluster)| PreviousCluster {
            summary_id: format!("s{n}"),
            child_hash: cluster.child_hash.clone(),
            children: cluster.members.clone(),
        })
        .collect()
}

fn replan(
    points: &[Point],
    previous: &[PreviousCluster],
    state: &TreeState,
    now: Millis,
) -> ChangePlan {
    plan_level(
        &request(points, previous, Some(state), now),
        &params(),
        &detect_cold,
    )
    .unwrap()
}

#[test]
fn three_separated_groups_form_three_clusters_every_run() {
    // Arrange
    let points = groups(&[4, 4, 4]);

    // Act
    let first = plan(&points);
    let second = plan(&points);

    // Assert
    let expected: BTreeSet<Vec<String>> = [0, 1, 2]
        .map(|axis_index| ids(&group(axis_index, 4)))
        .into();
    assert_eq!(as_sets(&first), expected);
    assert_eq!(first, second);
    assert!(first.full_build);
    assert_eq!(first.regenerate.len(), 3);
    assert!(first.keep.is_empty() && first.dissolved.is_empty());
}

#[test]
fn a_cluster_hash_is_the_store_hash_of_its_members() {
    // Arrange
    let points = group(0, 3);

    // Act
    let planned = plan(&points);

    // Assert
    assert_eq!(
        planned.regenerate[0].cluster.child_hash,
        child_hash(LEVEL, SCOPE, &ids(&points))
    );
}

#[test]
fn a_link_needs_the_time_window_and_the_similarity_and_only_the_nearest_count() {
    // Arrange
    let a = point("a", axis(0), 0);
    let at_window = point("b", axis(0), WINDOW.0);
    let past_window = point("b", axis(0), WINDOW.0 + 1);
    let slanted = point("b", vec![1.0, 1.0, 0.0, 0.0], 0);
    let threshold = cosine(&a.vector, &slanted.vector);
    let above_threshold = f32::from_bits(threshold.to_bits() + 1);
    let cases = [
        (
            "exactly at the window",
            [a.clone(), at_window],
            1,
            0.5,
            vec![(0, 1)],
        ),
        ("past the window", [a.clone(), past_window], 1, 0.5, vec![]),
        (
            "exactly at the similarity",
            [a.clone(), slanted.clone()],
            1,
            threshold,
            vec![(0, 1)],
        ),
        (
            "above the similarity",
            [a, slanted],
            1,
            above_threshold,
            vec![],
        ),
    ];

    for (name, points, knn, min_similarity, expected) in cases {
        // Act
        let edges = build_graph(&points, knn, min_similarity, WINDOW);

        // Assert
        assert_eq!(edges, expected, "{name}");
    }
}

#[test]
fn with_one_neighbour_only_the_nearest_is_considered() {
    // Arrange
    let points = [
        point("a", axis(0), 0),
        point("b", axis(0), 0),
        point("c", vec![1.0, 1.0, 0.0, 0.0], 0),
    ];

    // Act
    let one = build_graph(&points, 1, 0.5, WINDOW);
    let two = build_graph(&points, 2, 0.5, WINDOW);

    // Assert
    assert_eq!(one, [(0, 1), (0, 2)]);
    assert_eq!(two, [(0, 1), (0, 2), (1, 2)]);
}

#[test]
fn equal_similarities_are_broken_by_id_and_the_edges_come_back_in_pair_order() {
    // Arrange
    let points: Vec<Point> = (1..=4)
        .map(|n| point(&format!("p{n}"), axis(0), 0))
        .collect();

    // Act
    let edges = build_graph(&points, 2, 0.5, WINDOW);

    // Assert
    assert_eq!(edges, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]);
}

#[test]
fn too_few_items_make_no_cluster_and_no_panic() {
    for count in [0, 1, 2] {
        // Arrange
        let points = group(0, count);

        // Act
        let planned = plan(&points);

        // Assert
        assert_eq!(planned.cluster_count(), 0, "{count} items");
        assert!(planned.dissolved.is_empty());
        assert_eq!(planned.next_state.baseline_item_count, count as u32);
        assert_eq!(planned.next_state.unattached_count, count as u32);
    }
}

#[test]
fn a_cluster_over_the_maximum_splits_by_time_into_even_parts() {
    // Arrange
    let points = group(0, 10);
    let small = TreeParams {
        max_cluster_size: 4,
        ..params()
    };

    // Act
    let planned = plan_level(&request(&points, &[], None, NOW), &small, &detect_cold).unwrap();

    // Assert
    let sizes: Vec<usize> = clusters_of(&planned)
        .iter()
        .map(|c| c.members.len())
        .collect();
    assert_eq!(sizes, [4, 3, 3]);
    let first: Vec<&str> = clusters_of(&planned)[0]
        .members
        .iter()
        .map(String::as_str)
        .collect();
    assert_eq!(first, ["g0-00", "g0-01", "g0-02", "g0-03"]);
}

#[test]
fn a_clustering_that_only_returns_singletons_for_linked_items_is_a_failed_build() {
    // Arrange
    let points = group(0, 4);
    let singletons = |count: usize, _: &[Edge]| (0..count).collect::<Vec<_>>();

    // Act
    let outcome = plan_level(&request(&points, &[], None, NOW), &params(), &singletons);

    // Assert
    assert_eq!(outcome.unwrap_err(), ClusteringFailed { level: LEVEL });
}

#[test]
fn unlinked_items_are_not_a_failed_build() {
    // Arrange
    let points = vec![
        point("a", axis(0), 0),
        point("b", axis(1), 0),
        point("c", axis(2), 0),
    ];
    let singletons = |count: usize, _: &[Edge]| (0..count).collect::<Vec<_>>();

    // Act
    let outcome = plan_level(&request(&points, &[], None, NOW), &params(), &singletons);

    // Assert
    assert_eq!(outcome.unwrap().cluster_count(), 0);
}

#[test]
fn a_second_full_build_over_the_same_items_keeps_every_summary() {
    // Arrange
    let points = groups(&[4, 4, 4]);
    let first = plan(&points);
    let previous = stored(&first);

    // Act
    let second = plan_level(
        &request(&points, &previous, None, NOW),
        &params(),
        &detect_cold,
    )
    .unwrap();

    // Assert
    assert_eq!(second.keep.len(), 3);
    assert!(second.regenerate.is_empty() && second.dissolved.is_empty());
}

#[test]
fn a_full_build_pairs_each_new_cluster_with_the_live_summary_it_overlaps_most() {
    // Arrange
    let points = group(0, 8);
    let halves = |name: &str, members: &[Point]| PreviousCluster {
        summary_id: name.to_string(),
        child_hash: child_hash(LEVEL, SCOPE, &ids(members)),
        children: ids(members),
    };
    let previous = [
        halves("s-low", &points[..4]),
        halves("s-high", &points[4..]),
    ];

    // Act
    let planned = plan_level(
        &request(&points, &previous, None, NOW),
        &params(),
        &detect_cold,
    )
    .unwrap();

    // Assert: the merge takes one old row, and the other is left to dissolve.
    assert_eq!(planned.regenerate.len(), 1);
    assert_eq!(planned.regenerate[0].replaces.as_deref(), Some("s-high"));
    assert_eq!(planned.dissolved, ["s-low"]);
}

fn settled() -> (Vec<Point>, Vec<PreviousCluster>, TreeState) {
    let points = groups(&[10, 10, 10]);
    let first = plan(&points);
    (points.clone(), stored(&first), first.next_state)
}

#[test]
fn one_new_item_joins_the_nearest_cluster_and_only_that_cluster_regenerates() {
    // Arrange
    let (mut points, previous, state) = settled();
    points.push(point("new", axis(1), 20));

    // Act
    let planned = replan(&points, &previous, &state, NOW);

    // Assert
    assert!(!planned.full_build);
    assert_eq!(planned.keep.len(), 2);
    assert_eq!(planned.regenerate.len(), 1);
    let joined = &planned.regenerate[0];
    assert!(joined.cluster.members.contains(&"new".to_string()));
    assert!(joined
        .cluster
        .members
        .iter()
        .all(|id| id == "new" || id.starts_with("g1-")));
    let old = previous
        .iter()
        .find(|old| old.children.iter().all(|id| id.starts_with("g1-")))
        .unwrap();
    assert_eq!(joined.replaces.as_deref(), Some(old.summary_id.as_str()));
    assert_eq!(planned.next_state, state);
}

#[test]
fn an_item_unlike_every_cluster_waits_and_leaves_the_clusters_alone() {
    // Arrange
    let (mut points, previous, state) = settled();
    points.push(point("odd", axis(3), 20));

    // Act
    let planned = replan(&points, &previous, &state, NOW);

    // Assert
    assert!(!planned.full_build);
    assert_eq!(planned.keep.len(), 3);
    assert!(planned.regenerate.is_empty());
}

#[test]
fn more_change_than_the_ratio_allows_clusters_again_and_resets_the_baseline() {
    // Arrange
    let (mut points, previous, state) = settled();
    points.extend((0..4).map(|n| point(&format!("new{n}"), axis(0), 30 + n)));
    let later = Millis(NOW.0 + 1);

    // Act
    let planned = replan(&points, &previous, &state, later);

    // Assert
    assert!(planned.full_build);
    assert_eq!(planned.next_state.last_full_build_at, later);
    assert_eq!(planned.next_state.baseline_item_count, 34);
}

#[test]
fn many_small_changes_leave_the_baseline_until_they_add_up_to_a_full_build() {
    // Arrange
    let (mut points, mut previous, state) = settled();
    let mut full_builds = Vec::new();

    // Act
    for n in 0..5 {
        points.push(point(&format!("new{n}"), axis(n as usize % 3), 30 + n));
        let planned = replan(&points, &previous, &state, NOW);
        full_builds.push((planned.full_build, planned.next_state.clone()));
        previous = stored(&planned);
    }

    // Assert: three of thirty is the ratio; the fourth item crosses it.
    let flags: Vec<bool> = full_builds.iter().map(|(full, _)| *full).collect();
    assert_eq!(flags, [false, false, false, true, true]);
    assert!(full_builds[..3].iter().all(|(_, next)| *next == state));
    assert_eq!(full_builds[3].1.baseline_item_count, 34);
}

#[test]
fn a_level_is_clustered_again_once_the_interval_has_passed() {
    // Arrange
    let (points, previous, state) = settled();
    let just_before = Millis(NOW.0 + INTERVAL.0 - 1);
    let at_interval = Millis(NOW.0 + INTERVAL.0);

    // Act
    let before = replan(&points, &previous, &state, just_before);
    let at = replan(&points, &previous, &state, at_interval);

    // Assert
    assert!(!before.full_build);
    assert!(at.full_build);
    assert_eq!(at.next_state.last_full_build_at, at_interval);
}

#[test]
fn too_many_waiting_items_cluster_the_level_again() {
    // Arrange
    let (mut points, previous, state) = settled();
    let few_changes = TreeParams {
        recluster_change_ratio: 1.0,
        max_unattached: 2,
        ..params()
    };
    points.extend((0..3).map(|n| point(&format!("odd{n}"), axis(3), 30 + n)));

    // Act
    let planned = plan_level(
        &request(&points, &previous, Some(&state), NOW),
        &few_changes,
        &detect_cold,
    )
    .unwrap();

    // Assert
    assert!(planned.full_build);
    assert_eq!(planned.cluster_count(), 4);
}

#[test]
fn an_attach_that_overfills_a_cluster_clusters_the_level_again() {
    // Arrange
    let (mut points, previous, state) = settled();
    let tight = TreeParams {
        max_cluster_size: 10,
        recluster_change_ratio: 1.0,
        ..params()
    };
    points.push(point("new", axis(0), 20));

    // Act
    let planned = plan_level(
        &request(&points, &previous, Some(&state), NOW),
        &tight,
        &detect_cold,
    )
    .unwrap();

    // Assert
    assert!(planned.full_build);
}

#[test]
fn a_cluster_that_loses_its_members_dissolves_and_one_that_loses_some_regenerates() {
    // Arrange
    let (points, previous, state) = settled();
    let loose = TreeParams {
        recluster_change_ratio: 1.0,
        ..params()
    };
    let remaining: Vec<Point> = points
        .iter()
        .filter(|p| !p.id.starts_with("g0-") || p.id.ends_with(['0', '1']))
        .filter(|p| {
            !p.id.starts_with("g1-") || p.id.ends_with(['0', '1', '2', '3', '4', '5', '6', '7'])
        })
        .cloned()
        .collect();

    // Act
    let planned = plan_level(
        &request(&remaining, &previous, Some(&state), NOW),
        &loose,
        &detect_cold,
    )
    .unwrap();

    // Assert
    assert!(!planned.full_build);
    assert_eq!(planned.dissolved.len(), 1);
    assert_eq!(planned.regenerate.len(), 1);
    assert_eq!(planned.keep.len(), 1);
    assert_eq!(planned.regenerate[0].cluster.members.len(), 8);
}

#[test]
fn recursion_stops_at_the_depth_a_small_level_a_level_that_does_not_shrink_or_a_partial_one() {
    // Arrange
    let three_clusters = plan(&groups(&[4, 4, 4]));
    let two_clusters = plan(&groups(&[4, 4]));
    let cases = [
        ("a full level 1", 1, 12, &three_clusters, true, true),
        ("level 3 is the last", 3, 12, &three_clusters, true, false),
        ("below the minimum", 1, 12, &two_clusters, true, false),
        ("does not shrink", 1, 3, &three_clusters, true, false),
        ("only partly built", 1, 12, &three_clusters, false, false),
    ];

    for (name, level, inputs, plan, fully_built, expected) in cases {
        // Act
        let descend = should_descend(level, inputs, plan, fully_built, &params());

        // Assert
        assert_eq!(descend, expected, "{name}");
    }
}

#[test]
fn a_configured_depth_beyond_the_maximum_is_capped() {
    // Arrange
    let planned = plan(&groups(&[4, 4, 4]));
    let deep = TreeParams {
        max_depth: MAX_DEPTH + 1,
        ..params()
    };

    // Act
    let descend = should_descend(MAX_DEPTH, 12, &planned, true, &deep);

    // Assert
    assert!(!descend);
}

#[test]
fn a_point_from_an_input_takes_its_id_vector_and_valid_time() {
    // Arrange
    let input = SummaryInput {
        id: liam_store::NodeId::from_raw("node-1"),
        kind: "note".to_string(),
        label: "label".to_string(),
        content: liam_store::RecordedText::for_test("text"),
        valid_from: Millis(42),
        scope: None,
        origin_events: Vec::new(),
        sources: Vec::new(),
        vector: axis(2),
    };

    // Act
    let converted = Point::from(&input);

    // Assert
    assert_eq!(converted, point("node-1", axis(2), 42));
}
