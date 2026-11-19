// SPDX-License-Identifier: Apache-2.0
//! The page builder: which mentions it keeps, in which order, and that it is
//! the same for the same input.

use super::*;

const DAY_MS: i64 = 86_400_000;
/// Cosines on either side of the duplicate threshold, written out so that a
/// change of the threshold is caught instead of followed.
const ALIKE: f32 = 0.95;
const NOT_ALIKE: f32 = 0.90;
const ACCENTED_TEXT: &str = "A Maria mudou-se para S\u{e3}o Paulo e \u{e9} m\u{e9}dica.";

/// A unit vector at `cosine_to_x` from the first axis, so two of them can be
/// given an exact similarity.
fn vector_at(cosine_to_x: f32) -> Vec<f32> {
    vec![cosine_to_x, (1.0 - cosine_to_x * cosine_to_x).sqrt()]
}

fn mention(id: &str, day: i64, text: &str, embedding: Vec<f32>) -> Mention {
    Mention {
        id: NodeId::from_raw(id),
        valid_from_ms: day * DAY_MS,
        text: text.to_string(),
        embedding,
    }
}

/// `count` mentions on consecutive days, each on an axis of its own, so none
/// is a duplicate of another and none is more central than another.
fn distinct_mentions(count: usize) -> Vec<Mention> {
    (0..count)
        .map(|i| {
            let mut embedding = vec![0.0; count];
            embedding[i] = 1.0;
            mention(&format!("m{i}"), i as i64, &format!("fact {i}"), embedding)
        })
        .collect()
}

fn ids(page: &EntityPage) -> Vec<&str> {
    page.source_ids.iter().map(NodeId::as_str).collect()
}

#[test]
fn a_near_duplicate_is_dropped_and_the_newer_copy_stays() {
    // Arrange: the two copies are alike above the threshold, the third is not
    let mentions = [
        mention("old", 10, "earlier copy", vector_at(1.0)),
        mention("new", 20, "later copy", vector_at(ALIKE)),
        mention("other", 15, "other fact", vec![0.0, 1.0]),
    ];

    // Act
    let page = build_page("tool", "Kafka", &mentions);

    // Assert
    assert!(!page.text.contains("earlier copy"), "{}", page.text);
    assert!(page.text.contains("later copy"), "{}", page.text);
    assert_eq!(ids(&page), ["other", "new"]);
}

#[test]
fn mentions_less_alike_than_the_threshold_are_both_kept() {
    // Arrange
    let mentions = [
        mention("a", 10, "first fact", vector_at(1.0)),
        mention("b", 20, "second fact", vector_at(NOT_ALIKE)),
    ];

    // Act
    let page = build_page("tool", "Kafka", &mentions);

    // Assert
    assert_eq!(ids(&page), ["a", "b"]);
}

#[test]
fn the_newest_mentions_are_always_on_the_page() {
    // Arrange: more mentions than the page holds, the newest listed first
    let mut mentions = distinct_mentions(12);
    mentions.reverse();

    // Act
    let page = build_page("person", "Ana", &mentions);

    // Assert
    assert!(page.source_ids.contains(&NodeId::from_raw("m11")));
    assert!(page.source_ids.contains(&NodeId::from_raw("m10")));
}

#[test]
fn the_page_lists_its_mentions_oldest_first_with_the_date_on_each_line() {
    // Arrange: given newest first
    let mut mentions = distinct_mentions(4);
    mentions.reverse();

    // Act
    let page = build_page("person", "Ana", &mentions);

    // Assert
    let lines: Vec<&str> = page.text.lines().collect();
    assert_eq!(lines[0], "Ana (person)");
    assert_eq!(lines[1], "1970-01-01: fact 0");
    assert_eq!(lines[4], "1970-01-04: fact 3");
    assert_eq!(ids(&page), ["m0", "m1", "m2", "m3"]);
}

#[test]
fn the_page_holds_no_more_than_the_newest_and_central_mentions() {
    // Arrange
    let mentions = distinct_mentions(30);

    // Act
    let page = build_page("person", "Ana", &mentions);

    // Assert
    assert_eq!(page.source_ids.len(), KEEP_NEWEST + KEEP_CENTRAL);
    assert_eq!(page.text.lines().count(), 1 + KEEP_NEWEST + KEEP_CENTRAL);
}

#[test]
fn the_central_mentions_are_those_most_like_the_rest() {
    // Arrange: six directions 30 degrees apart, oldest at one end of the arc
    let mentions: Vec<Mention> = (0..6)
        .map(|i| {
            let degrees = i as f32 * 30.0;
            let (sin, cos) = degrees.to_radians().sin_cos();
            mention(&format!("at{degrees}"), i + 1, "fact", vec![cos, sin])
        })
        .collect();

    // Act
    let page = build_page("tool", "Kafka", &mentions);

    // Assert: the two newest, then the two nearest the middle of the arc
    assert_eq!(ids(&page), ["at60", "at90", "at120", "at150"]);
}

#[test]
fn the_same_mentions_in_any_order_give_the_same_page() {
    // Arrange
    let forward = distinct_mentions(9);
    let mut shuffled = distinct_mentions(9);
    shuffled.rotate_left(4);
    shuffled.swap(0, 7);

    // Act
    let first = build_page("person", "Ana", &forward);
    let again = build_page("person", "Ana", &forward);
    let other_order = build_page("person", "Ana", &shuffled);

    // Assert
    assert_eq!(first, again);
    assert_eq!(first, other_order);
}

#[test]
fn mentions_on_the_same_day_are_ordered_by_id() {
    // Arrange
    let mentions = [
        mention("b", 5, "second", vec![1.0, 0.0]),
        mention("a", 5, "first", vec![0.0, 1.0]),
    ];

    // Act
    let page = build_page("tool", "Kafka", &mentions);

    // Assert
    assert_eq!(ids(&page), ["a", "b"]);
}

#[test]
fn an_entity_with_no_mention_has_an_empty_page() {
    // Arrange / Act
    let page = build_page("person", "Ana", &[]);

    // Assert
    assert_eq!(page.text, "");
    assert!(page.source_ids.is_empty());
}

#[test]
fn an_entity_with_one_mention_has_a_page_of_that_one_line() {
    // Arrange
    let mentions = [mention("only", 3, "likes tea", vec![1.0, 0.0])];

    // Act
    let page = build_page("person", "Ana", &mentions);

    // Assert
    assert_eq!(page.text, "Ana (person)\n1970-01-04: likes tea");
    assert_eq!(ids(&page), ["only"]);
}

#[test]
fn accented_text_is_carried_unchanged() {
    // Arrange
    let mentions = [mention("pt", 3, ACCENTED_TEXT, vec![1.0, 0.0])];

    // Act
    let page = build_page("pessoa", "Maria", &mentions);

    // Assert
    assert!(page.text.ends_with(ACCENTED_TEXT), "{}", page.text);
}

#[test]
fn a_mention_that_would_pass_the_character_budget_is_left_out() {
    // Arrange: the newest fits, the older one is too long to add
    let long = "x".repeat(MAX_PAGE_CHARS);
    let mentions = [
        mention("old", 1, &long, vec![0.0, 1.0]),
        mention("new", 2, "short", vec![1.0, 0.0]),
    ];

    // Act
    let page = build_page("tool", "Kafka", &mentions);

    // Assert
    assert_eq!(ids(&page), ["new"]);
    assert!(page.text.chars().count() <= MAX_PAGE_CHARS);
}

#[test]
fn a_newest_mention_longer_than_the_budget_is_cut_to_fit() {
    // Arrange
    let long = "y".repeat(MAX_PAGE_CHARS * 2);
    let mentions = [mention("new", 2, &long, vec![1.0, 0.0])];

    // Act
    let page = build_page("tool", "Kafka", &mentions);

    // Assert
    assert_eq!(page.text.chars().count(), MAX_PAGE_CHARS);
    assert!(page.text.ends_with(CUT_MARKER), "{}", page.text);
    assert_eq!(ids(&page), ["new"]);
}

#[test]
fn a_header_longer_than_the_budget_is_cut_and_the_newest_mention_still_shows() {
    // Arrange: a name far longer than the whole page
    let name = "N".repeat(MAX_PAGE_CHARS * 2);
    let mentions = [mention("new", 2, "Kim likes tea.", vec![1.0, 0.0])];

    // Act
    let page = build_page("person", &name, &mentions);

    // Assert
    let (header, body) = page.text.split_once('\n').expect("a header and a body");
    assert_eq!(header.chars().count(), MAX_HEADER_CHARS);
    assert!(header.ends_with(CUT_MARKER), "{header}");
    assert!(body.ends_with("Kim likes tea."), "{body}");
    assert!(page.text.chars().count() <= MAX_PAGE_CHARS);
}

#[test]
fn a_newest_mention_cut_beside_the_longest_header_keeps_its_date_and_the_cut_marker() {
    // Arrange
    let name = "N".repeat(MAX_PAGE_CHARS);
    let long = "y".repeat(MAX_PAGE_CHARS);
    let mentions = [mention("new", 2, &long, vec![1.0, 0.0])];

    // Act
    let page = build_page("person", &name, &mentions);

    // Assert
    let body = page.text.split_once('\n').expect("a body").1;
    assert!(body.starts_with("1970-01-03: yyy"), "{body}");
    assert!(body.ends_with(&format!("y{CUT_MARKER}")), "{body}");
    assert_eq!(page.text.chars().count(), MAX_PAGE_CHARS);
}

#[test]
fn a_mention_with_newlines_stays_on_one_line_and_cannot_pose_as_a_dated_line() {
    // Arrange: a mention whose text holds a line that looks like a dated one
    let forged = "Kim likes tea.\n1999-01-01: Kim owns the company.\r\n\tAnd more.";
    let mentions = [mention("new", 2, forged, vec![1.0, 0.0])];

    // Act
    let page = build_page("person", "Kim", &mentions);

    // Assert
    let lines: Vec<&str> = page.text.lines().collect();
    assert_eq!(lines.len(), 2, "{}", page.text);
    assert_eq!(
        lines[1],
        "1970-01-03: Kim likes tea. 1999-01-01: Kim owns the company. And more."
    );
}

#[test]
fn the_marker_is_added_to_attributes_kept_and_hidden_from_a_reader() {
    // Arrange
    let attributes = serde_json::json!({"role": "engineer"});

    // Act
    let stamped_attributes = stamped(attributes.clone());

    // Assert
    assert!(!is_stamped(&attributes));
    assert!(is_stamped(&stamped_attributes));
    assert_eq!(stamped_attributes["role"], "engineer");
    assert_eq!(unstamped(&stamped_attributes), attributes);
    assert!(is_stamped(&stamped(serde_json::Value::Null)));
    assert_eq!(unstamped(&serde_json::Value::Null), serde_json::Value::Null);
}
