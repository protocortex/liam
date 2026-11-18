// SPDX-License-Identifier: Apache-2.0
//! Pure formatting for the `clusters` tool. Sync + dependency-light, like
//! `ask.rs`, so the budget fit and the format are unit-testable without a
//! store or model.

use liam_store::ClusterMember;

/// Resolve a caller's `k`/`members` argument: absent means "no narrowing",
/// `0` is clamped to `1` (an explicit "show nothing" request is not
/// meaningful for a listing tool). Never widens past what the store
/// returned; that clamp is `narrow_groups`' `.take()` below, implicitly.
fn clamp_narrow(n: Option<usize>) -> Option<usize> {
    n.map(|n| n.max(1))
}

/// Apply the caller's `k` (cluster count) and `members` (per-cluster member
/// count) requests. Both only narrow: the token budget in `render_clusters`
/// is the one thing allowed to shrink the result further, and neither
/// argument can grow it past what `groups` already holds.
pub fn narrow_groups(
    groups: &[Vec<ClusterMember>],
    k: Option<usize>,
    members: Option<usize>,
) -> Vec<Vec<ClusterMember>> {
    let k = clamp_narrow(k).unwrap_or(groups.len());
    let members = clamp_narrow(members);
    groups
        .iter()
        .take(k)
        .map(|g| match members {
            Some(m) => g.iter().take(m).cloned().collect(),
            None => g.clone(),
        })
        .collect()
}

/// One member line, byte-identical to `recall`'s per-hit prefix minus the
/// content line, so a handle copied out of `clusters` feeds straight into
/// `relate`.
fn render_member(m: &ClusterMember) -> String {
    format!("[{} {}] {}", m.kind, m.id.handle(), m.label)
}

fn render_group(g: &[ClusterMember]) -> String {
    g.iter().map(render_member).collect::<Vec<_>>().join("\n")
}

/// States the totals and, when some clusters were withheld, how many and why
/// to ask for more: the client's only signal to raise `k`.
fn render_header(shown: usize, total: usize) -> String {
    if shown == total {
        format!("{total} cluster(s)")
    } else {
        format!(
            "{shown} of {total} clusters shown, {withheld} withheld; raise k to see more",
            withheld = total - shown
        )
    }
}

/// Render `groups` (already narrowed by the caller's `k`/`members`, see
/// `narrow_groups`) largest-first, keeping as many as fit `budget`, but
/// never fewer than one when any exist. Mirrors `fit_evidence_to_budget`'s
/// floor for `ask`: a header with zero clusters reads identically to a
/// genuinely empty store, so a misconfigured or tiny budget still shows one.
///
/// `count` is injected so this stays testable without a model; see
/// `fit_evidence_to_budget` for the same pattern.
pub fn render_clusters(
    groups: &[Vec<ClusterMember>],
    budget: usize,
    count: impl Fn(&str) -> usize,
) -> String {
    if groups.is_empty() {
        return "no clusters yet".to_string();
    }
    let total = groups.len();
    let mut kept = total;
    loop {
        let body = groups[..kept]
            .iter()
            .map(|g| render_group(g))
            .collect::<Vec<_>>()
            .join("\n\n");
        let header = render_header(kept, total);
        if count(&header) + count(&body) <= budget || kept <= 1 {
            return format!("{header}\n\n{body}");
        }
        kept -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use liam_store::NodeId;

    fn member(id: &str, kind: &str, label: &str) -> ClusterMember {
        ClusterMember {
            id: NodeId::from_raw(id.to_string()),
            kind: kind.to_string(),
            label: label.to_string(),
        }
    }

    fn chars(s: &str) -> usize {
        s.chars().count()
    }

    #[test]
    fn render_clusters_format_is_pinned() {
        // Arrange: one cluster, two members, budget with plenty of room.
        let groups = vec![vec![
            member("01ABCDEFGHJKMNPQRST0000A", "fact", "A"),
            member("01ABCDEFGHJKMNPQRST0000B", "fact", "B"),
        ]];

        // Act
        let out = render_clusters(&groups, 10_000, chars);

        // Assert: exact shape, so a future change to it is a deliberate edit.
        assert_eq!(
            out,
            "1 cluster(s)\n\n[fact 01ABCDEFGHJKM] A\n[fact 01ABCDEFGHJKM] B"
        );
    }

    #[test]
    fn a_genuinely_empty_store_says_so_distinctly_from_a_too_small_budget() {
        // Arrange / Act
        let out = render_clusters(&[], 10_000, chars);

        // Assert: distinct from the too-small-budget floor message below, so
        // a client can tell "nothing to group" from "budget too tight".
        assert_eq!(out, "no clusters yet");
    }

    #[test]
    fn a_budget_too_small_for_even_one_cluster_still_shows_that_one() {
        // Arrange: a single oversized cluster, budget of 1 token.
        let groups = vec![vec![member("01ABCDEFGHJKMNPQRST0000A", "fact", "solo")]];

        // Act
        let out = render_clusters(&groups, 1, chars);

        // Assert: the floor from `fit_evidence_to_budget` applies here too:
        // never zero clusters when any exist, even though the budget is
        // exceeded.
        assert!(out.contains("solo"), "{out}");
        assert!(chars(&out) > 1, "the floor renders past the budget: {out}");
    }

    #[test]
    fn output_never_exceeds_budget_except_the_one_oversized_group_floor() {
        // Arrange: 3 clusters, content padded so each group's own size
        // dominates the header's length swing between truncated ("X of Y
        // shown...") and untruncated ("X cluster(s)") wording. Without the
        // padding, dropping the 3rd (tiny) group can cost MORE chars in a
        // longer header than it saves in body, which would make `kept=3`
        // legitimately win and defeat this test's premise.
        let pad = "x".repeat(200);
        let groups = vec![
            vec![
                member("01AAAAAAAAAAAAAAAAAAAAAAA", "fact", &format!("a1 {pad}")),
                member("01AAAAAAAAAAAAAAAAAAAAAAB", "fact", &format!("a2 {pad}")),
            ],
            vec![member(
                "01BBBBBBBBBBBBBBBBBBBBBBB",
                "fact",
                &format!("b1 {pad}"),
            )],
            vec![member(
                "01CCCCCCCCCCCCCCCCCCCCCCC",
                "fact",
                &format!("c1 {pad}"),
            )],
        ];
        let render_n = |kept: usize| {
            let header = render_header(kept, groups.len());
            let body = groups[..kept]
                .iter()
                .map(|g| render_group(g))
                .collect::<Vec<_>>()
                .join("\n\n");
            format!("{header}\n\n{body}")
        };
        let two = render_n(2);
        let three = render_n(3);
        assert!(
            chars(&three) > chars(&two),
            "fixture must make the 3rd cluster strictly grow the render: \
             two={two}\nthree={three}"
        );
        let budget = chars(&two);

        // Act
        let out = render_clusters(&groups, budget, chars);

        // Assert: fits within budget, and it's exactly the 2-cluster render.
        assert!(chars(&out) <= budget, "{out}");
        assert_eq!(out, two);
    }

    #[test]
    fn a_smaller_budget_shows_fewer_clusters_than_a_larger_one() {
        // Arrange
        let groups = vec![
            vec![member("01AAAAAAAAAAAAAAAAAAAAAAA", "fact", "a1")],
            vec![member("01BBBBBBBBBBBBBBBBBBBBBBB", "fact", "b1")],
            vec![member("01CCCCCCCCCCCCCCCCCCCCCCC", "fact", "c1")],
        ];
        let full = render_clusters(&groups, usize::MAX, chars);
        let small_budget = chars(&render_clusters(&groups[..1], usize::MAX, chars));

        // Act
        let small = render_clusters(&groups, small_budget, chars);
        let large = render_clusters(&groups, usize::MAX, chars);

        // Assert
        assert!(
            chars(&small) < chars(&large),
            "small: {small}\nlarge: {large}"
        );
        assert_eq!(large, full);
    }

    #[test]
    fn truncation_is_announced_with_the_count_withheld() {
        // Arrange: budget only fits the first cluster.
        let groups = vec![
            vec![member("01AAAAAAAAAAAAAAAAAAAAAAA", "fact", "a1")],
            vec![member("01BBBBBBBBBBBBBBBBBBBBBBB", "fact", "b1")],
            vec![member("01CCCCCCCCCCCCCCCCCCCCCCC", "fact", "c1")],
        ];
        let budget = chars(&render_clusters(&groups[..1], usize::MAX, chars));

        // Act
        let out = render_clusters(&groups, budget, chars);

        // Assert
        assert!(out.contains("1 of 3"), "{out}");
        assert!(out.contains("2 withheld"), "{out}");
    }

    #[test]
    fn no_group_identifier_is_ever_emitted() {
        // Arrange: distinct member ids so a leaked internal index would show
        // up as a stray small integer somewhere in the text.
        let groups = vec![
            vec![member("01AAAAAAAAAAAAAAAAAAAAAAA", "fact", "a1")],
            vec![member("01BBBBBBBBBBBBBBBBBBBBBBB", "fact", "b1")],
        ];

        // Act
        let out = render_clusters(&groups, usize::MAX, chars);

        // Assert: no "Group 0"/"Group 1"/"community"-style label anywhere.
        let lower = out.to_lowercase();
        assert!(!lower.contains("group 0"), "{out}");
        assert!(!lower.contains("group 1"), "{out}");
        assert!(!lower.contains("community"), "{out}");
    }

    #[test]
    fn narrow_groups_caps_the_cluster_count_to_k() {
        // Arrange
        let groups = vec![
            vec![member("01AAAAAAAAAAAAAAAAAAAAAAA", "fact", "a1")],
            vec![member("01BBBBBBBBBBBBBBBBBBBBBBB", "fact", "b1")],
            vec![member("01CCCCCCCCCCCCCCCCCCCCCCC", "fact", "c1")],
        ];

        // Act
        let out = narrow_groups(&groups, Some(2), None);

        // Assert
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn narrow_groups_caps_members_per_cluster_to_members() {
        // Arrange
        let groups = vec![vec![
            member("01AAAAAAAAAAAAAAAAAAAAAAA", "fact", "a1"),
            member("01BBBBBBBBBBBBBBBBBBBBBBB", "fact", "a2"),
            member("01CCCCCCCCCCCCCCCCCCCCCCC", "fact", "a3"),
        ]];

        // Act
        let out = narrow_groups(&groups, None, Some(1));

        // Assert
        assert_eq!(out[0].len(), 1);
        assert_eq!(out[0][0].label, "a1");
    }

    #[test]
    fn narrow_groups_treats_zero_as_one_not_as_nothing() {
        // Arrange
        let groups = vec![vec![member("01AAAAAAAAAAAAAAAAAAAAAAA", "fact", "a1")]];

        // Act
        let by_k = narrow_groups(&groups, Some(0), None);
        let by_members = narrow_groups(&groups, None, Some(0));

        // Assert: `k=0`/`members=0` must not render an empty listing.
        assert_eq!(by_k.len(), 1, "{by_k:?}");
        assert_eq!(by_members[0].len(), 1, "{by_members:?}");
    }

    #[test]
    fn narrow_groups_cannot_widen_past_what_the_store_returned() {
        // Arrange
        let groups = vec![vec![member("01AAAAAAAAAAAAAAAAAAAAAAA", "fact", "a1")]];

        // Act
        let out = narrow_groups(&groups, Some(50), Some(50));

        // Assert: an oversized ask is not an error, just a no-op cap.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 1);
    }
}
