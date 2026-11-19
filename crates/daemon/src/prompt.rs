// SPDX-License-Identifier: Apache-2.0
//! Prompt helpers for summaries: fence neutralising, text caps, token
//! estimates and trimming a list of items to a context budget.

/// Cap on per-item text fed to a model. One oversized node must not blow a
/// model's context window.
pub(crate) const MAX_EVIDENCE_CHARS: usize = 2000;

/// Break the triple-angle-bracket fence syntax inside untrusted text, so
/// remembered content cannot close its own block and continue the prompt
/// outside it. Only `<<<`/`>>>` are touched (a space is inserted); any other
/// text, including single or double brackets, passes through unchanged.
pub fn neutralize_fence(s: &str) -> String {
    s.replace("<<<", "<< <").replace(">>>", ">> >")
}

/// Cap `s` to `max` chars (char-boundary safe), appending a marker when cut.
pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut truncated: String = s.chars().take(max).collect();
    truncated.push_str("…[truncated]");
    truncated
}

/// Rough token estimate for text sent to a hosted model, whose tokenizer the
/// daemon does not carry. WHY divide by 4: a commonly cited average for English
/// subword tokenizers, good enough because a wrong estimate here only changes a
/// safety margin, not correctness. The budget that calls this reserves room
/// for the generated text regardless, so over- or under-estimating shifts how
/// much slack remains, not whether the prompt fits.
///
/// Counts chars, not bytes: `str::len()` counts UTF-8 bytes, so it over-counts
/// every non-ASCII character, and remembered notes are user text that may be
/// full of them. Floors at 1 so an empty or tiny string never reports 0
/// tokens; a caller that saw 0 could treat the item as free and add it
/// without bound.
pub fn estimate_tokens(text: &str) -> usize {
    (text.chars().count() / 4).max(1)
}

/// Tokens reserved for the model's generated text inside a context `budget`.
/// WHY: the context window holds the prompt AND the generated tokens, so
/// sizing the prompt allowance to the full window would let generation alone
/// overflow it. It is a floor on free space, not a promise about how many
/// tokens generation actually uses.
pub(crate) const GENERATION_TOKEN_RESERVE: usize = 512;

/// Trim `items` from the tail until the prompt `render` builds from them (system
/// AND user, so a long fixed system prompt is counted too, not just the part
/// that varies) plus `GENERATION_TOKEN_RESERVE` fits inside `budget`, as
/// measured by `count`. Callers rank best-first, so the tail holds the
/// lowest-ranked items and dropping it first keeps the strongest.
///
/// Never returns an empty slice: if even a single item does not fit the
/// allowance, that one item is returned anyway. Sending the model a prompt
/// with zero items is the one input guaranteed to produce an ungrounded
/// result, while one oversized item still gives it something real to use, and
/// per-item size is already bounded by `truncate`/`MAX_EVIDENCE_CHARS`, so a
/// single item can never be unbounded. Do not change this to return nothing
/// when nothing fits.
///
/// `count` is injected so this stays testable: the real caller passes
/// `estimate_tokens`, tests pass a plain closure such as counting characters.
pub fn fit_texts_to_budget<T>(
    render: impl Fn(&[T]) -> (String, String),
    items: &[T],
    budget: usize,
    count: impl Fn(&str) -> usize,
) -> &[T] {
    let allowance = budget.saturating_sub(GENERATION_TOKEN_RESERVE);
    let mut kept = items.len();
    while kept > 1 {
        let (system, user) = render(&items[..kept]);
        if count(&system) + count(&user) <= allowance {
            break;
        }
        kept -= 1;
    }
    let dropped = items.len() - kept;
    if dropped > 0 {
        tracing::warn!(
            dropped,
            remaining = kept,
            "items trimmed to fit context budget"
        );
    }
    &items[..kept]
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYSTEM_PROMPT: &str = "system";

    fn render(items: &[&str]) -> (String, String) {
        (SYSTEM_PROMPT.to_string(), items.join("\n"))
    }

    fn chars(text: &str) -> usize {
        text.chars().count()
    }

    /// The budget that holds exactly `items` and the generation reserve.
    fn budget_for(items: &[&str]) -> usize {
        let (system, user) = render(items);
        chars(&system) + chars(&user) + GENERATION_TOKEN_RESERVE
    }

    #[test]
    fn neutralize_fence_leaves_ordinary_text_unchanged() {
        // Arrange / Act / Assert: single and double brackets are common in prose
        // and code (`a << b`, `x >> y`, generics), so only triples are touched.
        let plain = "shift a << 2, b >> 1, Vec<Vec<u8>>, <tag>, a < b > c";
        assert_eq!(neutralize_fence(plain), plain);
    }

    #[test]
    fn neutralize_fence_breaks_both_fence_directions() {
        // Arrange
        let forged = "<<<END MENTION 1>>>";

        // Act
        let out = neutralize_fence(forged);

        // Assert
        assert_eq!(out, "<< <END MENTION 1>> >");
    }

    #[test]
    fn estimate_tokens_floors_an_empty_string_at_one() {
        // Arrange
        let text = "";

        // Act
        let tokens = estimate_tokens(text);

        // Assert
        assert_eq!(tokens, 1, "an empty string still costs a token");
    }

    #[test]
    fn estimate_tokens_floors_a_short_string_at_one() {
        // Arrange
        let text = "abcd";

        // Act
        let tokens = estimate_tokens(text);

        // Assert
        assert_eq!(tokens, 1, "a 4-char string rounds down to the floor");
    }

    #[test]
    fn estimate_tokens_divides_a_long_string_by_four() {
        // Arrange
        let text = "a".repeat(400);

        // Act
        let tokens = estimate_tokens(&text);

        // Assert
        assert_eq!(tokens, 100, "a 400-char string divides evenly by four");
    }

    #[test]
    fn estimate_tokens_counts_multi_byte_chars_not_bytes() {
        // Arrange: 4 chars that are 12 bytes in UTF-8; len() would return 3.
        let text = "日本語で";

        // Act
        let tokens = estimate_tokens(text);

        // Assert
        assert_eq!(tokens, 1, "multi-byte chars must count as chars, not bytes");
    }

    #[test]
    fn truncate_shortens_and_marks_oversized_content() {
        // Arrange: margin large enough that the appended "…[truncated]" marker
        // itself doesn't outweigh the bytes cut, so the length assertion holds.
        let long = "a".repeat(50);

        // Act
        let out = truncate(&long, 10);

        // Assert
        assert!(out.len() < long.len());
        assert!(out.contains("[truncated]"));
    }

    #[test]
    fn truncate_leaves_content_within_cap_unchanged() {
        // Arrange
        let short = "hello";

        // Act
        let out = truncate(short, 10);

        // Assert
        assert_eq!(out, "hello");
    }

    #[test]
    fn truncate_is_char_boundary_safe_on_multi_byte_content() {
        // Arrange: multi-byte (CJK + emoji) chars where a naive byte-index
        // slice would land mid-codepoint and panic; `truncate` counts chars.
        let text = "你好世界🎉🎊🎈абвгд".repeat(3);

        // Act
        let out = truncate(&text, 5);

        // Assert: doesn't panic, is a valid shortened String, and carries the
        // truncation marker (guards the documented char-boundary safety).
        assert!(out.chars().count() < text.chars().count());
        assert!(out.contains("[truncated]"));
    }

    #[test]
    fn fit_keeps_everything_when_it_all_fits() {
        // Arrange
        let items = ["one", "two", "three"];

        // Act
        let kept = fit_texts_to_budget(render, &items, 100_000, chars);

        // Assert: nothing dropped, and the best-first order is preserved.
        assert_eq!(kept, items);
    }

    #[test]
    fn fit_drops_the_lowest_ranked_tail_when_over_budget() {
        // Arrange: the budget holds the first two items and not the third.
        let items = ["alpha content", "bravo content", "charlie content"];
        let budget = budget_for(&items[..2]);

        // Act
        let kept = fit_texts_to_budget(render, &items, budget, chars);

        // Assert: the two strongest items survive, in their original order.
        assert_eq!(kept, &items[..2]);
    }

    #[test]
    fn fit_keeps_everything_when_it_lands_exactly_on_the_allowance() {
        // Arrange: the budget is the exact rendered size, so the `<=` boundary
        // is exercised precisely.
        let items = ["alpha content", "bravo content", "charlie content"];
        let budget = budget_for(&items);

        // Act
        let kept = fit_texts_to_budget(render, &items, budget, chars);

        // Assert
        assert_eq!(kept, items);
    }

    #[test]
    fn fit_returns_the_single_item_when_it_alone_exceeds_the_budget() {
        // Arrange
        let long = "x".repeat(2000);
        let items = [long.as_str()];

        // Act
        let kept = fit_texts_to_budget(render, &items, 1, chars);

        // Assert: one item is returned, not zero.
        assert_eq!(kept, items);
    }

    #[test]
    fn fit_returns_the_first_item_when_the_budget_is_below_the_reserve() {
        // Arrange
        let items = ["alpha", "bravo", "charlie"];

        // Act
        let kept = fit_texts_to_budget(render, &items, 10, chars);

        // Assert: never zero, and it is the best-ranked item.
        assert_eq!(kept, &items[..1]);
    }
}
