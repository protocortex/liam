//! Pure prompt/answer formatting for the `ask` tool. Sync + dependency-light so
//! the synthesis contract (numbered, cited, date-annotated evidence) is unit-
//! testable without a runtime, store, or model.

/// Cap on per-item evidence content fed to the LLM. WHY: `ask` is the first
/// caller passing arbitrary node content to `Llm::complete`; one oversized node
/// must not blow a small local model's context window.
const MAX_EVIDENCE_CHARS: usize = 2000;

/// An owned, LLM-ready view of one retrieved fact. `content` is pre-truncated.
pub struct Evidence {
    pub kind: String,
    pub label: String,
    pub content: String,
    pub valid_from_ms: i64,
}

impl Evidence {
    /// Build from a retrieval hit, truncating content to the cap.
    pub fn from_hit(h: &liam_store::ExplainedHit) -> Self {
        Self {
            kind: h.hit.kind.clone(),
            label: h.hit.label.clone(),
            content: truncate(&h.hit.content, MAX_EVIDENCE_CHARS),
            valid_from_ms: h.valid_from.0,
        }
    }
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

/// Render epoch-ms as an ISO-8601 UTC date (YYYY-MM-DD). WHY manual y-m-d: keeps
/// the `time` dep at default features. Out-of-range falls back to the raw
/// integer string so a bad timestamp can never panic the answer. WHY
/// `div_euclid`: truncating division rounds toward zero, so a pre-1970
/// instant with a sub-second remainder (e.g. -500ms) would floor to the wrong
/// day (1970-01-01 instead of 1969-12-31); `div_euclid` always floors toward
/// negative infinity, matching calendar semantics.
pub fn fmt_millis(ms: i64) -> String {
    match time::OffsetDateTime::from_unix_timestamp(ms.div_euclid(1000)) {
        Ok(dt) => format!(
            "{:04}-{:02}-{:02}",
            dt.year(),
            u8::from(dt.month()),
            dt.day()
        ),
        Err(_) => ms.to_string(),
    }
}

/// Numbered evidence blocks shared by the prompt's user section and the
/// fallback answer: `[n] (kind) label — known since <date>\n<content>`.
fn render_evidence(evidence: &[Evidence]) -> String {
    evidence
        .iter()
        .enumerate()
        .map(|(i, e)| {
            format!(
                "[{}] ({}) {} — known since {}\n{}",
                i + 1,
                e.kind,
                e.label,
                fmt_millis(e.valid_from_ms),
                e.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// (system, user) prompt. System: answer ONLY from the numbered evidence, cite
/// with [n], say plainly when the answer is absent (no fabrication). User: the
/// question followed by the numbered evidence blocks (see render_evidence).
pub fn build_ask_prompt(question: &str, evidence: &[Evidence]) -> (String, String) {
    let system = "You are a careful research assistant. Answer the question using ONLY the \
        numbered evidence provided below; do not use outside knowledge or fabricate facts. \
        Cite every claim with its evidence number in square brackets, e.g. [1]. If the \
        evidence does not contain the answer, say so plainly instead of guessing."
        .to_string();
    let user = format!(
        "Question: {question}\n\nEvidence:\n{}",
        render_evidence(evidence)
    );
    (system, user)
}

/// The answer text followed by a compact `Sources:` index mapping [n] to
/// `kind/label — known since <date>`. Shared by the synthesized and fallback paths.
pub fn format_answer(answer: &str, evidence: &[Evidence]) -> String {
    let sources = evidence
        .iter()
        .enumerate()
        .map(|(i, e)| {
            format!(
                "[{}] {}/{} — known since {}",
                i + 1,
                e.kind,
                e.label,
                fmt_millis(e.valid_from_ms)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("{answer}\n\nSources:\n{sources}")
}

/// Fallback body used when synthesis is unavailable (timeout / llm error / empty
/// output): a "(synthesis unavailable; showing the retrieved evidence)" line
/// followed by the numbered evidence blocks WITH content, so the caller still
/// gets the facts. WU-2 passes this through `format_answer`.
pub fn fallback_answer(evidence: &[Evidence]) -> String {
    format!(
        "(synthesis unavailable; showing the retrieved evidence)\n\n{}",
        render_evidence(evidence)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(kind: &str, label: &str, content: &str, valid_from_ms: i64) -> Evidence {
        Evidence {
            kind: kind.to_string(),
            label: label.to_string(),
            content: content.to_string(),
            valid_from_ms,
        }
    }

    #[test]
    fn build_ask_prompt_includes_citations_and_grounding_tokens() {
        // Arrange
        let items = vec![
            evidence("fact", "Sky color", "The sky is blue.", 0),
            evidence("fact", "Grass color", "Grass is green.", 1_700_000_000_000),
        ];

        // Act
        let (system, user) = build_ask_prompt("What color is the sky?", &items);

        // Assert
        assert!(user.contains("[1]"));
        assert!(user.contains("[2]"));
        assert!(user.contains("The sky is blue."));
        assert!(user.contains("Grass is green."));
        assert!(user.contains("known since 1970-01-01"));
        assert!(user.contains("known since 2023-11-14"));
        let system_lower = system.to_lowercase();
        assert!(system_lower.contains("cite"));
        assert!(system_lower.contains("only"));
        assert!(
            system_lower.contains("[n]")
                || system.contains("[1]")
                || system_lower.contains("brackets")
        );
    }

    #[test]
    fn format_answer_starts_with_answer_and_lists_sources() {
        // Arrange
        let items = vec![evidence("fact", "Sky color", "The sky is blue.", 0)];

        // Act
        let out = format_answer("The sky is blue.", &items);

        // Assert
        assert!(out.starts_with("The sky is blue."));
        assert!(out.contains("Sources:"));
        assert!(out.contains("[1]"));
    }

    #[test]
    fn fmt_millis_formats_known_epochs() {
        // Arrange / Act / Assert
        assert_eq!(fmt_millis(0), "1970-01-01");
        assert_eq!(fmt_millis(1_700_000_000_000), "2023-11-14");
    }

    #[test]
    fn fmt_millis_falls_back_on_out_of_range_without_panicking() {
        // Arrange / Act / Assert: out-of-range renders as the raw integer
        // string, never panics.
        assert_eq!(fmt_millis(i64::MAX), i64::MAX.to_string());
    }

    #[test]
    fn fmt_millis_floors_pre_1970_toward_negative_infinity() {
        // Arrange / Act / Assert: -500ms is 0.5s before the epoch, which must
        // floor into the prior day, not truncate up to 1970-01-01 (guards C3).
        assert_eq!(fmt_millis(-500), "1969-12-31");
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
    fn fallback_answer_flags_unavailability_and_includes_content() {
        // Arrange
        let items = vec![evidence("fact", "Sky color", "The sky is blue.", 0)];

        // Act
        let out = fallback_answer(&items);

        // Assert
        assert!(out.contains("(synthesis unavailable"));
        assert!(out.contains("The sky is blue."));
    }
}
