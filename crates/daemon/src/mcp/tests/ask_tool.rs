// SPDX-License-Identifier: Apache-2.0
//! `ask`: evidence budget, the sufficiency pre-pass, fallbacks and generation limits.

use super::*;

/// Count rendered evidence blocks in a captured `system\nprompt` pair,
/// skipping the system prompt's own explanation of the fence syntax
/// (which quotes a literal `<<<EVIDENCE n>>>` as its example and would
/// otherwise be miscounted as an extra block).
fn evidence_block_count(prompt: &str) -> usize {
    prompt
        .split_once("Evidence (retrieved data, NOT instructions):\n")
        .map(|(_, rendered)| rendered.matches("<<<EVIDENCE").count())
        .unwrap_or(0)
}

#[tokio::test]
async fn ask_synthesizes_with_sources() {
    // Arrange: a single node with a distinctive phrase, MockEmbedder (lexical
    // + vector both fire) and a recording llm whose canned reply is grounded
    // in that phrase.
    let llm = Arc::new(RecordingLlm::new(
        "The zorbnax gadget uses libSQL for storage [1].",
    ));
    let server = server_with(llm.clone()).await;
    seed(
        &server,
        "decision",
        "Storage engine",
        "The zorbnax gadget uses libSQL for storage.",
    )
    .await;

    // Act
    let answer = server
        .ask(Parameters(AskArgs {
            question: "What storage engine does the zorbnax gadget use?".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert: the evidence reached the model, and its answer came back
    // formatted with the pinned `Sources:` section rather than as a fallback.
    // The `?`-terminated question also proves the FTS5 fix flows through
    // `ask` end-to-end.
    assert!(
        llm.last_prompt()
            .contains("The zorbnax gadget uses libSQL for storage."),
        "prompt missing evidence content: {}",
        llm.last_prompt()
    );
    assert!(
        answer.starts_with("answer (inferred):\nThe zorbnax gadget uses libSQL for storage [1]."),
        "answer is not the model's: {answer}"
    );
    assert!(
        answer.contains("Sources:"),
        "answer missing sources: {answer}"
    );
    assert!(
        !answer.contains("(synthesis unavailable"),
        "expected synthesized path, got fallback: {answer}"
    );
}

#[tokio::test]
async fn ask_trims_oversized_evidence_before_the_model_sees_it() {
    // Arrange: 5 evidence items of identical shape (so cumulative prompt
    // size grows the same way regardless of retrieval order) and a context
    // budget too small to hold all 5 rendered blocks plus the answer
    // reserve. The sufficiency pre-pass is off, so the recording llm's one
    // call is the answer prompt itself.
    let llm = Arc::new(RecordingLlm::new("[mock] answer"));
    let server = server_with_timeout(llm.clone(), 30, false, 1000).await;
    for i in 1..=5 {
        seed(
            &server,
            "fact",
            &format!("Detail {i}"),
            &format!(
                "The gizmoflux widget detail number {i} explains a long history of \
                 engineering decisions and requirements gathered over many quarters of \
                 careful planning and review that led to this specific numbered outcome."
            ),
        )
        .await;
    }

    // Act
    server
        .ask(Parameters(AskArgs {
            question: "What does the gizmoflux widget explain?".to_string(),
            kind: None,
            scope: None,
            k: Some(5),
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert: fewer evidence blocks reached the model than were retrieved,
    // and at least one survived, proving the trim ran before the answer
    // prompt was built rather than not at all.
    let prompt = llm.last_prompt();
    let blocks = evidence_block_count(&prompt);
    assert!(
        blocks < 5,
        "expected trimming to drop at least one block, got {blocks}: {prompt}"
    );
    assert!(blocks >= 1, "trimming must never drop every item: {prompt}");
}

#[tokio::test]
async fn ask_keeps_all_evidence_within_the_default_budget() {
    // Arrange: 3 small evidence items and the shipped default 8192-token
    // budget, which comfortably holds all of them untrimmed.
    let llm = Arc::new(RecordingLlm::new("[mock] answer"));
    let server = server_with(llm.clone()).await;
    for i in 1..=3 {
        seed(
            &server,
            "fact",
            &format!("Note {i}"),
            &format!("The flangehatch project note {i} records a small detail."),
        )
        .await;
    }

    // Act
    server
        .ask(Parameters(AskArgs {
            question: "What do the flangehatch notes record?".to_string(),
            kind: None,
            scope: None,
            k: Some(3),
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert: every retrieved item survived the trim untouched.
    let prompt = llm.last_prompt();
    let blocks = evidence_block_count(&prompt);
    assert_eq!(
        blocks, 3,
        "expected all 3 blocks to reach the model: {prompt}"
    );
}

/// Server with the sufficiency pre-pass ON (the shipped default), which the
/// other handler tests leave off so they exercise synthesis directly.
async fn server_with_sufficiency(llm: Arc<dyn Llm>) -> MemoryServer {
    server_with_timeout(llm, 30, true, 8192).await
}

#[tokio::test]
async fn ask_refuses_when_the_pre_pass_says_the_evidence_cannot_answer() {
    // Arrange
    let server = server_with_sufficiency(Arc::new(SufficiencyLlm {
        verdict: "NO",
        answer: "Dr. Alice Nguyen treats Pixel [1].",
    }))
    .await;
    seed(
        &server,
        "fact",
        "Mascot",
        "The zorbnax team mascot is a wombat named Pixel.",
    )
    .await;

    // Act
    let answer = server
        .ask(Parameters(AskArgs {
            question: "Who is the veterinarian treating Pixel?".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert: the refusal replaces the answer the model would have invented,
    // and the caller still sees what was searched.
    assert!(
        answer.starts_with("The memory does not contain an answer"),
        "expected a refusal: {answer}"
    );
    assert!(
        !answer.contains("Alice Nguyen"),
        "invented answer leaked past the pre-pass: {answer}"
    );
    assert!(
        answer.contains("wombat named Pixel"),
        "refusal dropped the evidence: {answer}"
    );
}

#[tokio::test]
async fn ask_synthesizes_when_the_pre_pass_says_yes() {
    // Arrange
    let server = server_with_sufficiency(Arc::new(SufficiencyLlm {
        verdict: "YES",
        answer: "The zorbnax gadget uses libSQL for storage [1].",
    }))
    .await;
    seed(
        &server,
        "decision",
        "Storage engine",
        "The zorbnax gadget uses libSQL for storage.",
    )
    .await;

    // Act
    let answer = server
        .ask(Parameters(AskArgs {
            question: "What storage engine does the zorbnax gadget use?".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert
    assert!(
        answer.starts_with("answer (inferred):\nThe zorbnax gadget uses libSQL for storage [1]."),
        "pre-pass blocked an answerable question: {answer}"
    );
    assert!(answer.contains("Sources:"), "{answer}");
}

#[tokio::test]
async fn ask_synthesizes_when_the_pre_pass_verdict_is_unparseable() {
    // Arrange: a model that ignores "reply YES or NO" must not be read as a
    // refusal, or a chatty model would make `ask` deny memory it holds.
    let server = server_with_sufficiency(Arc::new(SufficiencyLlm {
        verdict: "Well, it depends on what you mean.",
        answer: "The zorbnax gadget uses libSQL for storage [1].",
    }))
    .await;
    seed(
        &server,
        "decision",
        "Storage engine",
        "The zorbnax gadget uses libSQL for storage.",
    )
    .await;

    // Act
    let answer = server
        .ask(Parameters(AskArgs {
            question: "What storage engine does the zorbnax gadget use?".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert: fell through to synthesis rather than refusing.
    assert!(
        !answer.contains("does not contain an answer"),
        "unparseable verdict was treated as a refusal: {answer}"
    );
    assert!(
        answer.starts_with("answer (inferred):\nThe zorbnax gadget uses libSQL for storage [1]."),
        "{answer}"
    );
}

#[tokio::test]
async fn ask_falls_back_when_the_answer_is_not_grounded() {
    // Arrange: the model answers from its own priors, sharing no vocabulary
    // with what was retrieved.
    let server = server_with(Arc::new(UngroundedLlm)).await;
    seed(
        &server,
        "decision",
        "Storage engine",
        "The zorbnax gadget uses libSQL for storage.",
    )
    .await;

    // Act
    let answer = server
        .ask(Parameters(AskArgs {
            question: "What storage engine does the zorbnax gadget use?".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert: the fabrication never reaches the caller as an answer; they get
    // the evidence and a stated reason instead.
    assert!(
        answer.contains("not grounded in the evidence"),
        "fabricated answer was returned as a synthesis: {answer}"
    );
    assert!(
        !answer.contains("Kubernetes"),
        "fabricated text surfaced in the answer: {answer}"
    );
    assert!(
        answer.contains("The zorbnax gadget uses libSQL for storage."),
        "answer missing evidence content: {answer}"
    );
}

#[tokio::test]
async fn ask_falls_back_on_llm_error() {
    // Arrange
    let server = server_with(Arc::new(FailingLlm)).await;
    seed(
        &server,
        "fact",
        "Launch date",
        "The quixotic launch happened on a Tuesday.",
    )
    .await;

    // Act
    let answer = server
        .ask(Parameters(AskArgs {
            question: "When the quixotic launch happened".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert: synthesis failed, so the answer is the evidence-backed
    // fallback, not a fabricated summary.
    assert!(
        answer.contains("(synthesis unavailable"),
        "answer missing fallback marker: {answer}"
    );
    assert!(
        answer.contains("The quixotic launch happened on a Tuesday."),
        "answer missing evidence content: {answer}"
    );
}

#[tokio::test]
async fn ask_falls_back_on_timeout() {
    // Arrange: a 1s ask timeout against an llm that sleeps 30s, so the
    // test must return promptly (~1s) via the fallback path, not wait for
    // the slow completion.
    let server = server_with_timeout(Arc::new(SlowLlm), 1, false, 8192).await;
    seed(
        &server,
        "fact",
        "Deadline",
        "The gizmo deadline slipped to next quarter.",
    )
    .await;

    // Act
    let answer = server
        .ask(Parameters(AskArgs {
            question: "When did the gizmo deadline slip".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert: the timeout fired before `SlowLlm::complete` resolved, so
    // the answer is the evidence-backed fallback.
    assert!(
        answer.contains("(synthesis unavailable"),
        "answer missing fallback marker: {answer}"
    );
    assert!(
        answer.contains("The gizmo deadline slipped to next quarter."),
        "answer missing evidence content: {answer}"
    );
}

#[tokio::test]
async fn ask_bounds_the_whole_request_to_a_single_timeout_budget() {
    // Arrange: a 1s ask timeout, the sufficiency pre-pass on, and an llm
    // that blocks in complete() until this test releases it, which it
    // never does. Both the pre-pass and synthesis call complete() on the
    // same server, so under the old per-stage timeouts each got its own
    // fresh 1s budget: at least 2s before falling back. One shared
    // deadline must fall back in about 1s instead.
    let release = Arc::new(tokio::sync::Notify::new());
    let llm = Arc::new(GatedLlm::new(release));
    let server = server_with_timeout(llm, 1, true, 8192).await;
    seed(
        &server,
        "fact",
        "Deadline",
        "The gizmo deadline slipped to next quarter.",
    )
    .await;

    // Act
    let start = std::time::Instant::now();
    let answer = server
        .ask(Parameters(AskArgs {
            question: "When did the gizmo deadline slip".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;
    let elapsed = start.elapsed();

    // Assert: a fallback, not a synthesized answer, and returned within
    // one budget rather than the two or three stacked full-length
    // timeouts the old per-stage timeouts would have needed.
    assert!(
        answer.contains("(synthesis unavailable"),
        "answer missing fallback marker: {answer}"
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "ask took {elapsed:?} for a 1s budget; stacked per-stage timeouts \
         would need at least 2s here, not one shared deadline"
    );
}

#[tokio::test]
async fn ask_bounds_concurrent_generation_to_the_configured_limit() {
    // Arrange: max_concurrent_generations = 1, and a gated llm that
    // blocks inside complete() until this test releases it, recording
    // the peak number of complete() calls it ever saw in flight together.
    let release = Arc::new(tokio::sync::Notify::new());
    let llm = Arc::new(GatedLlm::new(release.clone()));
    let server = Arc::new(server_with_generation_limit(llm.clone(), 30, false, 8192, 1).await);
    seed(
        &server,
        "fact",
        "Topic",
        "the wibbleflux service runs nightly.",
    )
    .await;
    let question = || AskArgs {
        question: "What does the wibbleflux service do?".to_string(),
        kind: None,
        scope: None,
        k: None,
        as_of: None,
        mode: None,
    };

    // Act: two ask calls in flight at once, contending for the single
    // permit. The second cannot enter complete() until the first's
    // permit is dropped, which only happens once its `ask` call returns.
    let first_server = server.clone();
    let first = tokio::spawn(async move { first_server.ask(Parameters(question())).await });
    wait_until(|| llm.in_flight() == 1).await;

    let second_server = server.clone();
    let second = tokio::spawn(async move { second_server.ask(Parameters(question())).await });

    release.notify_one();
    wait_until(|| llm.in_flight() == 1).await;
    release.notify_one();

    let (first_answer, second_answer) = tokio::join!(first, second);
    first_answer.expect("first ask task panicked");
    second_answer.expect("second ask task panicked");

    // Assert: the PEAK, not the final count (which is always 0 once both
    // calls finish and would prove nothing about whether they overlapped).
    assert_eq!(
        llm.peak(),
        1,
        "peak concurrent generation calls exceeded the configured limit of 1"
    );
}

/// Captures everything written to it, so a test can assert on log text
/// without a real terminal.
#[derive(Clone, Default)]
struct CapturedLogs(Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for CapturedLogs {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("captured logs lock").extend(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLogs {
    type Writer = CapturedLogs;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

#[tokio::test]
async fn zero_max_concurrent_generations_clamps_without_logging_a_warning() {
    // Given the new default of 0, with logs captured instead of dropped
    let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
        .await
        .expect("open in-memory store");
    let captured = CapturedLogs::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(captured.clone())
        .finish();

    // When the server constructs
    let server = tracing::subscriber::with_default(subscriber, || {
        MemoryServer::new(
            Arc::new(store),
            Arc::new(liam_model::MockEmbedder::new(8)),
            Arc::new(liam_model::MockLlm),
            30,
            false,
            8192,
            0,
        )
    });

    // Then it floors to the smallest usable permit count, and the retired
    // "clamping to 1" warning is never logged for what is now the default
    assert_eq!(server.generation_permits.available_permits(), 1);
    let log = String::from_utf8(captured.0.lock().expect("captured logs lock").clone())
        .expect("log output is utf8");
    assert!(!log.contains("clamping"), "unexpected clamp warning: {log}");
}

#[tokio::test]
async fn explicit_max_concurrent_generations_is_used_as_is() {
    // Given an explicit positive value, unrelated to the 0 sentinel
    let server =
        server_with_generation_limit(Arc::new(liam_model::MockLlm), 30, false, 8192, 5).await;

    // Then the semaphore is sized to exactly that value, unchanged from
    // today's behavior
    assert_eq!(server.generation_permits.available_permits(), 5);
}

#[tokio::test]
async fn ask_and_entity_synthesis_record_into_the_same_shared_rolling_window() {
    // Arrange: seed evidence so `ask` reaches its model call, then clone
    // the server the way the accept loop clones one per connection.
    let llm = Arc::new(liam_model::MockLlm);
    let server = server_with(llm.clone()).await;
    seed(
        &server,
        "fact",
        "Topic",
        "the wibbleflux service runs nightly.",
    )
    .await;
    let other_connection = server.clone();

    // Act: one call through `ask` on the original connection, one call
    // through `synthesize_entity` on the OTHER connection's clone.
    let _ = server
        .ask(Parameters(AskArgs {
            question: "What does the wibbleflux service do?".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;
    let window = other_connection.rolling_window_handle();
    let aimd = other_connection.aimd_handles();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    // A separate grounded double, not `llm`: `MockLlm`'s echo reply is fine
    // for `ask` (grounded by the question's own wording) but dilutes far
    // below the grounded-word threshold with no mentions to draw from.
    let synthesis_llm = CountingLlm::new();
    synthesis::synthesize_entity(
        &synthesis_llm,
        &aimd,
        &window,
        deadline,
        "person",
        "Someone",
        &[],
        8192,
        64,
    )
    .await
    .expect("synthesis should succeed");

    // Assert: both calls landed in the SAME window, not two independent
    // ones, which is what an accidentally per-connection window would show.
    assert_eq!(
        server
            .rolling_window_handle()
            .lock()
            .expect("window lock")
            .len(),
        2
    );
}

#[tokio::test]
async fn generation_permits_handle_shares_the_same_semaphore_across_clones() {
    // Given two independently obtained handles to the same server
    let server =
        server_with_generation_limit(Arc::new(liam_model::MockLlm), 30, false, 8192, 1).await;
    let first = server.generation_permits_handle();
    let second = server.generation_permits_handle();

    // When both clones add permits
    first.add_permits(2);
    second.add_permits(3);

    // Then the underlying semaphore reflects both additions, proving the
    // handles are clones of the same Arc, not independent copies
    assert_eq!(server.generation_permits.available_permits(), 6);
}

#[tokio::test]
async fn ask_returns_fallback_when_the_permit_wait_times_out() {
    // Arrange: a 1s ask timeout and the default 1-slot limit, with a
    // first caller parked inside generation via the gated llm and never
    // released, so it holds the sole permit for the rest of the test.
    let release = Arc::new(tokio::sync::Notify::new());
    let llm = Arc::new(GatedLlm::new(release));
    let server = Arc::new(server_with_generation_limit(llm.clone(), 1, false, 8192, 1).await);
    seed(
        &server,
        "fact",
        "Topic",
        "the wibbleflux service runs nightly.",
    )
    .await;
    let question = || AskArgs {
        question: "What does the wibbleflux service do?".to_string(),
        kind: None,
        scope: None,
        k: None,
        as_of: None,
        mode: None,
    };

    let holder = server.clone();
    let _first = tokio::spawn(async move { holder.ask(Parameters(question())).await });
    wait_until(|| llm.in_flight() == 1).await;

    // Act: the second call cannot acquire a permit within the 1s ask
    // timeout, since the first is held for the rest of the test.
    let answer = server.ask(Parameters(question())).await;

    // Assert: the fallback, not a hang. `_first` is left un-awaited and
    // un-notified; the test runtime drops it when the test ends.
    assert!(
        answer.contains("(synthesis unavailable"),
        "answer missing fallback marker: {answer}"
    );
    assert!(
        answer.contains("the wibbleflux service runs nightly."),
        "answer missing evidence content: {answer}"
    );
}

#[tokio::test]
async fn acquire_generation_permit_returns_ok_when_a_permit_is_available() {
    // Arrange: the default single-slot semaphore, untouched.
    let server = server_with(Arc::new(liam_model::MockLlm)).await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

    // Act
    let result = server.acquire_generation_permit(deadline).await;

    // Assert
    assert!(result.is_ok(), "expected a permit to be granted");
}

#[tokio::test]
async fn acquire_generation_permit_returns_err_when_the_semaphore_is_closed() {
    // Arrange: close the semaphore instead of exhausting it, mirroring a
    // shutdown path rather than ordinary contention.
    let server = server_with(Arc::new(liam_model::MockLlm)).await;
    server.generation_permits_handle().close();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

    // Act
    let result = server.acquire_generation_permit(deadline).await;

    // Assert
    assert_eq!(result.err(), Some("no generation slot is available"));
}

#[tokio::test(start_paused = true)]
async fn acquire_generation_permit_returns_err_when_the_deadline_is_already_past() {
    // Arrange: hold the sole permit directly so the callee's own acquire
    // attempt can never complete, then advance the paused clock past the
    // deadline instead of waiting on a real timer.
    let server = server_with(Arc::new(liam_model::MockLlm)).await;
    let _held = server
        .generation_permits_handle()
        .try_acquire_owned()
        .expect("sole permit should be free at test start");
    let deadline = tokio::time::Instant::now() + Duration::from_millis(50);
    tokio::time::advance(Duration::from_millis(100)).await;

    // Act
    let result = server.acquire_generation_permit(deadline).await;

    // Assert
    assert_eq!(
        result.err(),
        Some("timed out waiting for a generation slot")
    );
}

#[tokio::test]
async fn passes_sufficiency_check_returns_true_on_an_explicit_yes() {
    // Arrange
    let server = server_with(Arc::new(SufficiencyLlm {
        verdict: "YES",
        answer: "unused",
    }))
    .await;
    let evidence = vec![sufficiency_evidence()];
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

    // Act
    let sufficient = server
        .passes_sufficiency_check("Who is the mascot?", &evidence, deadline)
        .await;

    // Assert
    assert!(
        sufficient,
        "an explicit YES verdict must proceed to synthesis"
    );
}

#[tokio::test]
async fn passes_sufficiency_check_returns_false_on_an_explicit_no() {
    // Arrange
    let server = server_with(Arc::new(SufficiencyLlm {
        verdict: "NO",
        answer: "unused",
    }))
    .await;
    let evidence = vec![sufficiency_evidence()];
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

    // Act
    let sufficient = server
        .passes_sufficiency_check("Who is the mascot?", &evidence, deadline)
        .await;

    // Assert
    assert!(!sufficient, "an explicit NO verdict must refuse");
}

#[tokio::test]
async fn passes_sufficiency_check_fails_open_on_a_timeout() {
    // Arrange: a deadline that has already elapsed by the time the
    // never-returning `SlowLlm::complete` is polled.
    let server = server_with(Arc::new(SlowLlm)).await;
    let evidence = vec![sufficiency_evidence()];
    let deadline = tokio::time::Instant::now() + Duration::from_millis(50);

    // Act
    let sufficient = server
        .passes_sufficiency_check("Who is the mascot?", &evidence, deadline)
        .await;

    // Assert: a timeout is not an explicit refusal, so synthesis proceeds.
    assert!(sufficient, "a timeout must fail open");
}

#[tokio::test]
async fn passes_sufficiency_check_fails_open_on_an_llm_error() {
    // Arrange
    let server = server_with(Arc::new(FailingLlm)).await;
    let evidence = vec![sufficiency_evidence()];
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

    // Act
    let sufficient = server
        .passes_sufficiency_check("Who is the mascot?", &evidence, deadline)
        .await;

    // Assert: an LLM error is not an explicit refusal, so synthesis proceeds.
    assert!(sufficient, "an LLM error must fail open");
}

#[tokio::test]
async fn passes_sufficiency_check_fails_open_on_an_unparseable_reply() {
    // Arrange: a model that ignores "reply YES or NO" must not be read
    // as a refusal, or a chatty model would deny memory the store holds.
    let server = server_with(Arc::new(SufficiencyLlm {
        verdict: "Well, it depends on what you mean.",
        answer: "unused",
    }))
    .await;
    let evidence = vec![sufficiency_evidence()];
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

    // Act
    let sufficient = server
        .passes_sufficiency_check("Who is the mascot?", &evidence, deadline)
        .await;

    // Assert
    assert!(sufficient, "an unparseable verdict must fail open");
}

/// One evidence item shared by the `passes_sufficiency_check` tests
/// above: its content is unused by `SufficiencyLlm`/`FailingLlm`/
/// `SlowLlm`, only its presence in the prompt matters.
fn sufficiency_evidence() -> ask::Evidence {
    ask::Evidence::recorded(
        "fact",
        "Mascot",
        "The team mascot is a wombat named Pixel.",
        0,
    )
}

#[tokio::test]
async fn ask_falls_back_on_empty_answer() {
    // Arrange: llm "succeeds" but returns only whitespace.
    let server = server_with(Arc::new(EmptyLlm)).await;
    seed(
        &server,
        "fact",
        "Weather",
        "It rained on the picnic in Brooklyn.",
    )
    .await;

    // Act
    let answer = server
        .ask(Parameters(AskArgs {
            question: "What happened at the picnic".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert: a blank successful answer is treated as no synthesis, so
    // the answer is the evidence-backed fallback.
    assert!(
        answer.contains("(synthesis unavailable"),
        "answer missing fallback marker: {answer}"
    );
    assert!(
        answer.contains("It rained on the picnic in Brooklyn."),
        "answer missing evidence content: {answer}"
    );
}

#[tokio::test]
async fn ask_no_matches_returns_no_relevant_memory() {
    // Arrange: fresh server, nothing remembered.
    let server = server_with(Arc::new(liam_model::MockLlm)).await;

    // Act
    let answer = server
        .ask(Parameters(AskArgs {
            question: "Anything at all".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert
    assert_eq!(answer, "no relevant memory");
}
