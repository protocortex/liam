// SPDX-License-Identifier: Apache-2.0
//! The generation semaphore and the rolling window that entity synthesis
//! shares across the connections of one server.

use super::*;

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
    // Given the default of 0, with logs captured instead of dropped
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
            8192,
            0,
        )
    });

    // Then it floors to the smallest usable permit count, and no clamp warning
    // is logged for what is the default
    assert_eq!(server.generation_permits.available_permits(), 1);
    let log = String::from_utf8(captured.0.lock().expect("captured logs lock").clone())
        .expect("log output is utf8");
    assert!(!log.contains("clamping"), "unexpected clamp warning: {log}");
}

#[tokio::test]
async fn explicit_max_concurrent_generations_is_used_as_is() {
    // Given an explicit positive value, unrelated to the 0 sentinel
    let server = server_with_generation_limit(Arc::new(liam_model::MockLlm), 30, 8192, 5).await;

    // Then the semaphore is sized to exactly that value
    assert_eq!(server.generation_permits.available_permits(), 5);
}

#[tokio::test]
async fn generation_permits_handle_shares_the_same_semaphore_across_clones() {
    // Given two independently obtained handles to the same server
    let server = server_with_generation_limit(Arc::new(liam_model::MockLlm), 30, 8192, 1).await;
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
async fn entity_syntheses_on_two_connections_record_into_the_same_rolling_window() {
    // Arrange: clone the server the way the accept loop clones one per
    // connection.
    let server = server_with(Arc::new(liam_model::MockLlm)).await;
    let other_connection = server.clone();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let synthesis_llm = CountingLlm::new();

    // Act: one synthesis through each connection's handles.
    for connection in [&server, &other_connection] {
        synthesis::synthesize_entity(
            &synthesis_llm,
            &connection.aimd_handles(),
            &connection.rolling_window_handle(),
            deadline,
            "person",
            "Someone",
            &[],
            8192,
            64,
        )
        .await
        .expect("synthesis should succeed");
    }

    // Assert: both calls landed in the SAME window, not two independent ones,
    // which is what an accidentally per-connection window would show.
    assert_eq!(
        server
            .rolling_window_handle()
            .lock()
            .expect("window lock")
            .len(),
        2
    );
}
