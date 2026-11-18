// SPDX-License-Identifier: Apache-2.0
//! The timing samples `recall` and `ask` record.

use super::*;

/// Under a millisecond, to show timings keep their fraction.
const TICK: Duration = Duration::from_micros(250);

fn metered(server: MemoryServer) -> (MemoryServer, Arc<crate::metrics::sink::MemorySink>) {
    let clock = Arc::new(FixedClock::new(Millis(1_000)));
    let sink = Arc::new(crate::metrics::sink::MemorySink::new(clock.clone()));
    let metrics = Arc::new(
        Metrics::new(sink.clone(), clock, 2).with_monotonic(Arc::new(
            crate::metrics::monotonic::TickingMonotonic::new(TICK),
        )),
    );
    (server.with_metrics(metrics), sink)
}

/// Counts and feedback on, so the stages that need them have work to time.
fn with_usage_and_feedback(server: MemoryServer) -> MemoryServer {
    const TRACKED: usize = 100;
    const VERDICTS_PER_HOUR: u32 = 10;
    server
        .with_usage_recorder(Arc::new(UsageRecorder::new(TRACKED)))
        .with_feedback(VERDICTS_PER_HOUR, TRACKED)
}

fn reported(
    sink: &crate::metrics::sink::MemorySink,
    metric: crate::metrics::sink::MetricId,
) -> bool {
    use crate::metrics::sink::MetricsSink;
    sink.snapshot()
        .readings
        .iter()
        .any(|reading| reading.metric == metric)
}

#[tokio::test]
async fn recall_times_itself_and_the_arms_it_ran_and_not_ask() {
    use crate::metrics::sink::MetricId;
    // Arrange
    let (server, sink) = metered(plain_server().await);
    seed(&server, "fact", "One", "zorbnax note one").await;

    // Act
    server
        .recall(Parameters(RecallArgs {
            query: "zorbnax note".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert
    assert!(reported(&sink, MetricId::RecallLatencyP95Ms));
    assert!(reported(&sink, MetricId::ArmLatencyP95Ms));
    assert!(!reported(&sink, MetricId::AskLatencyP95Ms));
}

fn samples_of(
    sink: &crate::metrics::sink::MemorySink,
    metric: crate::metrics::sink::MetricId,
) -> u64 {
    use crate::metrics::sink::MetricsSink;
    sink.snapshot()
        .readings
        .iter()
        .filter(|reading| reading.metric == metric)
        .map(|reading| reading.samples)
        .sum()
}

fn latest(sink: &crate::metrics::sink::MemorySink, metric: crate::metrics::sink::MetricId) -> f64 {
    use crate::metrics::sink::MetricsSink;
    sink.snapshot()
        .readings
        .iter()
        .find(|reading| reading.metric == metric)
        .map_or(f64::NAN, |reading| reading.value)
}

#[tokio::test]
async fn recall_records_one_sample_per_stage_and_the_stages_fit_in_the_total() {
    use crate::metrics::sink::MetricId;
    // Arrange
    let (server, sink) = metered(with_usage_and_feedback(plain_server().await));
    seed(&server, "fact", "One", "zorbnax note one").await;
    let stages = [
        MetricId::StageEmbedP50Ms,
        MetricId::StageStoreSearchP50Ms,
        MetricId::StageRecordUsageP50Ms,
        MetricId::StageIssueHandlesP50Ms,
    ];

    // Act
    server
        .recall(Parameters(RecallArgs {
            query: "zorbnax note".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert
    for stage in stages {
        assert_eq!(samples_of(&sink, stage), 1, "{stage}");
    }
    assert_eq!(samples_of(&sink, MetricId::RecallLatencyP50Ms), 1);
    let staged: f64 = stages.into_iter().map(|stage| latest(&sink, stage)).sum();
    assert!(staged <= latest(&sink, MetricId::RecallLatencyP50Ms));
}

#[tokio::test]
async fn a_server_without_usage_or_feedback_records_no_such_stage_sample() {
    use crate::metrics::sink::MetricId;
    // Arrange
    let (server, sink) = metered(plain_server().await);
    seed(&server, "fact", "One", "zorbnax note one").await;

    // Act
    server
        .recall(Parameters(RecallArgs {
            query: "zorbnax note".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert
    assert_eq!(samples_of(&sink, MetricId::StageRecordUsageP50Ms), 0);
    assert_eq!(samples_of(&sink, MetricId::StageIssueHandlesP50Ms), 0);
    assert_eq!(samples_of(&sink, MetricId::StageEmbedP50Ms), 1);
}

#[tokio::test]
async fn ask_records_one_sample_per_stage() {
    use crate::metrics::sink::MetricId;
    // Arrange
    let llm = Arc::new(RecordingLlm::new(
        "The zorbnax gadget uses libSQL for storage [1].",
    ));
    let server = server_with(llm).await;
    let (server, sink) = metered(with_usage_and_feedback(server));
    seed(
        &server,
        "decision",
        "Storage engine",
        "The zorbnax gadget uses libSQL for storage.",
    )
    .await;
    let stages = [
        MetricId::StageEmbedP50Ms,
        MetricId::StageStoreSearchP50Ms,
        MetricId::StageRecordUsageP50Ms,
        MetricId::StageIssueHandlesP50Ms,
    ];

    // Act
    server
        .ask(Parameters(ask_args_for(
            "What storage engine does the zorbnax gadget use?",
        )))
        .await;

    // Assert
    for stage in stages {
        assert_eq!(samples_of(&sink, stage), 1, "{stage}");
    }
    assert_eq!(samples_of(&sink, MetricId::AskLatencyP50Ms), 1);
}

#[tokio::test]
async fn ask_times_itself() {
    use crate::metrics::sink::MetricId;
    // Arrange
    let (server, sink) = metered(plain_server().await);

    // Act
    server
        .ask(Parameters(AskArgs {
            question: "anything?".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert
    assert!(reported(&sink, MetricId::AskLatencyP50Ms));
    assert!(!reported(&sink, MetricId::RecallLatencyP50Ms));
}

/// Rewrites the recall and ask schema fixture from the live tool list.
/// Run it by hand with `--ignored` when an input schema changes on purpose.
#[tokio::test]
#[ignore = "writes the recall and ask schema fixture"]
async fn capture_recall_ask_schemas() {
    // Arrange
    let server = plain_server().await;

    // Act
    let schemas = recall_ask_schemas(&server);
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/mcp/fixtures/recall_ask_schemas.json"
    );
    std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap()).unwrap();
    std::fs::write(path, serde_json::to_string_pretty(&schemas).unwrap() + "\n").unwrap();

    // Assert
    assert!(schemas["recall"].is_object() && schemas["ask"].is_object());
}

#[tokio::test]
async fn recall_and_ask_list_the_schemas_clients_already_rely_on() {
    // Arrange
    let server = plain_server().await;
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/recall_ask_schemas.json"))
            .expect("fixture is JSON");

    // Act
    let listed = recall_ask_schemas(&server);

    // Assert
    assert_eq!(listed, fixture);
}

/// The `recall` and `ask` input schemas as listed by the server, by tool name.
fn recall_ask_schemas(server: &MemoryServer) -> serde_json::Value {
    let tools = server.tool_router.list_all();
    let mut out = serde_json::Map::new();
    for name in ["recall", "ask"] {
        let tool = tools
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} must be listed"));
        out.insert(
            name.to_string(),
            serde_json::to_value(&tool.input_schema).unwrap(),
        );
    }
    serde_json::Value::Object(out)
}
