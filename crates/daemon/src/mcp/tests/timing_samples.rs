// SPDX-License-Identifier: Apache-2.0
//! The timing samples `recall` records.

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
async fn recall_times_itself_and_the_arms_it_ran() {
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

/// Rewrites the recall schema fixture from the live tool list.
/// Run it by hand with `--ignored` when an input schema changes on purpose.
#[tokio::test]
#[ignore = "writes the recall schema fixture"]
async fn capture_recall_schemas() {
    // Arrange
    let server = plain_server().await;

    // Act
    let schemas = recall_schemas(&server);
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/mcp/fixtures/recall_schema.json"
    );
    std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap()).unwrap();
    std::fs::write(path, serde_json::to_string_pretty(&schemas).unwrap() + "\n").unwrap();

    // Assert
    assert!(schemas["recall"].is_object());
}

#[tokio::test]
async fn recall_lists_the_schema_clients_already_rely_on() {
    // Arrange
    let server = plain_server().await;
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/recall_schema.json"))
            .expect("fixture is JSON");

    // Act
    let listed = recall_schemas(&server);

    // Assert
    assert_eq!(listed, fixture);
}

/// The `recall` input schema as listed by the server, by tool name.
fn recall_schemas(server: &MemoryServer) -> serde_json::Value {
    let tools = server.tool_router.list_all();
    let tool = tools
        .iter()
        .find(|t| t.name == "recall")
        .expect("recall must be listed");
    serde_json::json!({ "recall": tool.input_schema })
}
