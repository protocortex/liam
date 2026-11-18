// SPDX-License-Identifier: Apache-2.0
//! Rollbacks over a real database file and a real WAL: a regressed or failed
//! run is undone, concurrent client writes survive, a crash at any point of the
//! rollback is finished by the next run, and a rollback that cannot complete
//! says so instead of leaving the log and the store disagreeing.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use liam_store::{Millis, NewNode, RetentionPolicy};

use super::rollback::{MARKER_APPENDED, ROLLBACK_MARKED, ROLLBACK_PENDING, ROLLBACK_RESTORED};
use super::tests::*;
use super::*;

const INVALID_THRESHOLD: f32 = 2.0;
const CLIENT_NOTE: &str = "client note written during the run";
const RUN_NOTE: &str = "note written by the run's maintenance";
const SCRAP: &str = "scrap";
const WAIT: Duration = Duration::from_millis(300);
const POLL: Duration = Duration::from_millis(20);
const CLIENT: &str = "client:mcp";

/// Writes under the source the run hands it, as a tagged sweep or resynthesis would.
struct TaggedWrite;

#[async_trait]
impl Maintenance for TaggedWrite {
    async fn run(&self, store: &DefaultGraph, source: &str) -> anyhow::Result<()> {
        let written = NewNode::now("fact", "note", RUN_NOTE).with_producer(source);
        store.insert(written).await.unwrap();
        Ok(())
    }
}

/// Damages the snapshot while the run is going, so its checksum fails.
struct CorruptsSnapshot {
    snapshot_dir: PathBuf,
    calls: AtomicUsize,
}

#[async_trait]
impl Maintenance for CorruptsSnapshot {
    async fn run(&self, _store: &DefaultGraph, _source: &str) -> anyhow::Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        for run in std::fs::read_dir(&self.snapshot_dir).unwrap().flatten() {
            for entry in std::fs::read_dir(run.path()).unwrap().flatten() {
                if entry.path().extension().is_some_and(|ext| ext == "db") {
                    std::fs::write(entry.path(), b"not the snapshot").unwrap();
                }
            }
        }
        Ok(())
    }
}

/// What a client does once, between the snapshot and the probe check.
enum Client {
    Writes(&'static str),
    SweepsScrap,
}

/// Probes that regress once the baseline is taken, and let a client act when
/// they first see the run past its baseline.
struct ProbesWithClient {
    inner: Arc<Probes>,
    client: Client,
    acted: AtomicBool,
}

#[async_trait]
impl ProbeRunner for ProbesWithClient {
    async fn run(&self, store: &DefaultGraph, probes: &[Probe]) -> Result<ProbeReport, ProbeError> {
        let past_baseline = self
            .inner
            .regress_once_present
            .as_ref()
            .is_some_and(|p| p.exists());
        if past_baseline && !self.acted.swap(true, Ordering::SeqCst) {
            match self.client {
                Client::Writes(text) => {
                    store.insert(note(text)).await.unwrap();
                }
                Client::SweepsScrap => {
                    let policy = RetentionPolicy::keep(SCRAP, Millis(0)).without_reclaim();
                    store.gc(&policy).await.unwrap();
                }
            }
        }
        self.inner.run(store, probes).await
    }
}

fn client_during(env: &Env, client: Client) -> Arc<ProbesWithClient> {
    Arc::new(ProbesWithClient {
        inner: env.regressing(),
        client,
        acted: AtomicBool::new(false),
    })
}

#[tokio::test]
async fn a_probe_regression_restores_the_store_and_keeps_client_writes_only() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let (before, no_links) = state(&shared).await;
    let probes = client_during(&env, Client::Writes(CLIENT_NOTE));
    let run = env.consolidator(Arc::new(TaggedWrite), probes);

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::RolledBack));
    let (contents, links) = state(&shared).await;
    let mut expected = before;
    expected.insert(CLIENT_NOTE.to_owned());
    assert_eq!(contents, expected, "the run's tagged write is cancelled");
    assert_eq!(links, no_links, "the run's own edges are gone");
    assert_eq!(env.snapshot_files(), 0);
}

/// One of the tick's write paths, run under the source the run hands it.
enum Path {
    Sweep,
    Repair,
    Resynthesis,
    Clusters,
}

/// Runs one path as the daemon's tick does and records that it did work.
struct TickPath {
    path: Path,
    worked: AtomicBool,
}

#[async_trait]
impl Maintenance for TickPath {
    async fn run(&self, store: &DefaultGraph, source: &str) -> anyhow::Result<()> {
        let worked = match self.path {
            Path::Sweep => {
                let policy = RetentionPolicy::keep(SCRAP, Millis(0)).without_reclaim();
                store.gc_from(&policy, source).await.unwrap().nodes_removed > 0
            }
            Path::Repair => store.repair_superseded_mentions_from(source).await.unwrap() > 0,
            Path::Resynthesis => {
                let mut page = NewNode::entity("person", "Kim").with_producer(source);
                page.content = RESYNTHESIZED.to_owned();
                store.upsert_by(page).await.unwrap();
                true
            }
            Path::Clusters => store.refresh_communities().await.unwrap(),
        };
        self.worked.store(worked, Ordering::SeqCst);
        Ok(())
    }
}

const ORIGINAL_PAGE: &str = "page before the run";
const RESYNTHESIZED: &str = "page written by the run";

/// A store with the fixtures every path needs: a scrap node to sweep, an
/// entity a fact mentions that a client then superseded, and two linked notes.
async fn tick_fixtures(env: &Env) -> SharedStore {
    let shared = env.seeded().await;
    {
        let store = shared.get().await.unwrap();
        let scrap = NewNode::now(SCRAP, "scrap", "scrap to sweep").with_producer(CLIENT);
        store.insert(scrap).await.unwrap();
        let mut page = NewNode::entity("person", "Kim").with_producer(CLIENT);
        page.content = ORIGINAL_PAGE.to_owned();
        let entity = store.upsert_by(page).await.unwrap();
        let fact = store.insert(note("Kim likes tea")).await.unwrap();
        store
            .relate(&entity, &fact, liam_store::relation::MENTIONS)
            .await
            .unwrap();
        store
            .supersede(&fact, note("Kim likes coffee"))
            .await
            .unwrap();
        let first = store.insert(note("linked one")).await.unwrap();
        let second = store.insert(note("linked two")).await.unwrap();
        store
            .relate(&first, &second, liam_store::relation::MENTIONS)
            .await
            .unwrap();
    }
    shared
}

/// Rolls back a run whose maintenance used `path`, with a client write landing
/// during it, and checks that the path did work, that nothing of it is left and
/// that the client write stands.
async fn assert_a_rollback_removes(path: Path) {
    // Arrange
    let env = Env::new();
    let shared = tick_fixtures(&env).await;
    let (before, links_before) = state(&shared).await;
    let tick = Arc::new(TickPath {
        path,
        worked: AtomicBool::new(false),
    });
    let probes = client_during(&env, Client::Writes(CLIENT_NOTE));
    let run = env.consolidator(tick.clone(), probes);

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::RolledBack));
    assert!(tick.worked.load(Ordering::SeqCst), "the path did work");
    let (contents, links) = state(&shared).await;
    let mut expected = before;
    expected.insert(CLIENT_NOTE.to_owned());
    assert_eq!(contents, expected, "only the client write is new");
    assert_eq!(links, links_before);
}

#[tokio::test]
async fn a_rollback_removes_what_the_sweep_wrote() {
    assert_a_rollback_removes(Path::Sweep).await;
}

#[tokio::test]
async fn a_rollback_removes_what_the_provenance_repair_wrote() {
    assert_a_rollback_removes(Path::Repair).await;
}

#[tokio::test]
async fn a_rollback_removes_the_page_resynthesis_wrote_and_reopens_the_old_one() {
    assert_a_rollback_removes(Path::Resynthesis).await;
}

#[tokio::test]
async fn a_rollback_leaves_the_clusters_stale_so_they_are_computed_again() {
    // Arrange
    let env = Env::new();
    let shared = tick_fixtures(&env).await;
    let tick = Arc::new(TickPath {
        path: Path::Clusters,
        worked: AtomicBool::new(false),
    });
    let run = env.consolidator(tick.clone(), env.regressing());

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::RolledBack));
    assert!(
        tick.worked.load(Ordering::SeqCst),
        "the run computed clusters"
    );
    let recomputed = shared
        .get()
        .await
        .unwrap()
        .refresh_communities()
        .await
        .unwrap();
    assert!(recomputed, "the restore took the run's assignments away");
}

#[tokio::test]
async fn a_rolled_back_store_equals_one_rebuilt_from_the_log() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let probes = client_during(&env, Client::Writes(CLIENT_NOTE));
    let run = env.consolidator(Arc::new(TaggedWrite), probes);
    run.consolidate(&shared).await.unwrap();
    let rolled_back = state(&shared).await;

    // Act
    let fresh = open_on(&env.dir.path().join("rebuilt.db"), &env.log, &env.embedder)
        .await
        .unwrap();
    fresh.catch_up().await.unwrap();
    let rebuilt = state(&SharedStore::new(fresh)).await;

    // Assert
    assert_eq!(rebuilt, rolled_back);
}

#[tokio::test]
async fn a_rolled_back_run_stays_gone_when_the_store_is_opened_again() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let run = env.consolidator(maintenance(), env.regressing());
    run.consolidate(&shared).await.unwrap();
    let after_rollback = state(&shared).await;
    drop(shared);

    // Act
    let reopened = env.open().await;

    // Assert
    assert_eq!(state(&reopened).await, after_rollback);
    assert!(after_rollback.1.is_empty());
}

#[tokio::test]
async fn a_failed_merge_rolls_the_run_back() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let before = state(&shared).await;
    let mut run = env.consolidator(maintenance(), honest());
    run.settings.dedup_similarity_threshold = INVALID_THRESHOLD;

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::RolledBack));
    assert_eq!(state(&shared).await, before);
}

#[tokio::test]
async fn a_client_sweep_of_a_row_the_merge_linked_replays_after_the_rollback() {
    // Arrange: the duplicate is the only "scrap" node, and the run links it.
    let env = Env::new();
    let shared = env.open().await;
    {
        let store = shared.get().await.unwrap();
        for i in 0..TOPICS {
            store.insert(note(&topic(i))).await.unwrap();
        }
        let scrap = NewNode::now(SCRAP, DUPLICATE, DUPLICATE).with_producer(CLIENT);
        store.insert(scrap).await.unwrap();
    }
    let probes = client_during(&env, Client::SweepsScrap);
    let run = env.consolidator(maintenance(), probes);

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::RolledBack));
    let (contents, links) = state(&shared).await;
    assert_eq!(
        contents.len(),
        TOPICS,
        "the client's sweep of the duplicate stands"
    );
    assert!(links.is_empty(), "the run's edge to it is gone");
}

#[tokio::test]
async fn a_corrupt_snapshot_ends_the_run_restore_failed_with_no_marker_and_the_store_unchanged() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let damage = Arc::new(CorruptsSnapshot {
        snapshot_dir: env.dir.path().join("snapshots"),
        calls: AtomicUsize::new(0),
    });
    let run = env.consolidator(damage.clone(), env.regressing());

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::RestoreFailed));
    assert_eq!(damage.calls.load(Ordering::SeqCst), 1, "no step ran again");
    assert!(!manifest.is_complete(ROLLBACK_PENDING));
    assert!(
        env.snapshot_files() > 0,
        "the snapshot is kept for inspection"
    );
    let live = state(&shared).await;
    assert_eq!(live.1, duplicate_links(), "the store keeps the run's edges");
    let fresh = open_on(&env.dir.path().join("check.db"), &env.log, &env.embedder)
        .await
        .unwrap();
    fresh.catch_up().await.unwrap();
    assert_eq!(
        state(&SharedStore::new(fresh)).await,
        live,
        "no marker is in the log, so a replay agrees with the live store"
    );
}

#[tokio::test]
async fn a_reopen_that_fails_leaves_the_slot_empty_and_the_run_restore_failed() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    env.reopen_fails.store(true, Ordering::SeqCst);
    let run = env.consolidator(maintenance(), env.regressing());

    // Act
    let result = run.consolidate(&shared).await;

    // Assert
    assert!(result.is_err());
    assert_eq!(env.manifest().outcome, Some(RunOutcome::RestoreFailed));
    assert!(shared.get().await.is_err());
}

#[tokio::test]
async fn a_replay_that_fails_after_the_restore_ends_the_run_restore_failed() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    env.poison_on_reopen.store(true, Ordering::SeqCst);
    let run = env.consolidator(maintenance(), env.regressing());

    // Act
    let manifest = run.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::RestoreFailed));
    assert!(
        shared.get().await.is_ok(),
        "the restored store is in the slot"
    );
}

#[tokio::test]
async fn a_marker_that_cannot_be_appended_is_retried_by_the_next_run_which_never_goes_forward() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let before = state(&shared).await;
    env.marker_fails.store(true, Ordering::SeqCst);
    let first = env.consolidator(maintenance(), env.regressing());
    assert!(first.consolidate(&shared).await.is_err());
    assert_eq!(
        state(&shared).await.1,
        duplicate_links(),
        "nothing was undone yet"
    );
    env.marker_fails.store(false, Ordering::SeqCst);

    // Act: honest probes would commit a run that went forward.
    let second = env.consolidator(maintenance(), honest());
    let manifest = second.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::RolledBack));
    assert_eq!(state(&shared).await, before);
}

#[tokio::test]
async fn a_crash_at_any_forward_step_of_a_regressed_run_ends_rolled_back() {
    for point in STEPS.iter().take(STEPS.len() - 1) {
        // Arrange
        let env = Env::new();
        let shared = env.seeded().await;
        let (before, no_links) = state(&shared).await;
        let mut crashing = env.consolidator(maintenance(), env.regressing());
        crashing.crash_after = Some(point);

        // Act
        let crashed = crashing.consolidate(&shared).await;
        let manifest = env
            .consolidator(maintenance(), env.regressing())
            .consolidate(&shared)
            .await
            .unwrap_or_else(|error| panic!("after {point}: {error:#}"));

        // Assert
        assert!(crashed.is_err(), "crash after {point}");
        assert_eq!(manifest.outcome, Some(RunOutcome::RolledBack), "{point}");
        assert_eq!(state(&shared).await, (before, no_links), "after {point}");
    }
}

#[tokio::test]
async fn a_crash_during_the_rollback_is_finished_by_the_next_run_even_with_probes_that_hold() {
    for point in [
        ROLLBACK_PENDING,
        MARKER_APPENDED,
        ROLLBACK_MARKED,
        ROLLBACK_RESTORED,
    ] {
        // Arrange
        let env = Env::new();
        let shared = env.seeded().await;
        let (before, no_links) = state(&shared).await;
        let mut crashing = env.consolidator(maintenance(), env.regressing());
        crashing.crash_after = Some(point);

        // Act
        let crashed = crashing.consolidate(&shared).await;
        let manifest = env
            .consolidator(maintenance(), honest())
            .consolidate(&shared)
            .await
            .unwrap_or_else(|error| panic!("after {point}: {error:#}"));

        // Assert
        assert!(crashed.is_err(), "crash at {point}");
        assert_eq!(manifest.outcome, Some(RunOutcome::RolledBack), "{point}");
        assert_eq!(state(&shared).await, (before, no_links), "after {point}");
    }
}

async fn crash_after_dedup(
    env: &Env,
) -> (SharedStore, (std::collections::BTreeSet<String>, Links)) {
    let shared = env.seeded().await;
    let before = state(&shared).await;
    let mut crashing = env.consolidator(maintenance(), honest());
    crashing.crash_after = Some("dedup");
    assert!(crashing.consolidate(&shared).await.is_err());
    assert_eq!(state(&shared).await.1, duplicate_links());
    (shared, before)
}

#[tokio::test]
async fn a_resume_under_another_embedder_rolls_the_unfinished_run_back() {
    // Arrange
    let env = Env::new();
    let (shared, before) = crash_after_dedup(&env).await;
    let mut other = env.consolidator(maintenance(), honest());
    other.identity.embedder.model = "other-model".into();

    // Act
    let refused = other.consolidate(&shared).await.unwrap();
    let fresh = env
        .consolidator(maintenance(), honest())
        .consolidate(&shared)
        .await
        .unwrap();

    // Assert
    assert_eq!(refused.outcome, Some(RunOutcome::RolledBack));
    assert_ne!(refused.run_id, fresh.run_id);
    assert_eq!(fresh.outcome, Some(RunOutcome::Committed));
    assert_eq!(before.1.len(), 0);
}

#[tokio::test]
async fn a_resume_under_another_llm_rolls_the_unfinished_run_back() {
    // Arrange
    let env = Env::new();
    let (shared, before) = crash_after_dedup(&env).await;
    let mut other = env.consolidator(maintenance(), honest());
    other.identity.llm_model = "llm/other.gguf".into();

    // Act
    let refused = other.consolidate(&shared).await.unwrap();

    // Assert
    assert_eq!(refused.outcome, Some(RunOutcome::RolledBack));
    assert_eq!(state(&shared).await, before);
}

#[tokio::test]
async fn a_finished_run_is_followed_by_a_fresh_one() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let first = env
        .consolidator(maintenance(), honest())
        .consolidate(&shared)
        .await
        .unwrap();

    // Act
    let second = env
        .consolidator(maintenance(), honest())
        .consolidate(&shared)
        .await
        .unwrap();

    // Assert
    assert_ne!(first.run_id, second.run_id);
    assert_eq!(second.outcome, Some(RunOutcome::Committed));
}

#[tokio::test]
async fn a_rollback_waits_for_a_request_in_flight_and_then_completes() {
    // Arrange
    let env = Env::new();
    let shared = Arc::new(env.seeded().await);
    let before = state(&shared).await;
    let reader = shared.get().await.unwrap();
    let run = Arc::new(env.consolidator(maintenance(), env.regressing()));
    let task = {
        let (run, shared) = (run.clone(), shared.clone());
        tokio::spawn(async move { run.consolidate(&shared).await })
    };
    while !shared.is_contended() {
        tokio::time::sleep(POLL).await;
    }

    // Act
    tokio::time::sleep(WAIT).await;
    let waiting = !task.is_finished();
    drop(reader);
    let manifest = task.await.unwrap().unwrap();

    // Assert
    assert!(waiting, "the restore waited for the open request");
    assert_eq!(manifest.outcome, Some(RunOutcome::RolledBack));
    assert_eq!(state(&shared).await, before);
}

/// What a run decided, by content: the probes it drew and the pairs it linked.
async fn decisions(seed: u64) -> (Vec<String>, Links) {
    let env = Env::new();
    let probes = honest();
    let run = env
        .consolidator(maintenance(), probes.clone())
        .with_seed(seed);
    let shared = env.seeded().await;
    run.consolidate(&shared).await.unwrap();
    let drawn = probes.first_queries.lock().unwrap().clone();
    (drawn, state(&shared).await.1)
}

#[tokio::test]
async fn two_runs_from_one_seed_make_the_same_decisions_by_content() {
    // Act
    let first = decisions(SEED).await;
    let second = decisions(SEED).await;

    // Assert
    assert_eq!(first.0.len(), PROBE_COUNT);
    assert_eq!(first.1, duplicate_links());
    assert_eq!(first, second);
}

#[tokio::test]
async fn another_seed_draws_other_probes_and_merges_alike() {
    // Act
    let first = decisions(SEED).await;
    let second = decisions(SEED + 1).await;

    // Assert
    assert_ne!(first.0, second.0);
    assert_eq!(first.1, second.1);
}

#[tokio::test]
async fn a_rollback_over_a_store_held_elsewhere_fails_and_leaves_the_store_in_its_slot() {
    // Arrange: a slot whose store another handle also holds
    let env = Env::new();
    let held = Arc::new(
        open_on(&env.database_path(), &env.log, &env.embedder)
            .await
            .unwrap(),
    );
    held.insert(note(CLIENT_NOTE)).await.unwrap();
    let shared = SharedStore::from_arc(held.clone());
    let run = env.consolidator(maintenance(), env.regressing());

    // Act
    let error = run.consolidate(&shared).await.unwrap_err();

    // Assert
    assert!(
        error.to_string().contains("held outside its slot"),
        "{error}"
    );
    assert!(
        shared.get().await.is_ok(),
        "the store stays available to clients"
    );
}

/// Sweeps one kind of node, as a retention rule would, under the run's source
/// or, when `anonymous`, as someone else's write.
struct SweepsKind {
    kind: &'static str,
    anonymous: bool,
}

#[async_trait]
impl Maintenance for SweepsKind {
    async fn run(&self, store: &DefaultGraph, source: &str) -> anyhow::Result<()> {
        tokio::time::sleep(Duration::from_millis(5)).await;
        let policy = RetentionPolicy::keep(self.kind, Millis(0)).without_reclaim();
        if self.anonymous {
            store.gc(&policy).await.unwrap();
        } else {
            store.gc_from(&policy, source).await.unwrap();
        }
        Ok(())
    }
}

/// Every node is probed, since the store holds fewer than this.
const PROBE_ALL: usize = 50;

async fn seeded_with_scrap(env: &Env) -> SharedStore {
    let shared = env.seeded().await;
    let store = shared.get().await.unwrap();
    store
        .insert(NewNode::now(SCRAP, "scrap note", "scrap note").with_producer(CLIENT))
        .await
        .unwrap();
    drop(store);
    shared
}

fn sweeping(env: &Env, kind: &'static str, anonymous: bool) -> Consolidator {
    env.consolidator_drawing(
        Arc::new(SweepsKind { kind, anonymous }),
        honest(),
        ConsolidationHookSet::default(),
        PROBE_ALL,
    )
}

#[tokio::test]
async fn a_probed_node_the_runs_own_sweep_removed_does_not_fail_the_run() {
    // Arrange
    let env = Env::new();
    let shared = seeded_with_scrap(&env).await;

    // Act
    let manifest = sweeping(&env, SCRAP, false)
        .consolidate(&shared)
        .await
        .unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::Committed));
    assert!(!state(&shared).await.0.contains("scrap note"));
}

#[tokio::test]
async fn a_probed_node_that_vanished_without_the_runs_sweep_fails_the_run() {
    // Arrange
    let env = Env::new();
    let shared = seeded_with_scrap(&env).await;
    let (mut expected, _) = state(&shared).await;
    expected.remove("scrap note");

    // Act
    let manifest = sweeping(&env, SCRAP, true)
        .consolidate(&shared)
        .await
        .unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::RolledBack));
    assert_eq!(
        state(&shared).await.0,
        expected,
        "a write that is not the run's stands after the rollback"
    );
}

#[tokio::test]
async fn a_sweep_that_leaves_no_probe_to_check_fails_the_run() {
    // Arrange: every node is old enough to be swept
    let env = Env::new();
    let shared = env.seeded().await;
    let (before, _) = state(&shared).await;

    // Act
    let manifest = sweeping(&env, "fact", false)
        .consolidate(&shared)
        .await
        .unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::RolledBack));
    assert_eq!(state(&shared).await.0, before, "the sweep was undone");
}

#[tokio::test]
async fn a_pending_rollback_is_finished_before_the_store_is_served() {
    // Arrange: a run that crashed right after committing to its rollback
    let env = Env::new();
    let shared = env.seeded().await;
    let (before, no_links) = state(&shared).await;
    let mut crashing = env.consolidator(Arc::new(TaggedWrite), env.regressing());
    crashing.crash_after = Some(ROLLBACK_PENDING);
    assert!(crashing.consolidate(&shared).await.is_err());

    // Act
    let manifest = env
        .consolidator(maintenance(), honest())
        .finish_pending_rollback(&shared)
        .await
        .unwrap();

    // Assert
    assert_eq!(manifest.unwrap().outcome, Some(RunOutcome::RolledBack));
    assert_eq!(state(&shared).await, (before, no_links));
}

#[tokio::test]
async fn a_run_that_crashed_going_forward_is_left_for_the_next_call_to_resume() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let mut crashing = env.consolidator(maintenance(), honest());
    crashing.crash_after = Some("dedup");
    assert!(crashing.consolidate(&shared).await.is_err());
    let run = env.consolidator(maintenance(), honest());

    // Act
    let reconciled = run.finish_pending_rollback(&shared).await.unwrap();

    // Assert
    assert!(reconciled.is_none(), "a forward run is not touched");
    assert!(run.unfinished_run().unwrap().is_some());
}

#[tokio::test]
async fn a_store_with_no_run_on_disk_has_no_unfinished_run() {
    let env = Env::new();
    let shared = env.seeded().await;
    let run = env.consolidator(maintenance(), honest());

    assert!(run.unfinished_run().unwrap().is_none());
    assert!(run
        .finish_pending_rollback(&shared)
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn a_new_run_is_refused_while_the_last_one_ended_restore_failed() {
    // Arrange: a corrupt snapshot ends a run restore failed and is kept
    let env = Env::new();
    let shared = env.seeded().await;
    let damage = Arc::new(CorruptsSnapshot {
        snapshot_dir: env.dir.path().join("snapshots"),
        calls: AtomicUsize::new(0),
    });
    let failed = env
        .consolidator(damage, env.regressing())
        .consolidate(&shared)
        .await
        .unwrap();
    let kept = env.snapshot_files();

    // Act
    let refused = env
        .consolidator(maintenance(), honest())
        .consolidate(&shared)
        .await
        .unwrap_err();

    // Assert
    assert_eq!(failed.outcome, Some(RunOutcome::RestoreFailed));
    let message = format!("{refused:#}");
    assert!(message.contains(&failed.run_id), "{message}");
    assert!(message.contains("manual inspection"), "{message}");
    assert!(message.contains("remove"), "{message}");
    assert_eq!(env.snapshot_files(), kept, "the snapshot is not swept");
}

#[tokio::test]
async fn a_run_resumes_after_the_operator_clears_a_failed_restore() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;
    let damage = Arc::new(CorruptsSnapshot {
        snapshot_dir: env.dir.path().join("snapshots"),
        calls: AtomicUsize::new(0),
    });
    env.consolidator(damage, env.regressing())
        .consolidate(&shared)
        .await
        .unwrap();
    std::fs::remove_file(manifest_path_for(&env.database_path())).unwrap();

    // Act
    let manifest = env
        .consolidator(maintenance(), honest())
        .consolidate(&shared)
        .await
        .unwrap();

    // Assert
    assert_eq!(manifest.outcome, Some(RunOutcome::Committed));
}

/// Fails the maintenance step the way a stopping daemon does.
struct Interrupted;

#[async_trait]
impl Maintenance for Interrupted {
    async fn run(&self, _store: &DefaultGraph, _source: &str) -> anyhow::Result<()> {
        anyhow::bail!("the daemon is stopping")
    }
}

#[tokio::test]
async fn a_maintenance_step_that_errors_is_not_recorded_and_the_next_call_resumes_at_it() {
    // Arrange
    let env = Env::new();
    let shared = env.seeded().await;

    // Act
    let stopped = env
        .consolidator(Arc::new(Interrupted), honest())
        .consolidate(&shared)
        .await;
    let recorded = RunManifest::load(&manifest_path_for(&env.database_path()))
        .unwrap()
        .unwrap();
    let ran = maintenance();
    let resumed = env
        .consolidator(ran.clone(), honest())
        .consolidate(&shared)
        .await
        .unwrap();

    // Assert
    assert!(stopped.is_err());
    assert!(!recorded.is_complete("maintenance"));
    assert!(recorded.is_complete("dedup"), "earlier steps stay recorded");
    assert!(recorded.outcome.is_none());
    assert_eq!(resumed.outcome, Some(RunOutcome::Committed));
    assert_eq!(resumed.run_id, recorded.run_id, "the same run resumed");
    assert_eq!(ran.runs.load(Ordering::SeqCst), 1);
}
