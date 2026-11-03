// SPDX-License-Identifier: Apache-2.0
//! What a delete does about requests already sent to a hosted provider. The
//! provider holds its own copy, so the delete asks it to cancel and delete what
//! it can, and tells the operator about every copy it cannot reach.

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use liam_store::{LineageHits, LineageStore, LineageTarget, PurgeReport, Residual, Result};

use super::batch_store::{BatchState, BatchStore};
use super::files::{failed, hits, residual};
use super::job::JobStore;
use super::send_ledger::SendLedger;

const BATCH_STORE: &str = "hosted_batches";
const BATCH_ENTRY: &str = "batch_entry";
const SEND_ROW: &str = "send_row";

/// Whether the provider still holds the batch's requests once it was cancelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderCopies {
    /// The provider's delete endpoint removed them.
    #[cfg_attr(not(feature = "hosted"), allow(dead_code))]
    Deleted,
    /// The provider has no delete endpoint, so its copy ages out under its own
    /// retention terms.
    Kept,
}

/// Cancels a batch at its provider and deletes what the provider lets it.
#[async_trait]
pub trait BatchCanceller: Send + Sync {
    /// Stops the batch and removes its files where the provider has an endpoint
    /// for that. Safe to call again for a batch it already cancelled.
    async fn cancel(&self, provider: &str, batch_id: &str) -> anyhow::Result<ProviderCopies>;
}

/// Reaches no provider: every copy is reported as kept. Used where no hosted
/// transport is wired.
pub struct NoProviderDelete;

#[async_trait]
impl BatchCanceller for NoProviderDelete {
    async fn cancel(&self, _provider: &str, _batch_id: &str) -> anyhow::Result<ProviderCopies> {
        Ok(ProviderCopies::Kept)
    }
}

/// The canceller a delete under `config` uses: the hosted providers when the
/// build has them and `[hosted]` is on, otherwise none.
#[cfg(feature = "hosted")]
pub(super) fn canceller_for(config: &crate::config::Config) -> Arc<dyn BatchCanceller> {
    if config.hosted.enabled {
        crate::hosted::canceller(config)
    } else {
        Arc::new(NoProviderDelete)
    }
}

#[cfg(not(feature = "hosted"))]
pub(super) fn canceller_for(_config: &crate::config::Config) -> Arc<dyn BatchCanceller> {
    Arc::new(NoProviderDelete)
}

/// The batch table and the send ledger as one lineage store.
pub struct BatchLineage {
    batches: BatchStore,
    ledger: SendLedger,
    jobs: JobStore,
    canceller: Arc<dyn BatchCanceller>,
}

impl BatchLineage {
    pub fn new(
        batches: BatchStore,
        ledger: SendLedger,
        jobs: JobStore,
        canceller: Arc<dyn BatchCanceller>,
    ) -> Self {
        Self {
            batches,
            ledger,
            jobs,
            canceller,
        }
    }
}

fn events_of(target: &LineageTarget) -> HashSet<String> {
    target.event_ids.iter().cloned().collect()
}

fn notice_for(provider: &str) -> String {
    format!(
        "{provider} was sent requests built from the deleted data and may keep copies under its \
         own retention terms; ask the provider to delete them"
    )
}

#[async_trait]
impl LineageStore for BatchLineage {
    fn name(&self) -> &str {
        BATCH_STORE
    }

    async fn find(&self, target: &LineageTarget) -> Result<LineageHits> {
        let events = events_of(target);
        let entries = self
            .batches
            .citing(&events)
            .map_err(|e| failed(BATCH_STORE, e))?;
        let rows = self
            .ledger
            .citing(&events)
            .map_err(|e| failed(BATCH_STORE, e))?;
        let mut found = hits(BATCH_ENTRY, entries.len());
        found.rows.extend(hits(SEND_ROW, rows.len()).rows);
        Ok(found)
    }

    /// Cancels each batch holding a deleted request, records the notices for
    /// the copies a provider keeps, and only then drops the ledger rows and
    /// the table entries, so a crash in between reruns with the evidence intact.
    async fn purge(&self, target: &LineageTarget) -> Result<PurgeReport> {
        let events = events_of(target);
        let entries = self.batches.load().map_err(|e| failed(BATCH_STORE, e))?;
        let rows = self.ledger.load().map_err(|e| failed(BATCH_STORE, e))?;
        let hit_batches: BTreeSet<(String, String)> = entries
            .iter()
            .filter(|entry| entry.cites(&events))
            .map(|entry| (entry.provider.clone(), entry.batch_id.clone()))
            .collect();
        let mut kept_by: BTreeSet<String> = BTreeSet::new();
        let mut covered_keys: HashSet<String> = HashSet::new();
        for (provider, batch_id) in &hit_batches {
            let copies = self
                .canceller
                .cancel(provider, batch_id)
                .await
                .map_err(|e| failed(BATCH_STORE, e))?;
            match copies {
                ProviderCopies::Deleted => covered_keys.extend(
                    entries
                        .iter()
                        .filter(|entry| &entry.batch_id == batch_id)
                        .map(|entry| entry.cache_key.clone()),
                ),
                ProviderCopies::Kept => {
                    kept_by.insert(provider.clone());
                }
            }
        }
        let sent_rows: Vec<_> = rows.iter().filter(|row| row.cites(&events)).collect();
        kept_by.extend(
            sent_rows
                .iter()
                .filter(|row| !covered_keys.contains(&row.cache_key))
                .map(|row| row.provider.clone()),
        );
        let notices: Vec<String> = kept_by.iter().map(|p| notice_for(p)).collect();
        if !notices.is_empty() {
            self.jobs
                .add_notices(&target.job_id, &notices)
                .map_err(|e| failed(BATCH_STORE, e))?;
        }

        if !sent_rows.is_empty() {
            let left: Vec<_> = rows.iter().filter(|r| !r.cites(&events)).cloned().collect();
            self.ledger
                .replace(&left)
                .map_err(|e| failed(BATCH_STORE, e))?;
        }
        let cancelled: BTreeSet<&str> = hit_batches.iter().map(|(_, id)| id.as_str()).collect();
        let mut dropped = 0;
        let mut table = Vec::with_capacity(entries.len());
        for mut entry in entries {
            if entry.cites(&events) {
                dropped += 1;
                continue;
            }
            // Its batch is cancelled, so a survivor is sent again without the
            // deleted request.
            if entry.state.is_live() && cancelled.contains(entry.batch_id.as_str()) {
                entry.state = BatchState::Cancelled;
            }
            table.push(entry);
        }
        if dropped > 0 {
            self.batches
                .replace(table)
                .map_err(|e| failed(BATCH_STORE, e))?;
        }
        Ok(PurgeReport {
            removed: dropped + sent_rows.len(),
        })
    }

    async fn verify(&self, target: &LineageTarget) -> Result<Vec<Residual>> {
        let events = events_of(target);
        let entries = self
            .batches
            .citing(&events)
            .map_err(|e| failed(BATCH_STORE, e))?;
        let rows = self
            .ledger
            .citing(&events)
            .map_err(|e| failed(BATCH_STORE, e))?;
        let mut found: Vec<Residual> = entries
            .iter()
            .map(|entry| residual(BATCH_STORE, BATCH_ENTRY, entry.cache_key.clone()))
            .collect();
        found.extend(
            rows.iter()
                .map(|row| residual(BATCH_STORE, SEND_ROW, row.cache_key.clone())),
        );
        Ok(found)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::deletion::batch_store::tests::entry;
    use crate::deletion::batch_store::BatchEntry;
    use crate::deletion::job::{DeleteJob, JobStore};
    use crate::deletion::send_ledger::tests::row;
    use crate::deletion::target::DeleteTarget;

    const JOB: &str = "job-1";

    /// Answers every cancel the same way and remembers what it was asked.
    struct Scripted {
        answer: ProviderCopies,
        calls: Mutex<Vec<(String, String)>>,
    }

    impl Scripted {
        fn new(answer: ProviderCopies) -> Arc<Self> {
            Arc::new(Self {
                answer,
                calls: Mutex::new(Vec::new()),
            })
        }

        fn calls(&self) -> Vec<(String, String)> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl BatchCanceller for Scripted {
        async fn cancel(&self, provider: &str, batch_id: &str) -> anyhow::Result<ProviderCopies> {
            self.calls
                .lock()
                .unwrap()
                .push((provider.to_owned(), batch_id.to_owned()));
            Ok(self.answer)
        }
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        lineage: BatchLineage,
        jobs: JobStore,
        batches: BatchStore,
        ledger: SendLedger,
    }

    fn fixture(canceller: Arc<dyn BatchCanceller>) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let jobs = JobStore::new(dir.path().join("jobs.json"));
        let job = DeleteJob::new(
            JOB.to_owned(),
            DeleteTarget::Event {
                event_id: "e".to_owned(),
            },
            0,
            1,
        );
        jobs.save(&job).unwrap();
        let lineage = BatchLineage::new(
            BatchStore::new(dir.path().join("batches.json")),
            SendLedger::new(dir.path().join("sends.jsonl")),
            JobStore::new(dir.path().join("jobs.json")),
            canceller,
        );
        Fixture {
            batches: BatchStore::new(dir.path().join("batches.json")),
            ledger: SendLedger::new(dir.path().join("sends.jsonl")),
            _dir: dir,
            lineage,
            jobs,
        }
    }

    fn target(events: &[&str]) -> LineageTarget {
        LineageTarget {
            job_id: JOB.to_owned(),
            event_ids: events.iter().map(|id| (*id).to_owned()).collect(),
            ..LineageTarget::default()
        }
    }

    fn notices(fixture: &Fixture) -> Vec<String> {
        fixture.jobs.get(JOB).unwrap().unwrap().notices
    }

    #[tokio::test]
    async fn a_batch_holding_a_deleted_request_is_cancelled_and_the_survivor_is_sent_again() {
        // Arrange
        let canceller = Scripted::new(ProviderCopies::Deleted);
        let fixture = fixture(canceller.clone());
        let deleted = entry("k-alice", "batch-1", &["e-alice"]);
        let survivor = entry("k-bob", "batch-1", &["e-bob"]);
        let other = entry("k-carol", "batch-2", &["e-carol"]);
        fixture
            .batches
            .replace(vec![deleted, survivor.clone(), other.clone()])
            .unwrap();
        let asked = target(&["e-alice"]);

        // Act
        let found = fixture.lineage.find(&asked).await.unwrap();
        let report = fixture.lineage.purge(&asked).await.unwrap();
        let residuals = fixture.lineage.verify(&asked).await.unwrap();

        // Assert
        let table = fixture.batches.load().unwrap();
        let resend = BatchEntry {
            state: BatchState::Cancelled,
            ..survivor
        };
        assert_eq!(found.total(), 1);
        assert_eq!(report.removed, 1);
        assert_eq!(table, vec![resend, other]);
        assert!(residuals.is_empty(), "{residuals:?}");
        assert_eq!(
            canceller.calls(),
            [("anthropic".to_owned(), "batch-1".to_owned())]
        );
    }

    #[tokio::test]
    async fn a_provider_with_a_delete_endpoint_leaves_no_notice_and_one_without_leaves_one() {
        for (answer, expected) in [(ProviderCopies::Deleted, 0), (ProviderCopies::Kept, 1)] {
            // Arrange
            let canceller = Scripted::new(answer);
            let fixture = fixture(canceller.clone());
            fixture
                .batches
                .replace(vec![
                    entry("k1", "batch-1", &["e-alice"]),
                    entry("k2", "batch-1", &["e-alice"]),
                ])
                .unwrap();
            for key in ["k1", "k2"] {
                fixture
                    .ledger
                    .append(&row(key, "anthropic", &["e-alice"]))
                    .unwrap();
            }

            // Act
            fixture.lineage.purge(&target(&["e-alice"])).await.unwrap();

            // Assert
            assert_eq!(canceller.calls().len(), 1, "{answer:?}");
            assert_eq!(notices(&fixture).len(), expected, "{answer:?}");
            assert!(fixture.ledger.load().unwrap().is_empty(), "{answer:?}");
        }
    }

    #[tokio::test]
    async fn a_synchronous_send_is_named_by_the_ledger_and_noticed_before_its_row_goes() {
        // Arrange
        let canceller = Scripted::new(ProviderCopies::Deleted);
        let fixture = fixture(canceller.clone());
        fixture
            .ledger
            .append(&row("k1", "openai", &["e-alice"]))
            .unwrap();
        fixture
            .ledger
            .append(&row("k2", "openai", &["e-bob"]))
            .unwrap();
        let asked = target(&["e-alice"]);

        // Act
        let found = fixture.lineage.find(&asked).await.unwrap();
        let before = fixture.lineage.verify(&asked).await.unwrap();
        fixture.lineage.purge(&asked).await.unwrap();
        fixture.lineage.purge(&asked).await.unwrap();

        // Assert
        let notices = notices(&fixture);
        assert_eq!(found.total(), 1);
        assert_eq!(before.len(), 1);
        assert_eq!(before[0].id, "k1");
        assert_eq!(notices.len(), 1, "{notices:?}");
        assert!(notices[0].starts_with("openai "), "{notices:?}");
        assert_eq!(
            fixture.ledger.load().unwrap(),
            vec![row("k2", "openai", &["e-bob"])]
        );
        assert!(canceller.calls().is_empty(), "no batch holds the request");
    }

    #[tokio::test]
    async fn a_purge_with_nothing_to_remove_changes_no_file() {
        // Arrange
        let fixture = fixture(Scripted::new(ProviderCopies::Deleted));

        // Act
        let report = fixture.lineage.purge(&target(&["e-alice"])).await.unwrap();

        // Assert
        assert_eq!(report.removed, 0);
        assert!(!fixture.batches.exists());
        assert!(!fixture.ledger.exists());
        assert!(notices(&fixture).is_empty());
    }

    #[tokio::test]
    async fn a_failed_cancel_keeps_the_table_and_the_ledger_for_the_rerun() {
        // Arrange
        struct Refusing;
        #[async_trait]
        impl BatchCanceller for Refusing {
            async fn cancel(&self, _: &str, _: &str) -> anyhow::Result<ProviderCopies> {
                anyhow::bail!("the provider is unreachable")
            }
        }
        let fixture = fixture(Arc::new(Refusing));
        fixture
            .batches
            .replace(vec![entry("k1", "batch-1", &["e-alice"])])
            .unwrap();
        fixture
            .ledger
            .append(&row("k1", "anthropic", &["e-alice"]))
            .unwrap();

        // Act
        let error = fixture
            .lineage
            .purge(&target(&["e-alice"]))
            .await
            .unwrap_err();

        // Assert
        assert!(error.to_string().contains("unreachable"), "{error}");
        assert_eq!(fixture.batches.load().unwrap().len(), 1);
        assert_eq!(fixture.ledger.load().unwrap().len(), 1);
    }
}
