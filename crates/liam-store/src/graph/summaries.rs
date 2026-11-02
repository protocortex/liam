// SPDX-License-Identifier: Apache-2.0
//! The summary store: summaries of clusters of recorded nodes, the children and
//! source events each rests on, their vectors, and the state the tree builder
//! keeps between runs.
//!
//! The source events of a summary are materialized when it is written, as the
//! union of its children's, so asking where a summary comes from is a lookup
//! and not a walk down the tree.

use std::collections::{BTreeSet, HashSet};

use liam_log::event::{BasisRef, DerivedArtifact, LogPayload, ARTIFACT_KIND_SUMMARY};
use sha2::{Digest, Sha256};

use super::fact_write::OpenWrite;
use super::logged_plan::{event, STORE_TRUST};
use super::logged_write::{abandon, commit_or_abandon};
use super::projection::{steps_for, Step};
use super::{live_at, row_not_inferred, Graph};
use crate::backend::{Backend, BackendTx};
use crate::error::{Error, Result};
use crate::ids::{Millis, NodeId, FOREVER};
use crate::types::{
    ChildKind, ClosureReport, Derivation, EventRef, EventRole, InferredText, NewSummary,
    PutOutcome, RecordedText, Summary, SummaryChild, SummaryFailure, SummaryInput, SummaryInputs,
    TreeState,
};
use crate::value::{Row, Value};

const SUMMARY_COLUMNS: &str = "id, level, scope, child_hash, text, derivation, events_fed, \
     valid_min, valid_max, model_id, prompt_version, tx_from, tx_to";

/// The identity of a summary's child set: SHA-256 over the level, the scope and
/// the sorted child ids, as lowercase hex. Every part is length-prefixed so
/// `("ab", "c")` and `("a", "bc")` differ. The same children in another order
/// give the same hash.
pub fn child_hash(level: u8, scope: &str, child_ids: &[String]) -> String {
    let mut sorted: Vec<&str> = child_ids.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    let mut hasher = Sha256::new();
    let mut part = |bytes: &[u8]| {
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    };
    part(&[level]);
    part(scope.as_bytes());
    for id in sorted {
        part(id.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn encode_vector(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// The vector `bytes` hold. The column takes a blob of any length, so a length
/// that is not `dims` whole floats is an error and not a vector of another width.
fn decode_vector(bytes: &[u8], dims: usize) -> Result<Vec<f32>> {
    let (chunks, rest) = bytes.as_chunks::<{ std::mem::size_of::<f32>() }>();
    if !rest.is_empty() || chunks.len() != dims {
        return Err(Error::Dimension {
            expected: dims,
            got: bytes.len() / std::mem::size_of::<f32>(),
        });
    }
    Ok(chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn summary_from_row(row: &Row) -> Result<Summary> {
    let corrupt = |column: &str| Error::Backend(format!("summaries holds a bad {column}"));
    Ok(Summary {
        id: row.get_string(0)?,
        level: u8::try_from(row.get_i64(1)?).map_err(|_| corrupt("level"))?,
        scope: row.get_string(2)?,
        child_hash: row.get_string(3)?,
        text: InferredText::new(row.get_string(4)?),
        derivation: Derivation::parse(&row.get_string(5)?)?,
        events_fed: u32::try_from(row.get_i64(6)?).map_err(|_| corrupt("events_fed"))?,
        valid_min: Millis(row.get_i64(7)?),
        valid_max: Millis(row.get_i64(8)?),
        model_id: row.get_string(9)?,
        prompt_version: row.get_string(10)?,
        tx_from: Millis(row.get_i64(11)?),
        tx_to: Millis(row.get_i64(12)?),
    })
}

/// Records a failed generation, starting the count again when the prompt or the
/// model differs from the one that failed before.
const RECORD_FAILURE_SQL: &str =
    "INSERT INTO summary_failures (child_hash, failures, last_error, prompt_version, model_id)
     VALUES (?1, 1, ?2, ?3, ?4)
     ON CONFLICT(child_hash) DO UPDATE SET
       failures = CASE WHEN prompt_version = excluded.prompt_version
                        AND model_id = excluded.model_id THEN failures + 1 ELSE 1 END,
       last_error = excluded.last_error,
       prompt_version = excluded.prompt_version,
       model_id = excluded.model_id
     RETURNING failures";

impl<B: Backend> Graph<B> {
    /// Stores a summary with its children, source events and vector, and closes
    /// the summaries in `closes` that it replaces, all in one transaction. A
    /// live summary of the same child set makes this a no-op that names it,
    /// appends nothing and leaves `closes` alone. A closed summary of the same
    /// child set does not block a new one, so a child set that returns to an
    /// earlier value gets a new row.
    ///
    /// On a logged store the summary is appended to the log first, with
    /// `write_source` as its source, so a rolled-back run's summaries can be
    /// told apart. The vector is not logged; a replay leaves it to the re-embed
    /// pass.
    ///
    /// # Errors
    ///
    /// Fails when there are no children, when a child is not of the kind the
    /// level takes (nodes at level 1, summaries above), when there are no source
    /// events, when the vector is not `dims` wide, and when a summary in
    /// `closes` is not open.
    pub async fn put_summary(
        &self,
        summary: NewSummary,
        children: &[SummaryChild],
        source_event_ids: &[EventRef],
        embedding: &[f32],
        closes: &[String],
        write_source: &str,
    ) -> Result<PutOutcome> {
        let child_ids: Vec<String> = children.iter().map(|child| child.id.clone()).collect();
        let hash = child_hash(summary.level, &summary.scope, &child_ids);
        if children.is_empty() {
            return Err(Error::InvalidReference(format!(
                "the summary for child set {hash} has no children"
            )));
        }
        let kind = child_kind_of(summary.level);
        if children.iter().any(|child| child.kind != kind) {
            return Err(Error::InvalidReference(format!(
                "the summary for child set {hash} at level {} takes {} children",
                summary.level,
                kind.as_str()
            )));
        }
        if source_event_ids.is_empty() {
            return Err(Error::SummaryWithoutSources(hash));
        }
        self.check_dims(embedding)?;

        let OpenWrite { held, mut tx } = self.open_write().await?;
        let at = self.clock.now();
        let decided = async {
            if let Some(live) = live_with_hash(&mut *tx, &hash).await? {
                return Ok(PutOutcome::Unchanged(live));
            }
            for old in closes {
                require_open(&mut *tx, old).await?;
            }
            Ok(PutOutcome::Inserted(ulid::Ulid::gen().to_string()))
        }
        .await;
        let id = match decided {
            Ok(PutOutcome::Inserted(id)) => id,
            Ok(unchanged) => {
                abandon(tx).await;
                return Ok(unchanged);
            }
            Err(error) => {
                abandon(tx).await;
                return Err(error);
            }
        };
        let artifact = DerivedArtifact {
            artifact_id: id.clone(),
            kind: ARTIFACT_KIND_SUMMARY.to_string(),
            level: summary.level,
            scope: summary.scope,
            text: summary.text,
            child_ids,
            source_event_ids: logged_events(source_event_ids),
            events_fed: summary.events_fed,
            derivation: Derivation::Inferred.as_str().to_string(),
            model_id: summary.model_id,
            prompt_version: summary.prompt_version,
            closes: closes.to_vec(),
            valid_min: summary.valid_min.0,
            valid_max: summary.valid_max.0,
            at: at.0,
        };
        let event = event(
            [0; 32],
            (write_source, STORE_TRUST),
            at.0,
            at,
            LogPayload::DerivedArtifact(artifact),
        );
        let mut steps = steps_for(&event.payload);
        steps.push(Step::SummaryEmbedding {
            id: id.clone(),
            embedding: encode_vector(embedding),
            dims: self.dims,
        });
        match held {
            Some(mut held) => self.append_and_commit(tx, &mut held, event, &steps).await?,
            None => self.apply_and_commit(tx, &steps).await?,
        }
        Ok(PutOutcome::Inserted(id))
    }

    /// Ends the transaction time of the open summary `id` at `at`, for a
    /// cluster that dissolved with nothing to replace it. A logged store
    /// records the close with `write_source`.
    pub async fn close_summary(&self, id: &str, at: Millis, write_source: &str) -> Result<()> {
        let OpenWrite { held, mut tx } = self.open_write().await?;
        if let Err(error) = require_open(&mut *tx, id).await {
            abandon(tx).await;
            return Err(error);
        }
        let event = event(
            [0; 32],
            (write_source, STORE_TRUST),
            at.0,
            self.clock.now(),
            LogPayload::DerivedArtifactClosed {
                artifact_id: id.to_string(),
                at: at.0,
            },
        );
        let steps = steps_for(&event.payload);
        match held {
            Some(mut held) => self.append_and_commit(tx, &mut held, event, &steps).await,
            None => self.apply_and_commit(tx, &steps).await,
        }
    }

    /// The open summaries of `scope` at `level`, oldest first.
    pub async fn live_summaries(&self, level: u8, scope: &str) -> Result<Vec<Summary>> {
        let rows = self
            .backend
            .query(
                &format!(
                    "SELECT {SUMMARY_COLUMNS} FROM summaries
                     WHERE level = ?1 AND scope = ?2 AND tx_to = ?3 ORDER BY tx_from, id"
                ),
                &[i64::from(level).into(), scope.into(), FOREVER.into()],
            )
            .await?;
        rows.iter().map(summary_from_row).collect()
    }

    /// The children of `summary_id`, ordered by id.
    pub async fn summary_children(&self, summary_id: &str) -> Result<Vec<SummaryChild>> {
        let rows = self
            .backend
            .query(
                "SELECT child_id, child_kind FROM summary_children
                 WHERE parent_id = ?1 ORDER BY child_id",
                &[summary_id.into()],
            )
            .await?;
        rows.iter()
            .map(|row| {
                Ok(SummaryChild {
                    id: row.get_string(0)?,
                    kind: ChildKind::parse(&row.get_string(1)?)?,
                })
            })
            .collect()
    }

    /// The source events of `summary_id`, ordered by event id.
    pub async fn summary_sources(&self, summary_id: &str) -> Result<Vec<EventRef>> {
        let rows = self
            .backend
            .query(
                "SELECT event_id, synthetic FROM summary_sources
                 WHERE summary_id = ?1 ORDER BY event_id",
                &[summary_id.into()],
            )
            .await?;
        rows.iter()
            .map(|row| {
                Ok(EventRef {
                    event_id: row.get_string(0)?,
                    synthetic: row.get_i64(1)? != 0,
                })
            })
            .collect()
    }

    /// Whether each child's events are among the summary's own, and how many of
    /// those are real log events.
    pub async fn check_closure(&self, summary_id: &str) -> Result<ClosureReport> {
        let own = self.summary_sources(summary_id).await?;
        let covered: HashSet<&str> = own.iter().map(|event| event.event_id.as_str()).collect();
        let children = self
            .backend
            .query(
                "SELECT child_id, child_kind FROM summary_children
                 WHERE parent_id = ?1 ORDER BY child_id",
                &[summary_id.into()],
            )
            .await?;
        let mut uncovered_children = Vec::new();
        for child in &children {
            let (child_id, kind) = (child.get_string(0)?, child.get_string(1)?);
            let events = match kind == ChildKind::Summary.as_str() {
                true => self.summary_sources(&child_id).await?,
                false => {
                    self.node_event_ids(&NodeId::from_raw(child_id.clone()), EventRole::Origin)
                        .await?
                }
            };
            if events
                .iter()
                .any(|event| !covered.contains(event.event_id.as_str()))
            {
                uncovered_children.push(child_id);
            }
        }
        let synthetic_events = own.iter().filter(|event| event.synthetic).count();
        Ok(ClosureReport {
            uncovered_children,
            real_events: own.len() - synthetic_events,
            synthetic_events,
        })
    }

    /// Counts a failed generation for `child_hash` and returns how many have
    /// failed under this prompt and model.
    pub async fn record_summary_failure(
        &self,
        child_hash: &str,
        error: &str,
        prompt_version: &str,
        model_id: &str,
    ) -> Result<u32> {
        let mut tx = self.backend.begin().await?;
        let counted = async {
            let rows = tx
                .query(
                    RECORD_FAILURE_SQL,
                    &[
                        child_hash.into(),
                        error.into(),
                        prompt_version.into(),
                        model_id.into(),
                    ],
                )
                .await?;
            let row = rows.first().ok_or_else(|| {
                Error::Backend("recording a summary failure returned no count".into())
            })?;
            u32::try_from(row.get_i64(0)?)
                .map_err(|_| Error::Backend("summary_failures holds a bad count".into()))
        }
        .await;
        commit_or_abandon(tx, counted).await
    }

    /// What is recorded about failed generation for `child_hash`.
    pub async fn summary_failures(&self, child_hash: &str) -> Result<Option<SummaryFailure>> {
        let rows = self
            .backend
            .query(
                "SELECT failures, last_error, prompt_version, model_id
                 FROM summary_failures WHERE child_hash = ?1",
                &[child_hash.into()],
            )
            .await?;
        rows.first()
            .map(|row| {
                Ok(SummaryFailure {
                    failures: u32::try_from(row.get_i64(0)?)
                        .map_err(|_| Error::Backend("summary_failures holds a bad count".into()))?,
                    last_error: row.get_string(1)?,
                    prompt_version: row.get_string(2)?,
                    model_id: row.get_string(3)?,
                })
            })
            .transpose()
    }

    /// The open summaries that have no vector, as `(id, text)`, by id.
    pub(super) async fn summaries_missing_embeddings(&self) -> Result<Vec<(String, String)>> {
        let rows = self
            .backend
            .query(
                "SELECT s.id, s.text FROM summaries s
                 WHERE s.tx_to = ?1
                   AND NOT EXISTS (SELECT 1 FROM summary_embeddings e WHERE e.summary_id = s.id)
                 ORDER BY s.id",
                &[FOREVER.into()],
            )
            .await?;
        rows.iter()
            .map(|row| Ok((row.get_string(0)?, row.get_string(1)?)))
            .collect()
    }

    /// Stores `embedding` for the summary `id` unless it has one, and says
    /// whether it did.
    pub(super) async fn store_summary_embedding_if_absent(
        &self,
        id: &str,
        embedding: &[f32],
    ) -> Result<bool> {
        let stored = self
            .backend
            .execute(
                "INSERT OR IGNORE INTO summary_embeddings (summary_id, dims, embedding)
                 VALUES (?1, ?2, ?3)",
                &[
                    id.into(),
                    (self.dims as i64).into(),
                    encode_vector(embedding).into(),
                ],
            )
            .await?;
        Ok(stored == 1)
    }

    /// The vectors of the open summaries of `scope` at `level`, oldest first.
    /// This is the only read of the table.
    pub async fn summary_embeddings(
        &self,
        scope: &str,
        level: u8,
    ) -> Result<Vec<(String, Vec<f32>)>> {
        let rows = self
            .backend
            .query(
                "SELECT e.summary_id, e.dims, e.embedding
                 FROM summary_embeddings e JOIN summaries s ON s.id = e.summary_id
                 WHERE s.scope = ?1 AND s.level = ?2 AND s.tx_to = ?3
                 ORDER BY s.tx_from, s.id",
                &[scope.into(), i64::from(level).into(), FOREVER.into()],
            )
            .await?;
        rows.iter()
            .map(|row| {
                let dims = usize::try_from(row.get_i64(1)?)
                    .map_err(|_| Error::Backend("summary_embeddings holds a bad width".into()))?;
                let vector = decode_vector(row.get_blob(2)?, dims)?;
                self.check_dims(&vector)?;
                Ok((row.get_string(0)?, vector))
            })
            .collect()
    }

    pub async fn tree_state(&self, scope: &str, level: u8) -> Result<Option<TreeState>> {
        let rows = self
            .backend
            .query(
                "SELECT last_full_build_at, baseline_item_count, unattached_count
                 FROM summary_tree_state WHERE scope = ?1 AND level = ?2",
                &[scope.into(), i64::from(level).into()],
            )
            .await?;
        let count = |row: &Row, column| {
            u32::try_from(row.get_i64(column)?)
                .map_err(|_| Error::Backend("summary_tree_state holds a bad count".into()))
        };
        rows.first()
            .map(|row| {
                Ok(TreeState {
                    scope: scope.to_string(),
                    level,
                    last_full_build_at: Millis(row.get_i64(0)?),
                    baseline_item_count: count(row, 1)?,
                    unattached_count: count(row, 2)?,
                })
            })
            .transpose()
    }

    pub async fn put_tree_state(&self, state: &TreeState) -> Result<()> {
        self.backend
            .execute(
                "INSERT INTO summary_tree_state
                   (scope, level, last_full_build_at, baseline_item_count, unattached_count)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(scope, level) DO UPDATE SET
                   last_full_build_at = excluded.last_full_build_at,
                   baseline_item_count = excluded.baseline_item_count,
                   unattached_count = excluded.unattached_count",
                &[
                    state.scope.as_str().into(),
                    i64::from(state.level).into(),
                    state.last_full_build_at.into(),
                    i64::from(state.baseline_item_count).into(),
                    i64::from(state.unattached_count).into(),
                ],
            )
            .await?;
        Ok(())
    }

    /// The live recorded and extracted nodes of `scope` with a stored vector, in
    /// insertion order. `scope` is the exact scope, '' for the unscoped group, so
    /// a summary never mixes scopes. Entity pages are left out, being composed.
    pub async fn summary_inputs(&self, scope: &str) -> Result<SummaryInputs> {
        let rows = self
            .backend
            .query(
                &format!(
                    "SELECT id, kind, label, content, valid_from, scope FROM nodes
                     WHERE COALESCE(scope, '') = ?2 AND {live}{recorded}
                     ORDER BY rowid",
                    live = live_at("nodes", 1),
                    recorded = row_not_inferred("nodes", false),
                ),
                &[self.clock.now().into(), scope.into()],
            )
            .await?;
        let ids = rows
            .iter()
            .map(|row| row.get_string(0))
            .collect::<Result<Vec<_>>>()?;
        let mut listed = self
            .listed_events(&ids.iter().map(String::as_str).collect::<Vec<_>>())
            .await?;
        let mut items = Vec::with_capacity(rows.len());
        let mut skipped_no_vector = 0;
        for (row, id) in rows.iter().zip(ids) {
            let Some(vector) = self.backend.vector_get(&id, self.dims).await? else {
                skipped_no_vector += 1;
                continue;
            };
            let origin_events = listed.remove(&id).unwrap_or_default().origin;
            items.push(SummaryInput {
                id: NodeId::from_raw(id),
                kind: row.get_string(1)?,
                label: row.get_string(2)?,
                content: RecordedText::new(row.get_string(3)?),
                valid_from: Millis(row.get_i64(4)?),
                scope: match row.0.get(5) {
                    Some(Value::Text(text)) => Some(text.clone()),
                    _ => None,
                },
                origin_events,
                vector,
            });
        }
        Ok(SummaryInputs {
            items,
            skipped_no_vector,
        })
    }
}

async fn live_with_hash(tx: &mut dyn BackendTx, hash: &str) -> Result<Option<String>> {
    let rows = tx
        .query(
            "SELECT id FROM summaries WHERE child_hash = ?1 AND tx_to = ?2",
            &[hash.into(), FOREVER.into()],
        )
        .await?;
    rows.first().map(|row| row.get_string(0)).transpose()
}

/// The kind of child a summary of `level` is made of.
fn child_kind_of(level: u8) -> ChildKind {
    match level {
        1 => ChildKind::Node,
        _ => ChildKind::Summary,
    }
}

fn logged_events(events: &[EventRef]) -> Vec<BasisRef> {
    events
        .iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|event| BasisRef {
            event_id: event.event_id.clone(),
            synthetic: event.synthetic,
        })
        .collect()
}

/// Projects a logged artifact: closes the artifacts it replaces, then stores it
/// with its children and source events. Closing one that is already closed or
/// gone changes nothing, so a replay can run it again, and a replacement can
/// follow a run that was rolled back.
pub(super) async fn apply_artifact(
    tx: &mut dyn BackendTx,
    artifact: &DerivedArtifact,
) -> Result<()> {
    if artifact.kind != ARTIFACT_KIND_SUMMARY {
        return Err(Error::Constraint(format!(
            "artifact {} has the unknown kind {}",
            artifact.artifact_id, artifact.kind
        )));
    }
    let derivation = Derivation::parse(&artifact.derivation)?;
    for old in &artifact.closes {
        close_if_open(tx, old, artifact.at).await?;
    }
    let hash = child_hash(artifact.level, &artifact.scope, &artifact.child_ids);
    tx.execute(
        &format!(
            "INSERT INTO summaries ({SUMMARY_COLUMNS})
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)"
        ),
        &[
            artifact.artifact_id.as_str().into(),
            i64::from(artifact.level).into(),
            artifact.scope.as_str().into(),
            hash.into(),
            artifact.text.as_str().into(),
            derivation.as_str().into(),
            i64::from(artifact.events_fed).into(),
            artifact.valid_min.into(),
            artifact.valid_max.into(),
            artifact.model_id.as_str().into(),
            artifact.prompt_version.as_str().into(),
            artifact.at.into(),
            FOREVER.into(),
        ],
    )
    .await?;
    let kind = child_kind_of(artifact.level);
    for child in &artifact.child_ids {
        tx.execute(
            "INSERT INTO summary_children (parent_id, child_id, child_kind) VALUES (?1, ?2, ?3)",
            &[
                artifact.artifact_id.as_str().into(),
                child.as_str().into(),
                kind.as_str().into(),
            ],
        )
        .await?;
    }
    for event in &artifact.source_event_ids {
        tx.execute(
            "INSERT OR IGNORE INTO summary_sources (summary_id, event_id, synthetic)
             VALUES (?1, ?2, ?3)",
            &[
                artifact.artifact_id.as_str().into(),
                event.event_id.as_str().into(),
                i64::from(event.synthetic).into(),
            ],
        )
        .await?;
    }
    Ok(())
}

/// Ends the open summary `id` at `at`; one that is not open is left alone.
pub(super) async fn close_if_open(tx: &mut dyn BackendTx, id: &str, at: i64) -> Result<()> {
    tx.execute(
        "UPDATE summaries SET tx_to = ?1 WHERE id = ?2 AND tx_to = ?3",
        &[at.into(), id.into(), FOREVER.into()],
    )
    .await?;
    Ok(())
}

pub(super) async fn store_embedding(
    tx: &mut dyn BackendTx,
    id: &str,
    dims: usize,
    embedding: &[u8],
) -> Result<()> {
    tx.execute(
        "INSERT INTO summary_embeddings (summary_id, dims, embedding) VALUES (?1, ?2, ?3)",
        &[id.into(), (dims as i64).into(), embedding.to_vec().into()],
    )
    .await?;
    Ok(())
}

async fn require_open(tx: &mut dyn BackendTx, id: &str) -> Result<()> {
    let open = tx
        .query(
            "SELECT 1 FROM summaries WHERE id = ?1 AND tx_to = ?2",
            &[id.into(), FOREVER.into()],
        )
        .await?;
    match open.is_empty() {
        true => Err(Error::SummaryNotOpen(id.to_string())),
        false => Ok(()),
    }
}
