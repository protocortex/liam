// SPDX-License-Identifier: Apache-2.0
//! The tree builder: which items form a cluster at one level, and which
//! summaries of that level to keep, regenerate or close.
//!
//! Everything here is a pure function of the loaded items and the stored
//! state, so a run can be repeated and compared. Clusters are named by their
//! member ids and never by the community numbers the clustering returns, which
//! are not stable between runs. Each cluster's identity is the `child_hash` of
//! its members, so an unchanged cluster keeps its summary.

use std::collections::{BTreeMap, BTreeSet};

use liam_store::cluster::{self, Edge};
use liam_store::{child_hash, Millis, SummaryInput, TreeState};

/// The deepest tree the builder plans, whatever the configuration asks for.
pub const MAX_DEPTH: u8 = 3;

/// How items are linked, grouped and kept stable from one run to the next.
#[derive(Clone, Debug)]
pub struct TreeParams {
    /// Nearest neighbours considered for each item.
    pub knn: usize,
    pub min_similarity: f32,
    pub time_window: Millis,
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
    pub max_depth: u8,
    /// The share of a level's baseline that may change before it is clustered again.
    pub recluster_change_ratio: f64,
    /// The age of the last full build after which the level is clustered again.
    pub recluster_interval: Millis,
    /// The least similarity to a cluster's centroid at which a new item joins it.
    pub min_attach_similarity: f32,
    /// How many new items may wait unattached before the level is clustered again.
    pub max_unattached: u32,
}

/// An item of one level: a node at level 1, a summary above that.
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    pub id: String,
    pub vector: Vec<f32>,
    pub time: Millis,
}

impl From<&SummaryInput> for Point {
    fn from(input: &SummaryInput) -> Self {
        Self {
            id: input.id.as_str().to_string(),
            vector: input.vector.clone(),
            time: input.valid_from,
        }
    }
}

/// Groups `node_count` items by the edges between them, as one community number
/// each. Production uses [`detect_cold`]; a test passes its own.
pub type ClusterFn<'a> = &'a dyn Fn(usize, &[Edge]) -> Vec<usize>;

pub fn detect_cold(node_count: usize, edges: &[Edge]) -> Vec<usize> {
    cluster::detect(node_count, edges, None)
}

/// A level that had links between its items came back as all singletons. The
/// clustering degrades to that on error, so it is a failed build and not a
/// level that does not group.
#[derive(Debug, PartialEq, Eq)]
pub struct ClusteringFailed {
    pub level: u8,
}

impl std::fmt::Display for ClusteringFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "clustering level {} returned only singletons for items that are linked",
            self.level
        )
    }
}

impl std::error::Error for ClusteringFailed {}

/// A live summary of the level, with the children it was built from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreviousCluster {
    pub summary_id: String,
    pub child_hash: String,
    pub children: Vec<String>,
}

/// A cluster of the level to be summarized.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedCluster {
    /// Sorted ids of the members.
    pub members: Vec<String>,
    pub child_hash: String,
}

/// A cluster with no live summary, or with one that its members have outgrown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Regeneration {
    pub cluster: PlannedCluster,
    /// The live summary the new one stands in for, to be closed with it.
    pub replaces: Option<String>,
}

/// What to do with one level.
#[derive(Clone, Debug, PartialEq)]
pub struct ChangePlan {
    /// Clusters whose live summary already covers exactly their members.
    pub keep: Vec<PlannedCluster>,
    pub regenerate: Vec<Regeneration>,
    /// Live summaries whose cluster no longer exists and that no new one replaces.
    pub dissolved: Vec<String>,
    /// The state to store once the level is built.
    pub next_state: TreeState,
    pub full_build: bool,
}

impl ChangePlan {
    pub fn cluster_count(&self) -> usize {
        self.keep.len() + self.regenerate.len()
    }
}

/// One level to plan: its items, its live summaries and the state it was left in.
pub struct LevelRequest<'a> {
    pub scope: &'a str,
    pub level: u8,
    pub points: &'a [Point],
    pub previous: &'a [PreviousCluster],
    pub state: Option<&'a TreeState>,
    pub now: Millis,
}

/// The cosine similarity of `a` and `b`, 0 when either has no length.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let (norm_a, norm_b) = (norm(a), norm(b));
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Links each item to its `knn` nearest others, keeping a link when the pair is
/// at least `min_similarity` alike and no more than `time_window` apart. The
/// nearest are chosen by score, highest first, then by id. Pairs come back
/// once, as indices in ascending order.
pub fn build_graph(
    points: &[Point],
    knn: usize,
    min_similarity: f32,
    time_window: Millis,
) -> Vec<(usize, usize)> {
    let mut pairs = BTreeSet::new();
    for (i, point) in points.iter().enumerate() {
        let mut scored: Vec<(f32, usize)> = (0..points.len())
            .filter(|&j| j != i)
            .map(|j| (cosine(&point.vector, &points[j].vector), j))
            .collect();
        scored.sort_by(|(score_a, a), (score_b, b)| {
            score_b
                .total_cmp(score_a)
                .then_with(|| points[*a].id.cmp(&points[*b].id))
        });
        for &(score, j) in scored.iter().take(knn) {
            let apart = point.time.0.abs_diff(points[j].time.0);
            if score >= min_similarity && apart <= time_window.0.unsigned_abs() {
                pairs.insert((i.min(j), i.max(j)));
            }
        }
    }
    pairs.into_iter().collect()
}

/// Splits `members` into the fewest chunks of at most `max` that differ in
/// size by one, keeping their order.
fn split_evenly(members: Vec<usize>, max: usize) -> Vec<Vec<usize>> {
    let chunks = members.len().div_ceil(max.max(1));
    if chunks <= 1 {
        return vec![members];
    }
    let (base, extra) = (members.len() / chunks, members.len() % chunks);
    let mut rest = members.as_slice();
    (0..chunks)
        .map(|chunk| {
            let (taken, tail) = rest.split_at(base + usize::from(chunk < extra));
            rest = tail;
            taken.to_vec()
        })
        .collect()
}

/// The clusters of `points` as indices, each in time order and the clusters
/// ordered by their first member. `None` when the clustering failed.
fn cluster_points(
    points: &[Point],
    params: &TreeParams,
    cluster_fn: ClusterFn<'_>,
) -> Option<Vec<Vec<usize>>> {
    if points.len() < params.min_cluster_size {
        return Some(Vec::new());
    }
    let edges: Vec<Edge> = build_graph(
        points,
        params.knn,
        params.min_similarity,
        params.time_window,
    )
    .into_iter()
    .map(|(a, b)| Edge(a, b))
    .collect();
    let membership = cluster_fn(points.len(), &edges);
    if membership.len() != points.len() {
        return None;
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, community) in membership.iter().enumerate() {
        groups.entry(*community).or_default().push(index);
    }
    if !edges.is_empty() && groups.len() == points.len() {
        return None;
    }
    let by_time = |a: &usize, b: &usize| {
        (points[*a].time, &points[*a].id).cmp(&(points[*b].time, &points[*b].id))
    };
    let mut clusters: Vec<Vec<usize>> = groups
        .into_values()
        .filter(|members| members.len() >= params.min_cluster_size)
        .flat_map(|mut members| {
            members.sort_by(by_time);
            split_evenly(members, params.max_cluster_size)
        })
        .filter(|members| members.len() >= params.min_cluster_size)
        .collect();
    clusters.sort_by(|a, b| by_time(&a[0], &b[0]));
    Some(clusters)
}

fn planned(scope: &str, level: u8, mut members: Vec<String>) -> PlannedCluster {
    members.sort_unstable();
    PlannedCluster {
        child_hash: child_hash(level, scope, &members),
        members,
    }
}

fn jaccard(a: &[String], b: &[String]) -> f64 {
    let (a, b): (BTreeSet<_>, BTreeSet<_>) = (a.iter().collect(), b.iter().collect());
    let shared = a.intersection(&b).count();
    shared as f64 / (a.len() + b.len() - shared).max(1) as f64
}

fn leftover(points: &[Point], clusters: &[PlannedCluster]) -> u32 {
    let clustered: usize = clusters.iter().map(|cluster| cluster.members.len()).sum();
    u32::try_from(points.len().saturating_sub(clustered)).unwrap_or(u32::MAX)
}

/// Clusters the level from scratch. A cluster whose hash a live summary
/// already has is kept; each other takes over the live summary it overlaps
/// most, one for one; a live summary left over is dissolved.
fn full_plan(
    request: &LevelRequest<'_>,
    params: &TreeParams,
    cluster_fn: ClusterFn<'_>,
) -> Result<ChangePlan, ClusteringFailed> {
    let points = request.points;
    let clusters = cluster_points(points, params, cluster_fn)
        .ok_or(ClusteringFailed {
            level: request.level,
        })?
        .into_iter()
        .map(|members| {
            let ids = members.iter().map(|&i| points[i].id.clone()).collect();
            planned(request.scope, request.level, ids)
        })
        .collect::<Vec<_>>();
    let next_state = TreeState {
        scope: request.scope.to_string(),
        level: request.level,
        last_full_build_at: request.now,
        baseline_item_count: u32::try_from(points.len()).unwrap_or(u32::MAX),
        unattached_count: leftover(points, &clusters),
    };
    let live_hashes: BTreeSet<&str> = request
        .previous
        .iter()
        .map(|old| old.child_hash.as_str())
        .collect();
    let (keep, fresh): (Vec<_>, Vec<_>) = clusters
        .into_iter()
        .partition(|cluster| live_hashes.contains(cluster.child_hash.as_str()));
    let kept: BTreeSet<&str> = keep.iter().map(|c| c.child_hash.as_str()).collect();
    let open: Vec<&PreviousCluster> = request
        .previous
        .iter()
        .filter(|old| !kept.contains(old.child_hash.as_str()))
        .collect();

    let mut overlaps: Vec<(f64, usize, &str)> = Vec::new();
    for (new, cluster) in fresh.iter().enumerate() {
        for old in &open {
            let overlap = jaccard(&cluster.members, &old.children);
            if overlap > 0.0 {
                overlaps.push((overlap, new, old.summary_id.as_str()));
            }
        }
    }
    overlaps.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(b.2)));
    let mut replaces: BTreeMap<usize, &str> = BTreeMap::new();
    let mut taken: BTreeSet<&str> = BTreeSet::new();
    for (_, new, old) in overlaps {
        if !replaces.contains_key(&new) && taken.insert(old) {
            replaces.insert(new, old);
        }
    }
    let dissolved = open
        .iter()
        .filter(|old| !taken.contains(old.summary_id.as_str()))
        .map(|old| old.summary_id.clone())
        .collect();
    let regenerate = fresh
        .into_iter()
        .enumerate()
        .map(|(new, cluster)| Regeneration {
            cluster,
            replaces: replaces.get(&new).map(|id| (*id).to_string()),
        })
        .collect();
    Ok(ChangePlan {
        keep,
        regenerate,
        dissolved,
        next_state,
        full_build: true,
    })
}

/// A live cluster as it stands now: its members that still exist.
struct Slot<'a> {
    previous: &'a PreviousCluster,
    members: Vec<String>,
}

/// What an incremental pass found, before the policy decides to use it.
struct Incremental<'a> {
    slots: Vec<Slot<'a>>,
    dissolved: Vec<String>,
    /// Members of live clusters that are no longer items.
    removed: usize,
    /// New items no cluster was alike enough to take.
    waiting: usize,
}

fn centroid(members: &[String], by_id: &BTreeMap<&str, &Point>) -> Vec<f32> {
    let mut sum: Vec<f32> = Vec::new();
    for id in members {
        let vector = &by_id[id.as_str()].vector;
        sum.resize(vector.len().max(sum.len()), 0.0);
        for (total, value) in sum.iter_mut().zip(vector) {
            *total += value;
        }
    }
    sum
}

/// Drops the members that are gone from each live cluster, dissolves one left
/// under the minimum, and attaches each new item to the cluster whose centroid
/// it is nearest, when that is near enough.
fn incremental<'a>(request: &LevelRequest<'a>, params: &TreeParams) -> Incremental<'a> {
    let by_id: BTreeMap<&str, &Point> = request
        .points
        .iter()
        .map(|point| (point.id.as_str(), point))
        .collect();
    let mut slots = Vec::new();
    let mut dissolved = Vec::new();
    let mut removed = 0;
    for old in request.previous {
        let members: Vec<String> = old
            .children
            .iter()
            .filter(|id| by_id.contains_key(id.as_str()))
            .cloned()
            .collect();
        removed += old.children.len() - members.len();
        if members.len() < params.min_cluster_size {
            dissolved.push(old.summary_id.clone());
        } else {
            slots.push(Slot {
                previous: old,
                members,
            });
        }
    }
    // The members left by a dissolved cluster are items to place again.
    let known: BTreeSet<String> = slots
        .iter()
        .flat_map(|slot| slot.members.iter().cloned())
        .collect();
    let centroids: Vec<Vec<f32>> = slots
        .iter()
        .map(|slot| centroid(&slot.members, &by_id))
        .collect();
    let mut waiting = 0;
    for point in request
        .points
        .iter()
        .filter(|point| !known.contains(&point.id))
    {
        let nearest = centroids
            .iter()
            .enumerate()
            .map(|(slot, centroid)| (cosine(centroid, &point.vector), slot))
            .max_by(|(score_a, a), (score_b, b)| {
                score_a.total_cmp(score_b).then_with(|| {
                    slots[*b]
                        .previous
                        .child_hash
                        .cmp(&slots[*a].previous.child_hash)
                })
            });
        match nearest {
            Some((score, slot)) if score >= params.min_attach_similarity => {
                slots[slot].members.push(point.id.clone());
            }
            _ => waiting += 1,
        }
    }
    Incremental {
        slots,
        dissolved,
        removed,
        waiting,
    }
}

fn incremental_plan(
    request: &LevelRequest<'_>,
    state: &TreeState,
    pass: Incremental<'_>,
) -> ChangePlan {
    let mut keep = Vec::new();
    let mut regenerate = Vec::new();
    for slot in pass.slots {
        let cluster = planned(request.scope, request.level, slot.members);
        if cluster.child_hash == slot.previous.child_hash {
            keep.push(cluster);
        } else {
            regenerate.push(Regeneration {
                cluster,
                replaces: Some(slot.previous.summary_id.clone()),
            });
        }
    }
    ChangePlan {
        keep,
        regenerate,
        dissolved: pass.dissolved,
        next_state: state.clone(),
        full_build: false,
    }
}

/// Plans one level. It is clustered from scratch when there is no state to
/// compare with, when the items have drifted from the baseline by more than the
/// change ratio, when too many new items wait unattached, when the last full
/// build is older than the interval, or when an attach would overfill a
/// cluster. Otherwise the live clusters stay and only the changed ones are
/// regenerated. The baseline moves only on a full build, so small changes
/// cannot add up unseen.
///
/// # Errors
///
/// Fails when a full build is needed and the clustering fails.
pub fn plan_level(
    request: &LevelRequest<'_>,
    params: &TreeParams,
    cluster_fn: ClusterFn<'_>,
) -> Result<ChangePlan, ClusteringFailed> {
    let Some(state) = request.state.filter(|_| !request.previous.is_empty()) else {
        return full_plan(request, params, cluster_fn);
    };
    let aged =
        request.now.0.saturating_sub(state.last_full_build_at.0) >= params.recluster_interval.0;
    if aged {
        return full_plan(request, params, cluster_fn);
    }
    let pass = incremental(request, params);
    let baseline = state.baseline_item_count;
    let drift = request.points.len().abs_diff(baseline as usize) + pass.removed;
    let excess_waiting = pass.waiting.saturating_sub(state.unattached_count as usize);
    let overfull = pass
        .slots
        .iter()
        .any(|slot| slot.members.len() > params.max_cluster_size);
    if drift as f64 > params.recluster_change_ratio * f64::from(baseline)
        || excess_waiting > params.max_unattached as usize
        || overfull
    {
        return full_plan(request, params, cluster_fn);
    }
    Ok(incremental_plan(request, state, pass))
}

/// Whether to plan the level above `level`, which was planned from `inputs`
/// items. Not when the level is the deepest, has too few clusters to group,
/// did not shrink, or was only partly built, because a level built from a
/// partial one would summarize a part of the tree as if it were the whole.
pub fn should_descend(
    level: u8,
    inputs: usize,
    plan: &ChangePlan,
    fully_built: bool,
    params: &TreeParams,
) -> bool {
    let clusters = plan.cluster_count();
    fully_built
        && level < params.max_depth.min(MAX_DEPTH)
        && clusters >= params.min_cluster_size
        && clusters < inputs
}

#[cfg(test)]
mod tests;
