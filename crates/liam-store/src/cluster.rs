// SPDX-License-Identifier: AGPL-3.0-only
//! Community detection over the edge graph. Pure and database-free: the graph
//! loads the edges and stores the assignments.
//!
//! VERSION CHECK: confirm these against the leiden-rs version you pin.
//! `Partition::from_membership` and `run_with_initial_partition` both take
//! their argument BY VALUE; `run_with_initial_partition` renumbers the seed on
//! entry (`src/leiden.rs:500`), which is what makes a sparse seed safe, and
//! `Partition::renumber` allocates a vector sized by the largest community id
//! (`src/partition.rs:91`), which is what makes the dense id scheme in
//! `build_seed` load-bearing rather than stylistic.
//!
//! Failure policy, and the two halves differ on purpose. A COLD run degrades to
//! singletons, so a caller always has a valid partition to persist. A WARM run
//! degrades to a cold run, never to singletons: the recompute writes the
//! assignment together with a matching fingerprint, so singletons written after
//! a bad seed would look current and be served until the next edge write.

use leiden_rs::{GraphDataBuilder, Leiden, LeidenConfig, Partition};

/// An undirected edge between dense node indices. A repeated pair raises weight.
pub struct Edge(pub usize, pub usize);

/// Assign each of `node_count` nodes a community id. Seeded for stable ids.
///
/// `seed` warm-starts from the previous assignment, mapped onto this run's
/// index space. It must be exactly `node_count` long; a shorter one is rejected
/// by the library and falls through to a cold run.
///
/// Warm starting buys convergence speed and nothing else. It does NOT stabilise
/// community ids: the seed is renumbered on entry and the output on exit, so
/// the integers here are "which nodes group together in this run", never
/// handles to compare against a later call.
pub fn detect(node_count: usize, edges: &[Edge], seed: Option<Vec<usize>>) -> Vec<usize> {
    if node_count == 0 {
        return Vec::new();
    }
    let mut builder = GraphDataBuilder::new(node_count);
    for Edge(u, v) in edges {
        let _ = builder.add_edge(*u, *v, 1.0);
    }
    let Ok(graph) = builder.build() else {
        return singletons(node_count);
    };
    if let Some(seed) = seed {
        if let Ok(result) = Leiden::new(config())
            .run_with_initial_partition(&graph, Partition::from_membership(seed))
        {
            return memberships(&result, node_count);
        }
        // Deliberately falls through to the cold run below rather than
        // returning singletons. See the failure policy in the module doc.
        tracing::warn!(node_count, "warm cluster start failed, retrying cold");
    }
    let Ok(result) = Leiden::new(config()).run(&graph) else {
        return singletons(node_count);
    };
    memberships(&result, node_count)
}

/// One config for both entry points. A warm run that differed here would be
/// comparing against a differently-parameterised cold run, invisibly.
fn config() -> LeidenConfig {
    LeidenConfig {
        seed: Some(42),
        ..Default::default()
    }
}

fn memberships(result: &leiden_rs::LeidenOutput, node_count: usize) -> Vec<usize> {
    (0..node_count)
        .map(|i| result.partition.community_of(i))
        .collect()
}

fn singletons(node_count: usize) -> Vec<usize> {
    (0..node_count).collect()
}
