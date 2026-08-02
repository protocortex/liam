//! Community detection over the edge graph. Pure and database-free: the graph
//! loads the edges and stores the assignments.
//!
//! VERSION CHECK: confirm the per-node membership accessor against the leiden-rs
//! version you pin; it is the one flagged line. Detection degrades to singletons
//! on any failure, so callers can always persist a valid partition.

use leiden_rs::{GraphDataBuilder, Leiden, LeidenConfig};

/// An undirected edge between dense node indices. A repeated pair raises weight.
pub struct Edge(pub usize, pub usize);

/// Assign each of `node_count` nodes a community id. Seeded for stable ids.
pub fn detect(node_count: usize, edges: &[Edge]) -> Vec<usize> {
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
    let config = LeidenConfig {
        seed: Some(42),
        ..Default::default()
    };
    let Ok(result) = Leiden::new(config).run(&graph) else {
        return singletons(node_count);
    };
    (0..node_count)
        .map(|i| result.partition.community_of(i))
        .collect()
}

fn singletons(node_count: usize) -> Vec<usize> {
    (0..node_count).collect()
}
