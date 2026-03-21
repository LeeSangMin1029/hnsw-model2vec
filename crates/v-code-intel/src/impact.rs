//! Reverse BFS (callers direction) from a symbol.
//!
//! Answers "if I change this symbol, what else is affected?"
//! Traverses the callers adjacency list up to a configurable depth.

use crate::bfs::{bfs_generic, BfsDirection, HasIdx};
use crate::graph::CallGraph;

/// BFS result entry with depth.
pub struct BfsEntry {
    pub idx: u32,
    pub depth: u32,
    pub is_test: bool,
}

impl HasIdx for BfsEntry {
    fn idx(&self) -> u32 { self.idx }
}

/// Run depth-limited BFS on the callers direction (reverse).
pub fn bfs_reverse(graph: &CallGraph, seeds: &[u32], max_depth: u32) -> Vec<BfsEntry> {
    bfs_generic(graph, seeds, max_depth, BfsDirection::Reverse, |idx, depth| {
        Some(BfsEntry {
            idx,
            depth,
            is_test: graph.is_test[idx as usize],
        })
    })
}
