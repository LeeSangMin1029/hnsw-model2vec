//! Reverse BFS (callers direction) from a symbol.
//!
//! Answers "if I change this symbol, what else is affected?"
//! Traverses the callers adjacency list up to a configurable depth.

use crate::bfs::{bfs_generic, BfsDirection, BfsEntryExt, HasIdx};
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

impl BfsEntryExt for BfsEntry {
    fn depth(&self) -> u32 { self.depth }

    fn is_test(&self, _graph: &CallGraph) -> bool {
        self.is_test
    }

    fn depth_label(depth: u32) -> String {
        match depth {
            0 => "target".to_owned(),
            1 => "depth 1 (direct callers)".to_owned(),
            d => format!("depth {d} (transitive callers)"),
        }
    }

    fn extra_top_level_json(entries: &[Self]) -> Vec<(String, serde_json::Value)> {
        let prod_count = entries.iter().filter(|e| e.depth > 0 && !e.is_test).count();
        let test_count = entries.iter().filter(|e| e.depth > 0 && e.is_test).count();
        vec![
            ("prod_callers".to_owned(), serde_json::json!(prod_count)),
            ("test_callers".to_owned(), serde_json::json!(test_count)),
        ]
    }
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
