//! BFS forward traversal (context) — what a symbol calls.
//!
//! Shows the "context" of a symbol: what it calls (callees), what those
//! call, etc., up to a configurable depth.

use crate::bfs::{bfs_generic, BfsDirection, BfsEntryExt, HasIdx};
use crate::graph::CallGraph;

/// BFS result entry with depth and score.
pub struct BfsEntry {
    pub idx: u32,
    pub depth: u32,
    pub score: f64,
}

impl HasIdx for BfsEntry {
    fn idx(&self) -> u32 { self.idx }
}

impl BfsEntryExt for BfsEntry {
    fn depth(&self) -> u32 { self.depth }

    fn is_test(&self, graph: &CallGraph) -> bool {
        graph.is_test[self.idx as usize]
    }

    fn extra_json_fields(&self) -> Vec<(&'static str, serde_json::Value)> {
        vec![("sc", serde_json::Value::String(format!("{:.2}", self.score)))]
    }

    fn schema_suffix() -> &'static str { "sc=score" }

    fn extra_text_suffix(&self) -> String {
        format!("  (score={:.2})", self.score)
    }
}

/// Run depth-limited BFS on the callees direction.
pub fn bfs_forward(graph: &CallGraph, seeds: &[u32], max_depth: u32) -> Vec<BfsEntry> {
    bfs_generic(graph, seeds, max_depth, BfsDirection::Forward, |idx, depth| {
        let is_test = graph.is_test[idx as usize];
        let score = (1.0 / f64::from(depth + 1)) * if is_test { 0.1 } else { 1.0 };
        Some(BfsEntry { idx, depth, score })
    })
}
