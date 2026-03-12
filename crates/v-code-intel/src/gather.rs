//! Combined context + impact BFS for full symbol understanding.
//!
//! Merges forward BFS (callees) and reverse BFS (callers) results to produce
//! a single "read this code to understand the symbol" view.

use std::collections::BTreeMap;

use crate::bfs::{bfs_generic, BfsDirection, BfsEntryExt, HasIdx};
use crate::graph::CallGraph;

/// Unified BFS entry with direction and score.
pub struct GatherEntry {
    pub idx: u32,
    pub depth: u32,
    pub score: f64,
    pub direction: Direction,
}

impl HasIdx for GatherEntry {
    fn idx(&self) -> u32 { self.idx }
}

impl BfsEntryExt for GatherEntry {
    fn depth(&self) -> u32 { self.depth }

    fn is_test(&self, graph: &CallGraph) -> bool {
        graph.is_test[self.idx as usize]
    }

    fn extra_json_fields(&self) -> Vec<(&'static str, serde_json::Value)> {
        vec![
            ("sc", serde_json::Value::String(format!("{:.2}", self.score))),
            ("dir", serde_json::Value::String(self.direction.label().to_owned())),
        ]
    }

    fn schema_suffix() -> &'static str { "sc=score,dir=direction" }

    fn extra_text_suffix(&self) -> String {
        format!("  ({}, score={:.2})", self.direction.label(), self.score)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Direction {
    Forward,
    Reverse,
}

impl Direction {
    pub fn label(self) -> &'static str {
        match self {
            Self::Forward => "callee",
            Self::Reverse => "caller",
        }
    }
}

/// BFS in a given direction, collecting scored entries.
pub fn bfs_directed(
    graph: &CallGraph,
    seeds: &[u32],
    max_depth: u32,
    include_tests: bool,
    direction: Direction,
) -> Vec<GatherEntry> {
    let bfs_dir = match direction {
        Direction::Forward => BfsDirection::Forward,
        Direction::Reverse => BfsDirection::Reverse,
    };
    bfs_generic(graph, seeds, max_depth, bfs_dir, |idx, depth| {
        let is_test = graph.is_test[idx as usize];
        if !include_tests && is_test {
            return None;
        }
        let score = (1.0 / f64::from(depth + 1)) * if is_test { 0.1 } else { 1.0 };
        Some(GatherEntry { idx, depth, score, direction })
    })
}

/// Merge forward and reverse results, dedup by idx (keep higher score).
pub fn merge_entries(forward: Vec<GatherEntry>, reverse: Vec<GatherEntry>) -> Vec<GatherEntry> {
    let mut best: BTreeMap<u32, GatherEntry> = BTreeMap::new();

    for entry in forward.into_iter().chain(reverse) {
        best.entry(entry.idx)
            .and_modify(|existing| {
                if entry.score > existing.score {
                    *existing = GatherEntry {
                        idx: entry.idx,
                        depth: entry.depth,
                        score: entry.score,
                        direction: entry.direction,
                    };
                }
            })
            .or_insert(entry);
    }

    let mut results: Vec<GatherEntry> = best.into_values().collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}
