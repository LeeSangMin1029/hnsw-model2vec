//! Blast radius summary — aggregates BFS reverse results into actionable metrics.
//!
//! Given the output of `impact::bfs_reverse`, produces a summary of
//! total affected functions, unique files, and prod/test breakdown.

use std::collections::BTreeSet;

use crate::graph::CallGraph;
use crate::helpers::relative_path;
use crate::impact;

/// Aggregated blast radius summary.
pub struct BlastSummary {
    /// Total affected functions (excludes seed at depth 0).
    pub total_affected: usize,
    /// Unique file paths (relative) containing affected functions.
    pub affected_files: Vec<String>,
    /// Number of affected production functions.
    pub prod_count: usize,
    /// Number of affected test functions.
    pub test_count: usize,
}

/// Summarize blast radius from BFS reverse entries.
///
/// Counts only entries with `depth > 0` (the seed itself is excluded).
pub fn summarize_blast(graph: &CallGraph, entries: &[impact::BfsEntry]) -> BlastSummary {
    let mut files: BTreeSet<String> = BTreeSet::new();
    let mut prod_count = 0usize;
    let mut test_count = 0usize;

    for e in entries {
        if e.depth == 0 {
            continue;
        }
        let file = relative_path(&graph.files[e.idx as usize]);
        files.insert(file.to_owned());
        if e.is_test {
            test_count += 1;
        } else {
            prod_count += 1;
        }
    }

    BlastSummary {
        total_affected: prod_count + test_count,
        affected_files: files.into_iter().collect(),
        prod_count,
        test_count,
    }
}
