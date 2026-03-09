//! `v-hnsw gather` — combined context + impact for full symbol understanding.
//!
//! Merges forward BFS (callees) and reverse BFS (callers) results to produce
//! a single "read this code to understand the symbol" view.
//!
//! ## HOW TO EXTEND
//! - Add `--direction` flag to limit to forward/reverse only.
//! - Add output format variants (e.g., file-list for editor integration).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;

use super::graph::CallGraph;
use super::{cached_json, OutputFormat, HasIdx, bfs_generic, BfsDirection, BfsEntryExt, build_bfs_json, print_bfs_grouped};
use super::context::load_or_build_graph;

/// Unified BFS entry with direction and score.
pub(crate) struct GatherEntry {
    pub(crate) idx: u32,
    pub(crate) depth: u32,
    pub(crate) score: f64,
    pub(crate) direction: Direction,
}

impl HasIdx for GatherEntry {
    fn idx(&self) -> u32 { self.idx }
}

impl BfsEntryExt for GatherEntry {
    fn depth(&self) -> u32 { self.depth }

    fn is_test(&self, graph: &super::graph::CallGraph) -> bool {
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
pub(crate) enum Direction {
    Forward,
    Reverse,
}

impl Direction {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Forward => "callee",
            Self::Reverse => "caller",
        }
    }
}

/// BFS in a given direction, collecting scored entries.
pub(crate) fn bfs_directed(
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
pub(crate) fn merge_entries(forward: Vec<GatherEntry>, reverse: Vec<GatherEntry>) -> Vec<GatherEntry> {
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

/// `v-hnsw gather <db> <symbol> --depth N --k K [--include-tests] [--detail]`
pub fn run_gather(
    db: std::path::PathBuf,
    symbol: String,
    depth: u32,
    k: usize,
    format: OutputFormat,
    include_tests: bool,
    detail: bool,
) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        let key = format!("gather:{symbol}:{depth}:{k}:{include_tests}");
        return cached_json(&db, &key, || compute_gather_json(&db, &symbol, depth, k, include_tests));
    }

    let graph = load_or_build_graph(&db)?;
    let seeds = graph.resolve(&symbol);

    if seeds.is_empty() {
        println!("No symbol found matching \"{symbol}\".");
        return Ok(());
    }

    let forward = bfs_directed(&graph, &seeds, depth, include_tests, Direction::Forward);
    let reverse = bfs_directed(&graph, &seeds, depth, include_tests, Direction::Reverse);
    let mut entries = merge_entries(forward, reverse);
    entries.truncate(k);

    println!("Gather for \"{symbol}\" (depth={depth}, top {k}):\n");
    print_bfs_grouped(&graph, &entries);

    if detail {
        super::print_detail_annotations(&db, &graph, &entries);
    }

    Ok(())
}

fn compute_gather_json(db: &Path, symbol: &str, depth: u32, k: usize, include_tests: bool) -> Result<String> {
    let graph = load_or_build_graph(db)?;
    let seeds = graph.resolve(symbol);

    let forward = bfs_directed(&graph, &seeds, depth, include_tests, Direction::Forward);
    let reverse = bfs_directed(&graph, &seeds, depth, include_tests, Direction::Reverse);
    let mut entries = merge_entries(forward, reverse);
    entries.truncate(k);

    let json = build_bfs_json(&graph, &entries);
    Ok(serde_json::to_string(&json)?)
}

