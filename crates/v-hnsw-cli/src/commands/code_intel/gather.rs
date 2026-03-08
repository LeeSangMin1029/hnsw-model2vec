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
use super::{cached_json, relative_path, OutputFormat, format_lines_opt, format_lines_str_opt, HasIdx, bfs_generic, BfsDirection};
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
    print_gathered(&graph, &entries);

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

    let json = build_json(&graph, &entries);
    Ok(serde_json::to_string(&json)?)
}

// ── Output helpers ───────────────────────────────────────────────────────

fn print_gathered(graph: &CallGraph, entries: &[GatherEntry]) {
    let mut by_depth: BTreeMap<u32, Vec<&GatherEntry>> = BTreeMap::new();
    for e in entries {
        by_depth.entry(e.depth).or_default().push(e);
    }

    for (depth, items) in &by_depth {
        let label = match *depth {
            0 => "target".to_owned(),
            1 => "depth 1 (direct)".to_owned(),
            d => format!("depth {d}"),
        };
        println!("  [{label}]");
        for e in items {
            let i = e.idx as usize;
            let file = relative_path(&graph.files[i]);
            let name = &graph.names[i];
            let kind = &graph.kinds[i];
            let lines = format_lines_opt(graph.lines[i]);
            let dir_tag = e.direction.label();
            let test_marker = if graph.is_test[i] { " [test]" } else { "" };
            println!("    {file}{lines}  [{kind}] {name}{test_marker}  ({dir_tag}, score={:.2})", e.score);
        }
        println!();
    }
}

pub(crate) fn build_json(graph: &CallGraph, entries: &[GatherEntry]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert(
        "_s".to_owned(),
        serde_json::Value::String("f=file,l=lines,k=kind,n=name,d=depth,sc=score,dir=direction,t=test".to_owned()),
    );

    let items: Vec<serde_json::Value> = entries
        .iter()
        .map(|e| {
            let i = e.idx as usize;
            serde_json::json!({
                "f": relative_path(&graph.files[i]),
                "l": format_lines_str_opt(graph.lines[i]),
                "k": &graph.kinds[i],
                "n": &graph.names[i],
                "d": e.depth,
                "sc": format!("{:.2}", e.score),
                "dir": e.direction.label(),
                "t": graph.is_test[i],
            })
        })
        .collect();

    map.insert("results".to_owned(), serde_json::Value::Array(items));
    serde_json::Value::Object(map)
}

