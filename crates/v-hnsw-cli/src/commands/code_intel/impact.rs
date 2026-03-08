//! `v-hnsw impact` — reverse BFS (callers direction) from a symbol.
//!
//! Answers "if I change this symbol, what else is affected?"
//! Traverses the callers adjacency list up to a configurable depth.
//!
//! ## HOW TO EXTEND
//! - Add risk scoring by weighing test vs prod callers differently.
//! - Add `--exclude` flag to skip known stable callers.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;

use super::graph::CallGraph;
use super::{cached_json, relative_path, OutputFormat, format_lines_opt, format_lines_str_opt, HasIdx, bfs_generic, BfsDirection};
use super::context::load_or_build_graph;

/// BFS result entry with depth.
pub(crate) struct BfsEntry {
    pub(crate) idx: u32,
    pub(crate) depth: u32,
    pub(crate) is_test: bool,
}

impl HasIdx for BfsEntry {
    fn idx(&self) -> u32 { self.idx }
}

/// Run depth-limited BFS on the callers direction (reverse).
pub(crate) fn bfs_reverse(graph: &CallGraph, seeds: &[u32], max_depth: u32) -> Vec<BfsEntry> {
    bfs_generic(graph, seeds, max_depth, BfsDirection::Reverse, |idx, depth| {
        Some(BfsEntry {
            idx,
            depth,
            is_test: graph.is_test[idx as usize],
        })
    })
}

/// `v-hnsw impact <db> <symbol> --depth N [--include-tests] [--detail]`
pub fn run_impact(
    db: std::path::PathBuf,
    symbol: String,
    depth: u32,
    format: OutputFormat,
    include_tests: bool,
    detail: bool,
) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        let key = format!("impact:{symbol}:{depth}:{include_tests}");
        return cached_json(&db, &key, || compute_impact_json(&db, &symbol, depth, include_tests));
    }

    let graph = load_or_build_graph(&db)?;
    let seeds = graph.resolve(&symbol);

    if seeds.is_empty() {
        println!("No symbol found matching \"{symbol}\".");
        return Ok(());
    }

    let all_entries = bfs_reverse(&graph, &seeds, depth);

    // Summary counts (always based on all entries).
    let prod_count = all_entries.iter().filter(|e| e.depth > 0 && !e.is_test).count();
    let test_count = all_entries.iter().filter(|e| e.depth > 0 && e.is_test).count();

    let entries: Vec<BfsEntry> = if include_tests {
        all_entries
    } else {
        all_entries.into_iter().filter(|e| !e.is_test).collect()
    };

    println!("Impact of \"{symbol}\" (depth={depth}):");
    println!("  {prod_count} production callers, {test_count} test callers\n");
    print_grouped(&graph, &entries);

    if detail {
        super::print_detail_annotations(&db, &graph, &entries);
    }

    Ok(())
}

fn compute_impact_json(db: &Path, symbol: &str, depth: u32, include_tests: bool) -> Result<String> {
    let graph = load_or_build_graph(db)?;
    let seeds = graph.resolve(symbol);
    let all_entries = bfs_reverse(&graph, &seeds, depth);
    let entries: Vec<BfsEntry> = if include_tests {
        all_entries
    } else {
        all_entries.into_iter().filter(|e| !e.is_test).collect()
    };
    let json = build_json(&graph, &entries);
    Ok(serde_json::to_string(&json)?)
}

// ── Output helpers ───────────────────────────────────────────────────────

fn print_grouped(graph: &CallGraph, entries: &[BfsEntry]) {
    let mut by_depth: BTreeMap<u32, Vec<&BfsEntry>> = BTreeMap::new();
    for e in entries {
        by_depth.entry(e.depth).or_default().push(e);
    }

    for (depth, items) in &by_depth {
        let label = match *depth {
            0 => "target".to_owned(),
            1 => "depth 1 (direct callers)".to_owned(),
            d => format!("depth {d} (transitive callers)"),
        };
        println!("  [{label}]");
        for e in items {
            let i = e.idx as usize;
            let file = relative_path(&graph.files[i]);
            let name = &graph.names[i];
            let kind = &graph.kinds[i];
            let lines = format_lines_opt(graph.lines[i]);
            let test_marker = if e.is_test { " [test]" } else { "" };
            println!("    {file}{lines}  [{kind}] {name}{test_marker}");
        }
        println!();
    }
}

pub(crate) fn build_json(graph: &CallGraph, entries: &[BfsEntry]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert(
        "_s".to_owned(),
        serde_json::Value::String("f=file,l=lines,k=kind,n=name,d=depth,t=test".to_owned()),
    );

    let prod_count = entries.iter().filter(|e| e.depth > 0 && !e.is_test).count();
    let test_count = entries.iter().filter(|e| e.depth > 0 && e.is_test).count();
    map.insert("prod_callers".to_owned(), serde_json::json!(prod_count));
    map.insert("test_callers".to_owned(), serde_json::json!(test_count));

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
                "t": e.is_test,
            })
        })
        .collect();

    map.insert("results".to_owned(), serde_json::Value::Array(items));
    serde_json::Value::Object(map)
}

