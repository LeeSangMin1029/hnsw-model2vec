//! `v-hnsw impact` — reverse BFS (callers direction) from a symbol.
//!
//! Answers "if I change this symbol, what else is affected?"
//! Traverses the callers adjacency list up to a configurable depth.
//!
//! ## HOW TO EXTEND
//! - Add risk scoring by weighing test vs prod callers differently.
//! - Add `--exclude` flag to skip known stable callers.

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;

use anyhow::Result;

use super::graph::CallGraph;
use super::{cached_json, relative_path, OutputFormat};
use super::context::load_or_build_graph;

/// BFS result entry with depth.
struct BfsEntry {
    idx: u32,
    depth: u32,
    is_test: bool,
}

/// Run depth-limited BFS on the callers direction (reverse).
fn bfs_reverse(graph: &CallGraph, seeds: &[u32], max_depth: u32) -> Vec<BfsEntry> {
    let mut visited = vec![false; graph.len()];
    let mut queue: VecDeque<(u32, u32)> = VecDeque::new();
    let mut results = Vec::new();

    for &seed in seeds {
        if (seed as usize) < graph.len() && !visited[seed as usize] {
            visited[seed as usize] = true;
            queue.push_back((seed, 0));
        }
    }

    while let Some((idx, depth)) = queue.pop_front() {
        results.push(BfsEntry {
            idx,
            depth,
            is_test: graph.is_test[idx as usize],
        });

        if depth < max_depth {
            for &caller in &graph.callers[idx as usize] {
                if !visited[caller as usize] {
                    visited[caller as usize] = true;
                    queue.push_back((caller, depth + 1));
                }
            }
        }
    }

    results
}

/// `v-hnsw impact <db> <symbol> --depth N`
pub fn run_impact(
    db: std::path::PathBuf,
    symbol: String,
    depth: u32,
    format: OutputFormat,
) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        let key = format!("impact:{symbol}:{depth}");
        return cached_json(&db, &key, || compute_impact_json(&db, &symbol, depth));
    }

    let graph = load_or_build_graph(&db)?;
    let seeds = graph.resolve(&symbol);

    if seeds.is_empty() {
        println!("No symbol found matching \"{symbol}\".");
        return Ok(());
    }

    let entries = bfs_reverse(&graph, &seeds, depth);

    // Summary counts.
    let prod_count = entries.iter().filter(|e| e.depth > 0 && !e.is_test).count();
    let test_count = entries.iter().filter(|e| e.depth > 0 && e.is_test).count();

    println!("Impact of \"{symbol}\" (depth={depth}):");
    println!("  {prod_count} production callers, {test_count} test callers\n");
    print_grouped(&graph, &entries);

    Ok(())
}

fn compute_impact_json(db: &Path, symbol: &str, depth: u32) -> Result<String> {
    let graph = load_or_build_graph(db)?;
    let seeds = graph.resolve(symbol);
    let entries = bfs_reverse(&graph, &seeds, depth);
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
            let lines = format_lines(graph.lines[i]);
            let test_marker = if e.is_test { " [test]" } else { "" };
            println!("    {file}{lines}  [{kind}] {name}{test_marker}");
        }
        println!();
    }
}

fn build_json(graph: &CallGraph, entries: &[BfsEntry]) -> serde_json::Value {
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
                "l": format_lines_str(graph.lines[i]),
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

fn format_lines(lines: Option<(usize, usize)>) -> String {
    if let Some((s, e)) = lines {
        format!(":{s}-{e}")
    } else {
        String::new()
    }
}

fn format_lines_str(lines: Option<(usize, usize)>) -> String {
    if let Some((s, e)) = lines {
        format!("{s}-{e}")
    } else {
        String::new()
    }
}
