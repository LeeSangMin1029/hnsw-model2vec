//! `v-hnsw context` — BFS forward traversal from a symbol.
//!
//! Shows the "context" of a symbol: what it calls (callees), what those
//! call, etc., up to a configurable depth. Useful for understanding the
//! dependency cone of a function before modifying it.
//!
//! ## HOW TO EXTEND
//! - Add scoring strategies by modifying `score_fn`.
//! - Add output formats by adding branches in `run_context`.

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;

use anyhow::Result;

use super::graph::CallGraph;
use super::{load_chunks, OutputFormat, cached_json, relative_path};

/// BFS result entry with depth and score.
struct BfsEntry {
    idx: u32,
    depth: u32,
    score: f64,
}

/// Run depth-limited BFS on the callees direction.
fn bfs_forward(graph: &CallGraph, seeds: &[u32], max_depth: u32) -> Vec<BfsEntry> {
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
        let is_test = graph.is_test[idx as usize];
        let score = (1.0 / f64::from(depth + 1)) * if is_test { 0.1 } else { 1.0 };

        results.push(BfsEntry { idx, depth, score });

        if depth < max_depth {
            for &callee in &graph.callees[idx as usize] {
                if !visited[callee as usize] {
                    visited[callee as usize] = true;
                    queue.push_back((callee, depth + 1));
                }
            }
        }
    }

    results
}

/// `v-hnsw context <db> <symbol> --depth N --k K [--include-tests] [--detail]`
pub fn run_context(
    db: std::path::PathBuf,
    symbol: String,
    depth: u32,
    k: usize,
    format: OutputFormat,
    include_tests: bool,
    detail: bool,
) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        let key = format!("context:{symbol}:{depth}:{k}:{include_tests}");
        return cached_json(&db, &key, || compute_context_json(&db, &symbol, depth, k, include_tests));
    }

    let graph = load_or_build_graph(&db)?;
    let seeds = graph.resolve(&symbol);

    if seeds.is_empty() {
        println!("No symbol found matching \"{symbol}\".");
        return Ok(());
    }

    let mut entries = bfs_forward(&graph, &seeds, depth);
    if !include_tests {
        entries.retain(|e| !graph.is_test[e.idx as usize]);
    }
    entries.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    entries.truncate(k);

    println!("Context of \"{symbol}\" (depth={depth}, top {k}):\n");
    print_grouped(&graph, &entries);

    if detail {
        print_detail_annotations(&db, &graph, &entries);
    }

    Ok(())
}

/// Print reasoning annotations for BFS entries that have reason data.
fn print_detail_annotations(db: &std::path::Path, graph: &CallGraph, entries: &[BfsEntry]) {
    use std::collections::HashSet;
    use super::reason;

    let mut found = false;
    let mut seen = HashSet::new();
    for e in entries {
        let name = &graph.names[e.idx as usize];
        if !seen.insert(name.as_str()) {
            continue;
        }
        if let Ok(Some(entry)) = reason::load_reason(db, name) {
            if !found {
                println!("  [reasoning]");
                found = true;
            }
            println!("    {name}: {}", reason::one_line_summary(&entry));
        }
    }
    if found {
        println!();
    }
}

fn compute_context_json(db: &Path, symbol: &str, depth: u32, k: usize, include_tests: bool) -> Result<String> {
    let graph = load_or_build_graph(db)?;
    let seeds = graph.resolve(symbol);

    let mut entries = bfs_forward(&graph, &seeds, depth);
    if !include_tests {
        entries.retain(|e| !graph.is_test[e.idx as usize]);
    }
    entries.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    entries.truncate(k);

    let json = build_json(&graph, &entries);
    Ok(serde_json::to_string(&json)?)
}

// ── Shared helpers ───────────────────────────────────────────────────────

/// Load graph from cache or build from chunks.
pub(super) fn load_or_build_graph(db: &Path) -> Result<CallGraph> {
    if let Some(g) = CallGraph::load(db) {
        return Ok(g);
    }

    let chunks = load_chunks(db)?;
    let g = CallGraph::build(&chunks);
    let _ = g.save(db);
    Ok(g)
}

/// Print BFS results grouped by depth.
fn print_grouped(graph: &CallGraph, entries: &[BfsEntry]) {
    let mut by_depth: BTreeMap<u32, Vec<&BfsEntry>> = BTreeMap::new();
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
            let lines = format_lines(graph.lines[i]);
            let test_marker = if graph.is_test[i] { " [test]" } else { "" };
            println!("    {file}{lines}  [{kind}] {name}{test_marker}  (score={:.2})", e.score);
        }
        println!();
    }
}

/// Build JSON output for BFS results.
fn build_json(graph: &CallGraph, entries: &[BfsEntry]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert(
        "_s".to_owned(),
        serde_json::Value::String("f=file,l=lines,k=kind,n=name,d=depth,sc=score,t=test".to_owned()),
    );

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
                "sc": format!("{:.2}", e.score),
                "t": graph.is_test[i],
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
