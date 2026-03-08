//! `v-hnsw trace` — find shortest call path between two symbols.
//!
//! Uses BFS on the call graph (callees direction) to find the shortest
//! path from symbol A to symbol B. Useful for understanding how two
//! pieces of code are connected.
//!
//! ## HOW TO EXTEND
//! - Add `--reverse` flag to search in callers direction.
//! - Add `--all-paths` to show all shortest paths (same length).

use std::collections::VecDeque;
use std::path::Path;

use anyhow::Result;

use super::graph::CallGraph;
use super::{cached_json, relative_path, OutputFormat, format_lines_opt, format_lines_str_opt};
use super::context::load_or_build_graph;

/// BFS to find shortest path from any source index to any target index.
///
/// Returns the path as a list of chunk indices (source first, target last),
/// or `None` if no path exists.
pub(crate) fn bfs_shortest_path(graph: &CallGraph, sources: &[u32], targets: &[u32]) -> Option<Vec<u32>> {
    let len = graph.len();
    let mut visited = vec![false; len];
    let mut parent: Vec<Option<u32>> = vec![None; len];
    let mut queue: VecDeque<u32> = VecDeque::new();

    // Mark all targets for O(1) lookup.
    let mut is_target = vec![false; len];
    for &t in targets {
        if (t as usize) < len {
            is_target[t as usize] = true;
        }
    }

    // Seed with sources.
    for &s in sources {
        if (s as usize) < len && !visited[s as usize] {
            visited[s as usize] = true;
            queue.push_back(s);
        }
    }

    // BFS through callees.
    while let Some(idx) = queue.pop_front() {
        if is_target[idx as usize] {
            // Reconstruct path.
            let mut path = vec![idx];
            let mut current = idx;
            while let Some(p) = parent[current as usize] {
                path.push(p);
                current = p;
            }
            path.reverse();
            return Some(path);
        }

        for &callee in &graph.callees[idx as usize] {
            if !visited[callee as usize] {
                visited[callee as usize] = true;
                parent[callee as usize] = Some(idx);
                queue.push_back(callee);
            }
        }
    }

    None
}

/// `v-hnsw trace <db> <from> <to>`
pub fn run_trace(
    db: std::path::PathBuf,
    from: String,
    to: String,
    format: OutputFormat,
) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        let key = format!("trace:{from}:{to}");
        return cached_json(&db, &key, || compute_trace_json(&db, &from, &to));
    }

    let graph = load_or_build_graph(&db)?;
    let sources = graph.resolve(&from);
    let targets = graph.resolve(&to);

    if sources.is_empty() {
        println!("No symbol found matching \"{from}\".");
        return Ok(());
    }
    if targets.is_empty() {
        println!("No symbol found matching \"{to}\".");
        return Ok(());
    }

    match bfs_shortest_path(&graph, &sources, &targets) {
        Some(path) => {
            println!("Call path from \"{from}\" to \"{to}\" ({} hops):\n", path.len() - 1);
            print_path(&graph, &path);
        }
        None => {
            println!("No call path found from \"{from}\" to \"{to}\".");
        }
    }

    Ok(())
}

fn compute_trace_json(db: &Path, from: &str, to: &str) -> Result<String> {
    let graph = load_or_build_graph(db)?;
    let sources = graph.resolve(from);
    let targets = graph.resolve(to);

    let json = match bfs_shortest_path(&graph, &sources, &targets) {
        Some(path) => build_json(&graph, &path),
        None => serde_json::json!({ "path": null, "hops": null }),
    };

    Ok(serde_json::to_string(&json)?)
}

// ── Output helpers ───────────────────────────────────────────────────────

fn print_path(graph: &CallGraph, path: &[u32]) {
    for (step, &idx) in path.iter().enumerate() {
        let i = idx as usize;
        let file = relative_path(&graph.files[i]);
        let name = &graph.names[i];
        let kind = &graph.kinds[i];
        let lines = format_lines_opt(graph.lines[i]);
        let test_marker = if graph.is_test[i] { " [test]" } else { "" };

        let arrow = if step == 0 { "  " } else { "-> " };
        let indent = if step == 0 { "" } else { &"   ".repeat(step) };
        println!("  {indent}{arrow}{file}{lines}  [{kind}] {name}{test_marker}");
    }
    println!();
}

pub(crate) fn build_json(graph: &CallGraph, path: &[u32]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert(
        "_s".to_owned(),
        serde_json::Value::String("f=file,l=lines,k=kind,n=name,t=test".to_owned()),
    );
    map.insert("hops".to_owned(), serde_json::json!(path.len() - 1));

    let items: Vec<serde_json::Value> = path
        .iter()
        .map(|&idx| {
            let i = idx as usize;
            serde_json::json!({
                "f": relative_path(&graph.files[i]),
                "l": format_lines_str_opt(graph.lines[i]),
                "k": &graph.kinds[i],
                "n": &graph.names[i],
                "t": graph.is_test[i],
            })
        })
        .collect();

    map.insert("path".to_owned(), serde_json::Value::Array(items));
    serde_json::Value::Object(map)
}

