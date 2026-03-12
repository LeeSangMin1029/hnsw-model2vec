//! Generic BFS traversal on the call graph.
//!
//! Provides `bfs_generic`, `BfsDirection`, `HasIdx`, and `BfsEntryExt` —
//! the building blocks for context, gather, impact, and trace modules.

use std::collections::{BTreeMap, VecDeque};

use crate::graph::CallGraph;
use crate::helpers::{format_lines_opt, format_lines_str_opt, relative_path};

// ── Shared trait for BFS entry types ─────────────────────────────────────

/// Trait for BFS entry types that carry a graph index.
///
/// Implemented by `context::BfsEntry`, `gather::GatherEntry`, and
/// `impact::BfsEntry` so that shared display logic can be reused.
pub trait HasIdx {
    fn idx(&self) -> u32;
}

/// Trait for BFS entry types that can be serialized to JSON and printed.
///
/// Each entry type provides its depth, test status, optional score/direction,
/// and a depth label for grouped text output.
pub trait BfsEntryExt: HasIdx {
    fn depth(&self) -> u32;
    fn is_test(&self, graph: &CallGraph) -> bool;

    /// Extra JSON fields beyond the common `f/l/k/n/d/t`.
    fn extra_json_fields(&self) -> Vec<(&'static str, serde_json::Value)> {
        Vec::new()
    }

    /// Schema suffix appended after the common `f=file,l=lines,k=kind,n=name,d=depth,t=test`.
    fn schema_suffix() -> &'static str { "" }

    /// Label for a given depth level in text output.
    fn depth_label(depth: u32) -> String {
        match depth {
            0 => "target".to_owned(),
            1 => "depth 1 (direct)".to_owned(),
            d => format!("depth {d}"),
        }
    }

    /// Extra text suffix after the standard `file:lines [kind] name [test]` line.
    fn extra_text_suffix(&self) -> String {
        String::new()
    }

    /// Extra top-level JSON fields inserted before `results`.
    fn extra_top_level_json(_entries: &[Self]) -> Vec<(String, serde_json::Value)>
    where Self: Sized
    {
        Vec::new()
    }
}

/// Build JSON output for any BFS entry type implementing `BfsEntryExt`.
pub fn build_bfs_json<E: BfsEntryExt>(
    graph: &CallGraph,
    entries: &[E],
) -> serde_json::Value {
    let mut map = serde_json::Map::new();

    let suffix = E::schema_suffix();
    let schema = if suffix.is_empty() {
        "f=file,l=lines,k=kind,n=name,d=depth,t=test".to_owned()
    } else {
        format!("f=file,l=lines,k=kind,n=name,d=depth,t=test,{suffix}")
    };
    map.insert("_s".to_owned(), serde_json::Value::String(schema));

    for (key, val) in E::extra_top_level_json(entries) {
        map.insert(key, val);
    }

    let items: Vec<serde_json::Value> = entries
        .iter()
        .map(|e| {
            let i = e.idx() as usize;
            let mut obj = serde_json::json!({
                "f": relative_path(&graph.files[i]),
                "l": format_lines_str_opt(graph.lines[i]),
                "k": &graph.kinds[i],
                "n": &graph.names[i],
                "d": e.depth(),
                "t": e.is_test(graph),
            });
            for (key, val) in e.extra_json_fields() {
                obj[key] = val;
            }
            obj
        })
        .collect();

    map.insert("results".to_owned(), serde_json::Value::Array(items));
    serde_json::Value::Object(map)
}

/// Print BFS results grouped by depth for any entry type implementing `BfsEntryExt`.
pub fn print_bfs_grouped<E: BfsEntryExt>(graph: &CallGraph, entries: &[E]) {
    let mut by_depth: BTreeMap<u32, Vec<&E>> = BTreeMap::new();
    for e in entries {
        by_depth.entry(e.depth()).or_default().push(e);
    }

    for (depth, items) in &by_depth {
        let label = E::depth_label(*depth);
        println!("  [{label}]");
        for e in items {
            let i = e.idx() as usize;
            let file = relative_path(&graph.files[i]);
            let name = &graph.names[i];
            let kind = &graph.kinds[i];
            let lines = format_lines_opt(graph.lines[i]);
            let test_marker = if e.is_test(graph) { " [test]" } else { "" };
            let suffix = e.extra_text_suffix();
            println!("    {file}{lines}  [{kind}] {name}{test_marker}{suffix}");
        }
        println!();
    }
}

// ── Generic BFS ──────────────────────────────────────────────────────────

/// Adjacency selector for BFS direction.
pub enum BfsDirection {
    /// Follow callees (forward traversal).
    Forward,
    /// Follow callers (reverse traversal).
    Reverse,
}

/// Run a depth-limited BFS on the call graph.
///
/// The `direction` parameter selects which adjacency list to follow.
/// For each visited node the `make_entry` callback produces the result entry;
/// returning `None` skips the node (useful for test filtering) but still
/// continues BFS through its neighbours.
pub fn bfs_generic<T>(
    graph: &CallGraph,
    seeds: &[u32],
    max_depth: u32,
    direction: BfsDirection,
    mut make_entry: impl FnMut(u32, u32) -> Option<T>,
) -> Vec<T> {
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
        if let Some(entry) = make_entry(idx, depth) {
            results.push(entry);
        }

        if depth < max_depth {
            let neighbours = match direction {
                BfsDirection::Forward => &graph.callees[idx as usize],
                BfsDirection::Reverse => &graph.callers[idx as usize],
            };
            for &next in neighbours {
                if !visited[next as usize] {
                    visited[next as usize] = true;
                    queue.push_back((next, depth + 1));
                }
            }
        }
    }

    results
}
