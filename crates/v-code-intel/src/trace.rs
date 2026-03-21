//! Shortest call path between two symbols.
//!
//! Uses bidirectional BFS on the call graph (callees + callers) to find
//! the shortest path from symbol A to symbol B.

use std::collections::VecDeque;

use crate::graph::CallGraph;
use crate::helpers::{format_lines_str_opt, relative_path};

/// BFS to find shortest path from any source index to any target index.
///
/// Traverses both callees and callers (bidirectional edges) so paths through
/// common ancestors (e.g. A ← parent → B) can be found.
/// Returns the path as a list of chunk indices (source first, target last),
/// or `None` if no path exists.
pub fn bfs_shortest_path(graph: &CallGraph, sources: &[u32], targets: &[u32]) -> Option<Vec<u32>> {
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

    // BFS through callees + callers (undirected traversal).
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

        let i = idx as usize;
        let neighbors = graph.callees[i].iter().chain(graph.callers[i].iter());
        for &next in neighbors {
            if !visited[next as usize] {
                visited[next as usize] = true;
                parent[next as usize] = Some(idx);
                queue.push_back(next);
            }
        }
    }

    None
}

/// Build JSON representation of a trace path.
pub fn build_json(graph: &CallGraph, path: &[u32]) -> serde_json::Value {
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
