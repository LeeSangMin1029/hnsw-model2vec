//! Execution flow tree — DFS-based callee traversal with tree structure.
//!
//! Given seed symbol(s), builds a recursive tree of callees up to a
//! configurable depth limit, then renders as indented text or JSON.

use std::collections::HashSet;

use serde_json::Value;

use crate::graph::CallGraph;
use crate::helpers::{apply_alias, build_path_aliases, format_lines_opt, relative_path};

/// A node in the execution flow tree.
pub struct FlowNode {
    pub idx: u32,
    pub children: Vec<FlowNode>,
    /// `true` when this node was already expanded elsewhere in the tree.
    pub backreference: bool,
}

/// Build execution flow tree via depth-limited DFS on callees.
///
/// Each seed becomes a root. A node already expanded elsewhere appears
/// as a childless leaf with `backreference = true` (no re-expansion).
pub fn build_flow_tree(graph: &CallGraph, seeds: &[u32], max_depth: u32) -> Vec<FlowNode> {
    let mut expanded = HashSet::new();
    seeds
        .iter()
        .map(|&idx| {
            expanded.insert(idx);
            build_subtree(graph, idx, max_depth, 0, &mut expanded)
        })
        .collect()
}

fn build_subtree(
    graph: &CallGraph,
    idx: u32,
    max_depth: u32,
    current_depth: u32,
    expanded: &mut HashSet<u32>,
) -> FlowNode {
    if current_depth >= max_depth {
        return FlowNode { idx, children: Vec::new(), backreference: false };
    }

    let callees = graph.callees.get(idx as usize).map_or(&[][..], |v| v.as_slice());
    let children: Vec<FlowNode> = callees
        .iter()
        .map(|&child_idx| {
            if expanded.contains(&child_idx) {
                FlowNode { idx: child_idx, children: Vec::new(), backreference: true }
            } else {
                expanded.insert(child_idx);
                build_subtree(graph, child_idx, max_depth, current_depth + 1, expanded)
            }
        })
        .collect();

    FlowNode { idx, children, backreference: false }
}

/// Render the flow tree as indented text with box-drawing characters.
///
/// Extracts a common directory prefix from all node paths, strips it from
/// each line, and prints `base: <prefix>` at the end.
///
/// ```text
/// build_indexes  indexing.rs:18-77
///   ├─→ StorageEngine::payload_store  ../v-hnsw-storage/src/engine.rs:374-376
///   └─→ build_bm25  indexing.rs:255-280
///
/// base: crates/v-hnsw-cli/src/commands/
/// ```
pub fn render_tree(graph: &CallGraph, nodes: &[FlowNode]) -> String {
    // Collect all file paths to build multi-base aliases.
    let mut all_paths: Vec<&str> = Vec::new();
    for node in nodes {
        collect_paths(graph, node, &mut all_paths);
    }
    let (alias_map, legend) = build_path_aliases(&all_paths);

    let mut buf = String::new();
    for node in nodes {
        let i = node.idx as usize;
        let name = &graph.names[i];
        let file = relative_path(&graph.files[i]);
        let short = apply_alias(file, &alias_map);
        let lines = format_lines_opt(graph.lines[i]);
        buf.push_str(&format!("{name}  {short}{lines}\n"));
        render_children(graph, &node.children, &mut buf, "  ", &alias_map);
    }
    if !legend.is_empty() {
        buf.push('\n');
        for (alias, dir) in &legend {
            buf.push_str(&format!("{alias} = {dir}\n"));
        }
    }
    buf
}

/// Recursively collect all relative file paths from a tree.
fn collect_paths<'a>(graph: &'a CallGraph, node: &FlowNode, out: &mut Vec<&'a str>) {
    out.push(relative_path(&graph.files[node.idx as usize]));
    for child in &node.children {
        collect_paths(graph, child, out);
    }
}

fn render_children(
    graph: &CallGraph,
    children: &[FlowNode],
    buf: &mut String,
    prefix: &str,
    alias_map: &std::collections::BTreeMap<String, String>,
) {
    let count = children.len();
    for (ci, child) in children.iter().enumerate() {
        let is_last = ci == count - 1;
        let connector = if is_last { "\u{2514}\u{2500}\u{2192} " } else { "\u{251c}\u{2500}\u{2192} " };
        let extension = if is_last { "    " } else { "\u{2502}   " };

        let i = child.idx as usize;
        let name = &graph.names[i];
        let file = relative_path(&graph.files[i]);
        let short = apply_alias(file, alias_map);
        let lines = format_lines_opt(graph.lines[i]);
        let backref = if child.backreference { "  \u{21a9}" } else { "" };

        buf.push_str(&format!("{prefix}{connector}{name}  {short}{lines}{backref}\n"));

        if !child.children.is_empty() {
            let next_prefix = format!("{prefix}{extension}");
            render_children(graph, &child.children, buf, &next_prefix, alias_map);
        }
    }
}

/// Convert the flow tree to a JSON value.
pub fn tree_to_json(graph: &CallGraph, nodes: &[FlowNode]) -> Value {
    Value::Array(nodes.iter().map(|n| node_to_json(graph, n)).collect())
}

fn node_to_json(graph: &CallGraph, node: &FlowNode) -> Value {
    let i = node.idx as usize;
    let mut obj = serde_json::json!({
        "name": &graph.names[i],
        "file": relative_path(&graph.files[i]),
        "kind": &graph.kinds[i],
    });
    if let Some((s, e)) = graph.lines[i] {
        obj["lines"] = serde_json::json!(format!("{s}-{e}"));
    }
    if node.backreference {
        obj["ref"] = serde_json::json!(true);
    }
    if !node.children.is_empty() {
        obj["children"] = Value::Array(
            node.children.iter().map(|c| node_to_json(graph, c)).collect(),
        );
    }
    obj
}
