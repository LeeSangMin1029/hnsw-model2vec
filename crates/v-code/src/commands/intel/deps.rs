//! `v-hnsw deps` — bidirectional file-level dependency graph.
//!
//! CLI command handler for the deps subcommand. Analysis logic lives in
//! `v_code_intel::deps`.

use anyhow::Result;

use super::load_chunks;
pub use v_code_intel::deps::{DepGraph, collect_transitive_files, merge_deps};
// Re-exported for tests only.
#[cfg(test)]
pub use v_code_intel::deps::{DepSet, resolve_symbol};

/// Run the `deps` subcommand.
pub fn run_deps(
    db: std::path::PathBuf,
    file: Option<String>,
    depth: usize,
) -> Result<()> {
    let chunks = load_chunks(&db)?;
    let graph = DepGraph::build(&chunks);

    if let Some(ref target) = file {
        print_file_deps(&graph, target, depth);
    } else {
        print_full_graph(&graph, depth);
    }
    Ok(())
}

/// Print a single file node (both directions).
fn print_node(file: &str, node: &v_code_intel::deps::FileNode) {
    let out_count = node.outgoing.len();
    let in_count = node.incoming.len();
    println!("{file}  (→{out_count} ←{in_count})");

    if !node.outgoing.is_empty() {
        for (dep, vias) in &merge_deps(&node.outgoing) {
            let via_str = vias.join(", ");
            println!("  → {dep} (via {via_str})");
        }
    }
    if !node.incoming.is_empty() {
        for (dep, vias) in &merge_deps(&node.incoming) {
            let via_str = vias.join(", ");
            println!("  ← {dep} (via {via_str})");
        }
    }
}

/// Print deps for a single file (bidirectional).
fn print_file_deps(graph: &DepGraph, target: &str, depth: usize) {
    let files = collect_transitive_files(graph, target, depth);

    if files.is_empty() {
        println!("No dependencies found for \"{target}\".");
        return;
    }

    for file in &files {
        if let Some(node) = graph.nodes.get(file) {
            print_node(file, node);
        }
    }
}

/// Print the full dependency graph (all files, both directions).
fn print_full_graph(graph: &DepGraph, _depth: usize) {
    if graph.nodes.is_empty() {
        println!("No dependencies found.");
        return;
    }

    let total_files = graph.nodes.len();
    let total_edges = graph.total_edges();
    println!("{total_files} files, {total_edges} dependency edges:\n");

    for (file, node) in &graph.nodes {
        if node.outgoing.is_empty() && node.incoming.is_empty() {
            continue;
        }
        print_node(file, node);
    }
}
