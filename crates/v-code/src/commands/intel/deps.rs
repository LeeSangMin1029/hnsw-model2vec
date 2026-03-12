//! `v-hnsw deps` — bidirectional file-level dependency graph.
//!
//! CLI command handler for the deps subcommand. Analysis logic lives in
//! `v_code_intel::deps`.

use std::path::Path;

use anyhow::Result;

use super::{cached_json, load_chunks, OutputFormat};
pub use v_code_intel::deps::{DepGraph, collect_transitive_files, merge_deps};
// Re-exported for tests only.
#[cfg(test)]
pub use v_code_intel::deps::{DepSet, resolve_symbol};

use std::collections::BTreeSet;

/// Schema descriptor for deps JSON output.
const DEPS_SCHEMA: &str = "o=outgoing(uses),i=incoming(used_by),n=name,v=via";

/// Run the `deps` subcommand.
pub fn run_deps(
    db: std::path::PathBuf,
    file: Option<String>,
    format: OutputFormat,
    depth: usize,
) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        let key = format!("deps:{}:{depth}", file.as_deref().unwrap_or("*"));
        return cached_json(&db, &key, || compute_deps_json(&db, file.as_deref(), depth));
    }

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
    println!("{file}  (\u{2192}{out_count} \u{2190}{in_count})");

    if !node.outgoing.is_empty() {
        for (dep, vias) in &merge_deps(&node.outgoing) {
            let via_str = vias.join(", ");
            println!("  \u{2192} {dep} (via {via_str})");
        }
    }
    if !node.incoming.is_empty() {
        for (dep, vias) in &merge_deps(&node.incoming) {
            let via_str = vias.join(", ");
            println!("  \u{2190} {dep} (via {via_str})");
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

/// Compute deps JSON output (bidirectional).
fn compute_deps_json(db: &Path, file: Option<&str>, depth: usize) -> Result<String> {
    let chunks = load_chunks(db)?;
    let graph = DepGraph::build(&chunks);

    let files: BTreeSet<String> = if let Some(target) = file {
        collect_transitive_files(&graph, target, depth)
    } else {
        graph.nodes.keys().cloned().collect()
    };

    let mut map = serde_json::Map::new();
    map.insert(
        "_s".to_owned(),
        serde_json::Value::String(DEPS_SCHEMA.to_owned()),
    );

    for f in &files {
        if let Some(node) = graph.nodes.get(f) {
            if node.outgoing.is_empty() && node.incoming.is_empty() {
                continue;
            }

            let outgoing: Vec<serde_json::Value> = merge_deps(&node.outgoing)
                .into_iter()
                .map(|(dep, vias)| serde_json::json!({"n": dep, "v": vias}))
                .collect();

            let incoming: Vec<serde_json::Value> = merge_deps(&node.incoming)
                .into_iter()
                .map(|(dep, vias)| serde_json::json!({"n": dep, "v": vias}))
                .collect();

            map.insert(
                f.clone(),
                serde_json::json!({"o": outgoing, "i": incoming}),
            );
        }
    }

    Ok(serde_json::to_string(&serde_json::Value::Object(map))?)
}

