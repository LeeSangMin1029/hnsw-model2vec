//! Code intelligence commands — structural queries on code-chunked databases.
//!
//! Provides `symbols`, `def`, `refs`, `impact`, and `trace`
//! subcommands that parse the structured text field of code chunks
//! (produced by `chunk_code`) and answer structural navigation queries.
//!
//! These commands are read-only and do not modify the database.

mod commands;
pub mod detail;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

// ── Re-exports: CLI command handlers ─────────────────────────────────────

pub use commands::{run_aliases, run_stats, run_symbols, run_context, run_blast, run_jump, run_trace, run_coverage};

// ── Re-exports: library types for submodules and external consumers ──────

#[cfg(test)]
pub use v_code_intel::context;
pub use v_code_intel::graph;
pub use v_code_intel::helpers::{format_lines_opt, format_lines_str_opt, relative_path};
pub use v_code_intel::impact;
pub use v_code_intel::loader::load_chunks;
use v_code_intel::loader::{DaemonBuildResult, DaemonHooks};

/// Load or build call graph with daemon support.
pub fn load_or_build_graph(
    db: &std::path::Path,
) -> anyhow::Result<v_code_intel::graph::CallGraph> {
    let hooks = DaemonHooks {
        try_graph_build: daemon_try_graph_build,
        spawn: daemon_spawn_and_wait,
    };
    v_code_intel::loader::load_or_build_graph(db, Some(&hooks))
}

/// Load or build call graph, also returning chunks if they were loaded.
///
/// Avoids double-loading chunks when the caller also needs them.
pub fn load_or_build_graph_with_chunks(
    db: &std::path::Path,
) -> anyhow::Result<(v_code_intel::graph::CallGraph, Option<Vec<v_code_intel::parse::ParsedChunk>>)> {
    let hooks = DaemonHooks {
        try_graph_build: daemon_try_graph_build,
        spawn: daemon_spawn_and_wait,
    };
    v_code_intel::loader::load_or_build_graph_with_chunks(db, Some(&hooks))
}

fn daemon_try_graph_build(db: &std::path::Path) -> DaemonBuildResult {
    let Some(canonical) = db.canonicalize().ok() else {
        return DaemonBuildResult::Unavailable;
    };

    // If daemon already built the graph (cached), use it immediately.
    if let Some(g) = v_code_intel::graph::CallGraph::load(&canonical) {
        return DaemonBuildResult::Ready(Box::new(g));
    }

    // Auto-start daemon if not running.
    if !v_hnsw_storage::daemon_client::is_running() {
        eprintln!("[graph] Daemon not running, starting...");
        if !v_hnsw_storage::daemon_client::spawn_daemon_and_wait(&canonical) {
            return DaemonBuildResult::Unavailable;
        }
        eprintln!("[graph] Daemon started");
    }

    // Blocking RPC — wait for daemon to build the graph (RA-based).
    let Some(db_str) = canonical.to_str() else {
        return DaemonBuildResult::Unavailable;
    };
    let params = serde_json::json!({"db": db_str});
    eprintln!("[graph] Building graph via daemon (RA)...");
    match v_hnsw_storage::daemon_client::daemon_rpc("graph/build", params, 300) {
        Ok(_) => {
            // Daemon saved graph.bin — load it.
            if let Some(g) = v_code_intel::graph::CallGraph::load(&canonical) {
                return DaemonBuildResult::Ready(Box::new(g));
            }
            DaemonBuildResult::Unavailable
        }
        Err(e) => {
            eprintln!("[graph] Daemon graph/build failed: {e}");
            DaemonBuildResult::Unavailable
        }
    }
}

fn daemon_spawn_and_wait(db: &std::path::Path) {
    if !v_hnsw_storage::daemon_client::is_running() {
        eprintln!("[graph] Starting daemon...");
        v_hnsw_storage::daemon_client::spawn_daemon_and_wait(db);
    }
}

pub use v_code_intel::parse::ParsedChunk;
#[cfg(test)]
pub use v_code_intel::parse;
#[cfg(test)]
pub use v_code_intel::reason;
pub use v_code_intel::stats::build_stats;
pub use v_code_intel::trace;

// ── Shared utilities (used by commands.rs) ────────────────────────────────

/// Print chunks grouped by file with path aliases.
pub(crate) fn print_grouped(
    chunks: &[&ParsedChunk],
    compact: bool,
    alias_map: &std::collections::BTreeMap<String, String>,
) {
    use v_code_intel::helpers::apply_alias;

    let files: Vec<&str> = chunks.iter().map(|c| relative_path(&c.file)).collect();

    let mut groups: BTreeMap<&str, Vec<&ParsedChunk>> = BTreeMap::new();
    for (c, file) in chunks.iter().zip(files.iter()) {
        groups.entry(file).or_default().push(c);
    }

    for (file, items) in &groups {
        let short = apply_alias(file, alias_map);
        println!("@ {short}");
        for c in items {
            let lines = format_lines_opt(c.lines);
            let test_marker = if v_code_intel::graph::is_test_chunk(c) { " [test]" } else { "" };
            let kind_tag = if c.kind == "function" { String::new() } else { format!("[{}] ", c.kind) };
            println!("  {lines} {kind_tag}{name}{test_marker}", name = c.name);
            if !compact {
                let sig = c.signature.as_deref().unwrap_or("");
                if !sig.is_empty() {
                    println!("    {sig}");
                }
            }
        }
        if !compact { println!(); }
    }
}

