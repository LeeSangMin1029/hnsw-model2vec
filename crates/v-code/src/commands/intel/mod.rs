//! Code intelligence commands — structural queries on code-chunked databases.
//!
//! Provides `symbols`, `def`, `refs`, `impact`, `gather`, and `trace`
//! subcommands that parse the structured text field of code chunks
//! (produced by `chunk_code`) and answer structural navigation queries.
//!
//! These commands are read-only and do not modify the database.

mod commands;
pub mod deps;
pub mod detail;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::fs;

use anyhow::Result;

// ── Re-exports: CLI command handlers ─────────────────────────────────────

pub use commands::{run_stats, run_symbols, run_context, run_blast, run_jump, run_trace, run_strings, run_flow, run_untested};

// ── Re-exports: library types for submodules and external consumers ──────

pub use v_code_intel::bfs::build_bfs_json;
#[cfg(test)]
pub use v_code_intel::context;
#[cfg(test)]
pub use v_code_intel::gather;
pub use v_code_intel::graph;
pub use v_code_intel::helpers::{format_lines_opt, format_lines_str_opt, relative_path, grouped_json};
pub use v_code_intel::impact;
pub use v_code_intel::loader::load_chunks;
use v_code_intel::loader::{DaemonBuildResult, DaemonHooks};

/// Load or build call graph with daemon support.
pub fn load_or_build_graph(
    db: &std::path::Path,
    lsp: Option<&mut v_code_intel::lsp::LspCallResolver>,
) -> anyhow::Result<v_code_intel::graph::CallGraph> {
    let hooks = DaemonHooks {
        try_graph_build: daemon_try_graph_build,
        spawn: daemon_spawn_and_wait,
    };
    v_code_intel::loader::load_or_build_graph(db, lsp, Some(&hooks))
}

fn daemon_try_graph_build(db: &std::path::Path) -> DaemonBuildResult {
    if !v_hnsw_storage::daemon_client::is_running() {
        return DaemonBuildResult::Unavailable;
    }

    let Some(canonical) = db.canonicalize().ok() else {
        return DaemonBuildResult::Unavailable;
    };
    let Some(db_str) = canonical.to_str() else {
        return DaemonBuildResult::Unavailable;
    };

    let params = serde_json::json!({"db": db_str});
    eprintln!("[graph] Waiting for daemon LSP graph build...");
    match v_hnsw_storage::daemon_client::daemon_rpc("graph/build", params, 300) {
        Ok(result) => {
            let nodes = result.get("nodes").and_then(|v| v.as_u64()).unwrap_or(0);
            let lsp = result.get("lsp_entries").and_then(|v| v.as_u64()).unwrap_or(0);
            eprintln!("[graph] Daemon graph ready: {nodes} nodes, {lsp} LSP entries");
            v_code_intel::graph::CallGraph::load(&canonical)
                .map(DaemonBuildResult::Ready)
                .unwrap_or(DaemonBuildResult::Unavailable)
        }
        Err(e) => {
            eprintln!("[graph] Daemon build failed: {e}");
            DaemonBuildResult::Unavailable
        }
    }
}

fn daemon_spawn_and_wait(db: &std::path::Path) {
    // Non-blocking: spawn daemon in background for next invocation.
    v_hnsw_storage::daemon_client::spawn_daemon(db);
}

pub use v_code_intel::parse::CodeChunk;
#[cfg(test)]
pub use v_code_intel::parse;
#[cfg(test)]
pub use v_code_intel::reason;
pub use v_code_intel::stats::{build_stats, stats_to_json};
pub use v_code_intel::trace;

// ── Output format ────────────────────────────────────────────────────────

/// Output format for code-intel commands.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
}

// ── Shared utilities (used by commands.rs) ────────────────────────────────

fn query_cache_dir(db: &Path) -> PathBuf {
    db.join("cache")
}

/// Print cached JSON if DB unchanged, otherwise compute and cache.
pub(super) fn cached_json(db: &Path, cache_key: &str, compute: impl FnOnce() -> Result<String>) -> Result<()> {
    let cache_dir = query_cache_dir(db);
    let mut hasher = std::hash::DefaultHasher::new();
    cache_key.hash(&mut hasher);
    let hash = hasher.finish();
    let cache_file = cache_dir.join(format!("{hash:x}.json"));

    let db_mtime = fs::metadata(db).and_then(|m| m.modified()).ok();
    if let Some(db_t) = db_mtime
        && let Ok(meta) = fs::metadata(&cache_file)
        && let Ok(cache_t) = meta.modified()
        && cache_t >= db_t
        && let Ok(content) = fs::read_to_string(&cache_file)
    {
        println!("{content}");
        return Ok(());
    }

    let output = compute()?;
    let _ = fs::create_dir_all(&cache_dir);
    let _ = fs::write(&cache_file, &output);
    println!("{output}");
    Ok(())
}

/// Print chunks grouped by parent directory.
pub(crate) fn print_grouped(chunks: &[&CodeChunk], compact: bool) {
    let mut groups: BTreeMap<String, Vec<&CodeChunk>> = BTreeMap::new();
    for c in chunks {
        let dir = parent_dir(&c.file);
        groups.entry(dir).or_default().push(c);
    }

    for (dir, items) in &groups {
        println!("  {dir}/");
        for c in items {
            let filename = file_name(&c.file);
            let lines = format_lines_opt(c.lines);
            println!("    {filename}{lines}  [{kind}] {name}",
                kind = c.kind, name = c.name);
            if !compact {
                let sig = c.signature.as_deref().unwrap_or("");
                if !sig.is_empty() {
                    println!("      {sig}");
                }
            }
        }
        if !compact { println!(); }
    }
}

fn parent_dir(path: &str) -> String {
    if let Some(idx) = path.rfind('/') {
        path[..idx].to_owned()
    } else {
        ".".to_owned()
    }
}

fn file_name(path: &str) -> &str {
    if let Some(idx) = path.rfind('/') {
        &path[idx + 1..]
    } else {
        path
    }
}
