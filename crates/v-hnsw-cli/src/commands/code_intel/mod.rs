//! Code intelligence commands — structural queries on code-chunked databases.
//!
//! Provides `symbols`, `def`, `refs`, `impact`, `gather`, and `trace`
//! subcommands that parse the structured text field of code chunks
//! (produced by `chunk_code`) and answer structural navigation queries.
//!
//! These commands are read-only and do not modify the database.

mod commands;
pub mod deps;
pub(crate) mod deps_html;
pub mod detail;
mod json_api;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::fs;

use anyhow::Result;

// ── Re-exports: CLI command handlers ─────────────────────────────────────

pub use commands::{run_stats, run_symbols, run_def, run_refs, run_gather, run_impact, run_trace};

// ── Re-exports: JSON API (daemon/handler) ────────────────────────────────

pub use json_api::{
    stats_as_json, def_as_json, refs_as_json, symbols_as_json,
    gather_as_json, impact_as_json, trace_as_json, detail_as_json,
};
pub(crate) use json_api::try_inline_lookup;

// ── Re-exports: library types for submodules and external consumers ──────

pub use v_hnsw_intel::bfs::{BfsEntryExt, HasIdx, build_bfs_json, print_bfs_grouped};
#[cfg(test)]
pub use v_hnsw_intel::context;
pub use v_hnsw_intel::gather;
pub use v_hnsw_intel::graph;
pub use v_hnsw_intel::helpers::{format_lines_opt, relative_path, grouped_json, grouped_json_refs};
#[cfg(test)]
pub use v_hnsw_intel::helpers::{format_lines_str_opt, lines_str, extract_crate_name};
pub use v_hnsw_intel::impact;
pub use v_hnsw_intel::loader::{load_chunks, load_or_build_graph};
pub use v_hnsw_intel::parse::CodeChunk;
#[cfg(test)]
pub use v_hnsw_intel::parse;
pub use v_hnsw_intel::reason;
pub use v_hnsw_intel::stats::{build_stats, stats_to_json};
pub use v_hnsw_intel::trace;

// ── Output format ────────────────────────────────────────────────────────

/// Output format for code-intel commands.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
    /// Standalone HTML with interactive D3.js force-directed graph.
    Html,
}

// ── Shared utilities (used by commands.rs and json_api.rs) ───────────────

/// Print reasoning annotations for entries that have reason data.
pub(crate) fn print_detail_annotations(
    db: &Path,
    graph: &graph::CallGraph,
    entries: &[impl HasIdx],
) {
    use std::collections::HashSet;

    let mut found = false;
    let mut seen = HashSet::new();
    for e in entries {
        let name = &graph.names[e.idx() as usize];
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
pub(crate) fn print_grouped(chunks: &[&CodeChunk], _label: Option<&str>) {
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
            let sig = c.signature.as_deref().unwrap_or("");
            println!("    {filename}{lines}  [{kind}] {name}",
                kind = c.kind, name = c.name);
            if !sig.is_empty() {
                println!("      {sig}");
            }
        }
        println!();
    }
}

fn find_refs<'a>(chunks: &'a [CodeChunk], name: &str) -> Vec<(&'a CodeChunk, Vec<&'static str>)> {
    let name_lower = name.to_lowercase();
    chunks
        .iter()
        .filter_map(|c| {
            let mut via = Vec::new();
            if c.calls.iter().any(|s| s.to_lowercase().contains(&name_lower)) {
                via.push("calls");
            }
            if c.types.iter().any(|s| s.to_lowercase().contains(&name_lower)) {
                via.push("types");
            }
            if c.signature.as_ref().is_some_and(|s| s.to_lowercase().contains(&name_lower)) {
                via.push("signature");
            }
            if c.name.to_lowercase().contains(&name_lower) {
                via.push("name");
            }
            if via.is_empty() { None } else { Some((c, via)) }
        })
        .collect()
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
