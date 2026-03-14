//! Code intelligence (v-code) daemon handler: graph/build.
//!
//! Uses the daemon's persistent LSP server (rust-analyzer) to resolve
//! call graphs without cold-start overhead.
//! Supports incremental resolution — only re-queries LSP for changed files.

use std::path::PathBuf;

use crate::state::DaemonState;
use v_code_intel::call_map_cache::CallMapCache;

pub fn handle_graph_build(
    params: serde_json::Value,
    state: &mut DaemonState,
) -> anyhow::Result<serde_json::Value> {
    #[derive(serde::Deserialize)]
    struct GraphBuildParams {
        db: String,
    }

    let p: GraphBuildParams = serde_json::from_value(params)
        .map_err(|e| anyhow::anyhow!("Invalid graph/build params: {e}"))?;
    let db_path = PathBuf::from(&p.db);
    let db = db_path
        .canonicalize()
        .unwrap_or_else(|_| db_path.clone());

    let chunks = v_code_intel::loader::load_chunks(&db)?;
    let project_root = v_code_intel::lsp::find_project_root(&db)
        .unwrap_or_else(|| PathBuf::from("."));

    // Load cached CallMap for incremental resolution.
    let old_cache = CallMapCache::load(&db);
    let changed_files = match &old_cache {
        Some(cache) => cache.changed_files(&chunks, &project_root),
        None => chunks.iter().map(|c| c.file.as_str()).collect(),
    };

    let total_files: std::collections::HashSet<&str> =
        chunks.iter().map(|c| c.file.as_str()).collect();

    let resolved_calls = if changed_files.is_empty() {
        eprintln!("[daemon] All files unchanged — using cached CallMap");
        old_cache.as_ref().map(CallMapCache::to_call_map).unwrap_or_default()
    } else {
        eprintln!("[daemon] {}/{} files changed — incremental LSP resolution",
            changed_files.len(), total_files.len());

        // Use daemon's persistent LSP server (no cold start).
        let resolver = state.ensure_lsp(&project_root)?;
        let new_calls = resolver
            .resolve_calls_for_files(&chunks, &project_root, &changed_files)
            .unwrap_or_default();

        // Build and save updated cache.
        let new_cache = CallMapCache::build(
            &chunks,
            old_cache.as_ref(),
            &new_calls,
            &changed_files,
            &project_root,
        );
        new_cache.save(&db);

        new_cache.to_call_map()
    };

    let entry_count = resolved_calls.len();

    let graph = if resolved_calls.is_empty() {
        v_code_intel::graph::CallGraph::build(&chunks)
    } else {
        v_code_intel::graph::CallGraph::build_with_resolved_calls(&chunks, &resolved_calls)
    };

    let _ = graph.save(&db);

    eprintln!(
        "[daemon] Graph built: {} nodes, {} LSP entries ({} files re-resolved)",
        graph.len(),
        entry_count,
        changed_files.len(),
    );

    Ok(serde_json::json!({
        "status": "ok",
        "nodes": graph.len(),
        "lsp_entries": entry_count,
        "files_resolved": changed_files.len(),
        "files_cached": total_files.len() - changed_files.len(),
    }))
}
