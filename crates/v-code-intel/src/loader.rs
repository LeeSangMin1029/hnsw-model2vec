//! Load code chunks from a database with bincode caching.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use v_hnsw_core::{PayloadStore, PayloadValue};
use v_hnsw_storage::StorageEngine;

use crate::call_map_cache::CallMapCache;
use crate::lsp::{self, CallMap, LspCallResolver};
use crate::parse::{self, CodeChunk};

/// Load all code chunks from the database, using a bincode cache.
pub fn load_chunks(path: &Path) -> Result<Vec<CodeChunk>> {
    let cache = cache_path(path);
    let db_mtime = fs::metadata(path)
        .and_then(|m| m.modified())
        .ok();

    // Try cache hit
    if let Some(db_t) = db_mtime
        && let Ok(cache_meta) = fs::metadata(&cache)
        && let Ok(cache_t) = cache_meta.modified()
        && cache_t >= db_t
        && let Ok(bytes) = fs::read(&cache)
    {
        let config = bincode::config::standard();
        if let Ok((chunks, _)) =
            bincode::decode_from_slice::<Vec<CodeChunk>, _>(&bytes, config)
        {
            return Ok(chunks);
        }
    }

    // Cache miss — load from DB and save
    let chunks = load_chunks_from_db(path)?;
    let config = bincode::config::standard();
    if let Ok(bytes) = bincode::encode_to_vec(&chunks, config) {
        let _ = fs::write(&cache, bytes);
    }
    Ok(chunks)
}

/// Load chunks directly from the database (no cache).
pub fn load_chunks_from_db(path: &Path) -> Result<Vec<CodeChunk>> {
    let engine = StorageEngine::open(path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    let vector_store = engine.vector_store();
    let payload_store = engine.payload_store();
    let ids: Vec<u64> = vector_store.id_map().keys().copied().collect();

    let mut chunks = Vec::new();
    for id in ids {
        if let Ok(Some(text)) = payload_store.get_text(id)
            && let Some(mut chunk) = parse::parse_chunk(&text)
        {
            if let Ok(Some(payload)) = payload_store.get_payload(id)
                && let Some(PayloadValue::StringList(imports)) = payload.custom.get("imports") {
                    chunk.imports.clone_from(imports);
                }
            chunks.push(chunk);
        }
    }
    Ok(chunks)
}

fn cache_path(db: &Path) -> PathBuf {
    db.join("cache").join("chunks.bin")
}

/// Optional daemon hooks for graph building.
///
/// Callers that have access to `v-daemon` can provide these to enable
/// daemon-assisted graph builds (persistent LSP, no cold start).
pub struct DaemonHooks {
    /// Try to build graph via running daemon. Returns `None` if unavailable.
    pub try_graph_build: fn(&Path) -> Option<crate::graph::CallGraph>,
    /// Spawn daemon in background for next invocation.
    pub spawn: fn(&Path),
}

/// Load graph from cache or build from chunks.
///
/// Resolution strategy (in order):
/// 1. Graph cache hit → return immediately
/// 2. Daemon running → delegate `graph/build` (uses persistent LSP, no cold start)
/// 3. CallMap cache + incremental LSP → re-resolve only changed files
/// 4. LSP resolver provided → use it directly
/// 5. Auto-start LSP → spawn, resolve, shutdown
/// 6. Tree-sitter only → heuristic fallback
pub fn load_or_build_graph(
    db: &Path,
    lsp: Option<&mut LspCallResolver>,
    daemon: Option<&DaemonHooks>,
) -> Result<crate::graph::CallGraph> {
    if let Some(g) = crate::graph::CallGraph::load(db) {
        return Ok(g);
    }

    // Try daemon first (persistent LSP, no cold start).
    if lsp.is_none() {
        if let Some(hooks) = daemon {
            if let Some(g) = (hooks.try_graph_build)(db) {
                return Ok(g);
            }
            // Daemon not running — auto-start for next time.
            (hooks.spawn)(db);
        }
    }

    let chunks = load_chunks(db)?;
    let project_root = lsp::find_project_root(db)
        .unwrap_or_else(|| PathBuf::from("."));

    // Try LSP-based call resolution with incremental caching.
    let resolved_calls: CallMap = if let Some(resolver) = lsp {
        resolve_incremental(db, &chunks, &project_root, resolver)
    } else {
        auto_resolve_incremental(db, &chunks, &project_root)
    };

    let g = if resolved_calls.is_empty() {
        crate::graph::CallGraph::build(&chunks)
    } else {
        crate::graph::CallGraph::build_with_resolved_calls(&chunks, &resolved_calls)
    };

    let _ = g.save(db);
    Ok(g)
}

/// Incremental call resolution using CallMap cache + LSP resolver.
///
/// Only re-resolves files that changed since last resolution.
fn resolve_incremental(
    db: &Path,
    chunks: &[CodeChunk],
    project_root: &Path,
    resolver: &mut LspCallResolver,
) -> CallMap {
    let old_cache = CallMapCache::load(db);

    let changed_files = match &old_cache {
        Some(cache) => cache.changed_files(chunks, project_root),
        None => chunks.iter().map(|c| c.file.as_str()).collect(),
    };

    if changed_files.is_empty() {
        if let Some(cache) = &old_cache {
            eprintln!("[graph] All files unchanged — using cached CallMap");
            return cache.to_call_map();
        }
    }

    let total_files: std::collections::HashSet<&str> =
        chunks.iter().map(|c| c.file.as_str()).collect();
    eprintln!("[graph] {}/{} files changed — incremental LSP resolution",
        changed_files.len(), total_files.len());

    // Resolve only changed files via LSP
    let new_calls = resolver
        .resolve_calls_for_files(chunks, project_root, &changed_files)
        .unwrap_or_default();

    // Build new cache merging old (unchanged) + new (changed)
    let new_cache = CallMapCache::build(
        chunks,
        old_cache.as_ref(),
        &new_calls,
        &changed_files,
        project_root,
    );
    new_cache.save(db);

    new_cache.to_call_map()
}

/// Auto-start an LSP server, incrementally resolve calls, then shut it down.
fn auto_resolve_incremental(
    db: &Path,
    chunks: &[parse::CodeChunk],
    project_root: &Path,
) -> CallMap {
    // If we have a complete cache with no changes, skip LSP entirely.
    if let Some(cache) = CallMapCache::load(db) {
        let changed = cache.changed_files(chunks, project_root);
        if changed.is_empty() {
            eprintln!("[graph] All files unchanged — using cached CallMap (no LSP needed)");
            return cache.to_call_map();
        }
    }

    let mut resolver = match LspCallResolver::start(project_root) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[graph] LSP not available: {e}");
            return CallMap::new();
        }
    };

    eprintln!("[graph] LSP started for: {}", project_root.display());

    let result = resolve_incremental(db, chunks, project_root, &mut resolver);

    let _ = resolver.shutdown();
    result
}
