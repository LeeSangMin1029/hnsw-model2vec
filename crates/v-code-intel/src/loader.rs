//! Load code chunks from a database with bincode caching.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use v_hnsw_core::{PayloadStore, PayloadValue};
use v_hnsw_storage::StorageEngine;

use crate::parse::{self, CodeChunk};

/// Load all code chunks from the database, using a bincode cache.
pub fn load_chunks(path: &Path) -> Result<Vec<CodeChunk>> {
    let cache = cache_path(path);
    // Use payload.dat mtime (not directory mtime) — directory mtime
    // doesn't update on Windows when files inside are modified.
    let db_mtime = fs::metadata(path.join("payload.dat"))
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

/// Result of a daemon graph build attempt.
pub enum DaemonBuildResult {
    /// Daemon returned a completed graph.
    Ready(crate::graph::CallGraph),
    /// Daemon is building asynchronously — don't cache the tree-sitter fallback.
    Building,
    /// Daemon not available.
    Unavailable,
}

/// Optional daemon hooks for graph building.
///
/// Callers that have access to `v-daemon` can provide these to enable
/// daemon-assisted graph builds (rustdoc + tree-sitter, no LSP).
pub struct DaemonHooks {
    /// Try to build graph via running daemon.
    pub try_graph_build: fn(&Path) -> DaemonBuildResult,
    /// Spawn daemon in background for next invocation.
    pub spawn: fn(&Path),
}

/// Load graph from cache or build from chunks.
///
/// Resolution strategy (in order):
/// 1. Graph cache hit → return immediately
/// 2. Daemon running → delegate `graph/build`
/// 3. Tree-sitter + rustdoc heuristic fallback
pub fn load_or_build_graph(
    db: &Path,
    daemon: Option<&DaemonHooks>,
) -> Result<crate::graph::CallGraph> {
    if let Some(g) = crate::graph::CallGraph::load(db) {
        return Ok(g);
    }

    // Try daemon first.
    let mut daemon_building = false;
    if let Some(hooks) = daemon {
        match (hooks.try_graph_build)(db) {
            DaemonBuildResult::Ready(g) => return Ok(g),
            DaemonBuildResult::Building => daemon_building = true,
            DaemonBuildResult::Unavailable => (hooks.spawn)(db),
        }
    }

    let chunks = load_chunks(db)?;

    // Try loading cached rustdoc type info for enrichment.
    let rustdoc = crate::rustdoc::load_cached(db);

    let g = crate::graph::CallGraph::build_with_rustdoc(&chunks, rustdoc.as_ref());

    // Don't persist tree-sitter fallback when daemon is building —
    // daemon will save the accurate graph.bin when done.
    if !daemon_building {
        let _ = g.save(db);
    }
    Ok(g)
}
