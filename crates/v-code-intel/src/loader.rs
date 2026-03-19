//! Load code chunks from a database with bincode caching.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use v_hnsw_core::{PayloadStore, PayloadValue};
use v_hnsw_storage::StorageEngine;

use crate::parse::{self, ParsedChunk};

/// Cache format version for `chunks.bin` — bump when `ParsedChunk` layout changes.
const CHUNKS_CACHE_VERSION: u8 = 1;

/// Load all code chunks from the database, using a bincode cache.
pub fn load_chunks(path: &Path) -> Result<Vec<ParsedChunk>> {
    let cache = cache_path(path);
    // Use payload.dat mtime (not directory mtime) — directory mtime
    // doesn't update on Windows when files inside are modified.
    let db_mtime = fs::metadata(path.join("payload.dat"))
        .and_then(|m| m.modified())
        .ok();

    // Try cache hit: version prefix byte + bincode payload.
    if let Some(db_t) = db_mtime
        && let Ok(cache_meta) = fs::metadata(&cache)
        && let Ok(cache_t) = cache_meta.modified()
        && cache_t >= db_t
        && let Ok(bytes) = fs::read(&cache)
        && bytes.first() == Some(&CHUNKS_CACHE_VERSION)
    {
        let config = bincode::config::standard();
        if let Ok((chunks, _)) =
            bincode::decode_from_slice::<Vec<ParsedChunk>, _>(&bytes[1..], config)
        {
            eprintln!("  chunks.bin cache hit: {} chunks from {:.1}MB", chunks.len(), bytes.len() as f64 / 1_048_576.0);
            return Ok(chunks);
        }
    }

    // Cache miss — load from DB and save
    eprintln!("  chunks.bin cache miss — loading from DB");
    let chunks = load_chunks_from_db(path)?;
    save_chunks_cache(&cache, &chunks);
    Ok(chunks)
}

/// Load chunks directly from the database (no cache).
pub fn load_chunks_from_db(path: &Path) -> Result<Vec<ParsedChunk>> {
    let engine = StorageEngine::open(path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    let vector_store = engine.vector_store();
    let payload_store = engine.payload_store();
    let mut ids: Vec<u64> = vector_store.id_map().keys().copied().collect();
    // Sort by ID for deterministic chunk ordering (HashMap iteration is random).
    ids.sort_unstable();

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

/// Save parsed chunks to the bincode cache file.
///
/// Called internally after DB load, and externally by `v-code add`
/// to pre-build the cache (avoids expensive text re-parsing on first verify).
pub fn save_chunks_cache(path: &Path, chunks: &[ParsedChunk]) {
    let config = bincode::config::standard();
    // Prepend version byte, then encode chunks.
    let mut bytes = vec![CHUNKS_CACHE_VERSION];
    if let Ok(chunk_bytes) = bincode::encode_to_vec(chunks, config) {
        bytes.extend_from_slice(&chunk_bytes);
        let _ = fs::write(path, bytes);
    }
}

/// Path to the chunks.bin cache file for a given database.
pub fn cache_path(db: &Path) -> PathBuf {
    db.join("cache").join("chunks.bin")
}

/// Result of a daemon graph build attempt.
pub enum DaemonBuildResult {
    /// Daemon returned a completed graph.
    Ready(Box<crate::graph::CallGraph>),
    /// Daemon is building asynchronously — don't cache the tree-sitter fallback.
    Building,
    /// Daemon not available.
    Unavailable,
}

/// Optional daemon hooks for graph building.
///
/// Callers that have access to `v-daemon` can provide these to enable
/// daemon-assisted graph builds.
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
/// 3. Tree-sitter + lightweight type inference fallback
pub fn load_or_build_graph(
    db: &Path,
    daemon: Option<&DaemonHooks>,
) -> Result<crate::graph::CallGraph> {
    let (g, _) = load_or_build_graph_with_chunks(db, daemon)?;
    Ok(g)
}

/// Load or build the call graph, also returning the parsed chunks if they were loaded.
///
/// Returns `(graph, Some(chunks))` when chunks were loaded for graph building,
/// or `(graph, None)` when the graph was loaded from cache (chunks not needed).
/// Callers that need both graph and chunks can avoid double-loading.
pub fn load_or_build_graph_with_chunks(
    db: &Path,
    daemon: Option<&DaemonHooks>,
) -> Result<(crate::graph::CallGraph, Option<Vec<ParsedChunk>>)> {
    if let Some(g) = crate::graph::CallGraph::load(db) {
        return Ok((g, None));
    }

    // Try daemon first.
    let mut daemon_building = false;
    if let Some(hooks) = daemon {
        match (hooks.try_graph_build)(db) {
            DaemonBuildResult::Ready(g) => return Ok((*g, None)),
            DaemonBuildResult::Building => daemon_building = true,
            DaemonBuildResult::Unavailable => (hooks.spawn)(db),
        }
    }

    let chunks = load_chunks(db)?;

    let g = crate::graph::CallGraph::build_full(&chunks);

    // Don't persist tree-sitter fallback when daemon is building —
    // daemon will save the accurate graph.bin when done.
    if !daemon_building {
        let _ = g.save(db);
    }
    Ok((g, Some(chunks)))
}
