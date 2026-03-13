//! Load code chunks from a database with bincode caching.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use v_hnsw_core::{PayloadStore, PayloadValue};
use v_hnsw_storage::StorageEngine;

use crate::mir;
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
            if let Ok(Some(payload)) = payload_store.get_payload(id) {
                if let Some(PayloadValue::StringList(imports)) = payload.custom.get("imports") {
                    chunk.imports.clone_from(imports);
                }
            }
            chunks.push(chunk);
        }
    }
    Ok(chunks)
}

fn cache_path(db: &Path) -> PathBuf {
    db.join("cache").join("chunks.bin")
}

/// Load graph from cache or build from chunks.
///
/// If a Cargo.toml is found near the DB (Rust project), automatically
/// collects MIR via `cargo rustc --emit=mir` for 100% accurate call resolution.
/// Falls back to tree-sitter heuristics for non-Rust projects or on MIR failure.
pub fn load_or_build_graph(db: &Path) -> Result<crate::graph::CallGraph> {
    if let Some(g) = crate::graph::CallGraph::load(db) {
        return Ok(g);
    }

    let chunks = load_chunks(db)?;

    // Try MIR-based graph if this is a Rust project.
    let cargo_root = find_cargo_root(db);
    let g = if let Some(project_root) = cargo_root {
        match mir::collect_workspace_mir(&project_root) {
            Ok(mir_fns) => {
                let mir_map = mir::build_mir_call_map(&mir_fns);
                if mir_map.is_empty() {
                    crate::graph::CallGraph::build(&chunks)
                } else {
                    crate::graph::CallGraph::build_with_mir(&chunks, &mir_map)
                }
            }
            Err(_) => {
                crate::graph::CallGraph::build(&chunks)
            }
        }
    } else {
        crate::graph::CallGraph::build(&chunks)
    };

    let _ = g.save(db);
    Ok(g)
}

/// Walk up from the DB path to find a Cargo.toml (Rust project root).
fn find_cargo_root(db: &Path) -> Option<PathBuf> {
    // Canonicalize to handle relative paths like `.v-hnsw-code.db`.
    let abs = db.canonicalize().ok()?;
    let start = if abs.is_dir() {
        abs
    } else {
        abs.parent()?.to_path_buf()
    };
    let mut dir = start.as_path();
    for _ in 0..10 {
        if dir.join("Cargo.toml").exists() {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?;
    }
    None
}
