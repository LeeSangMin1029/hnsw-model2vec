//! Shared search context: pre-loaded HNSW + storage for direct (non-daemon) search.

use std::path::Path;

use anyhow::{Context, Result};
use v_hnsw_core::VectorIndex;
use v_hnsw_graph::{HnswGraph, NormalizedCosineDistance};
use v_hnsw_storage::StorageEngine;

/// Pre-loaded HNSW index + storage engine for direct search.
pub struct SearchContext {
    engine: StorageEngine,
    hnsw: HnswGraph<NormalizedCosineDistance>,
    total_docs: usize,
}

impl SearchContext {
    /// Open a database for searching (read-only).
    ///
    /// `hint` is included in error messages (e.g. "v-hnsw add" or "v-code add").
    pub fn open(db_path: &Path, hint: &str) -> Result<Self> {
        let hnsw_path = db_path.join("hnsw.bin");
        if !hnsw_path.exists() {
            anyhow::bail!("HNSW index not found. Run '{hint}' first.");
        }
        let engine = StorageEngine::open(db_path).context("Failed to open storage")?;
        let hnsw: HnswGraph<NormalizedCosineDistance> =
            HnswGraph::load(&hnsw_path, NormalizedCosineDistance)
                .context("Failed to load HNSW index")?;
        let total_docs = hnsw.len();
        Ok(Self { engine, hnsw, total_docs })
    }

    /// Consume self, returning (engine, hnsw, total_docs).
    pub fn into_parts(self) -> (StorageEngine, HnswGraph<NormalizedCosineDistance>, usize) {
        (self.engine, self.hnsw, self.total_docs)
    }
}
