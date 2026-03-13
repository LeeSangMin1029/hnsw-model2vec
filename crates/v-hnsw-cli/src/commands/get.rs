//! Get command - Retrieve point details by ID.

use anyhow::{Context, Result};
use serde::Serialize;
use std::path::PathBuf;
use v_hnsw_core::{PayloadStore, VectorStore};
use v_hnsw_storage::StorageEngine;

/// Point details for JSON output.
#[derive(Debug, Serialize)]
pub(crate) struct PointOutput {
    pub(crate) id: u64,
    pub(crate) text: Option<String>,
    pub(crate) vector_preview: Vec<f32>,
    pub(crate) vector_dim: usize,
}

/// Run the get command.
pub fn run(path: PathBuf, ids: Vec<u64>) -> Result<()> {
    // Open storage
    let engine = StorageEngine::open(&path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    let vector_store = engine.vector_store();
    let payload_store = engine.payload_store();

    let mut results = Vec::new();

    for id in ids {
        let text = payload_store.get_text(id).ok().flatten();
        let vector = vector_store.get(id).ok();

        let (vector_preview, vector_dim) = if let Some(v) = vector {
            let preview: Vec<f32> = v.iter().take(5).copied().collect();
            (preview, v.len())
        } else {
            (Vec::new(), 0)
        };

        results.push(PointOutput {
            id,
            text,
            vector_preview,
            vector_dim,
        });
    }

    let json = serde_json::to_string_pretty(&results)
        .with_context(|| "failed to serialize output")?;
    println!("{json}");

    Ok(())
}
