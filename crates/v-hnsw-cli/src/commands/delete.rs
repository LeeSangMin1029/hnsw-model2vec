//! Delete command - Remove a point by ID.

use std::path::PathBuf;

use anyhow::{Context, Result};
use v_hnsw_storage::StorageEngine;

/// Run the delete command.
pub fn run(path: PathBuf, id: u64) -> Result<()> {
    // Check database exists
    if !path.exists() {
        anyhow::bail!("Database not found at {}", path.display());
    }

    // Open storage
    let mut engine = StorageEngine::open(&path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    // Remove the point
    engine.remove(id)
        .with_context(|| format!("failed to delete point {id}"))?;

    // Checkpoint
    engine.checkpoint()
        .with_context(|| "failed to checkpoint database")?;

    println!("Deleted point {id}");

    Ok(())
}
