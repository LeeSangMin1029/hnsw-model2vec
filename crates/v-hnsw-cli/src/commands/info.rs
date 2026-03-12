//! Info command - Display database information.

use std::path::PathBuf;

use anyhow::{Context, Result};
use v_hnsw_storage::StorageEngine;

use super::db_config::DbConfig;

/// Validate that a database directory exists.
fn require_db(path: &std::path::Path) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("Database not found at {}", path.display());
    }
    Ok(())
}

/// Run the info command.
pub fn run(path: PathBuf) -> Result<()> {
    require_db(&path)?;

    // Load config
    let config = DbConfig::load(&path)
        .with_context(|| format!("failed to load database config at {}", path.display()))?;

    // Open storage to get document count
    let engine = StorageEngine::open(&path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    let doc_count = engine.len();

    // Print info
    println!("Database: {}", path.display());
    println!("Version:  {}", config.version);
    println!("Dimension: {}", config.dim);
    println!("Metric:   {}", config.metric.to_uppercase());
    println!("M:        {}", config.m);
    println!("ef:       {}", config.ef_construction);
    println!("Documents: {doc_count}");
    println!("Korean:   {}", if config.korean { "enabled" } else { "disabled" });

    Ok(())
}
