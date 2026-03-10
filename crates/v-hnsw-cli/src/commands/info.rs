//! Info command - Display database information.

use std::path::PathBuf;

use anyhow::{Context, Result};
use v_hnsw_storage::StorageEngine;

use super::create::DbConfig;

/// Run the info command.
pub fn run(path: PathBuf) -> Result<()> {
    super::common::require_db(&path)?;

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
