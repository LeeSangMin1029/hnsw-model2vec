//! Insert command - Insert vectors from JSONL, Parquet, or fvecs/bvecs files.

mod embed_mode;
mod standard;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use v_hnsw_core::Payload;
use v_hnsw_storage::{StorageConfig, StorageEngine};

use super::create::DbConfig;

/// Auto-create the database when it does not exist (embed mode only).
fn auto_create_db(
    path: &PathBuf,
    dim: usize,
    embed_model: Option<&str>,
) -> Result<StorageEngine> {
    println!("Database not found — auto-creating at {}", path.display());

    let storage_config = StorageConfig {
        dim,
        initial_capacity: 10_000,
        checkpoint_threshold: 50_000,
    };
    let engine = StorageEngine::create(path, storage_config)
        .with_context(|| format!("failed to create storage at {}", path.display()))?;

    let db_config = DbConfig {
        version: DbConfig::CURRENT_VERSION,
        dim,
        metric: "cosine".to_string(),
        m: 16,
        ef_construction: 200,
        korean: false,
        embed_model: embed_model.map(|s| s.to_string()),
    };
    db_config.save(path)?;

    println!("  Dimension:  {dim}");
    println!("  Metric:     cosine");
    println!("  M:          16");
    println!("  ef:         200");
    if let Some(model) = embed_model {
        println!("  Model:      {model}");
    }
    println!();

    Ok(engine)
}

/// Update the embed_model in config if not already set.
fn update_embed_model(path: &Path, model_name: &str) -> Result<()> {
    let mut config = DbConfig::load(path)?;
    if config.embed_model.is_none() {
        config.embed_model = Some(model_name.to_string());
        config.save(path)?;
    }
    Ok(())
}

/// Build a [`Payload`] with current timestamp (for raw vector insert).
fn make_payload(source: Option<String>, tags: Option<Vec<String>>) -> Payload {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    Payload {
        source: source.unwrap_or_default(),
        tags: tags.unwrap_or_default(),
        created_at: now,
        source_modified_at: now,
        chunk_index: 0,
        chunk_total: 1,
        custom: HashMap::new(),
    }
}

/// Print final statistics.
fn print_stats(inserted: u64, skipped: u64, errors: u64, elapsed: std::time::Duration) {
    println!();
    println!("Insert completed:");
    println!("  Inserted: {inserted}");
    if skipped > 0 {
        println!("  Skipped:  {skipped}");
    }
    println!("  Errors:   {errors}");
    println!("  Elapsed:  {:.2}s", elapsed.as_secs_f64());
    if inserted > 0 {
        println!(
            "  Rate:     {:.0} vectors/s",
            inserted as f64 / elapsed.as_secs_f64()
        );
    }
}

/// Run the insert command.
pub fn run(
    path: PathBuf,
    input: PathBuf,
    vector_column: &str,
    embed: bool,
    text_column: &str,
    model_name: &str,
    batch_size: usize,
) -> Result<()> {
    if embed {
        embed_mode::run_embed(path, input, text_column, model_name, batch_size)
    } else {
        standard::run_standard(path, input, vector_column)
    }
}
