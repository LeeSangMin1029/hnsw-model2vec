//! Insert command - Insert vectors from JSONL, Parquet, or fvecs/bvecs files.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use v_hnsw_core::Payload;
use v_hnsw_storage::StorageEngine;

use super::create::DbConfig;
use super::readers;
use crate::is_interrupted;

/// Run the insert command.
pub fn run(path: PathBuf, input: PathBuf, vector_column: &str) -> Result<()> {
    // Check database exists
    if !path.exists() {
        anyhow::bail!("Database not found at {}", path.display());
    }

    // Load config
    let config = DbConfig::load(&path)?;

    // Open storage
    let mut engine = StorageEngine::open(&path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    // Open reader (auto-detect format)
    let mut reader = readers::open_reader(&input, vector_column)
        .with_context(|| format!("failed to open input file: {}", input.display()))?;

    let total = reader.count().unwrap_or(0);

    // Setup progress bar
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec})")
            .map_err(|e| anyhow::anyhow!("invalid progress template: {e}"))?
            .progress_chars("#>-"),
    );

    let start = Instant::now();
    let mut inserted = 0u64;
    let mut errors = 0u64;

    for record_result in reader.records() {
        if is_interrupted() {
            pb.abandon_with_message("Interrupted");
            println!("Inserted {inserted} vectors before interruption.");
            if let Err(e) = engine.checkpoint() {
                eprintln!("Warning: checkpoint failed: {e}");
            }
            return Ok(());
        }

        let record = match record_result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Record error: {e}");
                errors += 1;
                pb.inc(1);
                continue;
            }
        };

        // Validate dimension
        if record.vector.len() != config.dim {
            eprintln!(
                "ID {}: dimension mismatch: expected {}, got {}",
                record.id, config.dim, record.vector.len()
            );
            errors += 1;
            pb.inc(1);
            continue;
        }

        // Build payload
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let payload = Payload {
            source: record.source.unwrap_or_default(),
            tags: record.tags.unwrap_or_default(),
            created_at: now,
            source_modified_at: now,
            chunk_index: 0,
            chunk_total: 1,
            custom: HashMap::new(),
        };

        let text = record.text.unwrap_or_default();

        // Insert
        if let Err(e) = engine.insert(record.id, &record.vector, payload, &text) {
            eprintln!("ID {}: insert error: {e}", record.id);
            errors += 1;
            pb.inc(1);
            continue;
        }

        inserted += 1;
        pb.inc(1);
    }

    pb.finish_with_message("Done");

    // Final checkpoint
    engine
        .checkpoint()
        .with_context(|| "failed to checkpoint database")?;

    let elapsed = start.elapsed();

    println!();
    println!("Insert completed:");
    println!("  Inserted: {inserted}");
    println!("  Errors:   {errors}");
    println!("  Elapsed:  {:.2}s", elapsed.as_secs_f64());
    if inserted > 0 {
        println!(
            "  Rate:     {:.0} vectors/s",
            inserted as f64 / elapsed.as_secs_f64()
        );
    }

    Ok(())
}
