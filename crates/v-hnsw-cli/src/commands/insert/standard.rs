//! Standard (non-embed) insert with pre-computed vectors.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use v_hnsw_storage::StorageEngine;

use super::{make_payload, print_stats};
use crate::commands::common;
use crate::commands::create::DbConfig;
use crate::commands::readers::{self, ReaderConfig};
use crate::is_interrupted;

/// Run insert with pre-computed vectors.
pub fn run_standard(path: PathBuf, input: PathBuf, vector_column: &str) -> Result<()> {
    crate::commands::common::require_db(&path)?;

    let config = DbConfig::load(&path)?;

    let mut engine = StorageEngine::open_exclusive(&path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    let reader_cfg = ReaderConfig::with_vector(vector_column);
    let mut reader = readers::open_reader(&input, &reader_cfg)
        .with_context(|| format!("failed to open input file: {}", input.display()))?;

    let total = reader.count().unwrap_or(0);
    let pb = common::make_progress_bar(total as u64)?;

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

        if record.vector.len() != config.dim {
            eprintln!(
                "ID {}: dimension mismatch: expected {}, got {}",
                record.id, config.dim, record.vector.len()
            );
            errors += 1;
            pb.inc(1);
            continue;
        }

        let payload = make_payload(record.source, record.tags);
        let text = record.text.unwrap_or_default();

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

    engine
        .checkpoint()
        .with_context(|| "failed to checkpoint database")?;

    print_stats(inserted, 0, errors, start.elapsed());

    common::build_indexes(&path, &engine, &config)?;

    Ok(())
}
