//! Export command - Export database to JSONL file.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use v_hnsw_core::{PayloadStore, VectorStore};
use v_hnsw_storage::StorageEngine;

use crate::is_interrupted;

/// Export record for JSONL output.
#[derive(Debug, Serialize)]
struct ExportRecord {
    id: u64,
    vector: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
}

/// Run the export command.
pub fn run(path: PathBuf, output: PathBuf) -> Result<()> {
    super::common::require_db(&path)?;

    // Open storage
    let engine = StorageEngine::open(&path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    let doc_count = engine.len();
    if doc_count == 0 {
        println!("Database is empty, nothing to export");
        return Ok(());
    }

    // Create output file
    let file = File::create(&output)
        .with_context(|| format!("failed to create output file: {}", output.display()))?;
    let mut writer = BufWriter::new(file);

    // Setup progress bar
    let pb = ProgressBar::new(doc_count as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}")
            .map_err(|e| anyhow::anyhow!("invalid progress template: {e}"))?
            .progress_chars("#>-"),
    );

    let start = Instant::now();
    let mut exported = 0u64;

    let vector_store = engine.vector_store();
    let payload_store = engine.payload_store();
    let ids: Vec<u64> = vector_store.id_map().keys().copied().collect();

    for id in ids {
        if is_interrupted() {
            pb.abandon_with_message("Interrupted");
            println!("Exported {exported} records before interruption.");
            return Ok(());
        }

        // Get vector
        let vector = match vector_store.get(id) {
            Ok(v) => v.to_vec(),
            Err(_) => {
                pb.inc(1);
                continue;
            }
        };

        // Get text (optional)
        let text = payload_store.get_text(id).ok().flatten();

        let record = ExportRecord { id, vector, text };

        let line = serde_json::to_string(&record)
            .with_context(|| format!("failed to serialize record {id}"))?;

        writeln!(writer, "{line}")
            .with_context(|| format!("failed to write record {id}"))?;

        exported += 1;
        pb.inc(1);
    }

    pb.finish_with_message("Done");
    writer.flush()?;

    let elapsed = start.elapsed();

    println!();
    println!("Export completed:");
    println!("  Exported: {exported}");
    println!("  Output:   {}", output.display());
    println!("  Elapsed:  {:.2}s", elapsed.as_secs_f64());

    Ok(())
}
