//! Insert command - Insert vectors from JSONL, Parquet, or fvecs/bvecs files.

mod embed_mode;
mod standard;

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use v_hnsw_core::Payload;


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
