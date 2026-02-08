//! Pipeline parallelism for embedding + storage insertion.

use std::time::Instant;

use anyhow::{Context, Result};
use v_hnsw_embed::Model2VecModel;
use v_hnsw_storage::StorageEngine;

use crate::commands::common;
use crate::is_interrupted;

/// A record for batch processing.
#[derive(Clone)]
pub struct AddRecord {
    pub id: u64,
    pub text: String,
    pub source: String,
    pub title: Option<String>,
    pub tags: Vec<String>,
    pub chunk_index: usize,
    pub chunk_total: usize,
    pub source_modified_at: u64,
}

/// Embedded batch ready for storage insertion.
struct EmbeddedBatch {
    records: Vec<AddRecord>,
    embeddings: Vec<Vec<f32>>,
}

/// Process records in batches with embedding using pipeline parallelism.
/// Producer thread: prepare texts + embedding
/// Consumer thread: storage insertion
/// Returns (inserted, errors).
pub fn process_records(
    records: Vec<AddRecord>,
    model: &Model2VecModel,
    engine: &mut StorageEngine,
) -> Result<(u64, u64)> {
    if records.is_empty() {
        return Ok((0, 0));
    }

    let batch_size = 256;
    let max_chars = 8000;

    let pb = common::make_progress_bar(records.len() as u64)?;
    let start = Instant::now();

    // Bounded channel: buffer batches so producer can stay ahead during storage I/O
    let (sender, receiver) = crossbeam::channel::bounded::<EmbeddedBatch>(4);

    // Pipeline: Producer embeds, Consumer inserts
    let result = std::thread::scope(|scope| {
        let pb_ref = &pb;

        // Producer thread: prepare texts + embed
        let producer = scope.spawn(move || -> anyhow::Result<u64> {
            let mut producer_errors = 0u64;

            for chunk in records.chunks(batch_size) {
                if is_interrupted() {
                    break;
                }

                let texts: Vec<String> = chunk
                    .iter()
                    .map(|r| {
                        if r.text.len() > max_chars {
                            let mut end = max_chars;
                            while !r.text.is_char_boundary(end) {
                                end -= 1;
                            }
                            r.text[..end].to_string()
                        } else {
                            r.text.clone()
                        }
                    })
                    .collect();

                let embeddings = match common::embed_sorted(model, &texts) {
                    Ok(e) => e,
                    Err(e) => {
                        eprintln!("Embedding error: {e}");
                        producer_errors += chunk.len() as u64;
                        pb_ref.inc(chunk.len() as u64);
                        continue;
                    }
                };

                let batch = EmbeddedBatch {
                    records: chunk.to_vec(),
                    embeddings,
                };

                if sender.send(batch).is_err() {
                    break;
                }
            }

            drop(sender);
            Ok(producer_errors)
        });

        // Consumer (this thread): receive batches + insert into storage
        let mut inserted = 0u64;
        let mut consumer_errors = 0u64;

        for batch in receiver {
            if is_interrupted() {
                break;
            }

            let items: Vec<(u64, &[f32], _, &str)> = batch
                .records
                .iter()
                .zip(batch.embeddings.iter())
                .map(|(rec, emb)| {
                    let payload = common::make_payload(
                        &rec.source,
                        rec.title.as_deref(),
                        &rec.tags,
                        rec.chunk_index,
                        rec.chunk_total,
                        rec.source_modified_at,
                    );
                    (rec.id, emb.as_slice(), payload, rec.text.as_str())
                })
                .collect();

            if let Err(e) = engine.insert_batch(&items) {
                eprintln!("Insert error: {e}");
                consumer_errors += batch.records.len() as u64;
            } else {
                inserted += batch.records.len() as u64;
            }

            pb_ref.inc(batch.records.len() as u64);
        }

        let producer_errors = producer.join().unwrap_or_else(|_| Ok(0)).unwrap_or(0);

        (inserted, producer_errors + consumer_errors)
    });

    let (inserted, errors) = result;

    if !is_interrupted() {
        pb.finish_with_message("Done");
    }

    engine
        .checkpoint()
        .context("Failed to checkpoint database")?;

    let elapsed = start.elapsed();

    tracing::info!(
        inserted,
        errors,
        elapsed_secs = elapsed.as_secs_f64(),
        "Batch processing completed"
    );

    println!();
    println!("Add completed:");
    println!("  Inserted: {inserted}");
    println!("  Errors:   {errors}");
    println!("  Elapsed:  {:.2}s", elapsed.as_secs_f64());
    if inserted > 0 {
        println!(
            "  Rate:     {:.0} items/s",
            inserted as f64 / elapsed.as_secs_f64()
        );
    }

    Ok((inserted, errors))
}
