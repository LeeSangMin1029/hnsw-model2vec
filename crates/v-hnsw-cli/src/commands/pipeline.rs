//! Pipeline parallelism for embedding + storage insertion.

use std::time::Instant;

use anyhow::{Context, Result};
use v_hnsw_embed::EmbeddingModel;
use v_hnsw_storage::StorageEngine;

use super::common::make_progress_bar;
use crate::commands::ingest::{self, IngestRecord};
use crate::is_interrupted;

/// Embedded batch ready for storage insertion.
struct EmbeddedBatch {
    records: Vec<IngestRecord>,
    embeddings: Vec<Vec<f32>>,
}

/// Process records in batches with embedding using pipeline parallelism.
/// Producer thread: prepare texts + embedding
/// Consumer thread: storage insertion
/// Returns (inserted, errors, inserted_ids).
pub fn process_records(
    records: Vec<IngestRecord>,
    model: &dyn EmbeddingModel,
    engine: &mut StorageEngine,
) -> Result<(u64, u64, Vec<u64>)> {
    if records.is_empty() {
        return Ok((0, 0, Vec::new()));
    }

    let batch_size = 256;

    let pb = make_progress_bar(records.len() as u64)?;
    let start = Instant::now();

    // Bounded channel: buffer batches so producer can stay ahead during storage I/O
    let (sender, receiver) = crossbeam::channel::bounded::<EmbeddedBatch>(4);

    // Pipeline: Producer embeds, Consumer inserts
    let result = std::thread::scope(|scope| {
        let pb_ref = &pb;

        // Producer thread: prepare texts + embed (drain-based ownership transfer)
        let producer = scope.spawn(move || -> anyhow::Result<u64> {
            let mut producer_errors = 0u64;
            let mut remaining = records;

            while !remaining.is_empty() {
                if is_interrupted() {
                    break;
                }

                let batch_end = batch_size.min(remaining.len());
                let batch_records: Vec<IngestRecord> = remaining.drain(..batch_end).collect();

                let texts: Vec<String> = batch_records
                    .iter()
                    .map(|r| ingest::truncate_for_embed(&r.text).to_string())
                    .collect();

                let embeddings = match ingest::embed_sorted(model, &texts) {
                    Ok(e) => e,
                    Err(e) => {
                        eprintln!("Embedding error: {e}");
                        producer_errors += batch_records.len() as u64;
                        pb_ref.inc(batch_records.len() as u64);
                        continue;
                    }
                };

                let batch = EmbeddedBatch {
                    records: batch_records,
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
        let mut inserted_ids: Vec<u64> = Vec::new();

        for batch in receiver {
            if is_interrupted() {
                break;
            }

            let items: Vec<(u64, &[f32], _, &str)> = batch
                .records
                .iter()
                .zip(batch.embeddings.iter())
                .map(|(rec, emb)| {
                    let payload = ingest::make_payload(
                        &rec.source,
                        rec.title.as_deref(),
                        &rec.tags,
                        rec.chunk_index,
                        rec.chunk_total,
                        rec.source_modified_at,
                        &rec.custom,
                    );
                    (rec.id, emb.as_slice(), payload, rec.text.as_str())
                })
                .collect();

            if let Err(e) = engine.insert_batch(&items) {
                eprintln!("Insert error: {e}");
                consumer_errors += batch.records.len() as u64;
            } else {
                inserted += batch.records.len() as u64;
                inserted_ids.extend(batch.records.iter().map(|r| r.id));
            }

            pb_ref.inc(batch.records.len() as u64);
        }

        let producer_errors = producer.join().unwrap_or_else(|_| Ok(0)).unwrap_or(0);

        (inserted, producer_errors + consumer_errors, inserted_ids)
    });

    let (inserted, errors, inserted_ids) = result;

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
    if errors > 0 {
        println!("Inserted {inserted} chunks ({errors} errors) in {:.2}s", elapsed.as_secs_f64());
    } else {
        println!("Inserted {inserted} chunks in {:.2}s", elapsed.as_secs_f64());
    }

    Ok((inserted, errors, inserted_ids))
}

/// Insert records with zero vectors (no embedding).
///
/// Used by `v-code add` for fast indexing — real vectors are filled in
/// later by `v-code embed` when semantic search is needed.
pub fn process_records_text_only(
    records: Vec<IngestRecord>,
    dim: usize,
    engine: &mut StorageEngine,
) -> Result<(u64, u64, Vec<u64>)> {
    if records.is_empty() {
        return Ok((0, 0, Vec::new()));
    }

    let batch_size = 256;
    let zero_vec = vec![0.0f32; dim];

    let pb = make_progress_bar(records.len() as u64)?;
    let start = Instant::now();

    let mut inserted = 0u64;
    let mut errors = 0u64;
    let mut inserted_ids: Vec<u64> = Vec::new();

    for chunk in records.chunks(batch_size) {
        if is_interrupted() {
            break;
        }

        let items: Vec<(u64, &[f32], _, &str)> = chunk
            .iter()
            .map(|rec| {
                let payload = ingest::make_payload(
                    &rec.source,
                    rec.title.as_deref(),
                    &rec.tags,
                    rec.chunk_index,
                    rec.chunk_total,
                    rec.source_modified_at,
                    &rec.custom,
                );
                (rec.id, zero_vec.as_slice(), payload, rec.text.as_str())
            })
            .collect();

        if let Err(e) = engine.insert_batch(&items) {
            eprintln!("Insert error: {e}");
            errors += chunk.len() as u64;
        } else {
            inserted += chunk.len() as u64;
            inserted_ids.extend(chunk.iter().map(|r| r.id));
        }

        pb.inc(chunk.len() as u64);
    }

    if !is_interrupted() {
        pb.finish_with_message("Done");
    }

    engine
        .checkpoint()
        .context("Failed to checkpoint database")?;

    let elapsed = start.elapsed();
    println!();
    println!("Inserted {inserted} chunks (text-only) in {:.2}s", elapsed.as_secs_f64());

    Ok((inserted, errors, inserted_ids))
}
