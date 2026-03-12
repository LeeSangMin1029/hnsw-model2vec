//! Embed-mode insert: pipelined read+embed → storage insert.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use v_hnsw_core::Payload;
use v_hnsw_embed::{EmbeddingModel, Model2VecModel};

use super::{make_payload, print_stats};
use crate::commands::common;
use crate::commands::create::DbConfig;
use crate::commands::readers::{self, ReaderConfig};
use crate::is_interrupted;

/// A batch of embedded records ready for storage insertion.
struct EmbeddedBatch {
    records: Vec<readers::InputRecord>,
    embeddings: Vec<Vec<f32>>,
}

/// Run insert with automatic embedding using a pipelined architecture.
pub fn run_embed(
    path: PathBuf,
    input: PathBuf,
    _text_column: &str,
    model_name: &str,
    batch_size: usize,
) -> Result<()> {
    println!("Loading model2vec model: {model_name}");
    let model = Model2VecModel::from_pretrained(model_name)
        .map_err(|e| anyhow::anyhow!("failed to load model2vec model: {e}"))?;
    let dim = model.dim();
    println!("Model loaded (dim={dim}).");

    // Open or create DB.
    let mut engine = common::ensure_database(&path, dim, model_name, false, false)?;
    let model: Box<dyn EmbeddingModel> = Box::new(model);

    let reader_cfg = ReaderConfig {
        vector_column: None,
    };
    let mut reader = readers::open_reader(&input, &reader_cfg)
        .with_context(|| format!("failed to open input file: {}", input.display()))?;

    let total = reader.count().unwrap_or(0);
    let pb = common::make_progress_bar(total as u64)?;

    let start = Instant::now();
    let mut inserted = 0u64;
    let mut skipped = 0u64;
    let mut errors = 0u64;

    let (sender, receiver) = crossbeam::channel::bounded::<EmbeddedBatch>(8);

    std::thread::scope(|scope| {
        let pb_ref = &pb;

        let producer = scope.spawn(move || -> anyhow::Result<(u64, u64)> {
            let mut batch_texts: Vec<String> = Vec::with_capacity(batch_size);
            let mut batch_records: Vec<readers::InputRecord> = Vec::with_capacity(batch_size);
            let mut producer_skipped = 0u64;
            let mut producer_errors = 0u64;

            for record_result in reader.records() {
                if is_interrupted() {
                    break;
                }

                let record = match record_result {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("Record error: {e}");
                        producer_errors += 1;
                        pb_ref.inc(1);
                        continue;
                    }
                };

                let text = record.text.clone().unwrap_or_default();
                if text.is_empty() {
                    producer_skipped += 1;
                    pb_ref.inc(1);
                    continue;
                }

                let embed_text = common::truncate_for_embed(&text).to_string();

                batch_texts.push(embed_text);
                batch_records.push(record);

                if batch_texts.len() >= batch_size {
                    let embeddings = common::embed_sorted(model.as_ref(), &batch_texts)?;

                    let batch = EmbeddedBatch {
                        records: std::mem::replace(
                            &mut batch_records,
                            Vec::with_capacity(batch_size),
                        ),
                        embeddings,
                    };
                    batch_texts.clear();

                    if sender.send(batch).is_err() {
                        break;
                    }
                }
            }

            if !batch_texts.is_empty() {
                let embeddings = common::embed_sorted(model.as_ref(), &batch_texts)?;
                let _ = sender.send(EmbeddedBatch {
                    records: batch_records,
                    embeddings,
                });
            }

            Ok((producer_skipped, producer_errors))
        });

        for batch in receiver {
            let count = batch.records.len() as u64;

            let items: Vec<(u64, &[f32], Payload, &str)> = batch
                .records
                .iter()
                .zip(batch.embeddings.iter())
                .map(|(rec, emb)| {
                    let payload = make_payload(rec.source.clone(), rec.tags.clone());
                    let text = rec.text.as_deref().unwrap_or_default();
                    (rec.id, emb.as_slice(), payload, text)
                })
                .collect();

            if let Err(e) = engine.insert_batch(&items) {
                eprintln!("Batch insert error: {e}");
                errors += count;
            } else {
                inserted += count;
            }
            pb.inc(count);
        }

        match producer.join() {
            Ok(Ok((s, e))) => {
                skipped += s;
                errors += e;
            }
            Ok(Err(e)) => {
                eprintln!("Producer error: {e}");
            }
            Err(_) => {
                eprintln!("Producer thread panicked");
            }
        }
    });

    if is_interrupted() {
        pb.abandon_with_message("Interrupted");
        println!("Inserted {inserted} vectors before interruption.");
    } else {
        pb.finish_with_message("Done");
    }

    engine
        .checkpoint()
        .with_context(|| "failed to checkpoint database")?;

    print_stats(inserted, skipped, errors, start.elapsed());

    let config = DbConfig::load(&path)?;
    common::build_indexes(&path, &engine, &config)?;

    Ok(())
}
