//! Insert command - Insert vectors from JSONL, Parquet, or fvecs/bvecs files.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use v_hnsw_core::{Payload, PayloadStore, VectorIndex, VectorStore};
use v_hnsw_distance::{CosineDistance, DotProductDistance, L2Distance};
use v_hnsw_embed::{EmbeddingModel, Model2VecModel};
use v_hnsw_graph::{HnswConfig, HnswGraph};
use v_hnsw_search::{Bm25Index, KoreanBm25Tokenizer};
use v_hnsw_storage::{StorageConfig, StorageEngine};

use super::create::DbConfig;
use super::readers::{self, ReaderConfig};
use crate::is_interrupted;

/// Auto-create the database when it does not exist (embed mode only).
fn auto_create_db(path: &PathBuf, dim: usize, embed_model: Option<&str>) -> Result<StorageEngine> {
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
fn update_embed_model(path: &PathBuf, model_name: &str) -> Result<()> {
    let mut config = DbConfig::load(path)?;
    if config.embed_model.is_none() {
        config.embed_model = Some(model_name.to_string());
        config.save(path)?;
    }
    Ok(())
}

/// Build a progress bar with the standard template.
fn make_progress_bar(total: u64) -> Result<ProgressBar> {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec})")
            .map_err(|e| anyhow::anyhow!("invalid progress template: {e}"))?
            .progress_chars("#>-"),
    );
    Ok(pb)
}

/// Build a [`Payload`] with current timestamp.
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

/// Build and save HNSW and BM25 indexes after insertion.
fn build_and_save_indexes(path: &PathBuf, engine: &StorageEngine, config: &DbConfig) -> Result<()> {
    if engine.is_empty() {
        println!("No vectors to index, skipping index building.");
        return Ok(());
    }

    println!();
    println!("Building indexes...");

    // Build HNSW graph
    let hnsw_config = HnswConfig::builder()
        .dim(config.dim)
        .m(config.m)
        .ef_construction(config.ef_construction)
        .build()
        .with_context(|| "failed to create HNSW config")?;

    let hnsw_path = path.join("hnsw.bin");
    let vector_store = engine.vector_store();

    println!("  Building HNSW graph (metric={}, M={}, ef_construction={})...",
             config.metric, config.m, config.ef_construction);

    match config.metric.as_str() {
        "cosine" => {
            let mut hnsw = HnswGraph::new(hnsw_config, CosineDistance);
            for id in vector_store.id_map().keys() {
                if let Ok(vec) = vector_store.get(*id) {
                    let _ = hnsw.insert(*id, vec);
                }
            }
            hnsw.save(&hnsw_path)
                .with_context(|| format!("failed to save HNSW graph to {}", hnsw_path.display()))?;
        }
        "l2" => {
            let mut hnsw = HnswGraph::new(hnsw_config, L2Distance);
            for id in vector_store.id_map().keys() {
                if let Ok(vec) = vector_store.get(*id) {
                    let _ = hnsw.insert(*id, vec);
                }
            }
            hnsw.save(&hnsw_path)
                .with_context(|| format!("failed to save HNSW graph to {}", hnsw_path.display()))?;
        }
        "dot" => {
            let mut hnsw = HnswGraph::new(hnsw_config, DotProductDistance);
            for id in vector_store.id_map().keys() {
                if let Ok(vec) = vector_store.get(*id) {
                    let _ = hnsw.insert(*id, vec);
                }
            }
            hnsw.save(&hnsw_path)
                .with_context(|| format!("failed to save HNSW graph to {}", hnsw_path.display()))?;
        }
        _ => {
            anyhow::bail!("unknown distance metric: '{}'", config.metric);
        }
    }
    println!("  HNSW graph saved: {}", hnsw_path.display());

    // Build BM25 index
    println!("  Building BM25 index...");
    let bm25_path = path.join("bm25.bin");
    let mut bm25: Bm25Index<KoreanBm25Tokenizer> = Bm25Index::new(KoreanBm25Tokenizer::new());
    let payload_store = engine.payload_store();

    for id in vector_store.id_map().keys() {
        if let Ok(Some(text)) = payload_store.get_text(*id) {
            bm25.add_document(*id, &text);
        }
    }

    bm25.save(&bm25_path)
        .with_context(|| format!("failed to save BM25 index to {}", bm25_path.display()))?;
    println!("  BM25 index saved: {}", bm25_path.display());

    println!("Index building completed.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Embed-mode insert (pipelined: producer=read+embed, consumer=storage insert)
// ---------------------------------------------------------------------------

/// A batch of embedded records ready for storage insertion.
struct EmbeddedBatch {
    records: Vec<readers::InputRecord>,
    embeddings: Vec<Vec<f32>>,
}

/// Run insert with automatic embedding using a pipelined architecture.
///
/// A producer thread reads input records and embeds them,
/// while the consumer thread (main) inserts them into the storage engine.
/// This keeps both embedding and storage I/O busy concurrently.
fn run_embed(
    path: PathBuf,
    input: PathBuf,
    text_column: &str,
    model_name: &str,
    batch_size: usize,
) -> Result<()> {
    // Load model2vec model
    println!("Loading model2vec model: {model_name}");
    let model = Model2VecModel::from_pretrained(model_name)
        .map_err(|e| anyhow::anyhow!("failed to load model2vec model: {e}"))?;
    let dim = model.dim();
    // model2vec uses simple tokenization, so character limit is generous
    let max_chars = 8000;
    println!("Model loaded (dim={dim}).");

    // Open or create DB.
    let mut engine = if path.exists() {
        let config = DbConfig::load(&path)?;
        if config.dim != dim {
            anyhow::bail!(
                "dimension mismatch: database has dim={}, but model produces dim={dim}",
                config.dim,
            );
        }
        // Update embed_model if not set
        update_embed_model(&path, model_name)?;
        StorageEngine::open(&path)
            .with_context(|| format!("failed to open database at {}", path.display()))?
    } else {
        auto_create_db(&path, dim, Some(model_name))?
    };
    let model: Box<dyn EmbeddingModel> = Box::new(model);

    // Open reader (no vector column needed).
    let reader_cfg = ReaderConfig {
        vector_column: None,
        text_column,
    };
    let mut reader = readers::open_reader(&input, &reader_cfg)
        .with_context(|| format!("failed to open input file: {}", input.display()))?;

    let total = reader.count().unwrap_or(0);
    let pb = make_progress_bar(total as u64)?;

    let start = Instant::now();
    let mut inserted = 0u64;
    let mut skipped = 0u64;
    let mut errors = 0u64;

    // Bounded channel: buffer enough batches so the producer can stay ahead
    // of the consumer during checkpoint stalls.
    let (sender, receiver) = crossbeam::channel::bounded::<EmbeddedBatch>(8);

    // --- Pipeline ---
    // Producer thread: read records + embed.
    // Consumer (this thread): receive embedded batches + insert into engine.
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

                // Pre-truncate text to max_chars before tokenization.
                let embed_text = if text.len() > max_chars {
                    let mut end = max_chars;
                    while !text.is_char_boundary(end) {
                        end -= 1;
                    }
                    text[..end].to_string()
                } else {
                    text.clone()
                };

                batch_texts.push(embed_text);
                batch_records.push(record);

                if batch_texts.len() >= batch_size {
                    let embeddings = embed_sorted(model.as_ref(), &batch_texts)?;

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

            // Flush remaining partial batch.
            if !batch_texts.is_empty() {
                let embeddings = embed_sorted(model.as_ref(), &batch_texts)?;
                let _ = sender.send(EmbeddedBatch {
                    records: batch_records,
                    embeddings,
                });
            }

            // sender drops here, signalling end-of-stream.
            Ok((producer_skipped, producer_errors))
        });

        // Consumer: receive embedded batches and insert into storage engine.
        for batch in receiver {
            let count = batch.records.len() as u64;

            // Build tuples for engine.insert_batch()
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

        // Wait for producer and collect its counters.
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

    // Build and save indexes
    let config = DbConfig::load(&path)?;
    build_and_save_indexes(&path, &engine, &config)?;

    Ok(())
}

/// Embed texts with length-sorted batching to minimize padding waste.
///
/// Sorts texts by length, embeds them, then restores original order.
fn embed_sorted(model: &dyn EmbeddingModel, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    // Sort indices by text length
    let mut indices: Vec<usize> = (0..texts.len()).collect();
    indices.sort_by_key(|&i| texts[i].len());

    let sorted: Vec<&str> = indices.iter().map(|&i| texts[i].as_str()).collect();
    let sorted_embs = model
        .embed(&sorted)
        .map_err(|e| anyhow::anyhow!("embedding failed: {e}"))?;

    // Restore original order
    let mut embeddings = vec![Vec::new(); texts.len()];
    for (sorted_idx, &orig_idx) in indices.iter().enumerate() {
        embeddings[orig_idx] = sorted_embs[sorted_idx].clone();
    }
    Ok(embeddings)
}

// ---------------------------------------------------------------------------
// Standard (non-embed) insert
// ---------------------------------------------------------------------------

/// Run insert with pre-computed vectors.
fn run_standard(path: PathBuf, input: PathBuf, vector_column: &str) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("Database not found at {}", path.display());
    }

    let config = DbConfig::load(&path)?;

    let mut engine = StorageEngine::open(&path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    let reader_cfg = ReaderConfig::with_vector(vector_column);
    let mut reader = readers::open_reader(&input, &reader_cfg)
        .with_context(|| format!("failed to open input file: {}", input.display()))?;

    let total = reader.count().unwrap_or(0);
    let pb = make_progress_bar(total as u64)?;

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

        // Validate dimension.
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

    // Build and save indexes
    build_and_save_indexes(&path, &engine, &config)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

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
        run_embed(path, input, text_column, model_name, batch_size)
    } else {
        run_standard(path, input, vector_column)
    }
}
