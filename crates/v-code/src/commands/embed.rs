//! `v-code embed` — fill in real vectors for a text-only database.
//!
//! Loads the embedding model, reads all texts from the DB,
//! generates embeddings, and rebuilds the vector store + indexes.

use std::path::PathBuf;

use anyhow::{Context, Result};
use v_hnsw_core::PayloadStore;
use v_hnsw_embed::{EmbeddingModel, Model2VecModel};
use v_hnsw_storage::{MmapVectorStore, StorageEngine};

use v_hnsw_cli::commands::common::make_progress_bar;
use v_hnsw_cli::commands::db_config::DbConfig;
use v_hnsw_cli::commands::indexing;
use v_hnsw_cli::is_interrupted;

/// Run the embed command: load model, embed all texts, rebuild indexes.
pub fn run(db_path: PathBuf) -> Result<()> {
    let config = DbConfig::load(&db_path)
        .with_context(|| format!("Failed to load config from {}", db_path.display()))?;

    if config.embedded {
        println!("Database already has embeddings. Use `v-code add` to re-index.");
        return Ok(());
    }

    println!("Embedding vectors for: {}", db_path.display());

    // Load model
    let model = Model2VecModel::new()
        .context("Failed to load Model2Vec embedding model")?;
    let model_dim = model.dim();
    println!("Model: {} (dim={})", model.name(), model_dim);

    // Collect all (id, text) pairs from the DB
    let pairs = {
        let engine = StorageEngine::open(&db_path)
            .with_context(|| format!("Failed to open database at {}", db_path.display()))?;
        let pstore = engine.payload_store();
        let vstore = engine.vector_store();
        let ids: Vec<u64> = vstore.id_map().keys().copied().collect();

        let mut p = Vec::with_capacity(ids.len());
        for id in ids {
            if let Ok(Some(text)) = pstore.get_text(id) {
                p.push((id, text));
            }
        }
        p
    };
    // Engine dropped — mmap drops before file (field order fix), releasing lock.

    if pairs.is_empty() {
        println!("No texts found to embed.");
        return Ok(());
    }

    println!("Texts: {}", pairs.len());

    // Recreate vector store with model dimension
    println!("Creating vector store (dim={model_dim})...");
    let vectors_path = db_path.join("vectors.bin");
    std::fs::remove_file(&vectors_path)
        .with_context(|| format!("Failed to remove old {}", vectors_path.display()))?;

    let capacity = (pairs.len() + 1000).max(10_000) as u32;
    let mut vstore = MmapVectorStore::create(&vectors_path, model_dim, capacity)?;

    // Embed in batches and write to vector store
    let batch_size = 256;
    let pb = make_progress_bar(pairs.len() as u64)?;
    let start = std::time::Instant::now();
    let mut embedded = 0u64;
    let mut errors = 0u64;

    for chunk in pairs.chunks(batch_size) {
        if is_interrupted() {
            pb.abandon_with_message("Interrupted");
            break;
        }

        let texts: Vec<&str> = chunk.iter().map(|(_, t)| t.as_str()).collect();

        // Sort by length for efficient batching, then unsort
        let mut indexed: Vec<(usize, &str)> = texts.iter().enumerate().map(|(i, &t)| (i, t)).collect();
        indexed.sort_by_key(|(_, t)| t.len());

        let sorted_texts: Vec<&str> = indexed.iter().map(|(_, t)| *t).collect();
        let embeddings = match model.embed(&sorted_texts) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Embedding error: {e}");
                errors += chunk.len() as u64;
                pb.inc(chunk.len() as u64);
                continue;
            }
        };

        // Unsort back to original order
        let mut ordered = vec![Vec::new(); chunk.len()];
        for (sorted_idx, (orig_idx, _)) in indexed.iter().enumerate() {
            ordered[*orig_idx] = embeddings[sorted_idx].clone();
        }

        // Write vectors
        let batch: Vec<(u64, &[f32])> = chunk.iter()
            .zip(ordered.iter())
            .map(|((id, _), emb)| (*id, emb.as_slice()))
            .collect();
        if let Err(e) = vstore.insert_batch(&batch) {
            eprintln!("Vector write error: {e}");
            errors += chunk.len() as u64;
        } else {
            embedded += chunk.len() as u64;
        }

        pb.inc(chunk.len() as u64);
    }

    if !is_interrupted() {
        pb.finish_with_message("Embedding complete");
    }

    drop(vstore);

    let elapsed = start.elapsed();
    println!();
    println!("Embedded {embedded} vectors in {:.2}s", elapsed.as_secs_f64());
    if errors > 0 {
        println!("  ({errors} errors)");
    }

    // Update config
    let mut config = DbConfig::load(&db_path)?;
    config.dim = model_dim;
    config.embedded = true;
    config.embed_model = Some(model.name().to_owned());
    config.save(&db_path)?;

    // Build indexes (HNSW + SQ8 + BM25)
    let engine = StorageEngine::open(&db_path)?;
    indexing::build_indexes(&db_path, &engine, &config)?;

    println!("\nDone! Embeddings and indexes ready.");
    println!("Use: v-code find {} \"<query>\"", db_path.display());
    Ok(())
}
