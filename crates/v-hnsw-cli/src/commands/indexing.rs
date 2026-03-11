//! Index building and incremental updating (HNSW + BM25).

use std::path::Path;

use anyhow::{Context, Result};
use v_hnsw_core::{PayloadStore, VectorIndex, VectorStore};
use v_hnsw_graph::{HnswGraph, HnswSnapshot, NormalizedCosineDistance};
use v_hnsw_search::{Bm25Index, KoreanBm25Tokenizer};
use v_hnsw_storage::sq8::Sq8Params;
use v_hnsw_storage::sq8_store::Sq8VectorStore;
use v_hnsw_storage::StorageEngine;

use super::common::{ensure_korean_dict, make_progress_bar};
use super::create::DbConfig;
use crate::is_interrupted;

/// Build and save HNSW and BM25 indexes from storage data.
pub fn build_indexes(path: &Path, engine: &StorageEngine, config: &DbConfig) -> Result<()> {
    if engine.is_empty() {
        println!("No vectors to index, skipping index building.");
        return Ok(());
    }

    tracing::info!("Building indexes");
    println!();
    println!("Building indexes...");

    // Build HNSW graph
    let hnsw_config = config.to_hnsw_config()?;

    let hnsw_path = path.join("hnsw.bin");
    let vector_store = engine.vector_store();

    println!(
        "  Building HNSW graph (M={}, ef_construction={})...",
        config.m, config.ef_construction
    );

    let pb = make_progress_bar(vector_store.id_map().keys().len() as u64)?;

    let mut hnsw = HnswGraph::new(hnsw_config, NormalizedCosineDistance);
    for id in vector_store.id_map().keys() {
        if is_interrupted() {
            pb.abandon_with_message("Interrupted during HNSW build");
            return Ok(());
        }
        let _ = hnsw.build_insert(vector_store, *id);
        pb.inc(1);
    }

    pb.finish_with_message("HNSW build complete");

    hnsw.save(&hnsw_path)
        .with_context(|| format!("Failed to save HNSW graph to {}", hnsw_path.display()))?;
    println!("  HNSW graph saved: {}", hnsw_path.display());

    // Build SQ8 quantized vectors
    build_sq8(path, vector_store)?;

    // Build BM25 index
    println!("  Building BM25 index...");
    ensure_korean_dict()?;
    let bm25_path = path.join("bm25.bin");
    let mut bm25: Bm25Index<KoreanBm25Tokenizer> = Bm25Index::new(KoreanBm25Tokenizer::new());
    let payload_store = engine.payload_store();

    let pb = make_progress_bar(vector_store.id_map().keys().len() as u64)?;

    for id in vector_store.id_map().keys() {
        if is_interrupted() {
            pb.abandon_with_message("Interrupted during BM25 build");
            return Ok(());
        }
        if let Ok(Some(text)) = payload_store.get_text(*id) {
            bm25.add_document(*id, &text);
        }
        pb.inc(1);
    }

    pb.finish_with_message("BM25 build complete");

    bm25.build_fieldnorm_cache();
    bm25.save(&bm25_path)
        .with_context(|| format!("Failed to save BM25 index to {}", bm25_path.display()))?;
    println!("  BM25 index saved: {}", bm25_path.display());

    tracing::info!("Index building completed");
    println!("Index building completed.");
    Ok(())
}

/// Incrementally update HNSW and BM25 indexes for changed IDs only.
///
/// Falls back to full rebuild if index files don't exist yet.
pub fn update_indexes_incremental(
    path: &Path,
    engine: &StorageEngine,
    config: &DbConfig,
    added_ids: &[u64],
    removed_ids: &[u64],
) -> Result<()> {
    let hnsw_path = path.join("hnsw.bin");
    let bm25_path = path.join("bm25.bin");

    // Fallback to full rebuild if index files missing
    if !hnsw_path.exists() || !bm25_path.exists() {
        tracing::info!("Index files missing, falling back to full rebuild");
        println!("Index files not found, performing full rebuild...");
        return build_indexes(path, engine, config);
    }

    let total_changes = added_ids.len() + removed_ids.len();
    tracing::info!(added = added_ids.len(), removed = removed_ids.len(), "Incremental index update");
    println!("Updating indexes incrementally ({} additions, {} removals)...", added_ids.len(), removed_ids.len());

    // --- HNSW incremental update ---
    let mut hnsw: HnswGraph<NormalizedCosineDistance> = HnswGraph::load(&hnsw_path, NormalizedCosineDistance)
        .with_context(|| "Failed to load HNSW graph")?;

    let vector_store = engine.vector_store();

    for &id in removed_ids {
        let _ = hnsw.delete(id);
    }

    for &id in added_ids {
        let _ = hnsw.build_insert(vector_store, id);
    }

    hnsw.save(&hnsw_path)
        .with_context(|| "Failed to save HNSW graph")?;

    let hnsw_snap_path = path.join("hnsw.snap");
    HnswSnapshot::save(&hnsw, &hnsw_snap_path)
        .with_context(|| "Failed to save HNSW snapshot")?;
    println!("  HNSW graph updated ({total_changes} changes) + snapshot.");

    // --- SQ8 incremental update ---
    update_sq8(path, vector_store, added_ids, removed_ids)?;

    // --- BM25 incremental update ---
    if total_changes > 0 {
        ensure_korean_dict()?;
    }

    let mut bm25: Bm25Index<KoreanBm25Tokenizer> = Bm25Index::load_mutable(&bm25_path)
        .with_context(|| "Failed to load BM25 index")?;

    let payload_store = engine.payload_store();

    for &id in removed_ids {
        bm25.remove_document(id);
    }

    for &id in added_ids {
        if let Ok(Some(text)) = payload_store.get_text(id) {
            bm25.add_document(id, &text);
        }
    }

    bm25.build_fieldnorm_cache();
    bm25.save(&bm25_path)
        .with_context(|| "Failed to save BM25 index")?;
    bm25.save_snapshot(path)
        .with_context(|| "Failed to save BM25 snapshot")?;
    println!("  BM25 index updated ({total_changes} changes) + snapshot.");

    tracing::info!("Incremental index update completed");
    Ok(())
}

/// Build SQ8 quantized vector index from the f32 vector store.
///
/// Trains per-dimension min/max parameters, then quantizes all vectors.
fn build_sq8(path: &Path, vector_store: &v_hnsw_storage::MmapVectorStore) -> Result<()> {
    let id_map = vector_store.id_map();
    if id_map.is_empty() {
        return Ok(());
    }

    println!("  Building SQ8 quantized vectors...");

    let dim = vector_store.dim();

    // Collect all vectors for training
    let vectors: Vec<&[f32]> = id_map
        .keys()
        .filter_map(|&id| vector_store.get(id).ok())
        .collect();

    let params = Sq8Params::train(dim, &vectors)
        .with_context(|| "Failed to train SQ8 parameters")?;

    // Save params
    let params_path = path.join("sq8_params.bin");
    params
        .save(&params_path)
        .with_context(|| "Failed to save SQ8 params")?;

    // Create quantized store and insert all vectors
    let store_path = path.join("sq8_vectors.bin");
    let mut sq8_store = Sq8VectorStore::create(&store_path, dim, vectors.len() as u32 + 64)
        .with_context(|| "Failed to create SQ8 vector store")?;

    let mut buf = vec![0u8; dim];
    for (&id, &slot) in id_map {
        if let Ok(vec) = vector_store.get(id) {
            params.quantize_into(vec, &mut buf);
            sq8_store.insert_at(id, slot, &buf)?;
        }
    }

    sq8_store.flush()?;

    let f32_size = id_map.len() * dim * 4;
    let sq8_size = id_map.len() * dim;
    println!(
        "  SQ8 vectors saved: {:.1}x compression ({:.2}MB → {:.2}MB)",
        f32_size as f64 / sq8_size as f64,
        f32_size as f64 / 1_048_576.0,
        sq8_size as f64 / 1_048_576.0,
    );

    Ok(())
}

/// Incrementally update SQ8 index for changed vectors.
///
/// If SQ8 files don't exist, rebuilds from scratch.
/// Note: incremental updates reuse existing params (no re-training).
fn update_sq8(
    path: &Path,
    vector_store: &v_hnsw_storage::MmapVectorStore,
    added_ids: &[u64],
    removed_ids: &[u64],
) -> Result<()> {
    let params_path = path.join("sq8_params.bin");
    let store_path = path.join("sq8_vectors.bin");

    if !params_path.exists() || !store_path.exists() {
        // Full rebuild
        return build_sq8(path, vector_store);
    }

    if added_ids.is_empty() && removed_ids.is_empty() {
        return Ok(());
    }

    let params = Sq8Params::load(&params_path)
        .with_context(|| "Failed to load SQ8 params")?;

    let mut sq8_store = Sq8VectorStore::open(&store_path)
        .with_context(|| "Failed to open SQ8 vector store")?;

    // Restore id_map from the main vector store
    sq8_store.restore_id_map(vector_store.id_map());

    // Insert new vectors at the same slot positions as the main store.
    let id_map = vector_store.id_map();
    let mut buf = vec![0u8; params.dim()];
    for &id in added_ids {
        if let Some(&slot) = id_map.get(&id)
            && let Ok(vec) = vector_store.get(id)
        {
            params.quantize_into(vec, &mut buf);
            sq8_store.insert_at(id, slot, &buf)?;
        }
    }

    sq8_store.flush()?;
    println!(
        "  SQ8 vectors updated ({} additions).",
        added_ids.len()
    );

    Ok(())
}
