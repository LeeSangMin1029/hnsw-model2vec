//! Index building and incremental updating (HNSW + BM25).

use std::path::Path;

use anyhow::{Context, Result};
use v_hnsw_core::{PayloadStore, VectorIndex};
use v_hnsw_graph::{HnswGraph, HnswSnapshot, NormalizedCosineDistance};
use v_hnsw_search::{Bm25Index, KoreanBm25Tokenizer};
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

    bm25.save(&bm25_path)
        .with_context(|| "Failed to save BM25 index")?;
    bm25.save_snapshot(path)
        .with_context(|| "Failed to save BM25 snapshot")?;
    println!("  BM25 index updated ({total_changes} changes) + snapshot.");

    tracing::info!("Incremental index update completed");
    Ok(())
}
