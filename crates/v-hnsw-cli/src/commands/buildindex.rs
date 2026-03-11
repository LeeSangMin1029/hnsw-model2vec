//! BuildIndex command - Rebuild indexes from existing storage.

use std::path::PathBuf;
use std::time::Instant;

use std::collections::HashMap;

use anyhow::{Context, Result};
use v_hnsw_core::{DistanceMetric, PayloadStore, PointId, VectorStore};
use v_hnsw_graph::{DotProductDistance, HnswConfig, HnswGraph, HnswSnapshot, L2Distance, NormalizedCosineDistance};
use v_hnsw_search::{Bm25Index, KoreanBm25Tokenizer};
use v_hnsw_storage::StorageEngine;

use super::create::DbConfig;

pub fn run(path: PathBuf) -> Result<()> {
    super::common::require_db(&path)?;

    let config = DbConfig::load(&path)?;
    let engine = StorageEngine::open_exclusive(&path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    println!("Building indexes for {} vectors...", engine.len());
    let start = Instant::now();

    let hnsw_path = path.join("hnsw.bin");
    let bm25_path = path.join("bm25.bin");

    let hnsw_config = config.to_hnsw_config()?;

    let vector_store = engine.vector_store();

    // Sort by mmap slot for sequential read (cache-friendly)
    let ids = sorted_ids_by_slot(vector_store.id_map());

    println!("  Building HNSW graph (metric={}, M={}, ef_construction={})...",
             config.metric, config.m, config.ef_construction);

    // Init Korean dict before parallel scope (required by BM25 thread)
    super::common::ensure_korean_dict()?;

    let payload_store = engine.payload_store();

    let hnsw_snap_path = path.join("hnsw.snap");

    // Parallel HNSW + BM25 build
    std::thread::scope(|s| -> Result<()> {
        let hnsw_handle = s.spawn(|| -> Result<()> {
            match config.metric.as_str() {
                "cosine" => build_hnsw(hnsw_config, NormalizedCosineDistance, vector_store, &ids, &hnsw_path, &hnsw_snap_path),
                "l2" => build_hnsw(hnsw_config, L2Distance, vector_store, &ids, &hnsw_path, &hnsw_snap_path),
                "dot" => build_hnsw(hnsw_config, DotProductDistance, vector_store, &ids, &hnsw_path, &hnsw_snap_path),
                other => anyhow::bail!("Unknown metric: {other}"),
            }
        });

        let bm25_handle = s.spawn(|| -> Result<()> {
            build_bm25(payload_store, &ids, &bm25_path, &path)
        });

        #[expect(clippy::expect_used, reason = "thread panics are unrecoverable")]
        hnsw_handle.join().expect("HNSW thread panicked")?;
        #[expect(clippy::expect_used, reason = "thread panics are unrecoverable")]
        bm25_handle.join().expect("BM25 thread panicked")?;
        Ok(())
    })?;

    let hnsw_size = std::fs::metadata(&hnsw_path)?.len();
    let bm25_size = std::fs::metadata(&bm25_path)?.len();
    println!("  HNSW saved: {} ({:.2} MB)", hnsw_path.display(), hnsw_size as f64 / 1_000_000.0);
    println!("  BM25 saved: {} ({:.2} MB)", bm25_path.display(), bm25_size as f64 / 1_000_000.0);

    // Build SQ8 quantized vectors
    super::indexing::build_sq8(&path, vector_store)?;

    // Compress texts with FSST
    println!("  Compressing texts with FSST...");
    let texts = payload_store.all_text_bytes()?;
    v_hnsw_storage::compress_texts(&texts, &path)?;

    let elapsed = start.elapsed();
    println!("Index build completed in {:.2}s", elapsed.as_secs_f64());

    Ok(())
}

/// Sort IDs by mmap slot for sequential memory access during build.
pub(crate) fn sorted_ids_by_slot(id_map: &HashMap<PointId, u32>) -> Vec<u64> {
    let mut pairs: Vec<(u64, u32)> = id_map.iter().map(|(&id, &slot)| (id, slot)).collect();
    pairs.sort_unstable_by_key(|&(_, slot)| slot);
    pairs.into_iter().map(|(id, _)| id).collect()
}

/// Build HNSW graph using external store (no vector copy).
fn build_hnsw<D: DistanceMetric>(
    config: HnswConfig,
    distance: D,
    store: &dyn VectorStore,
    ids: &[u64],
    path: &std::path::Path,
    snap_path: &std::path::Path,
) -> Result<()> {
    let mut hnsw = HnswGraph::new(config, distance);
    for &id in ids {
        let _ = hnsw.build_insert(store, id);
    }
    hnsw.save(path).with_context(|| "failed to save HNSW")?;
    HnswSnapshot::save(&hnsw, snap_path).with_context(|| "failed to save HNSW snapshot")?;
    println!("  HNSW snapshot: {}", snap_path.display());
    Ok(())
}

/// Build BM25 index from payload text.
fn build_bm25(
    payload_store: &dyn PayloadStore,
    ids: &[u64],
    path: &std::path::Path,
    db_dir: &std::path::Path,
) -> Result<()> {
    let mut bm25: Bm25Index<KoreanBm25Tokenizer> = Bm25Index::new(KoreanBm25Tokenizer::new());
    let mut text_count = 0;
    for &id in ids {
        if let Ok(Some(text)) = payload_store.get_text(id)
            && !text.is_empty() {
                bm25.add_document(id, &text);
                text_count += 1;
            }
    }
    bm25.build_fieldnorm_cache();
    bm25.save(path).with_context(|| "failed to save BM25")?;
    bm25.save_snapshot(db_dir).with_context(|| "failed to save BM25 snapshot")?;
    println!("  BM25 built: {} documents (+ snapshot)", text_count);
    Ok(())
}
