//! BuildIndex command - Rebuild indexes from existing storage.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use v_hnsw_core::{PayloadStore, VectorIndex, VectorStore};
use v_hnsw_distance::{CosineDistance, DotProductDistance, L2Distance};
use v_hnsw_graph::{HnswConfig, HnswGraph};
use v_hnsw_search::{Bm25Index, KoreanBm25Tokenizer};
use v_hnsw_storage::StorageEngine;

use super::create::DbConfig;

pub fn run(path: PathBuf) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("Database not found at {}", path.display());
    }

    let config = DbConfig::load(&path)?;
    let engine = StorageEngine::open(&path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    println!("Building indexes for {} vectors...", engine.len());
    let start = Instant::now();

    // Build HNSW graph
    let hnsw_path = path.join("hnsw.bin");
    println!("  Building HNSW graph (metric={}, M={}, ef_construction={})...",
             config.metric, config.m, config.ef_construction);

    let hnsw_config = HnswConfig::builder()
        .dim(config.dim)
        .m(config.m)
        .ef_construction(config.ef_construction)
        .build()
        .with_context(|| "failed to create HNSW config")?;

    let vector_store = engine.vector_store();
    let ids: Vec<u64> = vector_store.id_map().keys().copied().collect();

    match config.metric.as_str() {
        "cosine" => {
            let mut hnsw = HnswGraph::new(hnsw_config, CosineDistance);
            for id in &ids {
                if let Ok(vec) = vector_store.get(*id) {
                    let _ = hnsw.insert(*id, vec);
                }
            }
            hnsw.save(&hnsw_path).with_context(|| "failed to save HNSW")?;
        }
        "l2" => {
            let mut hnsw = HnswGraph::new(hnsw_config, L2Distance);
            for id in &ids {
                if let Ok(vec) = vector_store.get(*id) {
                    let _ = hnsw.insert(*id, vec);
                }
            }
            hnsw.save(&hnsw_path).with_context(|| "failed to save HNSW")?;
        }
        "dot" => {
            let mut hnsw = HnswGraph::new(hnsw_config, DotProductDistance);
            for id in &ids {
                if let Ok(vec) = vector_store.get(*id) {
                    let _ = hnsw.insert(*id, vec);
                }
            }
            hnsw.save(&hnsw_path).with_context(|| "failed to save HNSW")?;
        }
        other => anyhow::bail!("Unknown metric: {other}"),
    }

    let hnsw_size = std::fs::metadata(&hnsw_path)?.len();
    println!("  HNSW saved: {} ({:.2} MB)", hnsw_path.display(), hnsw_size as f64 / 1_000_000.0);

    // Build BM25 index
    let bm25_path = path.join("bm25.bin");
    println!("  Building BM25 index...");

    let mut bm25: Bm25Index<KoreanBm25Tokenizer> = Bm25Index::new(KoreanBm25Tokenizer::new());
    let payload_store = engine.payload_store();

    let mut text_count = 0;
    for id in &ids {
        if let Ok(Some(text)) = payload_store.get_text(*id) {
            if !text.is_empty() {
                bm25.add_document(*id, &text);
                text_count += 1;
            }
        }
    }

    bm25.save(&bm25_path).with_context(|| "failed to save BM25")?;

    let bm25_size = std::fs::metadata(&bm25_path)?.len();
    println!("  BM25 saved: {} ({:.2} MB, {} documents)",
             bm25_path.display(), bm25_size as f64 / 1_000_000.0, text_count);

    let elapsed = start.elapsed();
    println!("Index build completed in {:.2}s", elapsed.as_secs_f64());

    Ok(())
}
