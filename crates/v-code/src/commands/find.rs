//! Code search with hybrid BM25+HNSW and cross-encoder reranking.
//!
//! Tries the shared daemon first for fast search (~2ms).
//! Falls back to direct search (loads model from scratch) if daemon unavailable.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use v_hnsw_core::VectorIndex;
use v_hnsw_embed::{EmbeddingModel, Model2VecModel};
use v_hnsw_graph::{HnswGraph, NormalizedCosineDistance};
use v_hnsw_search::{Bm25Index, CodeTokenizer, HybridSearchConfig, SimpleHybridSearcher};
use v_hnsw_storage::StorageEngine;

use v_hnsw_cli::commands::db_config::DbConfig;
use v_hnsw_cli::commands::search_result::{
    build_results, FindOutput, print_find_output, SearchResultItem,
};
use v_hnsw_cli::commands::serve;

/// Parameters for the find command.
pub struct FindParams {
    pub db: PathBuf,
    pub query: String,
    pub k: usize,
    pub full: bool,
    pub min_score: f32,
    pub no_rerank: bool,
}

/// Run the find command.
pub fn run(params: FindParams) -> Result<()> {
    let FindParams { db, query, k, full, min_score, no_rerank } = params;

    if !db.exists() {
        anyhow::bail!("Database not found at {}", db.display());
    }

    // Try daemon first (fast path: model already loaded, mmap indexes)
    if let Ok(result) = try_daemon_search(&db, &query, k) {
        return print_find_output(result, full, min_score);
    }

    // Auto-start daemon in background for next search
    auto_start_daemon(&db);

    // Fallback: direct search (loads model from scratch)
    run_direct(&db, &query, k, full, min_score, no_rerank)
}

// ── Daemon path ─────────────────────────────────────────────────────────

/// Try to search via running daemon.
fn try_daemon_search(db: &Path, query: &str, k: usize) -> Result<FindOutput> {
    let canonical = db.canonicalize()
        .with_context(|| format!("Database not found: {}", db.display()))?;
    let db_str = canonical.to_str().unwrap_or("");

    let params = serde_json::json!({
        "db": db_str,
        "query": query,
        "k": k,
    });

    let result = serve::daemon_rpc("search", params, 30)?;

    let results: Vec<SearchResultItem> =
        serde_json::from_value(result.get("results").cloned().unwrap_or_default())
            .unwrap_or_default();
    let elapsed_ms = result.get("elapsed_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);

    let config = DbConfig::load(db)?;

    Ok(FindOutput {
        results,
        query: query.to_string(),
        model: config.embed_model.unwrap_or_default(),
        total_docs: 0,
        elapsed_ms,
    })
}

/// Auto-start daemon in background if not running.
fn auto_start_daemon(db: &Path) {
    if serve::is_daemon_running() {
        return;
    }
    let db_str = db.to_string_lossy();
    let _ = v_hnsw_cli::commands::common::spawn_detached(&["serve", "--background", &db_str]);
}

// ── Direct search path (fallback) ───────────────────────────────────────

/// Pre-loaded HNSW + storage context.
struct SearchContext {
    engine: StorageEngine,
    hnsw: HnswGraph<NormalizedCosineDistance>,
    total_docs: usize,
}

impl SearchContext {
    fn open(db_path: &Path) -> Result<Self> {
        let hnsw_path = db_path.join("hnsw.bin");
        if !hnsw_path.exists() {
            anyhow::bail!("HNSW index not found. Run 'v-code add' first.");
        }
        let engine = StorageEngine::open(db_path).context("Failed to open storage")?;
        let hnsw: HnswGraph<NormalizedCosineDistance> =
            HnswGraph::load(&hnsw_path, NormalizedCosineDistance)
                .context("Failed to load HNSW index")?;
        let total_docs = hnsw.len();
        Ok(Self { engine, hnsw, total_docs })
    }
}

/// Run hybrid BM25+HNSW search with code tokenizer.
fn hybrid_search(
    hnsw: HnswGraph<NormalizedCosineDistance>,
    engine: &StorageEngine,
    db_path: &Path,
    query_vec: &[f32],
    query_text: &str,
    k: usize,
    ef: usize,
) -> Result<Vec<(u64, f32)>> {
    let bm25_path = db_path.join("bm25.bin");
    if !bm25_path.exists() {
        anyhow::bail!("BM25 index not found. Run 'v-code add' first.");
    }

    let alpha = v_hnsw_cli::commands::search_result::fusion_alpha(query_text);
    let hybrid_config = HybridSearchConfig::builder()
        .ef_search(ef)
        .dense_limit(k * 2)
        .sparse_limit(k * 2)
        .fusion_alpha(alpha)
        .build();

    let bm25: Bm25Index<CodeTokenizer> =
        Bm25Index::load(&bm25_path).context("Failed to load BM25 index")?;
    let searcher = SimpleHybridSearcher::new(hnsw, bm25, hybrid_config);
    searcher
        .search_ext(engine.vector_store(), query_vec, query_text, k)
        .context("Hybrid search failed")
}

/// Direct search without daemon (fallback).
fn run_direct(db: &Path, query: &str, k: usize, full: bool, min_score: f32, no_rerank: bool) -> Result<()> {
    let config = DbConfig::load(db)?;

    let model_name = config.embed_model.clone().ok_or_else(|| {
        anyhow::anyhow!("No embedding model in database config. Run 'v-code add' first.")
    })?;

    let t0 = Instant::now();
    eprintln!("Loading model2vec: {model_name}");
    let embed_model = Model2VecModel::from_pretrained(&model_name)
        .context("Failed to load model")?;
    eprintln!("  Model loaded: {:.0}ms", t0.elapsed().as_millis());

    if embed_model.dim() != config.dim {
        anyhow::bail!(
            "Model dimension ({}) doesn't match database ({})",
            embed_model.dim(),
            config.dim
        );
    }

    let t1 = Instant::now();
    let query_embedding = embed_model
        .embed(&[query])
        .context("Failed to embed query")?
        .into_iter()
        .next()
        .context("No embedding returned")?;
    eprintln!("  Query embed: {:.0}ms", t1.elapsed().as_millis());

    let t2 = Instant::now();
    let ctx = SearchContext::open(db)?;
    eprintln!("  HNSW+Storage load: {:.0}ms", t2.elapsed().as_millis());

    let start = Instant::now();
    let fetch_k = if no_rerank { k } else { k * 4 };

    let t3 = Instant::now();
    let mut results = hybrid_search(
        ctx.hnsw, &ctx.engine, db, &query_embedding, query, fetch_k, 200,
    )?;
    eprintln!("  BM25+Search: {:.0}ms", t3.elapsed().as_millis());

    // Rerank with cross-encoder
    if !no_rerank && results.len() > 1 {
        let t4 = Instant::now();
        v_hnsw_rerank::rerank_results(&mut results, query, ctx.engine.payload_store(), k)?;
        eprintln!("  Rerank: {:.0}ms", t4.elapsed().as_millis());
    }

    results.truncate(k);
    let results_with_text = build_results(&results, ctx.engine.payload_store());
    let elapsed = start.elapsed();

    let output = FindOutput {
        results: results_with_text,
        query: query.to_string(),
        model: model_name,
        total_docs: ctx.total_docs,
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    };

    print_find_output(output, full, min_score)
}
