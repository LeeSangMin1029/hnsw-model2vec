//! Code search with hybrid BM25+HNSW and cross-encoder reranking.

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
    build_results, FindOutput, print_find_output,
};

/// Parameters for the find command.
pub struct FindParams {
    pub db: PathBuf,
    pub query: String,
    pub k: usize,
    pub full: bool,
    pub min_score: f32,
    pub no_rerank: bool,
}

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

    let hybrid_config = HybridSearchConfig::builder()
        .ef_search(ef)
        .dense_limit(k * 2)
        .sparse_limit(k * 2)
        .fusion_alpha(0.7)
        .build();

    let bm25: Bm25Index<CodeTokenizer> =
        Bm25Index::load(&bm25_path).context("Failed to load BM25 index")?;
    let searcher = SimpleHybridSearcher::new(hnsw, bm25, hybrid_config);
    searcher
        .search_ext(engine.vector_store(), query_vec, query_text, k)
        .context("Hybrid search failed")
}

/// Run the find command.
pub fn run(params: FindParams) -> Result<()> {
    let FindParams { db, query, k, full, min_score, no_rerank } = params;

    if !db.exists() {
        anyhow::bail!("Database not found at {}", db.display());
    }

    let config = DbConfig::load(&db)?;

    // Load embedding model
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

    // Embed query
    let t1 = Instant::now();
    let query_embedding = embed_model
        .embed(&[query.as_str()])
        .context("Failed to embed query")?
        .into_iter()
        .next()
        .context("No embedding returned")?;
    eprintln!("  Query embed: {:.0}ms", t1.elapsed().as_millis());

    // Open DB
    let t2 = Instant::now();
    let ctx = SearchContext::open(&db)?;
    eprintln!("  HNSW+Storage load: {:.0}ms", t2.elapsed().as_millis());

    let start = Instant::now();

    // Fetch more candidates when reranking
    let fetch_k = if no_rerank { k } else { k * 4 };

    let t3 = Instant::now();
    let mut results = hybrid_search(
        ctx.hnsw, &ctx.engine, &db, &query_embedding, &query, fetch_k, 200,
    )?;
    eprintln!("  BM25+Search: {:.0}ms", t3.elapsed().as_millis());

    // Rerank with cross-encoder (shared utility from v-hnsw-rerank)
    if !no_rerank && results.len() > 1 {
        let t4 = Instant::now();
        v_hnsw_rerank::rerank_results(&mut results, &query, ctx.engine.payload_store(), k)?;
        eprintln!("  Rerank: {:.0}ms", t4.elapsed().as_millis());
    }

    // Build output using shared types
    results.truncate(k);
    let results_with_text = build_results(&results, ctx.engine.payload_store());
    let elapsed = start.elapsed();

    let output = FindOutput {
        results: results_with_text,
        query,
        model: model_name,
        total_docs: ctx.total_docs,
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    };

    print_find_output(output, full, min_score)
}
