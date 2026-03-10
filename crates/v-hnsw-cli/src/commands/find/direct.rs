//! Direct search without daemon (fallback mode).

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use v_hnsw_core::VectorIndex;
use v_hnsw_embed::{EmbeddingModel, Model2VecModel};
use v_hnsw_graph::{NormalizedCosineDistance, HnswGraph};
use v_hnsw_search::{Bm25Index, HybridSearchConfig, KoreanBm25Tokenizer, SimpleHybridSearcher};
use v_hnsw_storage::StorageEngine;

use crate::commands::common;
use crate::commands::create::DbConfig;
use crate::commands::serve;

use super::FindOutput;

/// Pre-loaded HNSW + storage context shared by both search paths.
struct SearchContext {
    engine: StorageEngine,
    hnsw: HnswGraph<NormalizedCosineDistance>,
    total_docs: usize,
}

impl SearchContext {
    fn open(db_path: &Path) -> Result<Self> {
        let hnsw_path = db_path.join("hnsw.bin");
        if !hnsw_path.exists() {
            anyhow::bail!("HNSW index not found. Run 'v-hnsw add' first to index data.");
        }
        let engine = StorageEngine::open(db_path).context("Failed to open storage")?;
        let hnsw: HnswGraph<NormalizedCosineDistance> =
            HnswGraph::load(&hnsw_path, NormalizedCosineDistance)
                .context("Failed to load HNSW index")?;
        let total_docs = hnsw.len();
        Ok(Self { engine, hnsw, total_docs })
    }
}

/// Load BM25 index and run hybrid search (dense + sparse).
#[expect(clippy::too_many_arguments)]
fn hybrid_search(
    hnsw: HnswGraph<NormalizedCosineDistance>,
    engine: &StorageEngine,
    db_path: &Path,
    query_vec: &[f32],
    query_text: &str,
    k: usize,
    fetch_k: usize,
    ef: usize,
) -> Result<Vec<(u64, f32)>> {
    let bm25_path = db_path.join("bm25.bin");
    if !bm25_path.exists() {
        anyhow::bail!("BM25 index not found. Run 'v-hnsw add' first to index data.");
    }
    common::ensure_korean_dict()?;
    let bm25: Bm25Index<KoreanBm25Tokenizer> =
        Bm25Index::load(&bm25_path).context("Failed to load BM25 index")?;

    let alpha = common::fusion_alpha(query_text);
    let hybrid_config = HybridSearchConfig::builder()
        .ef_search(ef)
        .dense_limit(k * 2)
        .sparse_limit(k * 2)
        .fusion_alpha(alpha)
        .build();
    let searcher = SimpleHybridSearcher::new(hnsw, bm25, hybrid_config);

    searcher
        .search_ext(engine.vector_store(), query_vec, query_text, fetch_k)
        .context("Hybrid search failed")
}

/// Apply tag filtering, build output, and print JSON.
#[expect(clippy::too_many_arguments)]
fn filter_and_output(
    results: &mut Vec<(u64, f32)>,
    engine: &StorageEngine,
    tags: &[String],
    k: usize,
    query: String,
    model: String,
    total_docs: usize,
    start: Instant,
    full: bool,
    min_score: f32,
) -> Result<()> {
    let payload_store = engine.payload_store();

    if !tags.is_empty() {
        let allowed_ids = payload_store.points_by_tags(tags);
        let allowed_set: HashSet<_> = allowed_ids.into_iter().collect();
        results.retain(|(id, _)| allowed_set.contains(id));
        results.truncate(k);
    }

    let results_with_text = common::build_results(results, payload_store);
    let elapsed = start.elapsed();

    let output = FindOutput {
        results: results_with_text,
        query,
        model,
        total_docs,
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    };

    super::print_output(output, full, min_score)
}

/// Spawn global daemon in background and wait for it to be ready.
///
/// `db_path` is passed so the daemon can preload that database on startup.
pub fn spawn_daemon(db_path: &Path) -> Result<()> {
    let canonical = db_path
        .canonicalize()
        .with_context(|| format!("Database not found: {}", db_path.display()))?;
    let path_str = canonical.to_str().context("Non-UTF8 path")?;

    eprintln!("Starting daemon for {}...", canonical.display());
    common::spawn_detached(&["serve", path_str, "--timeout", "300"])?;

    let start = Instant::now();
    let timeout = Duration::from_secs(60);

    eprintln!("Waiting for daemon to be ready...");

    loop {
        if start.elapsed() > timeout {
            anyhow::bail!("Timeout waiting for daemon to start ({}s)", timeout.as_secs());
        }

        if let Some(port) = serve::read_port_file()
            && let Ok(addr) = format!("127.0.0.1:{}", port).parse()
            && let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(500))
        {
            let ping = serde_json::json!({"id": 0, "method": "ping", "params": {}});
            if let Ok(json) = serde_json::to_string(&ping)
                && writeln!(stream, "{}", json).is_ok()
            {
                stream.flush().ok();
                let mut reader = BufReader::new(&stream);
                let mut response = String::new();
                if reader.read_line(&mut response).is_ok() && response.contains("ok") {
                    eprintln!("Daemon ready on port {}", port);
                    return Ok(());
                }
            }
        }

        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Raw vector search (dense-only or dense+BM25 hybrid).
#[expect(clippy::too_many_arguments)]
pub fn run_raw_vector(
    db_path: PathBuf,
    raw_vec: Vec<f32>,
    query_text: Option<String>,
    k: usize,
    tags: Vec<String>,
    full: bool,
    ef: usize,
    min_score: f32,
) -> Result<()> {
    let config = DbConfig::load(&db_path)?;

    if raw_vec.len() != config.dim {
        anyhow::bail!(
            "Vector dimension mismatch: expected {}, got {}",
            config.dim,
            raw_vec.len()
        );
    }

    let ctx = SearchContext::open(&db_path)?;
    let start = Instant::now();
    let fetch_k = if tags.is_empty() { k } else { k * 10 };

    let mut results = if let Some(ref text) = query_text {
        hybrid_search(ctx.hnsw, &ctx.engine, &db_path, &raw_vec, text, k, fetch_k, ef)?
    } else {
        ctx.hnsw
            .search_ext(ctx.engine.vector_store(), &raw_vec, fetch_k, ef)
            .context("HNSW search failed")?
    };

    filter_and_output(
        &mut results,
        &ctx.engine,
        &tags,
        k,
        query_text.unwrap_or_default(),
        String::new(),
        ctx.total_docs,
        start,
        full,
        min_score,
    )
}

/// Direct search without daemon (fallback).
pub fn run_direct(db_path: PathBuf, query: String, k: usize, tags: Vec<String>, full: bool, min_score: f32) -> Result<()> {
    let config = DbConfig::load(&db_path)?;

    // Try query cache first — skip model loading if cached
    let mut cache = common::QueryCache::load(&db_path);
    let (query_embedding, model_name) = if let Some(cached) = cache.get(&query) {
        eprintln!("  Cache hit! Skipping model load");
        (cached.clone(), String::new())
    } else {
        let model_name = config.embed_model.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "No embedding model specified in database config.\n\
                 The database may have been created with raw vectors.\n\
                 Use 'v-hnsw add' to create a new database with embedding."
            )
        })?;

        let t0 = Instant::now();
        eprintln!("Loading model2vec: {}", model_name);

        let embed_model = Model2VecModel::from_pretrained(&model_name)
            .context("Failed to initialize model2vec model")?;
        eprintln!("  Model loaded: {:.0}ms", t0.elapsed().as_millis());

        if embed_model.dim() != config.dim {
            anyhow::bail!(
                "Model dimension ({}) doesn't match database dimension ({}).\n\
                 The database was likely created with a different model.",
                embed_model.dim(),
                config.dim
            );
        }

        let t1 = Instant::now();
        let emb = embed_model
            .embed(&[query.as_str()])
            .context("Failed to embed query")?
            .into_iter()
            .next()
            .context("No embedding returned")?;
        eprintln!("  Query embed: {:.0}ms", t1.elapsed().as_millis());

        cache.insert(query.clone(), emb.clone());
        // Save cache immediately (direct mode has no persistent daemon)
        cache.save().ok();

        (emb, model_name)
    };

    let t2 = Instant::now();
    let ctx = SearchContext::open(&db_path)?;
    eprintln!("  HNSW+Storage load: {:.0}ms", t2.elapsed().as_millis());

    let start = Instant::now();
    let fetch_k = if tags.is_empty() { k } else { k * 10 };

    let t3 = Instant::now();
    let mut results = hybrid_search(
        ctx.hnsw, &ctx.engine, &db_path, &query_embedding, &query, k, fetch_k, 200,
    )?;
    eprintln!("  BM25+Search: {:.0}ms", t3.elapsed().as_millis());

    filter_and_output(
        &mut results,
        &ctx.engine,
        &tags,
        k,
        query,
        model_name,
        ctx.total_docs,
        start,
        full,
        min_score,
    )
}
