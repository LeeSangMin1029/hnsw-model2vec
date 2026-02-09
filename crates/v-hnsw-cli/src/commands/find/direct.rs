//! Direct search without daemon (fallback mode).

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::Command;
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

/// Spawn daemon in background and wait for it to be ready.
pub fn spawn_daemon(db_path: &Path) -> Result<()> {
    let canonical_path = db_path
        .canonicalize()
        .with_context(|| format!("Database not found: {}", db_path.display()))?;

    eprintln!("Starting daemon for {}...", canonical_path.display());

    let exe = std::env::current_exe()?;

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
        const CREATE_NO_WINDOW: u32 = 0x08000000;

        let path_str = canonical_path.to_str().context("Non-UTF8 path")?;
        Command::new(&exe)
            .args(["serve", path_str, "--timeout", "300"])
            .creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .context("Failed to spawn daemon")?;
    }

    #[cfg(not(windows))]
    {
        let path_str = canonical_path.to_str().context("Non-UTF8 path")?;
        Command::new(&exe)
            .args(["serve", path_str, "--timeout", "300"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .context("Failed to spawn daemon")?;
    }

    let start = Instant::now();
    let timeout = Duration::from_secs(60);

    eprintln!("Waiting for daemon to load model...");

    loop {
        if start.elapsed() > timeout {
            anyhow::bail!(
                "Timeout waiting for daemon to start ({}s)",
                timeout.as_secs()
            );
        }

        if let Some(port) = serve::read_port_file(&canonical_path)
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

/// BM25-only search (no model loading, fast cold start).
pub fn run_bm25_only(db_path: PathBuf, query: String, k: usize, tags: Vec<String>, full: bool) -> Result<()> {
    let bm25_path = db_path.join("bm25.bin");
    if !bm25_path.exists() {
        anyhow::bail!("BM25 index not found. Run 'v-hnsw add' first to index data.");
    }

    let t0 = Instant::now();
    common::ensure_korean_dict()?;
    let bm25: Bm25Index<KoreanBm25Tokenizer> =
        Bm25Index::load(&bm25_path).context("Failed to load BM25 index")?;
    eprintln!("  BM25 load: {:.0}ms", t0.elapsed().as_millis());

    let engine = StorageEngine::open(&db_path).context("Failed to open storage")?;
    let payload_store = engine.payload_store();

    let start = Instant::now();
    let total_docs = bm25.len();

    let fetch_k = if tags.is_empty() { k } else { k * 10 };
    let mut results = bm25.search(&query, fetch_k);

    if !tags.is_empty() {
        let allowed_ids = payload_store.points_by_tags(&tags);
        let allowed_set: std::collections::HashSet<_> = allowed_ids.into_iter().collect();
        results.retain(|(id, _)| allowed_set.contains(id));
        results.truncate(k);
    }

    let results_with_text = common::build_results(&results, payload_store);
    let elapsed = start.elapsed();

    let output = FindOutput {
        results: results_with_text,
        query,
        model: String::new(),
        total_docs,
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    };

    let output = if full { output } else { super::compact_output(output) };
    let json = serde_json::to_string_pretty(&output).context("Failed to serialize output")?;
    println!("{json}");

    Ok(())
}

/// Raw vector search (dense-only or dense+BM25 hybrid).
pub fn run_raw_vector(
    db_path: PathBuf,
    raw_vec: Vec<f32>,
    query_text: Option<String>,
    k: usize,
    tags: Vec<String>,
    full: bool,
    ef: usize,
) -> Result<()> {
    let config = DbConfig::load(&db_path)?;

    if raw_vec.len() != config.dim {
        anyhow::bail!(
            "Vector dimension mismatch: expected {}, got {}",
            config.dim,
            raw_vec.len()
        );
    }

    let hnsw_path = db_path.join("hnsw.bin");
    if !hnsw_path.exists() {
        anyhow::bail!("HNSW index not found. Run 'v-hnsw add' first to index data.");
    }

    let engine = StorageEngine::open(&db_path).context("Failed to open storage")?;
    let hnsw: HnswGraph<NormalizedCosineDistance> =
        HnswGraph::load(&hnsw_path, NormalizedCosineDistance).context("Failed to load HNSW index")?;
    let total_docs = hnsw.len();

    let start = Instant::now();

    let mut results = if let Some(ref text) = query_text {
        // Dense + BM25 hybrid with raw vector
        let bm25_path = db_path.join("bm25.bin");
        if !bm25_path.exists() {
            anyhow::bail!("BM25 index not found. Run 'v-hnsw add' first to index data.");
        }
        common::ensure_korean_dict()?;
        let bm25: Bm25Index<KoreanBm25Tokenizer> =
            Bm25Index::load(&bm25_path).context("Failed to load BM25 index")?;

        let alpha = common::fusion_alpha(text);
        let hybrid_config = HybridSearchConfig::builder()
            .ef_search(ef)
            .dense_limit(k * 2)
            .sparse_limit(k * 2)
            .fusion_alpha(alpha)
            .build();
        let searcher = SimpleHybridSearcher::new(hnsw, bm25, hybrid_config);

        let fetch_k = if tags.is_empty() { k } else { k * 10 };
        searcher
            .search_ext(engine.vector_store(), &raw_vec, text, fetch_k)
            .context("Hybrid search failed")?
    } else {
        // Dense-only with raw vector
        let fetch_k = if tags.is_empty() { k } else { k * 10 };
        hnsw.search_ext(engine.vector_store(), &raw_vec, fetch_k, ef)
            .context("HNSW search failed")?
    };

    let payload_store = engine.payload_store();

    if !tags.is_empty() {
        let allowed_ids = payload_store.points_by_tags(&tags);
        let allowed_set: std::collections::HashSet<_> = allowed_ids.into_iter().collect();
        results.retain(|(id, _)| allowed_set.contains(id));
        results.truncate(k);
    }

    let results_with_text = common::build_results(&results, payload_store);
    let elapsed = start.elapsed();

    let output = FindOutput {
        results: results_with_text,
        query: query_text.unwrap_or_default(),
        model: String::new(),
        total_docs,
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    };

    let output = if full { output } else { super::compact_output(output) };
    let json = serde_json::to_string_pretty(&output).context("Failed to serialize output")?;
    println!("{json}");

    Ok(())
}

/// Direct search without daemon (fallback).
pub fn run_direct(db_path: PathBuf, query: String, k: usize, tags: Vec<String>, full: bool) -> Result<()> {
    let config = DbConfig::load(&db_path)?;

    let hnsw_path = db_path.join("hnsw.bin");
    if !hnsw_path.exists() {
        anyhow::bail!("HNSW index not found. Run 'v-hnsw add' first to index data.");
    }

    let bm25_path = db_path.join("bm25.bin");
    if !bm25_path.exists() {
        anyhow::bail!("BM25 index not found. Run 'v-hnsw add' first to index data.");
    }

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
    let engine = StorageEngine::open(&db_path).context("Failed to open storage")?;
    let hnsw: HnswGraph<NormalizedCosineDistance> =
        HnswGraph::load(&hnsw_path, NormalizedCosineDistance).context("Failed to load HNSW index")?;
    eprintln!("  HNSW load: {:.0}ms", t2.elapsed().as_millis());

    let t3 = Instant::now();
    common::ensure_korean_dict()?;
    let bm25: Bm25Index<KoreanBm25Tokenizer> =
        Bm25Index::load(&bm25_path).context("Failed to load BM25 index")?;
    eprintln!("  BM25 load: {:.0}ms", t3.elapsed().as_millis());

    let start = Instant::now();
    let total_docs = hnsw.len();

    let alpha = common::fusion_alpha(&query);
    let hybrid_config = HybridSearchConfig::builder()
        .ef_search(200)
        .dense_limit(k * 2)
        .sparse_limit(k * 2)
        .fusion_alpha(alpha)
        .build();

    let searcher = SimpleHybridSearcher::new(hnsw, bm25, hybrid_config);

    let fetch_k = if tags.is_empty() { k } else { k * 10 };

    let t4 = Instant::now();
    let mut results = searcher
        .search_ext(engine.vector_store(), &query_embedding, &query, fetch_k)
        .context("Hybrid search failed")?;
    eprintln!("  Search: {:.0}ms", t4.elapsed().as_millis());

    let payload_store = engine.payload_store();

    if !tags.is_empty() {
        let t5 = Instant::now();
        let allowed_ids = payload_store.points_by_tags(&tags);
        let allowed_set: std::collections::HashSet<_> = allowed_ids.into_iter().collect();
        results.retain(|(id, _)| allowed_set.contains(id));
        results.truncate(k);
        eprintln!(
            "  Tag filter: {:.0}ms ({} results after filtering)",
            t5.elapsed().as_millis(),
            results.len()
        );
    }

    let results_with_text = common::build_results(&results, payload_store);

    let elapsed = start.elapsed();

    let output = FindOutput {
        results: results_with_text,
        query,
        model: model_name,
        total_docs,
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    };

    let output = if full { output } else { super::compact_output(output) };
    let json = serde_json::to_string_pretty(&output).context("Failed to serialize output")?;
    println!("{json}");

    Ok(())
}
