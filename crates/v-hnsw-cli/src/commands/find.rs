//! Find command - Unified search with hybrid HNSW + BM25 and RRF fusion.
//!
//! Uses daemon mode by default for fast searches. Falls back to direct mode
//! if daemon fails.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use v_hnsw_core::{PayloadStore, VectorIndex};
use v_hnsw_distance::CosineDistance;
use v_hnsw_embed::{Device, EmbeddingModel, FastEmbedModel, ModelType};
use v_hnsw_graph::HnswGraph;
use v_hnsw_search::{Bm25Index, HybridSearchConfig, SimpleHybridSearcher, WhitespaceTokenizer};
use v_hnsw_storage::StorageEngine;

use super::create::DbConfig;
use super::serve;

/// Search result for JSON output.
#[derive(Debug, Serialize, Deserialize)]
struct FindOutput {
    results: Vec<FindResult>,
    #[serde(default)]
    query: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    total_docs: usize,
    elapsed_ms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct FindResult {
    id: u64,
    score: f32,
    text: Option<String>,
    source: Option<String>,
    title: Option<String>,
}

/// JSON-RPC response for daemon communication.
#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    id: u64,
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcError {
    #[allow(dead_code)]
    code: i32,
    message: String,
}

/// Parse model name to ModelType.
fn parse_model_type(name: &str) -> Result<ModelType> {
    match name.to_lowercase().as_str() {
        "all-mini-lm-l6-v2" | "minilm" => Ok(ModelType::AllMiniLML6V2),
        "all-mini-lm-l12-v2" => Ok(ModelType::AllMiniLML12V2),
        "bge-small-en-v1.5" | "bge-small" => Ok(ModelType::BGESmallENV15),
        "bge-base-en-v1.5" | "bge-base" => Ok(ModelType::BGEBaseENV15),
        "bge-large-en-v1.5" | "bge-large" => Ok(ModelType::BGELargeENV15),
        "multilingual-e5-small" | "e5-small" => Ok(ModelType::MultilingualE5Small),
        "multilingual-e5-base" | "e5-base" => Ok(ModelType::MultilingualE5Base),
        "multilingual-e5-large" | "e5-large" => Ok(ModelType::MultilingualE5Large),
        other => anyhow::bail!(
            "Unknown model: '{}'. Available: all-mini-lm-l6-v2, bge-small-en-v1.5, bge-base-en-v1.5, multilingual-e5-small, etc.",
            other
        ),
    }
}

/// Try to connect to the daemon for a given database.
fn try_daemon_search(db_path: &PathBuf, query: &str, k: usize) -> Result<FindOutput> {
    // Canonicalize the path for consistent port file lookup
    let canonical_path = db_path.canonicalize()
        .with_context(|| format!("Database not found: {}", db_path.display()))?;

    // Read port from port file
    let port = serve::read_port_file(&canonical_path)
        .ok_or_else(|| anyhow::anyhow!("Daemon not running (no port file)"))?;

    // Connect to daemon
    let mut stream = TcpStream::connect_timeout(
        &format!("127.0.0.1:{}", port).parse()?,
        Duration::from_secs(2),
    ).context("Failed to connect to daemon")?;

    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;

    // Send search request
    let request = serde_json::json!({
        "id": 1,
        "method": "search",
        "params": {
            "query": query,
            "k": k
        }
    });

    writeln!(stream, "{}", serde_json::to_string(&request)?)?;
    stream.flush()?;

    // Read response
    let mut reader = BufReader::new(&stream);
    let mut response_line = String::new();
    reader.read_line(&mut response_line)?;

    let response: JsonRpcResponse = serde_json::from_str(&response_line)
        .context("Failed to parse daemon response")?;

    // Check for errors
    if let Some(err) = response.error {
        anyhow::bail!("Daemon error: {}", err.message);
    }

    // Parse result
    let result = response.result
        .ok_or_else(|| anyhow::anyhow!("Empty response from daemon"))?;

    let search_response: FindOutput = serde_json::from_value(result)
        .context("Failed to parse search results")?;

    Ok(search_response)
}

/// Spawn daemon in background and wait for it to be ready.
fn spawn_daemon(db_path: &PathBuf) -> Result<()> {
    let canonical_path = db_path.canonicalize()
        .with_context(|| format!("Database not found: {}", db_path.display()))?;

    eprintln!("Starting daemon for {}...", canonical_path.display());

    // Spawn daemon using serve command
    let exe = std::env::current_exe()?;

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
        const CREATE_NO_WINDOW: u32 = 0x08000000;

        Command::new(&exe)
            .args(["serve", canonical_path.to_str().unwrap()])
            .creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .context("Failed to spawn daemon")?;
    }

    #[cfg(not(windows))]
    {
        Command::new(&exe)
            .args(["serve", canonical_path.to_str().unwrap()])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .context("Failed to spawn daemon")?;
    }

    // Wait for daemon to be ready (poll with timeout)
    let start = Instant::now();
    let timeout = Duration::from_secs(60); // Model loading can take a while on first run

    eprintln!("Waiting for daemon to load model...");

    loop {
        if start.elapsed() > timeout {
            anyhow::bail!("Timeout waiting for daemon to start ({}s)", timeout.as_secs());
        }

        // Check if port file exists and daemon is connectable
        if let Some(port) = serve::read_port_file(&canonical_path) {
            if let Ok(mut stream) = TcpStream::connect_timeout(
                &format!("127.0.0.1:{}", port).parse().unwrap(),
                Duration::from_millis(500),
            ) {
                // Send ping to verify daemon is ready
                let ping = serde_json::json!({"id": 0, "method": "ping", "params": {}});
                if writeln!(stream, "{}", serde_json::to_string(&ping).unwrap()).is_ok() {
                    stream.flush().ok();
                    let mut reader = BufReader::new(&stream);
                    let mut response = String::new();
                    if reader.read_line(&mut response).is_ok() && response.contains("ok") {
                        eprintln!("Daemon ready on port {}", port);
                        return Ok(());
                    }
                }
            }
        }

        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Run the find command.
pub fn run(db_path: PathBuf, query: String, k: usize) -> Result<()> {
    // Check database exists
    if !db_path.exists() {
        anyhow::bail!("Database not found at {}", db_path.display());
    }

    // Try daemon mode first
    match try_daemon_search(&db_path, &query, k) {
        Ok(output) => {
            // Add query info
            let output = FindOutput {
                query: query.clone(),
                model: String::from("(daemon)"),
                total_docs: 0, // Not available from daemon
                ..output
            };
            let json = serde_json::to_string_pretty(&output)?;
            println!("{json}");
            return Ok(());
        }
        Err(_) => {
            // Daemon not running, try to start it
            eprintln!("Daemon not running, starting...");
        }
    }

    // Try to spawn daemon
    if spawn_daemon(&db_path).is_ok() {
        // Retry with daemon
        match try_daemon_search(&db_path, &query, k) {
            Ok(output) => {
                let output = FindOutput {
                    query: query.clone(),
                    model: String::from("(daemon)"),
                    total_docs: 0,
                    ..output
                };
                let json = serde_json::to_string_pretty(&output)?;
                println!("{json}");
                return Ok(());
            }
            Err(e) => {
                eprintln!("Daemon search failed: {}", e);
                eprintln!("Falling back to direct search...");
            }
        }
    } else {
        eprintln!("Failed to start daemon, using direct search...");
    }

    // Fallback: direct search (original implementation)
    run_direct(db_path, query, k)
}

/// Direct search without daemon (fallback).
fn run_direct(db_path: PathBuf, query: String, k: usize) -> Result<()> {
    // Load config
    let config = DbConfig::load(&db_path)?;

    // Check for HNSW index
    let hnsw_path = db_path.join("hnsw.bin");
    if !hnsw_path.exists() {
        anyhow::bail!(
            "HNSW index not found. Run 'v-hnsw add' first to index data."
        );
    }

    // Check for BM25 index
    let bm25_path = db_path.join("bm25.bin");
    if !bm25_path.exists() {
        anyhow::bail!(
            "BM25 index not found. Run 'v-hnsw add' first to index data."
        );
    }

    // Determine model from config (auto-detect)
    let model_name = config.embed_model.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "No embedding model specified in database config.\n\
             The database may have been created with raw vectors.\n\
             Use 'v-hnsw add' to create a new database with embedding."
        )
    })?;

    // Initialize embedding model with auto device detection
    let model_type = parse_model_type(&model_name)?;
    let device = Device::auto();

    let t0 = Instant::now();
    eprintln!("Loading model: {} (device={})", model_type.model_name(), device.name());

    let embed_model = FastEmbedModel::with_device(model_type, device)
        .context("Failed to initialize embedding model")?;
    eprintln!("  Model loaded: {:.0}ms", t0.elapsed().as_millis());

    // Check dimension matches
    if embed_model.dim() != config.dim {
        anyhow::bail!(
            "Model dimension ({}) doesn't match database dimension ({}).\n\
             The database was likely created with a different model.",
            embed_model.dim(),
            config.dim
        );
    }

    // Embed the query
    let t1 = Instant::now();
    let query_embedding = embed_model
        .embed(&[query.as_str()])
        .context("Failed to embed query")?
        .into_iter()
        .next()
        .context("No embedding returned")?;
    eprintln!("  Query embed: {:.0}ms", t1.elapsed().as_millis());

    // Load HNSW index
    let t2 = Instant::now();
    let hnsw: HnswGraph<CosineDistance> = HnswGraph::load(&hnsw_path, CosineDistance)
        .context("Failed to load HNSW index")?;
    eprintln!("  HNSW load: {:.0}ms", t2.elapsed().as_millis());

    // Load BM25 index
    let t3 = Instant::now();
    let bm25: Bm25Index<WhitespaceTokenizer> = Bm25Index::load(&bm25_path)
        .context("Failed to load BM25 index")?;
    eprintln!("  BM25 load: {:.0}ms", t3.elapsed().as_millis());

    let start = Instant::now();

    let total_docs = hnsw.len();

    // Create hybrid searcher with RRF fusion
    let hybrid_config = HybridSearchConfig::builder()
        .ef_search(200)
        .dense_limit(k * 2)  // Fetch more candidates for fusion
        .sparse_limit(k * 2)
        .dense_weight(0.7)   // Vector search weight
        .sparse_weight(0.3)  // BM25 weight
        .build();

    let searcher = SimpleHybridSearcher::new(hnsw, bm25, hybrid_config);

    // Perform hybrid search
    let t4 = Instant::now();
    let results = searcher.search(&query_embedding, &query, k)
        .context("Hybrid search failed")?;
    eprintln!("  Search: {:.0}ms", t4.elapsed().as_millis());

    // Load storage for text retrieval
    let engine = StorageEngine::open(&db_path)
        .context("Failed to open storage")?;
    let payload_store = engine.payload_store();

    // Build results with text
    let results_with_text: Vec<FindResult> = results
        .into_iter()
        .map(|(id, score)| {
            let text = payload_store.get_text(id).ok().flatten();

            // Get payload for source and title
            let payload = payload_store.get_payload(id).ok().flatten();
            let source = payload.as_ref()
                .map(|p| p.source.clone())
                .filter(|s: &String| !s.is_empty());
            let title = payload.as_ref()
                .and_then(|p| p.custom.get("title"))
                .and_then(|v| match v {
                    v_hnsw_core::PayloadValue::String(s) => Some(s.clone()),
                    _ => None,
                });

            FindResult {
                id,
                score,
                text,
                source,
                title,
            }
        })
        .collect();

    let elapsed = start.elapsed();

    // Output as JSON
    let output = FindOutput {
        results: results_with_text,
        query,
        model: model_name,
        total_docs,
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    };

    let json = serde_json::to_string_pretty(&output)
        .context("Failed to serialize output")?;
    println!("{json}");

    Ok(())
}
