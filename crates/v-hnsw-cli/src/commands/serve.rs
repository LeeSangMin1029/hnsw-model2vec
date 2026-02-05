//! Serve command - Daemon server for fast embedding search.
//!
//! Keeps the embedding model loaded in memory to avoid repeated model loading.
//! Uses TCP socket for cross-platform compatibility.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::is_interrupted;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use v_hnsw_core::{PayloadStore, VectorIndex};
use v_hnsw_distance::CosineDistance;
use v_hnsw_embed::{Device, EmbeddingModel, FastEmbedModel, ModelType};
use v_hnsw_graph::HnswGraph;
use v_hnsw_search::{Bm25Index, HybridSearchConfig, SimpleHybridSearcher, WhitespaceTokenizer};
use v_hnsw_storage::StorageEngine;

use super::create::DbConfig;

/// JSON-RPC request.
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    id: u64,
    method: String,
    params: serde_json::Value,
}

/// JSON-RPC response.
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

/// JSON-RPC error.
#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

/// Search request parameters.
#[derive(Debug, Deserialize)]
struct SearchParams {
    query: String,
    #[serde(default = "default_k")]
    k: usize,
}

fn default_k() -> usize {
    10
}

/// Search result.
#[derive(Debug, Serialize)]
struct SearchResult {
    id: u64,
    score: f32,
    text: Option<String>,
    source: Option<String>,
    title: Option<String>,
}

/// Search response.
#[derive(Debug, Serialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
    elapsed_ms: f64,
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
        other => anyhow::bail!("Unknown model: '{}'", other),
    }
}

/// Hash a path to create a unique identifier.
fn hash_path(path: &Path) -> String {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Get the PID file path for a database.
pub fn get_pid_file(db_path: &Path) -> PathBuf {
    let hash = hash_path(db_path);
    let cache_dir = directories::ProjectDirs::from("", "", "v-hnsw")
        .map(|d| d.cache_dir().to_path_buf())
        .unwrap_or_else(|| std::env::temp_dir().join("v-hnsw"));

    std::fs::create_dir_all(&cache_dir).ok();
    cache_dir.join(format!("{}.pid", hash))
}

/// Get the port file path for a database.
pub fn get_port_file(db_path: &Path) -> PathBuf {
    let hash = hash_path(db_path);
    let cache_dir = directories::ProjectDirs::from("", "", "v-hnsw")
        .map(|d| d.cache_dir().to_path_buf())
        .unwrap_or_else(|| std::env::temp_dir().join("v-hnsw"));

    std::fs::create_dir_all(&cache_dir).ok();
    cache_dir.join(format!("{}.port", hash))
}

/// Read the port from port file.
pub fn read_port_file(db_path: &Path) -> Option<u16> {
    let port_file = get_port_file(db_path);
    std::fs::read_to_string(&port_file)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Write the port to port file.
fn write_port_file(db_path: &Path, port: u16) -> Result<()> {
    let port_file = get_port_file(db_path);
    std::fs::write(&port_file, port.to_string())
        .with_context(|| format!("Failed to write port file: {}", port_file.display()))
}

/// Write PID file.
fn write_pid_file(db_path: &Path) -> Result<()> {
    let pid_file = get_pid_file(db_path);
    let pid = std::process::id();
    std::fs::write(&pid_file, pid.to_string())
        .with_context(|| format!("Failed to write PID file: {}", pid_file.display()))
}

/// Delete PID and port files.
fn cleanup_files(db_path: &Path) {
    let pid_file = get_pid_file(db_path);
    let port_file = get_port_file(db_path);
    let _ = std::fs::remove_file(&pid_file);
    let _ = std::fs::remove_file(&port_file);
}

/// Check if daemon is already running for this database.
pub fn is_daemon_running(db_path: &Path) -> bool {
    if let Some(port) = read_port_file(db_path) {
        // Try to connect
        TcpStream::connect_timeout(
            &format!("127.0.0.1:{}", port).parse().unwrap(),
            Duration::from_millis(100),
        )
        .is_ok()
    } else {
        false
    }
}

/// Loaded daemon state.
struct DaemonState {
    embed_model: FastEmbedModel,
    searcher: SimpleHybridSearcher<CosineDistance, WhitespaceTokenizer>,
    engine: StorageEngine,
}

impl DaemonState {
    fn new(db_path: &Path) -> Result<Self> {
        // Load config
        let config = DbConfig::load(db_path)?;

        // Check for HNSW and BM25 indexes
        let hnsw_path = db_path.join("hnsw.bin");
        let bm25_path = db_path.join("bm25.bin");

        if !hnsw_path.exists() {
            anyhow::bail!("HNSW index not found. Run 'v-hnsw add' first.");
        }
        if !bm25_path.exists() {
            anyhow::bail!("BM25 index not found. Run 'v-hnsw add' first.");
        }

        // Get model name from config
        let model_name = config.embed_model.ok_or_else(|| {
            anyhow::anyhow!("No embedding model specified in database config.")
        })?;

        // Initialize embedding model
        let model_type = parse_model_type(&model_name)?;
        let device = Device::auto();

        eprintln!("[daemon] Loading model: {} (device={})", model_type.model_name(), device.name());
        let t0 = Instant::now();
        let embed_model = FastEmbedModel::with_device(model_type, device)
            .context("Failed to initialize embedding model")?;
        eprintln!("[daemon] Model loaded: {:.0}ms", t0.elapsed().as_millis());

        // Check dimension
        if embed_model.dim() != config.dim {
            anyhow::bail!(
                "Model dimension ({}) doesn't match database dimension ({})",
                embed_model.dim(),
                config.dim
            );
        }

        // Warmup: run a dummy embed
        eprintln!("[daemon] Warming up model...");
        let _ = embed_model.embed(&["warmup"]);
        eprintln!("[daemon] Warmup complete");

        // Load HNSW index
        eprintln!("[daemon] Loading HNSW index...");
        let hnsw: HnswGraph<CosineDistance> = HnswGraph::load(&hnsw_path, CosineDistance)
            .context("Failed to load HNSW index")?;
        eprintln!("[daemon] HNSW loaded: {} vectors", hnsw.len());

        // Load BM25 index
        eprintln!("[daemon] Loading BM25 index...");
        let bm25: Bm25Index<WhitespaceTokenizer> = Bm25Index::load(&bm25_path)
            .context("Failed to load BM25 index")?;
        eprintln!("[daemon] BM25 loaded");

        // Create hybrid searcher
        let hybrid_config = HybridSearchConfig::builder()
            .ef_search(200)
            .dense_limit(20)
            .sparse_limit(20)
            .dense_weight(0.7)
            .sparse_weight(0.3)
            .build();

        let searcher = SimpleHybridSearcher::new(hnsw, bm25, hybrid_config);

        // Open storage for text retrieval
        let engine = StorageEngine::open(db_path)
            .context("Failed to open storage")?;

        Ok(Self {
            embed_model,
            searcher,
            engine,
        })
    }

    fn search(&self, query: &str, k: usize) -> Result<SearchResponse> {
        let start = Instant::now();

        // Embed the query
        let query_embedding = self.embed_model
            .embed(&[query])
            .context("Failed to embed query")?
            .into_iter()
            .next()
            .context("No embedding returned")?;

        // Perform hybrid search
        let results = self.searcher.search(&query_embedding, query, k)
            .context("Hybrid search failed")?;

        // Get payload store
        let payload_store = self.engine.payload_store();

        // Build results with text
        let results_with_text: Vec<SearchResult> = results
            .into_iter()
            .map(|(id, score)| {
                let text = payload_store.get_text(id).ok().flatten();
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

                SearchResult {
                    id,
                    score,
                    text,
                    source,
                    title,
                }
            })
            .collect();

        let elapsed = start.elapsed();

        Ok(SearchResponse {
            results: results_with_text,
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        })
    }
}

/// Handle a single client connection.
fn handle_client(
    stream: TcpStream,
    state: &DaemonState,
    last_activity: &mut Instant,
) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;

    let mut reader = BufReader::new(&stream);
    let mut writer = &stream;

    let mut line = String::new();
    reader.read_line(&mut line)?;

    *last_activity = Instant::now();

    // Parse JSON-RPC request
    let request: JsonRpcRequest = serde_json::from_str(&line)
        .with_context(|| format!("Failed to parse request: {}", line.trim()))?;

    let response = match request.method.as_str() {
        "search" => {
            let params: SearchParams = serde_json::from_value(request.params)
                .context("Invalid search params")?;

            match state.search(&params.query, params.k) {
                Ok(result) => JsonRpcResponse {
                    id: request.id,
                    result: Some(serde_json::to_value(result)?),
                    error: None,
                },
                Err(e) => JsonRpcResponse {
                    id: request.id,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -1,
                        message: e.to_string(),
                    }),
                },
            }
        }
        "ping" => JsonRpcResponse {
            id: request.id,
            result: Some(serde_json::json!({"status": "ok"})),
            error: None,
        },
        "shutdown" => {
            let response = JsonRpcResponse {
                id: request.id,
                result: Some(serde_json::json!({"status": "shutting_down"})),
                error: None,
            };
            let response_json = serde_json::to_string(&response)?;
            writeln!(writer, "{}", response_json)?;
            writer.flush()?;
            anyhow::bail!("Shutdown requested");
        }
        _ => JsonRpcResponse {
            id: request.id,
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: format!("Unknown method: {}", request.method),
            }),
        },
    };

    let response_json = serde_json::to_string(&response)?;
    writeln!(writer, "{}", response_json)?;
    writer.flush()?;

    Ok(())
}

/// Run the daemon server.
pub fn run(db_path: PathBuf, port: u16, timeout_secs: u64, background: bool) -> Result<()> {
    // Canonicalize path for consistent hashing
    let db_path = db_path.canonicalize()
        .with_context(|| format!("Database not found: {}", db_path.display()))?;

    // Check if daemon already running
    if is_daemon_running(&db_path) {
        if let Some(existing_port) = read_port_file(&db_path) {
            eprintln!("[daemon] Already running on port {}", existing_port);
            return Ok(());
        }
    }

    // Daemonize if requested (fork to background)
    if background {
        #[cfg(windows)]
        {
            // On Windows, use CREATE_NEW_PROCESS_GROUP
            use std::os::windows::process::CommandExt;
            const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
            const DETACHED_PROCESS: u32 = 0x00000008;

            let exe = std::env::current_exe()?;
            std::process::Command::new(exe)
                .args(["serve", db_path.to_str().unwrap(), "--port", &port.to_string(), "--timeout", &timeout_secs.to_string()])
                .creation_flags(CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .context("Failed to spawn background daemon")?;

            // Wait a bit for daemon to start
            std::thread::sleep(Duration::from_millis(500));
            return Ok(());
        }

        #[cfg(not(windows))]
        {
            use std::process::Command;

            let exe = std::env::current_exe()?;
            Command::new(exe)
                .args(["serve", db_path.to_str().unwrap(), "--port", &port.to_string(), "--timeout", &timeout_secs.to_string()])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .context("Failed to spawn background daemon")?;

            std::thread::sleep(Duration::from_millis(500));
            return Ok(());
        }
    }

    // Load state (model, indexes)
    let state = DaemonState::new(&db_path)?;

    // Bind to port (try specified port, then find available)
    let listener = TcpListener::bind(format!("127.0.0.1:{}", port))
        .or_else(|_| TcpListener::bind("127.0.0.1:0"))
        .context("Failed to bind to any port")?;

    let actual_port = listener.local_addr()?.port();

    // Write PID and port files
    write_pid_file(&db_path)?;
    write_port_file(&db_path, actual_port)?;

    // Set non-blocking for timeout checks
    listener.set_nonblocking(true)?;

    eprintln!("[daemon] Listening on 127.0.0.1:{}", actual_port);
    eprintln!("[daemon] Idle timeout: {}s", timeout_secs);
    eprintln!("[daemon] PID file: {}", get_pid_file(&db_path).display());
    eprintln!("[daemon] Ready for connections");

    let mut last_activity = Instant::now();
    let timeout = Duration::from_secs(timeout_secs);

    // Main loop (uses global INTERRUPTED flag from main.rs for Ctrl+C)
    loop {
        // Check for Ctrl+C via global flag
        if is_interrupted() {
            eprintln!("\n[daemon] Received shutdown signal");
            break;
        }

        // Check idle timeout
        if last_activity.elapsed() > timeout {
            eprintln!("[daemon] Idle timeout reached, shutting down");
            break;
        }

        // Accept connections with timeout
        match listener.accept() {
            Ok((stream, addr)) => {
                eprintln!("[daemon] Connection from {}", addr);
                if let Err(e) = handle_client(stream, &state, &mut last_activity) {
                    if e.to_string().contains("Shutdown requested") {
                        break;
                    }
                    eprintln!("[daemon] Client error: {}", e);
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No connection, sleep briefly
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                eprintln!("[daemon] Accept error: {}", e);
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }

    // Cleanup
    eprintln!("[daemon] Cleaning up...");
    cleanup_files(&db_path);
    eprintln!("[daemon] Shutdown complete");

    Ok(())
}
