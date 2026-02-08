//! Find command - Unified search with hybrid HNSW + BM25 and RRF fusion.
//!
//! Uses daemon mode by default for fast searches. Falls back to direct mode
//! if daemon fails.

mod direct;

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::commands::common::SearchResultItem;
use super::serve;

/// Search output for JSON formatting.
#[derive(Debug, Serialize, Deserialize)]
struct FindOutput {
    results: Vec<SearchResultItem>,
    #[serde(default)]
    query: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    total_docs: usize,
    elapsed_ms: f64,
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

/// Try to connect to the daemon for a given database.
fn try_daemon_search(
    db_path: &Path,
    query: &str,
    k: usize,
    tags: &[String],
) -> Result<FindOutput> {
    let canonical_path = db_path
        .canonicalize()
        .with_context(|| format!("Database not found: {}", db_path.display()))?;

    let port = serve::read_port_file(&canonical_path)
        .ok_or_else(|| anyhow::anyhow!("Daemon not running (no port file)"))?;

    let mut stream = TcpStream::connect_timeout(
        &format!("127.0.0.1:{}", port).parse()?,
        Duration::from_secs(2),
    )
    .context("Failed to connect to daemon")?;

    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;

    let mut params = serde_json::json!({
        "query": query,
        "k": k
    });

    if !tags.is_empty() {
        params["tags"] = serde_json::json!(tags);
    }

    let request = serde_json::json!({
        "id": 1,
        "method": "search",
        "params": params
    });

    writeln!(stream, "{}", serde_json::to_string(&request)?)?;
    stream.flush()?;

    let mut reader = BufReader::new(&stream);
    let mut response_line = String::new();
    reader.read_line(&mut response_line)?;

    let response: JsonRpcResponse =
        serde_json::from_str(&response_line).context("Failed to parse daemon response")?;

    if let Some(err) = response.error {
        anyhow::bail!("Daemon error: {}", err.message);
    }

    let result = response
        .result
        .ok_or_else(|| anyhow::anyhow!("Empty response from daemon"))?;

    let search_response: FindOutput =
        serde_json::from_value(result).context("Failed to parse search results")?;

    Ok(search_response)
}

/// Run the find command.
pub fn run(db_path: PathBuf, query: String, k: usize, tags: Vec<String>) -> Result<()> {
    if !db_path.exists() {
        anyhow::bail!("Database not found at {}", db_path.display());
    }

    // Try daemon mode first
    match try_daemon_search(&db_path, &query, k, &tags) {
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
        Err(_) => {
            eprintln!("Daemon not running, starting...");
        }
    }

    // Try to spawn daemon
    if direct::spawn_daemon(&db_path).is_ok() {
        match try_daemon_search(&db_path, &query, k, &tags) {
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

    // Fallback: direct search
    direct::run_direct(db_path, query, k, tags)
}
