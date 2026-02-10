//! Find command - Unified search: hybrid HNSW+BM25 and raw vector modes.
//!
//! Modes:
//!   find db "query"              — hybrid (auto-embed + BM25, daemon preferred)
//!   find db --vector "0.1,..."   — raw vector dense-only
//!   find db --vector "0.1,..." "query" — raw vector + BM25 hybrid

mod direct;

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::commands::common::SearchResultItem;
use super::serve;

/// Parameters for the find command.
pub struct FindParams {
    pub db: PathBuf,
    pub query: Option<String>,
    pub k: usize,
    pub tags: Vec<String>,
    pub full: bool,
    pub vector: Option<String>,
    pub ef: usize,
}

/// Search output for JSON formatting.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct FindOutput {
    results: Vec<SearchResultItem>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    query: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    model: String,
    #[serde(default, skip_serializing_if = "is_zero")]
    total_docs: usize,
    elapsed_ms: f64,
}

pub(super) fn is_zero(v: &usize) -> bool {
    *v == 0
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

/// Truncate text to max_len chars, appending "..." if truncated.
fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }
    let mut end = max_len;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &text[..end])
}

/// Compact the output: truncate text, strip home prefix from source.
pub(super) fn compact_output(mut output: FindOutput) -> FindOutput {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default()
        .replace('\\', "/");

    for item in &mut output.results {
        if let Some(ref text) = item.text {
            let cleaned = text.replace('\n', " ");
            item.text = Some(truncate_text(&cleaned, 150));
        }
        if let Some(ref source) = item.source {
            let short = source
                .strip_prefix(&home)
                .unwrap_or(source)
                .trim_start_matches('/');
            item.source = Some(short.to_string());
        }
    }
    output
}

/// Print FindOutput as JSON with optional compaction.
fn print_output(output: FindOutput, full: bool) -> Result<()> {
    let output = if full { output } else { compact_output(output) };
    let json = serde_json::to_string_pretty(&output)?;
    println!("{json}");
    Ok(())
}

/// Run the find command (unified entry point).
pub fn run(params: FindParams) -> Result<()> {
    let FindParams { db, query, k, tags, full, vector, ef } = params;

    if !db.exists() {
        anyhow::bail!("Database not found at {}", db.display());
    }

    // Validate: at least one of query or vector is required
    if query.is_none() && vector.is_none() {
        anyhow::bail!("At least one of <query> or --vector must be provided");
    }

    // Mode 1: Raw vector (--vector provided)
    if let Some(ref vec_str) = vector {
        let raw_vec = parse_vector(vec_str)?;
        return direct::run_raw_vector(db, raw_vec, query, k, tags, full, ef);
    }

    // Mode 3: Hybrid (default) — daemon preferred
    let query = query.unwrap(); // validated above
    run_hybrid(db, query, k, tags, full)
}

/// Parse a comma-separated vector string.
fn parse_vector(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<f32>()
                .with_context(|| format!("invalid number: '{part}'"))
        })
        .collect()
}

/// Hybrid search: try daemon first, then direct fallback.
fn run_hybrid(db_path: PathBuf, query: String, k: usize, tags: Vec<String>, full: bool) -> Result<()> {
    // Try daemon mode first
    match try_daemon_search(&db_path, &query, k, &tags) {
        Ok(output) => return print_output(output, full),
        Err(_) => eprintln!("Daemon not running, starting..."),
    }

    // Try to spawn daemon
    if direct::spawn_daemon(&db_path).is_ok() {
        match try_daemon_search(&db_path, &query, k, &tags) {
            Ok(output) => return print_output(output, full),
            Err(e) => {
                eprintln!("Daemon search failed: {}", e);
                eprintln!("Falling back to direct search...");
            }
        }
    } else {
        eprintln!("Failed to start daemon, using direct search...");
    }

    // Fallback: direct search
    direct::run_direct(db_path, query, k, tags, full)
}
