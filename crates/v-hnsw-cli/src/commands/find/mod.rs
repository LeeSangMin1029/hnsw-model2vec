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
    pub min_score: f32,
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

/// Try to connect to the global daemon and search a specific database.
fn try_daemon_search(
    db_path: &Path,
    query: &str,
    k: usize,
    tags: &[String],
) -> Result<FindOutput> {
    let canonical = db_path
        .canonicalize()
        .with_context(|| format!("Database not found: {}", db_path.display()))?;

    let port = serve::read_port_file()
        .ok_or_else(|| anyhow::anyhow!("Daemon not running (no port file)"))?;

    let addr: std::net::SocketAddr = format!("127.0.0.1:{}", port).parse()?;
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2))
        .context("Failed to connect to daemon")?;

    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;

    let mut params = serde_json::json!({
        "db": canonical.to_str().unwrap_or(""),
        "query": query,
        "k": k
    });
    if !tags.is_empty() {
        params["tags"] = serde_json::json!(tags);
    }

    let request = serde_json::json!({"id": 1, "method": "search", "params": params});
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

    serde_json::from_value(result).context("Failed to parse search results")
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

/// Print FindOutput as JSON with optional compaction and score filtering.
fn print_output(output: FindOutput, full: bool, min_score: f32) -> Result<()> {
    let mut output = if full { output } else { compact_output(output) };
    if min_score > 0.0 {
        output.results.retain(|item| item.score >= min_score);
    }
    let json = serde_json::to_string_pretty(&output)?;
    println!("{json}");
    Ok(())
}

/// Check if a query looks like a code symbol (no spaces, camelCase/snake_case/::).
fn looks_like_code_symbol(query: &str) -> bool {
    let trimmed = query.trim();
    // Must not contain spaces (natural language has spaces)
    if trimmed.contains(' ') {
        return false;
    }
    // Must have at least one identifier-like character pattern
    trimmed.contains("::")
        || trimmed.contains('_')
        || trimmed.chars().any(|c| c.is_uppercase())
        || trimmed.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Run the find command (unified entry point).
pub fn run(params: FindParams) -> Result<()> {
    let FindParams { db, query, k, tags, full, vector, ef, min_score } = params;

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
        return direct::run_raw_vector(db, raw_vec, query, k, tags, full, ef, min_score);
    }

    #[expect(clippy::unwrap_used, reason = "query presence validated by early return above")]
    let query = query.unwrap();

    // Code DB + code-like query → try code-intel first
    if looks_like_code_symbol(&query)
        && let Ok(config) = super::create::DbConfig::load(&db)
        && config.content_type == "code"
        && try_code_intel(&db, &query)?
    {
        return Ok(());
        // code-intel found nothing, fall through to vector search
    }

    // Hybrid vector search (default)
    run_hybrid(db, query, k, tags, full, min_score)
}

/// Try code-intel commands (def + refs) for a code symbol query.
///
/// Returns `true` if results were found and printed.
fn try_code_intel(db: &Path, query: &str) -> Result<bool> {
    let chunks = super::code_intel::load_chunks(db)?;
    let query_lower = query.to_lowercase();

    // Try def first
    let defs: Vec<&super::code_intel::parse::CodeChunk> = chunks.iter().filter(|c| {
        c.name.to_lowercase() == query_lower
            || c.name.to_lowercase().ends_with(&format!("::{query_lower}"))
    }).collect();

    if !defs.is_empty() {
        eprintln!("[code-intel] Found {} definition(s) for \"{}\"", defs.len(), query);
        println!("Definition of \"{query}\":\n");
        print_code_chunks(&defs);

        // Also show callers
        let callers: Vec<&super::code_intel::parse::CodeChunk> = chunks.iter().filter(|c| {
            c.calls.iter().any(|call| {
                let call_lower = call.to_lowercase();
                call_lower == query_lower
                    || call_lower.ends_with(&format!("::{query_lower}"))
                    || call_lower.contains(&query_lower)
            })
        }).collect();
        if !callers.is_empty() {
            println!("\n{} caller(s):\n", callers.len());
            print_code_chunks(&callers);
        }
        return Ok(true);
    }

    // Try symbol name search (partial match)
    let symbols: Vec<&super::code_intel::parse::CodeChunk> = chunks.iter().filter(|c| {
        c.name.to_lowercase().contains(&query_lower)
    }).collect();

    if !symbols.is_empty() {
        eprintln!("[code-intel] Found {} symbol(s) matching \"{}\"", symbols.len(), query);
        println!("Symbols matching \"{query}\":\n");
        print_code_chunks(&symbols);
        return Ok(true);
    }

    Ok(false)
}

/// Print code chunks grouped by directory (compact format).
fn print_code_chunks(chunks: &[&super::code_intel::parse::CodeChunk]) {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<&str, Vec<&super::code_intel::parse::CodeChunk>> = BTreeMap::new();
    for c in chunks {
        let dir = if let Some(idx) = c.file.rfind('/') { &c.file[..idx] } else { "." };
        groups.entry(dir).or_default().push(c);
    }
    for (dir, items) in &groups {
        println!("  {dir}/");
        for c in items {
            let filename = if let Some(idx) = c.file.rfind('/') { &c.file[idx + 1..] } else { &c.file };
            let lines = c.lines.map(|(s, e)| format!(":{s}-{e}")).unwrap_or_default();
            println!("    {filename}{lines}  [{kind}] {name}", kind = c.kind, name = c.name);
            if let Some(ref sig) = c.signature {
                println!("      {sig}");
            }
        }
        println!();
    }
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
fn run_hybrid(db_path: PathBuf, query: String, k: usize, tags: Vec<String>, full: bool, min_score: f32) -> Result<()> {
    // Try daemon mode first
    match try_daemon_search(&db_path, &query, k, &tags) {
        Ok(output) => return print_output(output, full, min_score),
        Err(_) => eprintln!("Daemon not running, starting..."),
    }

    // Try to spawn daemon
    if direct::spawn_daemon(&db_path).is_ok() {
        match try_daemon_search(&db_path, &query, k, &tags) {
            Ok(output) => return print_output(output, full, min_score),
            Err(e) => {
                eprintln!("Daemon search failed: {}", e);
                eprintln!("Falling back to direct search...");
            }
        }
    } else {
        eprintln!("Failed to start daemon, using direct search...");
    }

    // Fallback: direct search
    direct::run_direct(db_path, query, k, tags, full, min_score)
}
