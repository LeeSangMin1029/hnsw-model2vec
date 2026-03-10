//! Find command - Unified search: hybrid HNSW+BM25 and raw vector modes.
//!
//! Modes:
//!   find db "query"              — hybrid (auto-embed + BM25, daemon preferred)
//!   find db --vector "0.1,..."   — raw vector dense-only
//!   find db --vector "0.1,..." "query" — raw vector + BM25 hybrid

mod direct;

use std::path::{Path, PathBuf};

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
    #[serde(default, skip_serializing_if = "is_elapsed_zero")]
    elapsed_ms: f64,
}

pub(super) fn is_zero(v: &usize) -> bool {
    *v == 0
}

fn is_elapsed_zero(v: &f64) -> bool {
    *v == 0.0
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

    let mut params = serde_json::json!({
        "db": canonical.to_str().unwrap_or(""),
        "query": query,
        "k": k
    });
    if !tags.is_empty() {
        params["tags"] = serde_json::json!(tags);
    }

    let result = serve::daemon_rpc("search", params, 30)?;
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
        // Strip id/score in compact mode — reduces noise for AI consumers
        item.id = 0;
        item.score = 0.0;
        item.url = None;
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
    // Strip metadata noise in compact mode
    output.model = String::new();
    output.total_docs = 0;
    output.elapsed_ms = 0.0;
    output
}

/// Print FindOutput as JSON with optional compaction and score filtering.
pub(super) fn print_output(output: FindOutput, full: bool, min_score: f32) -> Result<()> {
    // Apply score filter BEFORE compaction (compact zeroes out scores)
    let mut output = output;
    if min_score > 0.0 {
        output.results.retain(|item| item.score >= min_score);
    }
    let output = if full { output } else { compact_output(output) };
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

    super::common::require_db(&db)?;

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
    super::code_intel::try_inline_lookup(db, query)
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

#[cfg(test)]
mod tests;

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
