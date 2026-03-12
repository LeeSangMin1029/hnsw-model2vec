//! Find command - Unified search: hybrid HNSW+BM25 and raw vector modes.
//!
//! Modes:
//!   find db "query"              — hybrid (auto-embed + BM25, daemon preferred)
//!   find db --vector "0.1,..."   — raw vector dense-only
//!   find db --vector "0.1,..." "query" — raw vector + BM25 hybrid

mod direct;

use std::path::PathBuf;

use anyhow::{Context, Result};

// Re-export shared types from search_result for backwards compat
pub use super::search_result::{FindOutput, compact_output, print_find_output, truncate_text};

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

    // Hybrid vector search (default)
    direct::run_direct(db, query, k, tags, full, min_score)
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
