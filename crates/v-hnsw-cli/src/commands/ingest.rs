//! Embedding + ingestion pipeline utilities.
//!
//! Shared types and functions for the `add` and `update` commands:
//! record construction, payload building, batched embedding with
//! length-sorted padding optimization, and batch insert.

use std::collections::HashMap;

use anyhow::{Context, Result};
use v_hnsw_core::{Payload, PayloadValue};
use v_hnsw_embed::{EmbeddingModel, Model2VecModel};
use v_hnsw_storage::StorageEngine;

/// A record for batch ingestion (shared by add and update commands).
#[derive(Clone)]
pub struct IngestRecord {
    pub id: u64,
    pub text: String,
    pub source: String,
    pub title: Option<String>,
    pub tags: Vec<String>,
    pub chunk_index: usize,
    pub chunk_total: usize,
    pub source_modified_at: u64,
    /// Extra custom fields to merge into payload (e.g., ast_hash).
    pub custom: HashMap<String, PayloadValue>,
}

/// Build payload from source info.
pub fn make_payload(
    source: &str,
    title: Option<&str>,
    tags: &[String],
    chunk_index: usize,
    chunk_total: usize,
    source_modified_at: u64,
    extra_custom: &HashMap<String, PayloadValue>,
) -> Payload {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut custom = extra_custom.clone();
    if let Some(t) = title {
        custom.insert("title".to_string(), PayloadValue::String(t.to_string()));
    }

    Payload {
        source: source.to_string(),
        tags: tags.to_vec(),
        created_at: now,
        source_modified_at,
        chunk_index: chunk_index as u32,
        chunk_total: chunk_total as u32,
        custom,
    }
}

/// Max character length for text sent to embedding model.
const EMBED_MAX_CHARS: usize = 8000;

/// Truncate text at a char boundary for embedding.
///
/// Returns a new string if truncation occurs, otherwise the original slice.
pub fn truncate_for_embed(text: &str) -> &str {
    if text.len() <= EMBED_MAX_CHARS {
        return text;
    }
    let mut end = EMBED_MAX_CHARS;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Embed texts with length-sorted batching to minimize padding waste.
pub fn embed_sorted(model: &dyn EmbeddingModel, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    // Sort indices by text length
    let mut indices: Vec<usize> = (0..texts.len()).collect();
    indices.sort_by_key(|&i| texts[i].len());

    let sorted: Vec<&str> = indices.iter().map(|&i| texts[i].as_str()).collect();
    let sorted_embs = model
        .embed(&sorted)
        .map_err(|e| anyhow::anyhow!("Embedding failed: {e}"))?;

    // Restore original order (consume sorted_embs to avoid clone)
    let mut embeddings = vec![Vec::new(); texts.len()];
    for (emb, &orig_idx) in sorted_embs.into_iter().zip(indices.iter()) {
        embeddings[orig_idx] = emb;
    }
    Ok(embeddings)
}

/// Embed a batch of records and insert into storage (no progress bar).
///
/// Used by `update::run_core` for per-file incremental ingestion.
/// For bulk ingestion with pipeline parallelism, use `pipeline::process_records`.
pub fn embed_and_insert(
    records: &[IngestRecord],
    model: &Model2VecModel,
    engine: &mut StorageEngine,
) -> Result<()> {
    let texts: Vec<String> = records
        .iter()
        .map(|r| truncate_for_embed(&r.text).to_string())
        .collect();

    let embeddings = embed_sorted(model, &texts).context("Embedding failed")?;

    let items: Vec<(u64, &[f32], _, &str)> = records
        .iter()
        .zip(embeddings.iter())
        .map(|(rec, emb)| {
            let payload = make_payload(
                &rec.source,
                rec.title.as_deref(),
                &rec.tags,
                rec.chunk_index,
                rec.chunk_total,
                rec.source_modified_at,
                &rec.custom,
            );
            (rec.id, emb.as_slice(), payload, rec.text.as_str())
        })
        .collect();

    engine
        .insert_batch(&items)
        .context("Failed to insert batch")?;

    Ok(())
}
