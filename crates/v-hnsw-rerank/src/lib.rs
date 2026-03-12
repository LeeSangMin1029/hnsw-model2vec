//! Cross-encoder reranker using candle (pure Rust, no ONNX).
//!
//! Uses `cross-encoder/ms-marco-TinyBERT-L-2-v2` for fast CPU reranking.

mod model;

pub use model::{CrossEncoderReranker, RerankResult};

use anyhow::{Context, Result};
use v_hnsw_core::PayloadStore;

/// Rerank search results using cross-encoder, replacing scores with reranker scores.
///
/// Takes raw `(id, score)` pairs, fetches text from `payload_store`,
/// runs cross-encoder reranking, and replaces `results` with top-k reranked entries.
pub fn rerank_results(
    results: &mut Vec<(u64, f32)>,
    query: &str,
    payload_store: &dyn PayloadStore,
    k: usize,
) -> Result<()> {
    let mut docs: Vec<(usize, String)> = Vec::new();
    for (i, &(id, _)) in results.iter().enumerate() {
        if let Ok(Some(text)) = payload_store.get_text(id) {
            docs.push((i, text));
        }
    }

    if docs.is_empty() {
        return Ok(());
    }

    let doc_texts: Vec<&str> = docs.iter().map(|(_, t)| t.as_str()).collect();

    let reranker = CrossEncoderReranker::new()
        .context("Failed to load cross-encoder reranker")?;

    let reranked = reranker
        .rerank(query, &doc_texts)
        .map_err(|e| anyhow::anyhow!("Reranking failed: {e:#}"))?;

    let mut scored: Vec<(u64, f32)> = reranked
        .iter()
        .map(|r| {
            let (orig_idx, _) = docs[r.index];
            let (id, _) = results[orig_idx];
            (id, r.score)
        })
        .collect();

    scored.truncate(k);
    *results = scored;
    Ok(())
}
