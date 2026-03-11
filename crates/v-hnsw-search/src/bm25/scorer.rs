//! BM25 scoring implementation.
//!
//! Implements the Okapi BM25 ranking function for term-document scoring.
//! Also provides shared scoring infrastructure (`ScoringCtx`, `PostingView`,
//! `accumulate_and_rank`) used by both `Bm25Index` and `Bm25Snapshot`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use v_hnsw_core::PointId;

use super::fieldnorm::FieldNormLut;

/// BM25 scoring parameters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct Bm25Params {
    /// Term frequency saturation parameter (default: 1.2).
    pub k1: f32,
    /// Document length normalization parameter (default: 0.75).
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

impl Bm25Params {
    /// Create new BM25 parameters with custom values.
    pub fn new(k1: f32, b: f32) -> Self {
        Self { k1, b }
    }

    /// Compute the IDF (inverse document frequency) component.
    ///
    /// Call once per query term, then reuse for all documents.
    #[inline]
    pub fn idf(&self, df: u32, total_docs: usize) -> f32 {
        if df == 0 || total_docs == 0 {
            return 0.0;
        }
        let n = total_docs as f32;
        let df_f = df as f32;
        ((n - df_f + 0.5) / (df_f + 0.5) + 1.0).ln()
    }

    /// Compute the TF normalization component (without IDF).
    ///
    /// Multiply by `idf()` to get the full BM25 score.
    #[inline]
    pub fn tf_norm(&self, tf: u32, doc_len: u32, avg_doc_len: f32) -> f32 {
        let tf_f = tf as f32;
        let doc_len_f = doc_len as f32;
        let length_norm = 1.0 - self.b + self.b * (doc_len_f / avg_doc_len);
        (tf_f * (self.k1 + 1.0)) / (tf_f + self.k1 * length_norm)
    }

    /// Compute the BM25 score for a single term in a document.
    ///
    /// # Parameters
    /// - `tf`: Term frequency in the document
    /// - `df`: Document frequency (number of documents containing the term)
    /// - `doc_len`: Length of the document (number of tokens)
    /// - `avg_doc_len`: Average document length in the corpus
    /// - `total_docs`: Total number of documents in the corpus
    ///
    /// # Returns
    /// The BM25 score contribution for this term.
    #[inline]
    pub fn score(
        &self,
        tf: u32,
        df: u32,
        doc_len: u32,
        avg_doc_len: f32,
        total_docs: usize,
    ) -> f32 {
        self.idf(df, total_docs) * self.tf_norm(tf, doc_len, avg_doc_len)
    }
}

// -- Shared scoring infrastructure --

/// Trait for types that represent a posting entry (doc_id + tf).
pub(crate) trait PostingView {
    fn doc_id(&self) -> PointId;
    fn tf(&self) -> u32;
}

/// Pre-computed scoring context for BM25 search.
///
/// Holds all state needed for scoring a posting, including optional
/// `FieldNormLut` for O(1) length normalization.
pub(crate) struct ScoringCtx<'a> {
    pub params: &'a Bm25Params,
    pub avg_doc_len: f32,
    pub fieldnorm_lut: Option<&'a FieldNormLut>,
    pub fieldnorm_codes: &'a HashMap<PointId, u8>,
    /// Fallback when `FieldNormLut` is not available.
    pub doc_lengths: Option<&'a HashMap<PointId, u32>>,
}

impl ScoringCtx<'_> {
    /// Score a single posting entry using FieldNorm LUT (fast) or fallback.
    #[inline]
    pub fn score(&self, doc_id: PointId, tf: u32, idf: f32) -> f32 {
        if let Some(lut) = self.fieldnorm_lut {
            let code = self.fieldnorm_codes.get(&doc_id).copied().unwrap_or(0);
            idf * lut.tf_norm(self.params.k1, tf, code)
        } else {
            let doc_len = self
                .doc_lengths
                .and_then(|dl| dl.get(&doc_id).copied())
                .unwrap_or(0);
            idf * self.params.tf_norm(tf, doc_len, self.avg_doc_len)
        }
    }
}

/// Vec accumulator ceiling: 256K entries = 1MB max allocation.
const MAX_VEC_ACCUMULATOR_ID: u64 = 256_000;

/// Accumulate BM25 scores and return top-k results.
///
/// Generic over posting type via `PostingView`. Chooses Vec (cache-friendly)
/// or HashMap (sparse IDs) accumulator based on `max_doc_id`.
pub(crate) fn accumulate_and_rank<P: PostingView>(
    ctx: &ScoringCtx<'_>,
    terms: &[(&[P], f32)],
    max_doc_id: u64,
    limit: usize,
) -> Vec<(PointId, f32)> {
    if max_doc_id <= MAX_VEC_ACCUMULATOR_ID {
        let len = max_doc_id as usize + 1;
        let mut scores = vec![0.0f32; len];
        let mut touched: Vec<PointId> = Vec::with_capacity(256);

        for &(postings, idf) in terms {
            for p in postings {
                let id = p.doc_id() as usize;
                if id < len {
                    if scores[id] == 0.0 {
                        touched.push(p.doc_id());
                    }
                    scores[id] += ctx.score(p.doc_id(), p.tf(), idf);
                }
            }
        }

        let mut results: Vec<(PointId, f32)> = touched
            .into_iter()
            .map(|id| (id, scores[id as usize]))
            .collect();
        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(limit);
        results
    } else {
        let mut score_map: HashMap<PointId, f32> = HashMap::new();
        for &(postings, idf) in terms {
            for p in postings {
                *score_map.entry(p.doc_id()).or_insert(0.0) +=
                    ctx.score(p.doc_id(), p.tf(), idf);
            }
        }
        let mut results: Vec<_> = score_map.into_iter().collect();
        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(limit);
        results
    }
}

/// Score only the specified documents (Dense-Guided BM25).
///
/// Requires postings to be sorted by doc_id for binary search.
pub(crate) fn score_documents_common<P: PostingView>(
    ctx: &ScoringCtx<'_>,
    terms: &[(&[P], f32)],
    doc_ids: &[PointId],
) -> Vec<(PointId, f32)> {
    let mut results = Vec::with_capacity(doc_ids.len());
    for &doc_id in doc_ids {
        let mut score = 0.0f32;
        for &(postings, idf) in terms {
            if let Ok(idx) = postings.binary_search_by_key(&doc_id, |p| p.doc_id()) {
                score += ctx.score(doc_id, postings[idx].tf(), idf);
            }
        }
        if score > 0.0 {
            results.push((doc_id, score));
        }
    }
    results
}