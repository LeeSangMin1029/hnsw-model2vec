//! SimpleMaxScore: lightweight top-k pruning for BM25 search.
//!
//! Classifies query terms into essential (must score all docs) and
//! non-essential (skippable when upper bound < threshold). No block
//! structures required — works directly on sorted posting lists.
//!
//! Based on Turtle & Flood (1995), simplified for small-to-medium
//! collections (50K–500K docs). Posting lists must be sorted by doc_id.

use std::collections::HashMap;

use v_hnsw_core::PointId;

use super::fieldnorm::FieldNormLut;
use super::index::PostingList;
use super::scorer::Bm25Params;

/// A resolved query term with pre-computed scoring bounds.
struct TermCursor<'a> {
    postings: &'a [super::index::Posting],
    idf: f32,
    /// Upper bound: idf * tf_norm(max_tf, min_doc_len) for this term.
    max_contribution: f32,
    /// Current position in the posting list.
    pos: usize,
}

impl<'a> TermCursor<'a> {
    fn is_exhausted(&self) -> bool {
        self.pos >= self.postings.len()
    }

    fn current_doc_id(&self) -> PointId {
        self.postings[self.pos].doc_id
    }

    /// Advance past the given doc_id. Posting list must be doc_id-sorted.
    fn advance_to(&mut self, target: PointId) {
        while self.pos < self.postings.len() && self.postings[self.pos].doc_id < target {
            self.pos += 1;
        }
    }
}

/// Run SimpleMaxScore top-k search.
///
/// Requires posting lists sorted by doc_id (ensured by `save_fst()`).
/// All allocations are function-local — safe for concurrent use.
///
/// # Arguments
/// - `terms`: (PostingList, IDF weight) pairs from `resolve_terms()`
/// - `k`: number of results
/// - `params`: BM25 scoring parameters
/// - `doc_lengths`: doc_id → token count map
/// - `avg_doc_len`: average document length
pub fn maxscore_search(
    terms: &[(&PostingList, f32)],
    k: usize,
    params: &Bm25Params,
    doc_lengths: &HashMap<PointId, u32>,
    avg_doc_len: f32,
    fieldnorm_lut: Option<&FieldNormLut>,
    fieldnorm_codes: &HashMap<PointId, u8>,
) -> Vec<(PointId, f32)> {
    if terms.is_empty() || k == 0 {
        return Vec::new();
    }

    // Build cursors with max_contribution bounds
    let mut cursors: Vec<TermCursor<'_>> = terms
        .iter()
        .filter(|(pl, _)| !pl.postings.is_empty())
        .map(|(pl, idf)| {
            let max_contribution = compute_max_contribution(pl, *idf, params, doc_lengths, avg_doc_len);
            TermCursor {
                postings: &pl.postings,
                idf: *idf,
                max_contribution,
                pos: 0,
            }
        })
        .collect();

    if cursors.is_empty() {
        return Vec::new();
    }

    // Sort by max_contribution ascending (low-impact terms first = non-essential)
    cursors.sort_by(|a, b| a.max_contribution.total_cmp(&b.max_contribution));

    // Prefix sums: non_essential_bound[i] = sum of max_contribution[0..i]
    let prefix_sums = build_prefix_sums(&cursors);

    // Top-k tracking: (score, doc_id) min-heap via sorted Vec
    let mut top_k: Vec<(f32, PointId)> = Vec::with_capacity(k + 1);
    let mut threshold: f32 = 0.0;

    // DAAT merge: iterate doc_ids in ascending order across all cursors
    loop {
        // Find minimum doc_id among non-exhausted cursors
        let min_doc = cursors
            .iter()
            .filter(|c| !c.is_exhausted())
            .map(|c| c.current_doc_id())
            .min();

        let Some(doc_id) = min_doc else {
            break;
        };

        // Find the essential boundary: smallest i where prefix_sums[i] >= threshold
        let essential_idx = find_essential_boundary(&prefix_sums, threshold);

        // Score essential terms (high max_contribution — always evaluated)
        let mut score = 0.0f32;

        // Inline scoring helper using LUT when available
        let score_term = |idf: f32, tf: u32| -> f32 {
            if let Some(lut) = fieldnorm_lut {
                let code = fieldnorm_codes.get(&doc_id).copied().unwrap_or(0);
                idf * lut.tf_norm(params.k1, tf, code)
            } else {
                let doc_len = doc_lengths.get(&doc_id).copied().unwrap_or(0);
                idf * params.tf_norm(tf, doc_len, avg_doc_len)
            }
        };

        for cursor in cursors[essential_idx..].iter_mut() {
            if !cursor.is_exhausted() && cursor.current_doc_id() == doc_id {
                let tf = cursor.postings[cursor.pos].tf;
                score += score_term(cursor.idf, tf);
                cursor.pos += 1;
            }
        }

        // Check if non-essential terms could push score above threshold
        let non_essential_upper = if essential_idx > 0 {
            prefix_sums[essential_idx - 1]
        } else {
            0.0
        };

        if score + non_essential_upper > threshold {
            // Score non-essential terms too
            for cursor in cursors[..essential_idx].iter_mut() {
                if !cursor.is_exhausted() && cursor.current_doc_id() == doc_id {
                    let tf = cursor.postings[cursor.pos].tf;
                    score += score_term(cursor.idf, tf);
                    cursor.pos += 1;
                }
            }

            // Update top-k
            if top_k.len() < k || score > threshold {
                top_k.push((score, doc_id));
                if top_k.len() > k {
                    // Remove minimum score entry
                    top_k.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                    top_k.truncate(k);
                    threshold = top_k.last().map(|&(s, _)| s).unwrap_or(0.0);
                }
            }
        } else {
            // Skip non-essential terms — advance past this doc_id
            for cursor in cursors[..essential_idx].iter_mut() {
                if !cursor.is_exhausted() && cursor.current_doc_id() == doc_id {
                    cursor.pos += 1;
                }
            }
        }

        // Advance all remaining cursors past this doc_id
        let next = doc_id + 1;
        for cursor in &mut cursors {
            cursor.advance_to(next);
        }
    }

    // Final sort by score descending
    top_k.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    top_k.into_iter().map(|(score, id)| (id, score)).collect()
}

/// Compute the maximum possible BM25 contribution for a term.
///
/// Uses max TF and min doc length from the posting list for a tight upper bound.
fn compute_max_contribution(
    pl: &PostingList,
    idf: f32,
    params: &Bm25Params,
    doc_lengths: &HashMap<PointId, u32>,
    avg_doc_len: f32,
) -> f32 {
    let mut max_tf = 0u32;
    let mut min_doc_len = u32::MAX;

    for p in &pl.postings {
        if p.tf > max_tf {
            max_tf = p.tf;
        }
        let dl = doc_lengths.get(&p.doc_id).copied().unwrap_or(1);
        if dl < min_doc_len {
            min_doc_len = dl;
        }
    }

    if min_doc_len == u32::MAX {
        min_doc_len = 1;
    }

    idf * params.tf_norm(max_tf, min_doc_len, avg_doc_len)
}

/// Build prefix sums of max_contribution (ascending order).
fn build_prefix_sums(cursors: &[TermCursor<'_>]) -> Vec<f32> {
    let mut sums = Vec::with_capacity(cursors.len());
    let mut running = 0.0f32;
    for c in cursors {
        running += c.max_contribution;
        sums.push(running);
    }
    sums
}

/// Find the essential boundary: first index where prefix_sums[i-1] < threshold.
///
/// Terms at `[boundary..]` are essential (must be scored for every doc).
/// Terms at `[..boundary]` are non-essential (skippable).
fn find_essential_boundary(prefix_sums: &[f32], threshold: f32) -> usize {
    // All terms with cumulative sum < threshold are non-essential
    prefix_sums
        .iter()
        .position(|&s| s >= threshold)
        .unwrap_or(prefix_sums.len())
}