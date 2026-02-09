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
        let doc_len = doc_lengths.get(&doc_id).copied().unwrap_or(0);

        for cursor in cursors[essential_idx..].iter_mut() {
            if !cursor.is_exhausted() && cursor.current_doc_id() == doc_id {
                let tf = cursor.postings[cursor.pos].tf;
                score += cursor.idf * params.tf_norm(tf, doc_len, avg_doc_len);
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
                    score += cursor.idf * params.tf_norm(tf, doc_len, avg_doc_len);
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::index::Posting;

    fn make_pl(entries: &[(PointId, u32)]) -> PostingList {
        let mut pl = PostingList::new();
        for &(doc_id, tf) in entries {
            pl.postings.push(Posting { doc_id, tf });
        }
        // Must be sorted by doc_id
        pl.postings.sort_unstable_by_key(|p| p.doc_id);
        pl
    }

    fn make_doc_lengths(docs: &[(PointId, u32)]) -> HashMap<PointId, u32> {
        docs.iter().copied().collect()
    }

    #[test]
    fn test_maxscore_basic() {
        let pl1 = make_pl(&[(1, 2), (2, 1), (3, 3)]);
        let pl2 = make_pl(&[(2, 1), (3, 1)]);
        let doc_lengths = make_doc_lengths(&[(1, 10), (2, 15), (3, 8)]);
        let params = Bm25Params::default();

        let terms: Vec<(&PostingList, f32)> = vec![
            (&pl1, params.idf(3, 3)),
            (&pl2, params.idf(2, 3)),
        ];

        let results = maxscore_search(&terms, 2, &params, &doc_lengths, 11.0);
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
        // Doc 3 should rank high (tf=3 in pl1, tf=1 in pl2, short doc)
        assert_eq!(results[0].0, 3);
    }

    #[test]
    fn test_maxscore_matches_brute_force() {
        // Verify MaxScore produces same ranking as brute-force
        let pl1 = make_pl(&[(1, 1), (2, 3), (3, 1), (4, 2)]);
        let pl2 = make_pl(&[(1, 2), (3, 1), (5, 1)]);
        let pl3 = make_pl(&[(2, 1), (4, 1), (5, 2)]);
        let doc_lengths = make_doc_lengths(&[
            (1, 10), (2, 20), (3, 10), (4, 15), (5, 12),
        ]);
        let params = Bm25Params::default();
        let avg_dl = 13.4;
        let total_docs = 5;

        let terms: Vec<(&PostingList, f32)> = vec![
            (&pl1, params.idf(pl1.df(), total_docs)),
            (&pl2, params.idf(pl2.df(), total_docs)),
            (&pl3, params.idf(pl3.df(), total_docs)),
        ];

        let maxscore_results = maxscore_search(&terms, 3, &params, &doc_lengths, avg_dl);

        // Brute-force scoring
        let mut brute_scores: HashMap<PointId, f32> = HashMap::new();
        for &(pl, idf) in &terms {
            for p in &pl.postings {
                let dl = doc_lengths.get(&p.doc_id).copied().unwrap_or(0);
                *brute_scores.entry(p.doc_id).or_insert(0.0) +=
                    idf * params.tf_norm(p.tf, dl, avg_dl);
            }
        }
        let mut brute_results: Vec<(PointId, f32)> = brute_scores.into_iter().collect();
        brute_results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        brute_results.truncate(3);

        // Same top-k doc IDs and order
        assert_eq!(maxscore_results.len(), brute_results.len());
        for (ms, bf) in maxscore_results.iter().zip(brute_results.iter()) {
            assert_eq!(ms.0, bf.0, "doc ID mismatch");
            assert!((ms.1 - bf.1).abs() < 1e-5, "score mismatch: {} vs {}", ms.1, bf.1);
        }
    }

    #[test]
    fn test_maxscore_empty() {
        let doc_lengths = HashMap::new();
        let params = Bm25Params::default();
        let terms: Vec<(&PostingList, f32)> = vec![];
        let results = maxscore_search(&terms, 10, &params, &doc_lengths, 0.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_maxscore_single_term() {
        let pl = make_pl(&[(1, 3), (5, 1), (10, 2)]);
        let doc_lengths = make_doc_lengths(&[(1, 10), (5, 20), (10, 10)]);
        let params = Bm25Params::default();

        let terms = vec![(&pl, params.idf(3, 10))];
        let results = maxscore_search(&terms, 2, &params, &doc_lengths, 13.3);
        assert_eq!(results.len(), 2);
        // Doc 1 should rank highest (tf=3, avg doc length)
        assert_eq!(results[0].0, 1);
    }
}
