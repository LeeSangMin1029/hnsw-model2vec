//! Reciprocal Rank Fusion (RRF) implementation.
//!
//! RRF combines multiple ranked lists into a single ranking.
//! Formula: RRF(d) = sum_r 1 / (k + rank_r(d))

use std::collections::HashMap;

use v_hnsw_core::PointId;

/// Reciprocal Rank Fusion combiner.
///
/// Merges multiple ranked result lists using the RRF formula.
/// Each document's final score is the sum of reciprocal ranks
/// across all lists where it appears.
#[derive(Debug, Clone, Copy)]
pub struct RrfFusion {
    /// Rank smoothing constant (default: 60).
    /// Higher values reduce the impact of top-ranked items.
    k: u32,
}

impl Default for RrfFusion {
    fn default() -> Self {
        Self { k: 60 }
    }
}

impl RrfFusion {
    /// Create a new RRF combiner with default k=60.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new RRF combiner with custom k parameter.
    pub fn with_k(k: u32) -> Self {
        Self { k }
    }

    /// Get the k parameter.
    pub fn k(&self) -> u32 {
        self.k
    }

    /// Fuse multiple ranked lists into a single ranking.
    ///
    /// # Parameters
    /// - `ranked_lists`: A slice of ranked result lists. Each list contains
    ///   `(PointId, score)` pairs, assumed to be sorted by score descending.
    /// - `limit`: Maximum number of results to return.
    ///
    /// # Returns
    /// Combined results sorted by RRF score descending.
    pub fn fuse(&self, ranked_lists: &[Vec<(PointId, f32)>], limit: usize) -> Vec<(PointId, f32)> {
        let mut rrf_scores: HashMap<PointId, f32> = HashMap::new();

        for list in ranked_lists {
            for (rank, (doc_id, _original_score)) in list.iter().enumerate() {
                // RRF formula: 1 / (k + rank), where rank is 1-indexed
                let rrf_contribution = 1.0 / (self.k as f32 + (rank as f32 + 1.0));
                *rrf_scores.entry(*doc_id).or_insert(0.0) += rrf_contribution;
            }
        }

        // Sort by RRF score descending
        let mut results: Vec<(PointId, f32)> = rrf_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        results
    }

    /// Fuse multiple weighted ranked lists.
    ///
    /// Each list can have a different weight applied to its RRF contribution.
    ///
    /// # Parameters
    /// - `weighted_lists`: A slice of `(weight, ranked_list)` pairs.
    /// - `limit`: Maximum number of results to return.
    pub fn fuse_weighted(
        &self,
        weighted_lists: &[(f32, Vec<(PointId, f32)>)],
        limit: usize,
    ) -> Vec<(PointId, f32)> {
        let mut rrf_scores: HashMap<PointId, f32> = HashMap::new();

        for (weight, list) in weighted_lists {
            for (rank, (doc_id, _original_score)) in list.iter().enumerate() {
                let rrf_contribution = weight / (self.k as f32 + (rank as f32 + 1.0));
                *rrf_scores.entry(*doc_id).or_insert(0.0) += rrf_contribution;
            }
        }

        let mut results: Vec<(PointId, f32)> = rrf_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_k() {
        let rrf = RrfFusion::new();
        assert_eq!(rrf.k(), 60);
    }

    #[test]
    fn test_custom_k() {
        let rrf = RrfFusion::with_k(100);
        assert_eq!(rrf.k(), 100);
    }

    #[test]
    fn test_fuse_empty() {
        let rrf = RrfFusion::new();
        let result = rrf.fuse(&[], 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_fuse_single_list() {
        let rrf = RrfFusion::with_k(60);
        let list = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
        let result = rrf.fuse(&[list], 10);

        assert_eq!(result.len(), 3);
        // First item should have highest RRF score
        assert_eq!(result[0].0, 1);
        // Score should be 1/(60+1) = ~0.0164
        assert!((result[0].1 - 1.0 / 61.0).abs() < 1e-6);
    }

    #[test]
    fn test_fuse_two_lists_same_order() {
        let rrf = RrfFusion::with_k(60);
        let list1 = vec![(1, 0.9), (2, 0.8)];
        let list2 = vec![(1, 0.95), (2, 0.85)];
        let result = rrf.fuse(&[list1, list2], 10);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 1); // Doc 1 is first in both
        assert_eq!(result[1].0, 2);

        // Doc 1 score: 1/61 + 1/61 = 2/61
        let expected_score = 2.0 / 61.0;
        assert!((result[0].1 - expected_score).abs() < 1e-6);
    }

    #[test]
    fn test_fuse_two_lists_different_order() {
        let rrf = RrfFusion::with_k(60);
        // Doc 1 is first in list1, Doc 2 is first in list2
        let list1 = vec![(1, 0.9), (2, 0.8)];
        let list2 = vec![(2, 0.95), (1, 0.85)];
        let result = rrf.fuse(&[list1, list2], 10);

        // Both should have the same score: 1/61 + 1/62
        assert_eq!(result.len(), 2);
        let expected_score = 1.0 / 61.0 + 1.0 / 62.0;
        assert!((result[0].1 - expected_score).abs() < 1e-6);
        assert!((result[1].1 - expected_score).abs() < 1e-6);
    }

    #[test]
    fn test_fuse_disjoint_lists() {
        let rrf = RrfFusion::with_k(60);
        let list1 = vec![(1, 0.9), (2, 0.8)];
        let list2 = vec![(3, 0.95), (4, 0.85)];
        let result = rrf.fuse(&[list1, list2], 10);

        assert_eq!(result.len(), 4);
        // Top items from each list should be tied
        let top_two: Vec<PointId> = result.iter().take(2).map(|(id, _)| *id).collect();
        assert!(top_two.contains(&1) && top_two.contains(&3));
    }

    #[test]
    fn test_fuse_with_limit() {
        let rrf = RrfFusion::new();
        let list1 = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
        let list2 = vec![(4, 0.95), (5, 0.85), (6, 0.75)];
        let result = rrf.fuse(&[list1, list2], 2);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_fuse_item_in_one_list_only() {
        let rrf = RrfFusion::with_k(60);
        // Doc 1 in both, Doc 2 only in list1, Doc 3 only in list2
        let list1 = vec![(1, 0.9), (2, 0.8)];
        let list2 = vec![(1, 0.95), (3, 0.85)];
        let result = rrf.fuse(&[list1, list2], 10);

        assert_eq!(result.len(), 3);
        // Doc 1 should be first (appears in both at rank 1)
        assert_eq!(result[0].0, 1);
        // Score: 1/61 + 1/61 = 2/61
        let expected_score = 2.0 / 61.0;
        assert!((result[0].1 - expected_score).abs() < 1e-6);
    }

    #[test]
    fn test_fuse_weighted_equal_weights() {
        let rrf = RrfFusion::with_k(60);
        let list1 = vec![(1, 0.9), (2, 0.8)];
        let list2 = vec![(1, 0.95), (2, 0.85)];

        let unweighted = rrf.fuse(&[list1.clone(), list2.clone()], 10);
        let weighted = rrf.fuse_weighted(&[(1.0, list1), (1.0, list2)], 10);

        assert_eq!(unweighted.len(), weighted.len());
        for i in 0..unweighted.len() {
            assert_eq!(unweighted[i].0, weighted[i].0);
            assert!((unweighted[i].1 - weighted[i].1).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fuse_weighted_different_weights() {
        let rrf = RrfFusion::with_k(60);
        // Weight list1 at 2x, list2 at 1x
        let list1 = vec![(1, 0.9)];
        let list2 = vec![(2, 0.95)];
        let result = rrf.fuse_weighted(&[(2.0, list1), (1.0, list2)], 10);

        assert_eq!(result.len(), 2);
        // Doc 1 should be first (weighted 2x)
        assert_eq!(result[0].0, 1);
        // Score for doc 1: 2 * 1/61
        assert!((result[0].1 - 2.0 / 61.0).abs() < 1e-6);
        // Score for doc 2: 1 * 1/61
        assert!((result[1].1 - 1.0 / 61.0).abs() < 1e-6);
    }
}
