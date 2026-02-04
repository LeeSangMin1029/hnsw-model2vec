//! Reranking trait and implementations.
//!
//! Rerankers refine search results using more sophisticated scoring.

use v_hnsw_core::PointId;

/// A trait for reranking search results.
///
/// Rerankers take initial search results and re-score them using
/// additional signals (e.g., cross-encoder models, custom scoring).
pub trait Reranker: Send + Sync {
    /// Rerank search results.
    ///
    /// # Parameters
    /// - `query`: The original query text
    /// - `candidates`: List of (id, score, text) tuples to rerank
    ///
    /// # Returns
    /// Reranked results sorted by new score descending.
    fn rerank(
        &self,
        query: &str,
        candidates: &[(PointId, f32, String)],
    ) -> v_hnsw_core::Result<Vec<(PointId, f32)>>;
}

/// A passthrough reranker that returns results unchanged.
///
/// Useful as a default or for testing when no actual reranking is needed.
#[derive(Debug, Clone, Copy, Default)]
pub struct PassthroughReranker;

impl PassthroughReranker {
    /// Create a new passthrough reranker.
    pub fn new() -> Self {
        Self
    }
}

impl Reranker for PassthroughReranker {
    fn rerank(
        &self,
        _query: &str,
        candidates: &[(PointId, f32, String)],
    ) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        // Simply strip the text and return the original scores
        let results = candidates
            .iter()
            .map(|(id, score, _text)| (*id, *score))
            .collect();
        Ok(results)
    }
}

/// A score boosting reranker that adjusts scores based on text length.
///
/// Shorter documents get a slight boost. Useful as a simple heuristic
/// when brevity is preferred.
#[derive(Debug, Clone, Copy)]
pub struct LengthBoostReranker {
    /// Optimal document length (chars). Docs closer to this get boosted.
    optimal_length: usize,
    /// Maximum boost factor (e.g., 1.2 = 20% max boost).
    max_boost: f32,
}

impl Default for LengthBoostReranker {
    fn default() -> Self {
        Self {
            optimal_length: 200,
            max_boost: 1.2,
        }
    }
}

impl LengthBoostReranker {
    /// Create a new length boost reranker.
    pub fn new(optimal_length: usize, max_boost: f32) -> Self {
        Self {
            optimal_length,
            max_boost,
        }
    }
}

impl Reranker for LengthBoostReranker {
    fn rerank(
        &self,
        _query: &str,
        candidates: &[(PointId, f32, String)],
    ) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        let mut results: Vec<(PointId, f32)> = candidates
            .iter()
            .map(|(id, score, text)| {
                let length = text.len();
                // Calculate boost based on how close to optimal length
                let length_ratio = if length < self.optimal_length {
                    length as f32 / self.optimal_length as f32
                } else {
                    self.optimal_length as f32 / length as f32
                };
                // Boost is between 1.0 and max_boost
                let boost = 1.0 + (self.max_boost - 1.0) * length_ratio;
                (*id, score * boost)
            })
            .collect();

        // Sort by new score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passthrough_reranker() {
        let reranker = PassthroughReranker::new();
        let candidates = vec![
            (1, 0.9, "hello world".to_string()),
            (2, 0.8, "goodbye world".to_string()),
        ];

        let results = reranker.rerank("query", &candidates).expect("rerank should succeed");

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1);
        assert!((results[0].1 - 0.9).abs() < f32::EPSILON);
        assert_eq!(results[1].0, 2);
        assert!((results[1].1 - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_passthrough_empty() {
        let reranker = PassthroughReranker::new();
        let candidates: Vec<(PointId, f32, String)> = vec![];

        let results = reranker.rerank("query", &candidates).expect("rerank should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_length_boost_reranker() {
        let reranker = LengthBoostReranker::new(100, 1.5);
        let candidates = vec![
            (1, 1.0, "a".repeat(100)),    // Optimal length
            (2, 1.0, "a".repeat(50)),     // Half optimal
            (3, 1.0, "a".repeat(200)),    // Double optimal
        ];

        let results = reranker.rerank("query", &candidates).expect("rerank should succeed");

        // Doc 1 should have highest boost (optimal length)
        assert_eq!(results[0].0, 1);
        // All scores should be between 1.0 and 1.5
        for (_, score) in &results {
            assert!(*score >= 1.0 && *score <= 1.5);
        }
    }

    #[test]
    fn test_length_boost_reorders() {
        let reranker = LengthBoostReranker::new(100, 2.0);
        let candidates = vec![
            (1, 0.9, "a".repeat(1000)), // Very long, will get penalized
            (2, 0.8, "a".repeat(100)),   // Optimal length, will get boosted
        ];

        let results = reranker.rerank("query", &candidates).expect("rerank should succeed");

        // Doc 2 should now be first after boosting
        assert_eq!(results[0].0, 2);
    }
}
