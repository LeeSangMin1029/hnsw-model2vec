//! Convex Combination Fusion implementation.
//!
//! Combines dense and sparse ranked lists using score-level interpolation:
//! `final_score = alpha * dense_score + (1 - alpha) * sparse_score`
//!
//! Based on Bruch et al. (ACM TOIS 2023): outperforms RRF on both
//! in-domain and out-of-domain benchmarks.

use std::collections::HashMap;

use v_hnsw_core::PointId;

/// Convex Combination fusion combiner.
///
/// Merges dense and sparse result lists by normalizing scores to [0, 1]
/// and computing a weighted linear combination.
#[derive(Debug, Clone, Copy)]
pub struct ConvexFusion {
    /// Interpolation weight for dense results (0.0–1.0).
    /// Higher alpha = more weight on dense (vector) search.
    alpha: f32,
}

impl Default for ConvexFusion {
    fn default() -> Self {
        Self { alpha: 0.5 }
    }
}

impl ConvexFusion {
    /// Create a new fusion combiner with default alpha=0.5.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new fusion combiner with custom alpha.
    ///
    /// - `alpha = 0.5`: equal weight (default)
    /// - `alpha = 0.7`: dense-heavy
    /// - `alpha = 0.3`: sparse-heavy
    pub fn with_alpha(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Get the alpha parameter.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Fuse dense and sparse result lists using convex combination.
    ///
    /// # Parameters
    /// - `dense`: Dense search results `(id, distance)` — lower distance is better.
    /// - `sparse`: Sparse search results `(id, score)` — higher score is better.
    /// - `limit`: Maximum number of results to return.
    ///
    /// # Returns
    /// Combined results sorted by fused score descending.
    pub fn fuse(
        &self,
        dense: &[(PointId, f32)],
        sparse: &[(PointId, f32)],
        limit: usize,
    ) -> Vec<(PointId, f32)> {
        if dense.is_empty() && sparse.is_empty() {
            return Vec::new();
        }

        let dense_scores = normalize(dense, true);   // distance: invert
        let sparse_scores = normalize(sparse, false); // score: keep direction

        // Merge into combined scores
        let mut combined: HashMap<PointId, f32> = HashMap::new();

        for (&id, &score) in &dense_scores {
            *combined.entry(id).or_insert(0.0) += self.alpha * score;
        }

        let beta = 1.0 - self.alpha;
        for (&id, &score) in &sparse_scores {
            *combined.entry(id).or_insert(0.0) += beta * score;
        }

        // Sort by score descending
        let mut results: Vec<(PointId, f32)> = combined.into_iter().collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        results
    }
}

/// Min-max normalize scores to [0, 1].
/// If `invert`, lower values → higher scores (for distance metrics).
pub(super) fn normalize(results: &[(PointId, f32)], invert: bool) -> HashMap<PointId, f32> {
    let mut scores = HashMap::with_capacity(results.len());
    if results.is_empty() {
        return scores;
    }

    let (min_val, max_val) = min_max_values(results);
    let range = max_val - min_val;

    for &(id, val) in results {
        let normalized = if range <= f32::EPSILON {
            1.0
        } else if invert {
            1.0 - (val - min_val) / range
        } else {
            (val - min_val) / range
        };
        scores.insert(id, normalized);
    }

    scores
}

/// Get min and max values from results.
fn min_max_values(results: &[(PointId, f32)]) -> (f32, f32) {
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &(_, val) in results {
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }
    (min_val, max_val)
}