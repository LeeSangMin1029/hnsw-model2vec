//! Cosine distance with SIMD optimization.

use v_hnsw_core::DistanceMetric;

use crate::dot::dot_product;

/// Cosine distance metric.
///
/// Returns `1.0 - cosine_similarity(a, b)`.
/// Range: [0.0, 2.0] where 0.0 = identical direction.
#[derive(Debug, Clone, Copy, Default)]
pub struct CosineDistance;

impl DistanceMetric for CosineDistance {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        cosine_distance(a, b)
    }

    fn name(&self) -> &'static str {
        "cosine"
    }
}

/// Compute cosine distance using SIMD-accelerated dot product and norms.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = dot_product(a, a).sqrt();
    let norm_b = dot_product(b, b).sqrt();
    let denom = norm_a * norm_b;

    if denom == 0.0 {
        return 1.0;
    }

    let similarity = dot / denom;
    // Clamp to handle floating point imprecision
    1.0 - similarity.clamp(-1.0, 1.0)
}

/// Pre-normalized cosine distance (for vectors already normalized to unit length).
///
/// Much faster since it skips norm computation.
#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)]
pub struct NormalizedCosineDistance;

impl DistanceMetric for NormalizedCosineDistance {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let dot = dot_product(a, b);
        1.0 - dot.clamp(-1.0, 1.0)
    }

    fn name(&self) -> &'static str {
        "cosine_normalized"
    }
}

/// Normalize a vector to unit length in-place.
#[allow(dead_code)]
pub fn normalize(v: &mut [f32]) {
    let norm = dot_product(v, v).sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}
