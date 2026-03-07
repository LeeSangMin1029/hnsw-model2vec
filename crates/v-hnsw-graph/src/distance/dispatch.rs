//! Runtime SIMD feature detection and automatic dispatch.

use v_hnsw_core::DistanceMetric;

use super::cosine::CosineDistance;
use super::dot::DotProductDistance;
use super::l2::L2Distance;

/// Distance metric enum for runtime selection.
#[derive(Debug, Clone, Copy)]
pub enum AutoDistance {
    L2,
    Cosine,
    DotProduct,
}

impl DistanceMetric for AutoDistance {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            AutoDistance::L2 => L2Distance.distance(a, b),
            AutoDistance::Cosine => CosineDistance.distance(a, b),
            AutoDistance::DotProduct => DotProductDistance.distance(a, b),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            AutoDistance::L2 => "l2",
            AutoDistance::Cosine => "cosine",
            AutoDistance::DotProduct => "dot",
        }
    }
}

