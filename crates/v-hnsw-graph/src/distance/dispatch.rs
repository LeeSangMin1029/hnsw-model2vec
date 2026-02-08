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

/// Report which SIMD features are available at runtime.
#[allow(dead_code)]
pub fn simd_features() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return "avx512f";
        }
        if is_x86_feature_detected!("avx2") {
            return "avx2";
        }
        if is_x86_feature_detected!("sse4.1") {
            return "sse4.1";
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return "neon";
    }

    "none"
}
