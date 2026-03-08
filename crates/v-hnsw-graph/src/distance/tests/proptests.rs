//! Property-based tests for distance functions.

use proptest::prelude::*;
use v_hnsw_core::DistanceMetric;

use crate::distance::cosine::CosineDistance;
use crate::distance::fallback;
use crate::distance::l2::L2Distance;

proptest! {
    #[test]
    fn l2_non_negative(
        a in proptest::collection::vec(-100.0f32..100.0, 1..256),
    ) {
        let b = vec![0.0; a.len()];
        let dist = L2Distance.distance(&a, &b);
        prop_assert!(dist >= 0.0, "L2 distance must be non-negative, got {}", dist);
    }

    #[test]
    fn l2_identity(
        a in proptest::collection::vec(-100.0f32..100.0, 1..256),
    ) {
        let dist = L2Distance.distance(&a, &a);
        prop_assert!((dist - 0.0).abs() < 1e-5, "L2(a, a) must be ~0, got {}", dist);
    }

    #[test]
    fn l2_symmetry(
        a in proptest::collection::vec(-100.0f32..100.0, 1..128),
        b in proptest::collection::vec(-100.0f32..100.0, 1..128),
    ) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];
        let d1 = L2Distance.distance(a, b);
        let d2 = L2Distance.distance(b, a);
        prop_assert!((d1 - d2).abs() < 1e-5, "L2 must be symmetric: {} vs {}", d1, d2);
    }

    #[test]
    fn l2_simd_matches_fallback(
        a in proptest::collection::vec(-100.0f32..100.0, 1..512),
        b in proptest::collection::vec(-100.0f32..100.0, 1..512),
    ) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];
        let simd_result = L2Distance.distance(a, b);
        let fallback_result = fallback::l2_squared_fallback(a, b);
        let tolerance = fallback_result.abs() * 1e-5 + 1e-6;
        prop_assert!(
            (simd_result - fallback_result).abs() < tolerance,
            "SIMD {} vs fallback {}", simd_result, fallback_result
        );
    }

    #[test]
    fn cosine_range(
        a in proptest::collection::vec(0.1f32..100.0, 2..128),
        b in proptest::collection::vec(0.1f32..100.0, 2..128),
    ) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];
        let dist = CosineDistance.distance(a, b);
        prop_assert!(dist >= -0.001 && dist <= 2.001,
            "Cosine distance must be in [0, 2], got {}", dist);
    }
}
