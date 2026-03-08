//! Tests for the AutoDistance runtime dispatch enum.

use v_hnsw_core::DistanceMetric;

use crate::distance::cosine::CosineDistance;
use crate::distance::dispatch::AutoDistance;
use crate::distance::dot::DotProductDistance;
use crate::distance::l2::L2Distance;

#[test]
fn auto_l2_matches_direct() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let auto_dist = AutoDistance::L2.distance(&a, &b);
    let direct_dist = L2Distance.distance(&a, &b);
    assert!((auto_dist - direct_dist).abs() < 1e-6,
        "AutoDistance::L2 should match L2Distance: {auto_dist} vs {direct_dist}");
}

#[test]
fn auto_cosine_matches_direct() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let auto_dist = AutoDistance::Cosine.distance(&a, &b);
    let direct_dist = CosineDistance.distance(&a, &b);
    assert!((auto_dist - direct_dist).abs() < 1e-6,
        "AutoDistance::Cosine should match CosineDistance: {auto_dist} vs {direct_dist}");
}

#[test]
fn auto_dot_matches_direct() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let auto_dist = AutoDistance::DotProduct.distance(&a, &b);
    let direct_dist = DotProductDistance.distance(&a, &b);
    assert!((auto_dist - direct_dist).abs() < 1e-6,
        "AutoDistance::DotProduct should match DotProductDistance: {auto_dist} vs {direct_dist}");
}

#[test]
fn auto_l2_name() {
    assert_eq!(AutoDistance::L2.name(), "l2");
}

#[test]
fn auto_cosine_name() {
    assert_eq!(AutoDistance::Cosine.name(), "cosine");
}

#[test]
fn auto_dot_name() {
    assert_eq!(AutoDistance::DotProduct.name(), "dot");
}

#[test]
fn auto_l2_identical_zero() {
    let a = vec![1.0, 2.0, 3.0];
    let dist = AutoDistance::L2.distance(&a, &a);
    assert!(dist.abs() < 1e-6, "L2 self-distance should be ~0, got {dist}");
}

#[test]
fn auto_cosine_identical_zero() {
    let a = vec![1.0, 2.0, 3.0];
    let dist = AutoDistance::Cosine.distance(&a, &a);
    assert!(dist.abs() < 1e-6, "cosine self-distance should be ~0, got {dist}");
}

#[test]
fn auto_dot_self_negative() {
    let a = vec![1.0, 2.0, 3.0];
    let dist = AutoDistance::DotProduct.distance(&a, &a);
    // dot(a,a) = 1+4+9 = 14, distance = -14
    assert!((dist - (-14.0)).abs() < 1e-6, "expected -14.0, got {dist}");
}

#[test]
fn auto_large_vectors() {
    let a: Vec<f32> = (0..300).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..300).map(|i| (i as f32 * 0.2).cos()).collect();

    let l2 = AutoDistance::L2.distance(&a, &b);
    let cos = AutoDistance::Cosine.distance(&a, &b);
    let dot = AutoDistance::DotProduct.distance(&a, &b);

    assert!(l2 >= 0.0, "L2 should be non-negative: {l2}");
    assert!(cos >= -1e-6 && cos <= 2.0 + 1e-6, "cosine out of range: {cos}");
    // dot can be any value
    assert!(!dot.is_nan(), "dot should not be NaN");
}

#[test]
fn auto_distance_is_copy() {
    let d = AutoDistance::L2;
    let d2 = d; // Copy
    assert_eq!(d.name(), d2.name());
}
