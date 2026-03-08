//! Tests for cosine distance.

use v_hnsw_core::DistanceMetric;

use crate::distance::cosine::{cosine_distance, CosineDistance, NormalizedCosineDistance};

// ---------------------------------------------------------------------------
// CosineDistance basic tests
// ---------------------------------------------------------------------------

#[test]
fn identical_vectors_zero_distance() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let dist = CosineDistance.distance(&a, &a);
    assert!(dist.abs() < 1e-6, "identical vectors should have distance ~0, got {dist}");
}

#[test]
fn orthogonal_vectors_distance_one() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let dist = CosineDistance.distance(&a, &b);
    assert!((dist - 1.0).abs() < 1e-6, "orthogonal vectors should have distance ~1, got {dist}");
}

#[test]
fn opposite_vectors_distance_two() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![-1.0, 0.0, 0.0];
    let dist = CosineDistance.distance(&a, &b);
    assert!((dist - 2.0).abs() < 1e-6, "opposite vectors should have distance ~2, got {dist}");
}

#[test]
fn parallel_same_direction_distance_zero() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![2.0, 4.0, 6.0]; // same direction, different magnitude
    let dist = CosineDistance.distance(&a, &b);
    assert!(dist.abs() < 1e-6, "parallel same-dir vectors should have distance ~0, got {dist}");
}

#[test]
fn zero_vector_returns_one() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![1.0, 2.0, 3.0];
    let dist = cosine_distance(&a, &b);
    assert!((dist - 1.0).abs() < 1e-6, "zero vector should give distance 1.0, got {dist}");
}

#[test]
fn both_zero_vectors_returns_one() {
    let a = vec![0.0, 0.0];
    let b = vec![0.0, 0.0];
    let dist = cosine_distance(&a, &b);
    assert!((dist - 1.0).abs() < 1e-6, "both zero vectors should give distance 1.0, got {dist}");
}

#[test]
fn cosine_distance_range() {
    // Cosine distance should always be in [0.0, 2.0] for non-zero vectors
    let cases: Vec<(Vec<f32>, Vec<f32>)> = vec![
        (vec![1.0, 0.0], vec![0.0, 1.0]),
        (vec![1.0, 1.0], vec![1.0, -1.0]),
        (vec![3.0, 4.0], vec![4.0, 3.0]),
        (vec![-1.0, -2.0], vec![1.0, 2.0]),
    ];
    for (a, b) in &cases {
        let dist = CosineDistance.distance(a, b);
        assert!(dist >= -1e-6 && dist <= 2.0 + 1e-6,
            "cosine distance out of range [0,2]: {dist} for {:?} vs {:?}", a, b);
    }
}

#[test]
fn cosine_symmetry() {
    let a = vec![1.0, 3.0, -2.0, 5.0];
    let b = vec![-1.0, 2.0, 4.0, -3.0];
    let d1 = CosineDistance.distance(&a, &b);
    let d2 = CosineDistance.distance(&b, &a);
    assert!((d1 - d2).abs() < 1e-6, "cosine distance should be symmetric: {d1} vs {d2}");
}

#[test]
fn cosine_name() {
    assert_eq!(CosineDistance.name(), "cosine");
}

// ---------------------------------------------------------------------------
// NormalizedCosineDistance tests
// ---------------------------------------------------------------------------

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 { return v.to_vec(); }
    v.iter().map(|x| x / norm).collect()
}

#[test]
fn normalized_identical_vectors() {
    let a = normalize(&[1.0, 2.0, 3.0]);
    let dist = NormalizedCosineDistance.distance(&a, &a);
    assert!(dist.abs() < 1e-6, "identical normalized vectors should have distance ~0, got {dist}");
}

#[test]
fn normalized_orthogonal_vectors() {
    let a = normalize(&[1.0, 0.0]);
    let b = normalize(&[0.0, 1.0]);
    let dist = NormalizedCosineDistance.distance(&a, &b);
    assert!((dist - 1.0).abs() < 1e-6, "orthogonal normalized vectors should have distance ~1, got {dist}");
}

#[test]
fn normalized_matches_full_cosine() {
    let a_raw = vec![3.0, 4.0, 1.0, -2.0];
    let b_raw = vec![1.0, -1.0, 2.0, 3.0];
    let a = normalize(&a_raw);
    let b = normalize(&b_raw);

    let full = CosineDistance.distance(&a, &b);
    let norm = NormalizedCosineDistance.distance(&a, &b);
    assert!((full - norm).abs() < 1e-5,
        "normalized cosine should match full cosine for unit vectors: {full} vs {norm}");
}

#[test]
fn normalized_cosine_name() {
    assert_eq!(NormalizedCosineDistance.name(), "cosine_normalized");
}

// ---------------------------------------------------------------------------
// Large vector test (exercises SIMD tail handling)
// ---------------------------------------------------------------------------

#[test]
fn cosine_large_vector() {
    // 257 elements to test SIMD remainder paths (256 = 32*8, +1 remainder)
    let a: Vec<f32> = (0..257).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..257).map(|i| (i as f32 * 0.2).cos()).collect();
    let dist = CosineDistance.distance(&a, &b);
    assert!(dist >= -1e-6 && dist <= 2.0 + 1e-6, "large vector cosine out of range: {dist}");
}

#[test]
fn cosine_single_element() {
    let a = vec![3.0];
    let b = vec![5.0];
    let dist = CosineDistance.distance(&a, &b);
    assert!(dist.abs() < 1e-6, "same-sign single element should be ~0: {dist}");

    let c = vec![-5.0];
    let dist2 = CosineDistance.distance(&a, &c);
    assert!((dist2 - 2.0).abs() < 1e-6, "opposite-sign single element should be ~2: {dist2}");
}
