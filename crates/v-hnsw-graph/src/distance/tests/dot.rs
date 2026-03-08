//! Tests for dot product distance.

use v_hnsw_core::DistanceMetric;

use crate::distance::dot::{dot_product, DotProductDistance};

#[test]
fn dot_product_known_values() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    // dot = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    let result = dot_product(&a, &b);
    assert!((result - 32.0).abs() < 1e-6, "expected 32.0, got {result}");
}

#[test]
fn dot_product_distance_is_negated() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let dist = DotProductDistance.distance(&a, &b);
    assert!((dist - (-32.0)).abs() < 1e-6, "expected -32.0, got {dist}");
}

#[test]
fn dot_product_orthogonal_is_zero() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let result = dot_product(&a, &b);
    assert!(result.abs() < 1e-6, "orthogonal vectors should have dot product ~0, got {result}");
}

#[test]
fn dot_product_self() {
    let a = vec![3.0, 4.0];
    // dot(a, a) = 9 + 16 = 25
    let result = dot_product(&a, &a);
    assert!((result - 25.0).abs() < 1e-6, "expected 25.0, got {result}");
}

#[test]
fn dot_product_negative_values() {
    let a = vec![-1.0, -2.0];
    let b = vec![3.0, 4.0];
    // dot = -3 + -8 = -11
    let result = dot_product(&a, &b);
    assert!((result - (-11.0)).abs() < 1e-6, "expected -11.0, got {result}");
}

#[test]
fn dot_product_symmetry() {
    let a = vec![1.0, -3.0, 5.0, 2.0];
    let b = vec![-2.0, 4.0, 1.0, -1.0];
    let d1 = dot_product(&a, &b);
    let d2 = dot_product(&b, &a);
    assert!((d1 - d2).abs() < 1e-6, "dot product should be symmetric: {d1} vs {d2}");
}

#[test]
fn dot_product_zero_vector() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![1.0, 2.0, 3.0];
    let result = dot_product(&a, &b);
    assert!(result.abs() < 1e-6, "dot product with zero vector should be ~0, got {result}");
}

#[test]
fn dot_product_single_element() {
    let a = vec![7.0];
    let b = vec![3.0];
    let result = dot_product(&a, &b);
    assert!((result - 21.0).abs() < 1e-6, "expected 21.0, got {result}");
}

#[test]
fn dot_product_large_vector() {
    // 257 elements to test SIMD remainder handling
    let a: Vec<f32> = (0..257).map(|i| i as f32).collect();
    let b: Vec<f32> = vec![1.0; 257];
    let result = dot_product(&a, &b);
    // sum of 0..256 = 256*257/2 = 32896
    let expected = (0..257).map(|i| i as f32).sum::<f32>();
    assert!((result - expected).abs() < 1e-2, "expected {expected}, got {result}");
}

#[test]
fn dot_product_distance_name() {
    assert_eq!(DotProductDistance.name(), "dot");
}

#[test]
fn dot_product_distance_higher_similarity_lower_distance() {
    // More similar vectors should have lower (more negative) distance
    let query = vec![1.0, 0.0, 0.0];
    let similar = vec![1.0, 0.1, 0.0]; // close to query direction
    let dissimilar = vec![0.0, 1.0, 0.0]; // orthogonal

    let d_similar = DotProductDistance.distance(&query, &similar);
    let d_dissimilar = DotProductDistance.distance(&query, &dissimilar);

    assert!(d_similar < d_dissimilar,
        "similar vector should have lower distance: {d_similar} vs {d_dissimilar}");
}

// Test exact 8-element boundary (AVX2 register size)
#[test]
fn dot_product_exact_avx_boundary() {
    let a = vec![1.0; 8];
    let b = vec![2.0; 8];
    let result = dot_product(&a, &b);
    assert!((result - 16.0).abs() < 1e-6, "expected 16.0, got {result}");
}

#[test]
fn dot_product_16_elements() {
    let a = vec![1.0; 16];
    let b = vec![3.0; 16];
    let result = dot_product(&a, &b);
    assert!((result - 48.0).abs() < 1e-6, "expected 48.0, got {result}");
}

#[test]
fn dot_product_non_multiple_of_8() {
    // 11 elements: 1 full AVX chunk + 3 remainder
    let a: Vec<f32> = (1..=11).map(|i| i as f32).collect();
    let b = vec![1.0; 11];
    let result = dot_product(&a, &b);
    let expected = (1..=11).sum::<i32>() as f32; // 66
    assert!((result - expected).abs() < 1e-4, "expected {expected}, got {result}");
}
