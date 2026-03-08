//! Tests for L2 (Euclidean squared) distance.

use v_hnsw_core::DistanceMetric;

use crate::distance::l2::{l2_squared, L2Distance};

#[test]
fn identical_vectors_zero() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let dist = L2Distance.distance(&a, &a);
    assert!(dist.abs() < 1e-6, "identical vectors should have distance 0, got {dist}");
}

#[test]
fn known_values_3d() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![1.0, 1.0, 1.0];
    let dist = l2_squared(&a, &b);
    assert!((dist - 3.0).abs() < 1e-6, "expected 3.0, got {dist}");
}

#[test]
fn known_values_2d() {
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];
    // L2 squared = 9 + 16 = 25
    let dist = l2_squared(&a, &b);
    assert!((dist - 25.0).abs() < 1e-6, "expected 25.0, got {dist}");
}

#[test]
fn single_dimension() {
    let a = vec![5.0];
    let b = vec![2.0];
    let dist = l2_squared(&a, &b);
    assert!((dist - 9.0).abs() < 1e-6, "expected 9.0, got {dist}");
}

#[test]
fn symmetry() {
    let a = vec![1.0, -3.0, 5.0, 2.0];
    let b = vec![-2.0, 4.0, 1.0, -1.0];
    let d1 = L2Distance.distance(&a, &b);
    let d2 = L2Distance.distance(&b, &a);
    assert!((d1 - d2).abs() < 1e-6, "L2 should be symmetric: {d1} vs {d2}");
}

#[test]
fn non_negative() {
    let a = vec![-10.0, 20.0, -30.0, 40.0];
    let b = vec![5.0, -15.0, 25.0, -35.0];
    let dist = L2Distance.distance(&a, &b);
    assert!(dist >= 0.0, "L2 distance must be non-negative, got {dist}");
}

#[test]
fn zero_vectors() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![0.0, 0.0, 0.0];
    let dist = l2_squared(&a, &b);
    assert!(dist.abs() < 1e-6, "zero vectors should have distance 0, got {dist}");
}

#[test]
fn negative_values() {
    let a = vec![-1.0, -2.0];
    let b = vec![-4.0, -6.0];
    // (-1 - -4)^2 + (-2 - -6)^2 = 9 + 16 = 25
    let dist = l2_squared(&a, &b);
    assert!((dist - 25.0).abs() < 1e-6, "expected 25.0, got {dist}");
}

#[test]
fn triangle_inequality_squared() {
    // For squared L2, sqrt(d(a,c)) <= sqrt(d(a,b)) + sqrt(d(b,c))
    let a = vec![0.0, 0.0];
    let b = vec![1.0, 0.0];
    let c = vec![2.0, 0.0];

    let dab = l2_squared(&a, &b).sqrt();
    let dbc = l2_squared(&b, &c).sqrt();
    let dac = l2_squared(&a, &c).sqrt();

    assert!(dac <= dab + dbc + 1e-6,
        "triangle inequality violated: {dac} > {dab} + {dbc}");
}

#[test]
fn l2_name() {
    assert_eq!(L2Distance.name(), "l2");
}

// ---------------------------------------------------------------------------
// SIMD boundary tests
// ---------------------------------------------------------------------------

#[test]
fn exact_avx_boundary_8_elements() {
    let a = vec![1.0; 8];
    let b = vec![2.0; 8];
    // Each diff = 1.0, diff^2 = 1.0, sum = 8.0
    let dist = l2_squared(&a, &b);
    assert!((dist - 8.0).abs() < 1e-6, "expected 8.0, got {dist}");
}

#[test]
fn exact_16_elements() {
    let a = vec![0.0; 16];
    let b = vec![1.0; 16];
    let dist = l2_squared(&a, &b);
    assert!((dist - 16.0).abs() < 1e-6, "expected 16.0, got {dist}");
}

#[test]
fn non_multiple_of_8() {
    // 11 elements: 1 full AVX chunk (8) + 3 remainder
    let a = vec![0.0; 11];
    let b: Vec<f32> = (1..=11).map(|i| i as f32).collect();
    let expected: f32 = (1..=11).map(|i| (i * i) as f32).sum(); // 1+4+9+16+25+36+49+64+81+100+121 = 506
    let dist = l2_squared(&a, &b);
    assert!((dist - expected).abs() < 1e-3, "expected {expected}, got {dist}");
}

#[test]
fn large_vector_257() {
    let a: Vec<f32> = (0..257).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..257).map(|i| (i as f32) * 0.02).collect();
    let dist = l2_squared(&a, &b);
    assert!(dist >= 0.0, "L2 distance must be non-negative for large vectors, got {dist}");

    // Verify against manual calculation
    let expected: f32 = (0..257).map(|i| {
        let d = (i as f32) * 0.01 - (i as f32) * 0.02;
        d * d
    }).sum();
    assert!((dist - expected).abs() < expected * 1e-5 + 1e-6,
        "expected ~{expected}, got {dist}");
}
