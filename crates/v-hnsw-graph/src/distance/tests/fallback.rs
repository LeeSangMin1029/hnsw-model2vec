//! Tests for portable (non-SIMD) fallback distance implementations.

use crate::distance::fallback::{dot_product_fallback, l2_squared_fallback};

// ---------------------------------------------------------------------------
// l2_squared_fallback
// ---------------------------------------------------------------------------

#[test]
fn l2_fallback_identical() {
    let a = vec![1.0, 2.0, 3.0];
    let dist = l2_squared_fallback(&a, &a);
    assert!(dist.abs() < 1e-6, "identical vectors should give 0, got {dist}");
}

#[test]
fn l2_fallback_known_values() {
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];
    let dist = l2_squared_fallback(&a, &b);
    assert!((dist - 25.0).abs() < 1e-6, "expected 25.0, got {dist}");
}

#[test]
fn l2_fallback_negative_values() {
    let a = vec![-1.0, -2.0, -3.0];
    let b = vec![1.0, 2.0, 3.0];
    // diffs: 2, 4, 6 => squares: 4, 16, 36 => sum: 56
    let dist = l2_squared_fallback(&a, &b);
    assert!((dist - 56.0).abs() < 1e-6, "expected 56.0, got {dist}");
}

#[test]
fn l2_fallback_single_element() {
    let a = vec![10.0];
    let b = vec![7.0];
    let dist = l2_squared_fallback(&a, &b);
    assert!((dist - 9.0).abs() < 1e-6, "expected 9.0, got {dist}");
}

#[test]
fn l2_fallback_symmetry() {
    let a = vec![1.0, -5.0, 3.0];
    let b = vec![-2.0, 4.0, 0.0];
    let d1 = l2_squared_fallback(&a, &b);
    let d2 = l2_squared_fallback(&b, &a);
    assert!((d1 - d2).abs() < 1e-6, "should be symmetric: {d1} vs {d2}");
}

#[test]
fn l2_fallback_non_negative() {
    let a = vec![-100.0, 200.0, -50.0];
    let b = vec![30.0, -40.0, 60.0];
    let dist = l2_squared_fallback(&a, &b);
    assert!(dist >= 0.0, "L2 fallback must be non-negative, got {dist}");
}

#[test]
fn l2_fallback_zeros() {
    let a = vec![0.0; 10];
    let b = vec![0.0; 10];
    let dist = l2_squared_fallback(&a, &b);
    assert!(dist.abs() < 1e-6, "both zero should give 0, got {dist}");
}

// ---------------------------------------------------------------------------
// dot_product_fallback
// ---------------------------------------------------------------------------

#[test]
fn dot_fallback_known_values() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = dot_product_fallback(&a, &b);
    assert!((result - 32.0).abs() < 1e-6, "expected 32.0, got {result}");
}

#[test]
fn dot_fallback_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let result = dot_product_fallback(&a, &b);
    assert!(result.abs() < 1e-6, "orthogonal should give ~0, got {result}");
}

#[test]
fn dot_fallback_self() {
    let a = vec![3.0, 4.0];
    let result = dot_product_fallback(&a, &a);
    assert!((result - 25.0).abs() < 1e-6, "expected 25.0, got {result}");
}

#[test]
fn dot_fallback_negative() {
    let a = vec![-1.0, -2.0];
    let b = vec![3.0, 4.0];
    let result = dot_product_fallback(&a, &b);
    assert!((result - (-11.0)).abs() < 1e-6, "expected -11.0, got {result}");
}

#[test]
fn dot_fallback_symmetry() {
    let a = vec![5.0, -3.0, 1.0];
    let b = vec![-2.0, 7.0, 4.0];
    let d1 = dot_product_fallback(&a, &b);
    let d2 = dot_product_fallback(&b, &a);
    assert!((d1 - d2).abs() < 1e-6, "should be symmetric: {d1} vs {d2}");
}

#[test]
fn dot_fallback_zero_vector() {
    let a = vec![0.0; 5];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = dot_product_fallback(&a, &b);
    assert!(result.abs() < 1e-6, "dot with zero should be ~0, got {result}");
}

#[test]
fn dot_fallback_single_element() {
    let a = vec![7.0];
    let b = vec![3.0];
    let result = dot_product_fallback(&a, &b);
    assert!((result - 21.0).abs() < 1e-6, "expected 21.0, got {result}");
}

// ---------------------------------------------------------------------------
// SIMD vs fallback consistency (if SIMD is available, they should match)
// ---------------------------------------------------------------------------

#[test]
fn l2_fallback_matches_simd() {
    use crate::distance::l2::l2_squared;
    let a: Vec<f32> = (0..100).map(|i| (i as f32) * 0.3).collect();
    let b: Vec<f32> = (0..100).map(|i| (i as f32) * -0.2 + 5.0).collect();

    let simd_result = l2_squared(&a, &b);
    let fallback_result = l2_squared_fallback(&a, &b);
    let tol = fallback_result.abs() * 1e-5 + 1e-6;
    assert!((simd_result - fallback_result).abs() < tol,
        "SIMD {simd_result} vs fallback {fallback_result}");
}

#[test]
fn dot_fallback_matches_simd() {
    use crate::distance::dot::dot_product;
    let a: Vec<f32> = (0..100).map(|i| (i as f32) * 0.3).collect();
    let b: Vec<f32> = (0..100).map(|i| (i as f32) * -0.2 + 5.0).collect();

    let simd_result = dot_product(&a, &b);
    let fallback_result = dot_product_fallback(&a, &b);
    let tol = fallback_result.abs() * 1e-5 + 1e-6;
    assert!((simd_result - fallback_result).abs() < tol,
        "SIMD {simd_result} vs fallback {fallback_result}");
}
