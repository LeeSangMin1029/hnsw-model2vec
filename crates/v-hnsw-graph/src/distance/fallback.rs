//! Portable (non-SIMD) distance implementations.
//! Used as fallback when no SIMD instructions are available.

/// L2 squared distance (portable).
#[inline]
pub fn l2_squared_fallback(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Dot product (portable).
#[inline]
pub fn dot_product_fallback(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Vector norm squared (portable).
#[inline]
#[allow(dead_code)]
pub fn norm_squared_fallback(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

/// Cosine distance (portable). Returns `1.0 - cosine_similarity`.
#[inline]
#[allow(dead_code)]
pub fn cosine_distance_fallback(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot = dot_product_fallback(a, b);
    let norm_a = norm_squared_fallback(a).sqrt();
    let norm_b = norm_squared_fallback(b).sqrt();
    let denom = norm_a * norm_b;
    if denom == 0.0 {
        1.0
    } else {
        1.0 - (dot / denom)
    }
}
