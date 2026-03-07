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

