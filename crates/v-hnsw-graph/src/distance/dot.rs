//! Dot product distance with SIMD optimization.
#![allow(unsafe_code)]

use v_hnsw_core::DistanceMetric;

use super::fallback;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Negative dot product distance metric.
///
/// Returns `-dot(a, b)` so that higher similarity = lower distance.
#[derive(Debug, Clone, Copy, Default, bincode::Encode, bincode::Decode)]
pub struct DotProductDistance;

impl DistanceMetric for DotProductDistance {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        -dot_product(a, b)
    }

    fn name(&self) -> &'static str {
        "dot"
    }
}

/// Compute dot product, dispatching to the best available SIMD.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We checked for AVX2 support above.
            return unsafe { dot_product_avx2(a, b) };
        }
    }

    fallback::dot_product_fallback(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        super::avx2_reduce!(a, b,
            |va, vb, sum| _mm256_fmadd_ps(va, vb, sum),
            |ai, bi| ai * bi
        )
    }
}

/// Horizontal sum of 8 floats in an AVX2 register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sum2 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum2, sum2);
    let sum_all = _mm_add_ss(sum2, shuf2);
    _mm_cvtss_f32(sum_all)
}
