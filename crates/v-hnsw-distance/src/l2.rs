//! L2 (Euclidean) squared distance with SIMD optimization.
#![allow(unsafe_code)]

use v_hnsw_core::DistanceMetric;

use crate::fallback;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// L2 squared distance metric.
///
/// Computes `sum((a[i] - b[i])^2)` with SIMD acceleration.
#[derive(Debug, Clone, Copy, Default)]
pub struct L2Distance;

impl DistanceMetric for L2Distance {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        l2_squared(a, b)
    }

    fn name(&self) -> &'static str {
        "l2"
    }
}

/// Compute L2 squared distance, dispatching to the best available SIMD.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We checked for AVX2 support above.
            return unsafe { l2_squared_avx2(a, b) };
        }
    }

    fallback::l2_squared_fallback(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum = _mm256_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let vb = _mm256_loadu_ps(b_ptr.add(offset));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        // Horizontal sum of 8 floats
        let mut result = hsum_avx2(sum);

        // Handle remaining elements
        let tail_start = chunks * 8;
        for i in 0..remainder {
            let d = a[tail_start + i] - b[tail_start + i];
            result += d * d;
        }

        result
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    // v = [a0, a1, a2, a3, a4, a5, a6, a7]
    let hi = _mm256_extractf128_ps(v, 1); // [a4, a5, a6, a7]
    let lo = _mm256_castps256_ps128(v); // [a0, a1, a2, a3]
    let sum128 = _mm_add_ps(lo, hi); // [a0+a4, a1+a5, a2+a6, a3+a7]
    let shuf = _mm_movehdup_ps(sum128); // [a1+a5, a1+a5, a3+a7, a3+a7]
    let sum2 = _mm_add_ps(sum128, shuf); // [a0+a1+a4+a5, -, a2+a3+a6+a7, -]
    let shuf2 = _mm_movehl_ps(sum2, sum2); // [a2+a3+a6+a7, -, -, -]
    let sum_all = _mm_add_ss(sum2, shuf2);
    _mm_cvtss_f32(sum_all)
}
