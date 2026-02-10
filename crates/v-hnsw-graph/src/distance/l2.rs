//! L2 (Euclidean) squared distance with SIMD optimization.
#![allow(unsafe_code)]

use v_hnsw_core::DistanceMetric;

use super::fallback;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use super::dot::hsum_avx2;

/// L2 squared distance metric.
///
/// Computes `sum((a[i] - b[i])^2)` with SIMD acceleration.
#[derive(Debug, Clone, Copy, Default, bincode::Encode, bincode::Decode)]
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
