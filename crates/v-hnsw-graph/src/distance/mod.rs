//! SIMD-optimized distance functions for v-hnsw.
//!
//! Provides L2 (Euclidean), Cosine, and Dot Product distance metrics
//! with runtime SIMD dispatch (AVX2/AVX-512 on x86, NEON on ARM).

pub(crate) mod fallback;
pub(crate) mod l2;
pub(crate) mod cosine;
pub(crate) mod dot;
pub(crate) mod dispatch;
mod prefetch;

/// AVX2 pairwise reduction: load 8-wide chunks, apply SIMD op, hsum, scalar tail.
///
/// `$simd_op(va, vb, sum)` — accumulates a pair of __m256 into the running sum.
/// `$scalar_op(a_i, b_i)` — scalar fallback for remaining elements.
#[cfg(target_arch = "x86_64")]
macro_rules! avx2_reduce {
    ($a:expr, $b:expr, |$va:ident, $vb:ident, $sum:ident| $simd_op:expr, |$ai:ident, $bi:ident| $scalar_op:expr) => {{
        use std::arch::x86_64::*;
        let len = $a.len();
        let chunks = len / 8;
        let remainder = len % 8;
        let mut $sum = _mm256_setzero_ps();
        let a_ptr = $a.as_ptr();
        let b_ptr = $b.as_ptr();
        for i in 0..chunks {
            let offset = i * 8;
            let $va = _mm256_loadu_ps(a_ptr.add(offset));
            let $vb = _mm256_loadu_ps(b_ptr.add(offset));
            $sum = $simd_op;
        }
        let mut result = $crate::distance::dot::hsum_avx2($sum);
        let tail_start = chunks * 8;
        for i in 0..remainder {
            let $ai = $a[tail_start + i];
            let $bi = $b[tail_start + i];
            result += $scalar_op;
        }
        result
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use avx2_reduce;

pub use l2::L2Distance;
pub use cosine::{CosineDistance, NormalizedCosineDistance};
pub use dot::{DotProductDistance, dot_product};
pub use dispatch::AutoDistance;
pub use prefetch::{prefetch_read, prefetch_vector};

#[cfg(test)]
mod tests;

