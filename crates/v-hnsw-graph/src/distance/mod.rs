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

pub use l2::L2Distance;
pub use cosine::{CosineDistance, NormalizedCosineDistance};
pub use dot::{DotProductDistance, dot_product};
pub use dispatch::AutoDistance;
pub use prefetch::{prefetch_read, prefetch_vector};

#[cfg(test)]
mod tests;

