//! GPU acceleration for v-hnsw via CubeCL.
//!
//! Supports CUDA (NVIDIA) and wgpu (Vulkan/Metal/DX12) backends
//! with automatic runtime selection and a pure-CPU fallback.
//!
//! # Quick start
//!
//! ```rust
//! use v_hnsw_gpu::{GpuContext, GpuDistance};
//!
//! let ctx = GpuContext::auto();
//! let query = &[1.0_f32, 2.0, 3.0];
//! let database = &[4.0_f32, 5.0, 6.0, 7.0, 8.0, 9.0]; // 2 vectors of dim 3
//! let distances = ctx.batch_l2_squared(query, database, 3).unwrap();
//! assert_eq!(distances.len(), 2);
//! ```

pub mod backend;
pub(crate) mod error;
pub(crate) mod kernels;

pub use backend::{GpuBackend, GpuContext};

// ---------------------------------------------------------------------------
// Core trait
// ---------------------------------------------------------------------------

/// GPU-accelerated batch distance computation.
///
/// Implementations are free to choose a GPU backend (CUDA, wgpu) or fall
/// back to CPU.  The trait is object-safe so callers can use `dyn GpuDistance`.
pub trait GpuDistance: Send + Sync {
    /// Compute L2 squared distances from one query vector to multiple
    /// database vectors.
    ///
    /// `database` is a **flattened** row-major matrix of shape `[n, dim]`.
    /// Returns a `Vec<f32>` with one distance per database vector.
    fn batch_l2_squared(
        &self,
        query: &[f32],
        database: &[f32],
        dim: usize,
    ) -> v_hnsw_core::Result<Vec<f32>>;

    /// Whether the underlying GPU device is reachable.
    fn is_available(&self) -> bool;

    /// Human-readable name of the active backend (e.g. `"cuda"`, `"wgpu"`, `"cpu"`).
    fn backend_name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Integration tests (CPU-only, no GPU required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_context_single() {
        let ctx = GpuContext::with_backend(GpuBackend::Cpu);
        let query = &[1.0_f32, 2.0, 3.0];
        let db = &[4.0_f32, 5.0, 6.0];
        let dists = ctx.batch_l2_squared(query, db, 3);
        assert!(dists.is_ok());
        let dists = dists.expect("should succeed");
        assert_eq!(dists.len(), 1);
        // (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
        assert!((dists[0] - 27.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_context_multiple() {
        let ctx = GpuContext::with_backend(GpuBackend::Cpu);
        let dim = 4;
        let n = 10;
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5).collect();
        let database: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.1).collect();

        let dists = ctx
            .batch_l2_squared(&query, &database, dim)
            .expect("should succeed");
        assert_eq!(dists.len(), n);

        // Verify against naive computation.
        for i in 0..n {
            let start = i * dim;
            let expected: f32 = (0..dim)
                .map(|d| {
                    let diff = query[d] - database[start + d];
                    diff * diff
                })
                .sum();
            assert!(
                (dists[i] - expected).abs() < 1e-5,
                "vec {i}: got {}, expected {}",
                dists[i],
                expected
            );
        }
    }

    #[test]
    fn test_empty_database() {
        let ctx = GpuContext::with_backend(GpuBackend::Cpu);
        let query = &[1.0_f32, 2.0, 3.0];
        let dists = ctx
            .batch_l2_squared(query, &[], 3)
            .expect("should succeed");
        assert!(dists.is_empty());
    }

    #[test]
    fn test_is_available() {
        let ctx = GpuContext::auto();
        assert!(ctx.is_available());
    }

    #[test]
    fn test_dimension_mismatch() {
        let ctx = GpuContext::with_backend(GpuBackend::Cpu);
        // query has 2 elements but dim is 3
        let result = ctx.batch_l2_squared(&[1.0, 2.0], &[3.0, 4.0, 5.0], 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_dim_error() {
        let ctx = GpuContext::with_backend(GpuBackend::Cpu);
        let result = ctx.batch_l2_squared(&[], &[], 0);
        assert!(result.is_err());
    }
}
