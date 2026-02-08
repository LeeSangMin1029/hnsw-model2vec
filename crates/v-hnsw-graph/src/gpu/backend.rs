//! GPU backend detection and dispatch.
//!
//! [`GpuContext`] wraps the selected compute backend and implements
//! [`GpuDistance`] by dispatching to CubeCL kernels (CUDA / wgpu) or
//! a pure-CPU fallback.

use super::GpuDistance;
#[cfg(feature = "gpu")]
use super::kernels::distance;

// ---------------------------------------------------------------------------
// Backend enum
// ---------------------------------------------------------------------------

/// Available GPU compute backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// NVIDIA CUDA (requires `cuda` feature).
    Cuda,
    /// wgpu / Vulkan / Metal / DX12 (requires `wgpu` feature).
    Wgpu,
    /// Pure-CPU fallback using SIMD-optimized `distance` module.
    Cpu,
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda => write!(f, "cuda"),
            Self::Wgpu => write!(f, "wgpu"),
            Self::Cpu => write!(f, "cpu"),
        }
    }
}

// ---------------------------------------------------------------------------
// GpuContext
// ---------------------------------------------------------------------------

/// GPU compute context that owns the selected backend.
///
/// Use [`GpuContext::auto`] to let the library pick the best available
/// backend, or [`GpuContext::with_backend`] to force a specific one.
#[derive(Debug)]
pub struct GpuContext {
    backend: GpuBackend,
}

impl GpuContext {
    /// Automatically detect the best available backend.
    ///
    /// Priority: CUDA > wgpu > CPU.
    #[must_use]
    pub fn auto() -> Self {
        let backend = Self::detect_backend();
        tracing::info!(backend = %backend, "GPU backend selected");
        Self { backend }
    }

    /// Create a context targeting a specific backend.
    ///
    /// If the requested backend is not compiled-in, falls back to CPU.
    #[must_use]
    pub fn with_backend(backend: GpuBackend) -> Self {
        let actual = match backend {
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda => GpuBackend::Cuda,
            #[cfg(not(feature = "cuda"))]
            GpuBackend::Cuda => {
                tracing::warn!("CUDA requested but `cuda` feature not enabled; falling back to CPU");
                GpuBackend::Cpu
            }
            #[cfg(feature = "wgpu")]
            GpuBackend::Wgpu => GpuBackend::Wgpu,
            #[cfg(not(feature = "wgpu"))]
            GpuBackend::Wgpu => {
                tracing::warn!("wgpu requested but `wgpu` feature not enabled; falling back to CPU");
                GpuBackend::Cpu
            }
            GpuBackend::Cpu => GpuBackend::Cpu,
        };

        tracing::info!(requested = %backend, actual = %actual, "GPU backend configured");
        Self { backend: actual }
    }

    /// Return the active backend.
    #[must_use]
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    // -- private helpers -----------------------------------------------------

    fn detect_backend() -> GpuBackend {
        #[cfg(feature = "cuda")]
        {
            return GpuBackend::Cuda;
        }

        #[cfg(feature = "wgpu")]
        {
            return GpuBackend::Wgpu;
        }

        #[allow(unreachable_code)]
        GpuBackend::Cpu
    }
}

// ---------------------------------------------------------------------------
// GpuDistance impl
// ---------------------------------------------------------------------------

impl GpuDistance for GpuContext {
    fn batch_l2_squared(
        &self,
        query: &[f32],
        database: &[f32],
        dim: usize,
    ) -> v_hnsw_core::Result<Vec<f32>> {
        // Validate inputs.
        if dim == 0 {
            return Err(v_hnsw_core::VhnswError::Gpu(
                "dimension must be > 0".into(),
            ));
        }
        if query.len() != dim {
            return Err(v_hnsw_core::VhnswError::Gpu(format!(
                "query length {} does not match dim {}",
                query.len(),
                dim,
            )));
        }
        if !database.len().is_multiple_of(dim) {
            return Err(v_hnsw_core::VhnswError::Gpu(format!(
                "database length {} is not a multiple of dim {}",
                database.len(),
                dim,
            )));
        }

        let n_vectors = database.len() / dim;

        match self.backend {
            GpuBackend::Cuda => self.dispatch_cuda(query, database, n_vectors, dim),
            GpuBackend::Wgpu => self.dispatch_wgpu(query, database, n_vectors, dim),
            GpuBackend::Cpu => Ok(super::cpu_batch_l2_squared(
                query, database, n_vectors, dim,
            )),
        }
    }

    fn is_available(&self) -> bool {
        true
    }

    fn backend_name(&self) -> &str {
        match self.backend {
            GpuBackend::Cuda => "cuda",
            GpuBackend::Wgpu => "wgpu",
            GpuBackend::Cpu => "cpu",
        }
    }
}

// ---------------------------------------------------------------------------
// Backend-specific dispatch
// ---------------------------------------------------------------------------

impl GpuContext {
    #[cfg(feature = "cuda")]
    fn dispatch_cuda(
        &self,
        query: &[f32],
        database: &[f32],
        n_vectors: usize,
        dim: usize,
    ) -> v_hnsw_core::Result<Vec<f32>> {
        use cubecl::cuda::CudaRuntime;
        use cubecl::prelude::Runtime as _;

        let device = <CudaRuntime as cubecl::prelude::Runtime>::Device::default();
        let client = CudaRuntime::client(&device);
        distance::launch_batch_l2_squared::<CudaRuntime>(
            &client, query, database, n_vectors, dim,
        )
    }

    #[cfg(not(feature = "cuda"))]
    fn dispatch_cuda(
        &self,
        _query: &[f32],
        _database: &[f32],
        _n_vectors: usize,
        _dim: usize,
    ) -> v_hnsw_core::Result<Vec<f32>> {
        Err(v_hnsw_core::VhnswError::Gpu(
            "CUDA backend not compiled (enable `cuda` feature)".into(),
        ))
    }

    #[cfg(feature = "wgpu")]
    fn dispatch_wgpu(
        &self,
        query: &[f32],
        database: &[f32],
        n_vectors: usize,
        dim: usize,
    ) -> v_hnsw_core::Result<Vec<f32>> {
        use cubecl::wgpu::WgpuRuntime;
        use cubecl::prelude::Runtime as _;

        let device = <WgpuRuntime as cubecl::prelude::Runtime>::Device::default();
        let client = WgpuRuntime::client(&device);
        distance::launch_batch_l2_squared::<WgpuRuntime>(
            &client, query, database, n_vectors, dim,
        )
    }

    #[cfg(not(feature = "wgpu"))]
    fn dispatch_wgpu(
        &self,
        _query: &[f32],
        _database: &[f32],
        _n_vectors: usize,
        _dim: usize,
    ) -> v_hnsw_core::Result<Vec<f32>> {
        Err(v_hnsw_core::VhnswError::Gpu(
            "wgpu backend not compiled (enable `wgpu` feature)".into(),
        ))
    }
}
