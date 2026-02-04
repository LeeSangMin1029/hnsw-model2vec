//! GPU-specific error helpers.
//!
//! Maps GPU backend errors into [`v_hnsw_core::VhnswError::Gpu`].

use v_hnsw_core::VhnswError;

/// Create a [`VhnswError::Gpu`] from any displayable error.
#[cfg_attr(
    not(any(feature = "cuda", feature = "wgpu")),
    allow(dead_code)
)]
pub(crate) fn gpu_err(msg: impl std::fmt::Display) -> VhnswError {
    VhnswError::Gpu(msg.to_string())
}
