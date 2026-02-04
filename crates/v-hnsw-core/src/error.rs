//! Error types for v-hnsw.

use crate::types::{Dim, PointId};

/// Top-level error type for v-hnsw operations.
#[derive(thiserror::Error, Debug)]
pub enum VhnswError {
    /// Vector dimension does not match the index configuration.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: Dim, got: Dim },

    /// Requested point was not found in the index.
    #[error("point not found: {0}")]
    PointNotFound(PointId),

    /// The index has reached its maximum capacity.
    #[error("index full: capacity {capacity}")]
    IndexFull { capacity: usize },

    /// An I/O or storage-layer error.
    #[error("storage error: {0}")]
    Storage(#[from] std::io::Error),

    /// A GPU operation failed.
    #[error("gpu error: {0}")]
    Gpu(String),

    /// A tokenizer operation failed.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// A quantization operation failed.
    #[error("quantization error: {0}")]
    Quantization(String),

    /// Payload serialization/deserialization failed.
    #[error("payload error: {0}")]
    Payload(String),

    /// WAL is corrupted or unreadable.
    #[error("wal error: {0}")]
    Wal(String),

    /// Batch operation was incomplete (crash during batch).
    #[error("incomplete batch: {batch_id}")]
    IncompleteBatch { batch_id: u64 },
}
