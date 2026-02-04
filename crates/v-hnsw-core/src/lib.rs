//! Core traits, types, and error definitions for v-hnsw.

mod error;
mod traits;
mod types;

pub use error::VhnswError;
pub use traits::{DistanceMetric, PayloadStore, Quantizer, VectorIndex, VectorStore};
pub use types::{Dim, LayerId, Payload, PayloadValue, PointId};

/// Convenience Result alias.
pub type Result<T> = std::result::Result<T, VhnswError>;
