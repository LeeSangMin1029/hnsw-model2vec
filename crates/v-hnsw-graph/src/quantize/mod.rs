//! Vector quantization for v-hnsw.
//!
//! Supports Scalar Quantization (SQ8) and Product Quantization (PQ)
//! for memory-efficient approximate nearest neighbor search.
//!
//! # Modules
//!
//! - [`sq8`] -- Scalar quantization mapping `f32` to `u8` per dimension.
//! - [`pq`] -- Product quantization with codebook-based compression.
//! - [`rescore`] -- 2-stage search: quantized oversample followed by full-precision rescore.

pub mod pq;
pub mod rescore;
pub mod sq8;

pub use pq::{PqEncoded, PqQuantizer};
pub use rescore::rescore;
pub use sq8::{Sq8Encoded, Sq8Quantizer};

#[cfg(test)]
mod tests;
