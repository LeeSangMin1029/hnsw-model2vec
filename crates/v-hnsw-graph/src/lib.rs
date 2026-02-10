//! HNSW graph implementation for v-hnsw.
//!
//! Hierarchical Navigable Small World graph with concurrent insert/search,
//! software prefetch, and delta-encoded neighbor compression.

pub mod distance;

mod config;
mod delta;
mod graph;
mod insert;
mod node;
mod search;
mod select;
mod snapshot;
mod store;

pub use config::{HnswConfig, HnswConfigBuilder};
pub use distance::{AutoDistance, CosineDistance, DotProductDistance, L2Distance, NormalizedCosineDistance};
pub use graph::HnswGraph;
pub use search::NodeGraph;
pub use snapshot::HnswSnapshot;
pub use store::InMemoryVectorStore;

#[cfg(test)]
mod proptests;
