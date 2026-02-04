//! HNSW graph implementation for v-hnsw.
//!
//! Hierarchical Navigable Small World graph with concurrent insert/search,
//! software prefetch, and delta-encoded neighbor compression.

mod config;
mod delta;
mod graph;
mod insert;
mod node;
mod search;
mod select;
mod store;

pub use config::{HnswConfig, HnswConfigBuilder};
pub use graph::HnswGraph;
pub use store::InMemoryVectorStore;

#[cfg(test)]
mod proptests;
