//! Persistence layer for v-hnsw.
//!
//! Memory-mapped vector storage with Write-Ahead Log (WAL)
//! for crash recovery. Supports vectors, metadata, and text chunks.
//! Cross-platform (Windows/Mac/Linux).

mod engine;
mod mmap_store;
mod payload_store;
#[cfg(test)]
mod tests;
mod wal;

pub use engine::{StorageConfig, StorageEngine};
pub use mmap_store::MmapVectorStore;
pub use payload_store::FilePayloadStore;
pub use wal::{Wal, WalRecord};
