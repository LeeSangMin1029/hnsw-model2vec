//! Persistence layer for v-hnsw.
//!
//! Memory-mapped vector storage with Write-Ahead Log (WAL)
//! for crash recovery. Supports vectors, metadata, and text chunks.
//! Cross-platform (Windows/Mac/Linux).

mod collection;
mod collection_manager;
mod engine;
mod manifest;
mod mmap_store;
mod payload_store;
#[cfg(test)]
mod tests;
mod wal;

pub use collection::Collection;
pub use collection_manager::CollectionManager;
pub use engine::{StorageConfig, StorageEngine};
pub use manifest::{CollectionInfo, Manifest, MANIFEST_VERSION};
pub use mmap_store::MmapVectorStore;
pub use payload_store::FilePayloadStore;
pub use wal::{Wal, WalRecord};
