//! Persistence layer for v-hnsw.
//!
//! Memory-mapped vector storage with Write-Ahead Log (WAL)
//! for crash recovery. Supports vectors, metadata, and text chunks.
//! Cross-platform (Windows/Mac/Linux).

mod collection;
mod collection_manager;
pub mod daemon_client;
pub mod db_config;
pub mod distance;
mod engine;
mod fsst_text;
mod manifest;
mod mmap_store;
mod payload_store;
pub mod sq8;
pub mod sq8_store;
#[cfg(test)]
mod tests;
mod wal;

pub use collection::Collection;
pub use collection_manager::CollectionManager;
pub use db_config::DbConfig;
pub use distance::{F32Dc, Sq8Dc, Sq8LutDc};
pub use engine::{StorageConfig, StorageEngine};
pub use fsst_text::compress_texts;
pub use manifest::{CollectionInfo, Manifest, MANIFEST_VERSION};
pub use mmap_store::MmapVectorStore;
pub use payload_store::FilePayloadStore;
pub use wal::{Wal, WalRecord};
