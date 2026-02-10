//! Core traits, types, and error definitions for v-hnsw.

mod error;
mod traits;
mod types;

pub use error::VhnswError;
pub use traits::{DistanceMetric, PayloadStore, VectorIndex, VectorStore};
pub use types::{Dim, LayerId, Payload, PayloadValue, PointId};

/// Convenience Result alias.
pub type Result<T> = std::result::Result<T, VhnswError>;

/// Cross-platform home directory (`USERPROFILE` on Windows, `HOME` elsewhere).
pub fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(std::path::PathBuf::from)
}

/// Base data directory: `~/.v-hnsw/`.
pub fn data_dir() -> std::path::PathBuf {
    home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".v-hnsw")
}

/// Korean dictionary directory: `~/.v-hnsw/dict/ko-dic/`.
pub fn ko_dic_dir() -> std::path::PathBuf {
    data_dir().join("dict").join("ko-dic")
}

/// Create a [`VhnswError::Storage`] from a string message.
pub fn storage_err(msg: &str) -> VhnswError {
    VhnswError::Storage(std::io::Error::other(msg))
}

/// Read a little-endian `u64` from a byte slice at the given offset.
pub fn read_le_u64(data: &[u8], offset: usize) -> Option<u64> {
    let bytes: [u8; 8] = data.get(offset..offset + 8)?.try_into().ok()?;
    Some(u64::from_le_bytes(bytes))
}
