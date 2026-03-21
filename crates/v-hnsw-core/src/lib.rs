//! Core traits, types, and error definitions for v-hnsw.

mod error;
pub mod interrupt;
mod traits;
mod types;

#[cfg(test)]
mod tests;

pub use error::VhnswError;
pub use traits::{DistanceMetric, PayloadStore, VectorIndex, VectorStore};
pub use types::{check_dimension, Dim, LayerId, Payload, PayloadValue, PointId};

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

/// Platform-aware cache directory for v-hnsw.
pub fn cache_dir() -> std::path::PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            return std::path::PathBuf::from(local).join("v-hnsw").join("cache");
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(cache) = std::env::var("XDG_CACHE_HOME") {
            return std::path::PathBuf::from(cache).join("v-hnsw");
        }
        if let Ok(home) = std::env::var("HOME") {
            return std::path::PathBuf::from(home).join(".cache").join("v-hnsw");
        }
    }
    std::env::temp_dir().join("v-hnsw")
}

/// Korean dictionary directory: `~/.v-hnsw/dict/ko-dic/`.
pub fn ko_dic_dir() -> std::path::PathBuf {
    data_dir().join("dict").join("ko-dic")
}

/// Create a [`VhnswError::Storage`] from a string message.
pub fn storage_err(msg: &str) -> VhnswError {
    VhnswError::Storage(std::io::Error::other(msg))
}

/// Strip Windows extended-length prefix (`\\?\` or `//?/`).
///
/// `canonicalize()` on Windows adds this prefix, which breaks `git ls-files`,
/// shell commands, and path comparison.
pub fn strip_unc_prefix(path: &str) -> &str {
    path.strip_prefix(r"\\?\")
        .or_else(|| path.strip_prefix("//?/"))
        .unwrap_or(path)
}

/// Like [`strip_unc_prefix`] but returns a `PathBuf`.
pub fn strip_unc_prefix_path(path: &std::path::Path) -> std::path::PathBuf {
    std::path::PathBuf::from(strip_unc_prefix(&path.to_string_lossy()))
}

/// Read a little-endian `u64` from a byte slice at the given offset.
pub fn read_le_u64(data: &[u8], offset: usize) -> Option<u64> {
    let bytes: [u8; 8] = data.get(offset..offset + 8)?.try_into().ok()?;
    Some(u64::from_le_bytes(bytes))
}
