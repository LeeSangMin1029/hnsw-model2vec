//! v-daemon client utilities — re-exports from v-hnsw-storage.
//!
//! The actual implementation lives in `v_hnsw_storage::daemon_client`
//! to avoid circular dependencies (v-hnsw-cli ↔ v-daemon).

pub use v_hnsw_storage::daemon_client::*;
