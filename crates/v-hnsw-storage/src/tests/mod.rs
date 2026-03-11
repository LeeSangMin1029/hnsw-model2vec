//! Integration tests for the storage engine.

#![allow(clippy::unwrap_used)] // Tests can use unwrap

mod collection;
mod collection_manager;
mod engine;
mod fsst_text;
mod manifest;
mod mmap_store;
mod payload_store;
mod sq8;
mod sq8_store;
mod wal;

use std::collections::HashMap;
use std::path::PathBuf;

use v_hnsw_core::Payload;

/// Get a unique test directory path.
fn test_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "v-hnsw-test-{}-{}-{}",
        name,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

/// Clean up test directory.
fn cleanup(dir: &PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}

/// Create a test payload.
fn test_payload(source: &str, chunk_idx: u32, chunk_total: u32) -> Payload {
    Payload {
        source: source.to_string(),
        tags: vec!["test".to_string()],
        created_at: 1700000000,
        source_modified_at: 1700000000,
        chunk_index: chunk_idx,
        chunk_total,
        custom: HashMap::new(),
    }
}

/// Create a test vector with predictable values.
fn test_vector(dim: usize, seed: f32) -> Vec<f32> {
    (0..dim).map(|i| seed + i as f32 * 0.01).collect()
}
