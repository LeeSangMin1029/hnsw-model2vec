//! Integration tests for the storage engine.

#![allow(clippy::unwrap_used)] // Tests can use unwrap

use std::collections::HashMap;
use std::path::PathBuf;

use v_hnsw_core::{Payload, PayloadStore, PayloadValue, VectorStore};

use crate::engine::{StorageConfig, StorageEngine};

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

#[test]
fn test_create_and_insert() {
    let dir = test_dir("create_insert");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 128,
        initial_capacity: 100,
        checkpoint_threshold: 10,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    let vector = test_vector(128, 1.0);
    let payload = test_payload("doc1.md", 0, 1);

    engine.insert(1, &vector, payload, "Hello, world!").unwrap();

    assert_eq!(engine.len(), 1);
    assert!(!engine.is_empty());

    cleanup(&dir);
}

#[test]
fn test_insert_and_get_vector() {
    let dir = test_dir("get_vector");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 64,
        initial_capacity: 100,
        checkpoint_threshold: 100,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    let vector = test_vector(64, 2.5);
    let payload = test_payload("doc2.md", 0, 1);

    engine.insert(42, &vector, payload, "Test text").unwrap();

    // Get vector via vector_store
    let retrieved = engine.vector_store().get(42).unwrap();
    assert_eq!(retrieved.len(), 64);
    assert!((retrieved[0] - 2.5).abs() < 0.001);
    assert!((retrieved[1] - 2.51).abs() < 0.001);

    cleanup(&dir);
}

#[test]
fn test_insert_and_get_payload() {
    let dir = test_dir("get_payload");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 32,
        initial_capacity: 100,
        checkpoint_threshold: 100,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    let vector = test_vector(32, 1.0);
    let mut payload = test_payload("notes/meeting.md", 2, 5);
    payload.tags = vec!["meeting".to_string(), "important".to_string()];
    payload.custom.insert("priority".to_string(), PayloadValue::Integer(1));

    engine.insert(100, &vector, payload.clone(), "Meeting notes here").unwrap();

    // Flush to make payload available via file read
    engine.checkpoint().unwrap();

    // Get payload via payload_store
    let retrieved = engine.payload_store().get_payload(100).unwrap().unwrap();
    assert_eq!(retrieved.source, "notes/meeting.md");
    assert_eq!(retrieved.chunk_index, 2);
    assert_eq!(retrieved.chunk_total, 5);
    assert_eq!(retrieved.tags.len(), 2);

    cleanup(&dir);
}

#[test]
fn test_insert_and_get_text() {
    let dir = test_dir("get_text");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 16,
        initial_capacity: 100,
        checkpoint_threshold: 100,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    let vector = test_vector(16, 0.5);
    let payload = test_payload("readme.md", 0, 1);
    let text = "This is a longer piece of text that represents the actual content chunk.";

    engine.insert(999, &vector, payload, text).unwrap();
    engine.checkpoint().unwrap();

    // Get text via payload_store
    let retrieved = engine.payload_store().get_text(999).unwrap().unwrap();
    assert_eq!(retrieved, text);

    cleanup(&dir);
}

#[test]
fn test_remove() {
    let dir = test_dir("remove");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 32,
        initial_capacity: 100,
        checkpoint_threshold: 100,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    // Insert two points
    engine.insert(1, &test_vector(32, 1.0), test_payload("a.md", 0, 1), "A").unwrap();
    engine.insert(2, &test_vector(32, 2.0), test_payload("b.md", 0, 1), "B").unwrap();

    assert_eq!(engine.len(), 2);

    // Remove one
    engine.remove(1).unwrap();

    assert_eq!(engine.len(), 1);

    // Verify point 1 is gone but point 2 remains
    assert!(engine.vector_store().get(1).is_err());
    assert!(engine.vector_store().get(2).is_ok());

    cleanup(&dir);
}

#[test]
fn test_checkpoint_and_reopen() {
    let dir = test_dir("checkpoint_reopen");
    cleanup(&dir);

    // Create and insert
    {
        let config = StorageConfig {
            dim: 64,
            initial_capacity: 100,
            checkpoint_threshold: 100,
        };

        let mut engine = StorageEngine::create(&dir, config).unwrap();

        for i in 0..5 {
            let vector = test_vector(64, i as f32);
            let payload = test_payload(&format!("doc{i}.md"), 0, 1);
            engine.insert(i, &vector, payload, &format!("Content {i}")).unwrap();
        }

        // Explicitly checkpoint before closing
        engine.checkpoint().unwrap();
    }

    // Reopen and verify
    {
        let engine = StorageEngine::open(&dir).unwrap();

        assert_eq!(engine.len(), 5);

        // Verify vectors
        for i in 0..5 {
            let v = engine.vector_store().get(i).unwrap();
            assert!((v[0] - i as f32).abs() < 0.001);
        }

        // Verify payload
        let p = engine.payload_store().get_payload(3).unwrap().unwrap();
        assert_eq!(p.source, "doc3.md");

        // Verify text
        let t = engine.payload_store().get_text(2).unwrap().unwrap();
        assert_eq!(t, "Content 2");
    }

    cleanup(&dir);
}

#[test]
fn test_wal_recovery() {
    let dir = test_dir("wal_recovery");
    cleanup(&dir);

    // Create, checkpoint some data, then add more WITHOUT checkpoint
    {
        let config = StorageConfig {
            dim: 32,
            initial_capacity: 100,
            checkpoint_threshold: 1000, // High threshold to prevent auto-checkpoint
        };

        let mut engine = StorageEngine::create(&dir, config).unwrap();

        // Insert and checkpoint first batch
        for i in 0..2 {
            let vector = test_vector(32, i as f32);
            let payload = test_payload(&format!("stable{i}.md"), 0, 1);
            engine.insert(i, &vector, payload, &format!("Stable {i}")).unwrap();
        }
        engine.checkpoint().unwrap();

        // Insert more WITHOUT checkpoint - these should be recovered from WAL
        for i in 0..3 {
            let vector = test_vector(32, (i + 100) as f32);
            let payload = test_payload(&format!("wal{i}.md"), 0, 1);
            engine.insert(i + 100, &vector, payload, &format!("WAL test {i}")).unwrap();
        }

        // Intentionally NOT calling checkpoint() - WAL should recover these
    }

    // Reopen - should have 2 (checkpointed) + 3 (WAL recovered) = 5
    {
        let engine = StorageEngine::open(&dir).unwrap();

        assert_eq!(engine.len(), 5);

        // Verify checkpointed data
        let v = engine.vector_store().get(0).unwrap();
        assert!((v[0] - 0.0).abs() < 0.001);

        // Verify WAL-recovered data
        let v = engine.vector_store().get(101).unwrap();
        assert!((v[0] - 101.0).abs() < 0.001);

        // After recovery, WAL-recovered payload should be accessible
        let p = engine.payload_store().get_payload(102);
        assert!(p.is_ok());
    }

    cleanup(&dir);
}

#[test]
fn test_replace_source() {
    let dir = test_dir("replace_source");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 16,
        initial_capacity: 100,
        checkpoint_threshold: 100,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    // Insert 3 chunks from doc1.md
    for i in 0u64..3 {
        let vector = test_vector(16, i as f32);
        let payload = test_payload("doc1.md", i as u32, 3);
        engine.insert(i, &vector, payload, &format!("Old chunk {i}")).unwrap();
    }

    // Insert 2 chunks from doc2.md (should remain untouched)
    for i in 0u64..2 {
        let vector = test_vector(16, (i + 10) as f32);
        let payload = test_payload("doc2.md", i as u32, 2);
        engine.insert(i + 100, &vector, payload, &format!("Doc2 chunk {i}")).unwrap();
    }

    engine.checkpoint().unwrap();
    assert_eq!(engine.len(), 5);

    // Replace doc1.md with 2 new chunks
    let new_chunks = vec![
        (50, test_vector(16, 50.0), test_payload("doc1.md", 0, 2), "New chunk 0".to_string()),
        (51, test_vector(16, 51.0), test_payload("doc1.md", 1, 2), "New chunk 1".to_string()),
    ];

    engine.replace_source("doc1.md", new_chunks).unwrap();
    engine.checkpoint().unwrap();

    // Old doc1.md chunks (0, 1, 2) should be gone
    assert!(engine.vector_store().get(0).is_err());
    assert!(engine.vector_store().get(1).is_err());
    assert!(engine.vector_store().get(2).is_err());

    // New doc1.md chunks should exist
    assert!(engine.vector_store().get(50).is_ok());
    assert!(engine.vector_store().get(51).is_ok());

    // doc2.md chunks should be untouched
    assert!(engine.vector_store().get(100).is_ok());
    assert!(engine.vector_store().get(101).is_ok());

    // Total: 2 (new doc1) + 2 (doc2) = 4
    assert_eq!(engine.len(), 4);

    cleanup(&dir);
}

#[test]
fn test_auto_checkpoint() {
    let dir = test_dir("auto_checkpoint");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 8,
        initial_capacity: 100,
        checkpoint_threshold: 3, // Low threshold for testing
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    // Insert 4 points - should trigger auto-checkpoint after 3
    for i in 0..4 {
        let vector = test_vector(8, i as f32);
        let payload = test_payload(&format!("auto{i}.md"), 0, 1);
        engine.insert(i, &vector, payload, &format!("Auto {i}")).unwrap();
    }

    // The 4th insert should have triggered checkpoint, resetting counter
    // Insert one more - should not trigger checkpoint yet
    engine.insert(10, &test_vector(8, 10.0), test_payload("extra.md", 0, 1), "Extra").unwrap();

    assert_eq!(engine.len(), 5);

    cleanup(&dir);
}

#[test]
fn test_multiple_sources_source_index() {
    let dir = test_dir("multi_source");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 8,
        initial_capacity: 100,
        checkpoint_threshold: 100,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    // Insert chunks from multiple sources
    let sources = ["alpha.md", "beta.md", "gamma.md"];
    let mut id = 0;
    for source in &sources {
        for chunk in 0..3 {
            let vector = test_vector(8, id as f32);
            let payload = test_payload(source, chunk, 3);
            engine.insert(id, &vector, payload, &format!("{source} chunk {chunk}")).unwrap();
            id += 1;
        }
    }

    engine.checkpoint().unwrap();
    assert_eq!(engine.len(), 9);

    // Verify source index works
    let beta_ids = engine.payload_store().points_by_source("beta.md");
    assert_eq!(beta_ids.len(), 3);

    // Replace only beta.md
    let new_beta = vec![
        (100, test_vector(8, 100.0), test_payload("beta.md", 0, 1), "New beta".to_string()),
    ];
    engine.replace_source("beta.md", new_beta).unwrap();
    engine.checkpoint().unwrap();

    // alpha (3) + gamma (3) + new beta (1) = 7
    assert_eq!(engine.len(), 7);

    // Verify alpha and gamma untouched
    let alpha_ids = engine.payload_store().points_by_source("alpha.md");
    assert_eq!(alpha_ids.len(), 3);

    let gamma_ids = engine.payload_store().points_by_source("gamma.md");
    assert_eq!(gamma_ids.len(), 3);

    cleanup(&dir);
}
