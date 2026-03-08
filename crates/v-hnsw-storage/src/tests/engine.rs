//! Engine-level integration tests.

use v_hnsw_core::{Payload, PayloadStore, PayloadValue, VectorStore};

use crate::engine::{StorageConfig, StorageEngine};

use super::{cleanup, test_dir, test_payload, test_vector};

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

#[test]
fn test_engine_open_nonexistent_dir_fails() {
    let result = StorageEngine::open("/tmp/v_hnsw_test_nonexistent_engine_999999");
    assert!(result.is_err());
}

#[test]
fn test_engine_empty_store() {
    let dir = test_dir("empty_store");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 32,
        initial_capacity: 10,
        checkpoint_threshold: 100,
    };

    let engine = StorageEngine::create(&dir, config).unwrap();
    assert_eq!(engine.len(), 0);
    assert!(engine.is_empty());

    cleanup(&dir);
}

#[test]
fn test_engine_insert_batch() {
    let dir = test_dir("insert_batch");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 8,
        initial_capacity: 100,
        checkpoint_threshold: 1000,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    // Need owned vectors for lifetime
    let v1 = test_vector(8, 1.0);
    let v2 = test_vector(8, 2.0);
    let v3 = test_vector(8, 3.0);
    let batch: Vec<(u64, &[f32], Payload, &str)> = vec![
        (1, &v1, test_payload("a.md", 0, 1), "text A"),
        (2, &v2, test_payload("b.md", 0, 1), "text B"),
        (3, &v3, test_payload("c.md", 0, 1), "text C"),
    ];

    engine.insert_batch(&batch).unwrap();
    assert_eq!(engine.len(), 3);

    // Verify vectors
    let v = engine.vector_store().get(2).unwrap();
    assert!((v[0] - 2.0).abs() < 0.001);

    cleanup(&dir);
}

#[test]
fn test_engine_insert_batch_empty() {
    let dir = test_dir("insert_batch_empty");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 8,
        initial_capacity: 10,
        checkpoint_threshold: 100,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();
    engine.insert_batch(&[]).unwrap();
    assert_eq!(engine.len(), 0);

    cleanup(&dir);
}

#[test]
fn test_engine_remove_nonexistent_fails() {
    let dir = test_dir("remove_nonexist");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 8,
        initial_capacity: 10,
        checkpoint_threshold: 100,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();
    let result = engine.remove(999);
    assert!(result.is_err());

    cleanup(&dir);
}

#[test]
fn test_engine_insert_same_id_updates() {
    let dir = test_dir("insert_update");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 4,
        initial_capacity: 10,
        checkpoint_threshold: 100,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    let v1 = vec![1.0, 2.0, 3.0, 4.0];
    let v2 = vec![10.0, 20.0, 30.0, 40.0];

    engine.insert(1, &v1, test_payload("a.md", 0, 1), "first").unwrap();
    engine.insert(1, &v2, test_payload("a.md", 0, 1), "second").unwrap();

    // Vector should be updated (same slot reused)
    let retrieved = engine.vector_store().get(1).unwrap();
    assert!((retrieved[0] - 10.0).abs() < 0.001);

    // Count should still be 1 (update, not new insert)
    assert_eq!(engine.len(), 1);

    cleanup(&dir);
}

#[test]
fn test_engine_checkpoint_frequency_with_low_threshold() {
    let dir = test_dir("ckpt_freq");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 4,
        initial_capacity: 100,
        checkpoint_threshold: 2,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    // Insert 5 points with threshold=2: should auto-checkpoint at 2 and 4
    for i in 0..5 {
        engine
            .insert(
                i,
                &test_vector(4, i as f32),
                test_payload(&format!("d{i}.md"), 0, 1),
                &format!("t{i}"),
            )
            .unwrap();
    }

    assert_eq!(engine.len(), 5);

    // After auto-checkpoints, data should be recoverable on reopen
    drop(engine);
    let engine = StorageEngine::open(&dir).unwrap();
    assert_eq!(engine.len(), 5);

    cleanup(&dir);
}

#[test]
fn test_engine_wal_recovery_with_removes() {
    let dir = test_dir("wal_recovery_remove");
    cleanup(&dir);

    {
        let config = StorageConfig {
            dim: 8,
            initial_capacity: 100,
            checkpoint_threshold: 1000,
        };

        let mut engine = StorageEngine::create(&dir, config).unwrap();

        // Insert 3, checkpoint
        for i in 0..3 {
            engine
                .insert(i, &test_vector(8, i as f32), test_payload("x.md", 0, 1), "x")
                .unwrap();
        }
        engine.checkpoint().unwrap();

        // Insert 2 more, remove 1 - WITHOUT checkpoint
        engine
            .insert(10, &test_vector(8, 10.0), test_payload("y.md", 0, 1), "y")
            .unwrap();
        engine
            .insert(11, &test_vector(8, 11.0), test_payload("z.md", 0, 1), "z")
            .unwrap();
        engine.remove(10).unwrap();

        // Drop without checkpoint
    }

    // Reopen: should have 3 (checkpointed) + 1 (WAL: inserted 10,11, removed 10) = 4
    let engine = StorageEngine::open(&dir).unwrap();
    assert_eq!(engine.len(), 4);

    // id 10 was removed
    assert!(engine.vector_store().get(10).is_err());
    // id 11 should exist
    assert!(engine.vector_store().get(11).is_ok());

    cleanup(&dir);
}

#[test]
fn test_engine_replace_source_empty_new_chunks() {
    let dir = test_dir("replace_empty");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 4,
        initial_capacity: 100,
        checkpoint_threshold: 100,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    // Insert 2 chunks from doc.md
    engine
        .insert(1, &test_vector(4, 1.0), test_payload("doc.md", 0, 2), "chunk0")
        .unwrap();
    engine
        .insert(2, &test_vector(4, 2.0), test_payload("doc.md", 1, 2), "chunk1")
        .unwrap();
    engine.checkpoint().unwrap();
    assert_eq!(engine.len(), 2);

    // Replace with empty (effectively delete all chunks)
    engine.replace_source("doc.md", vec![]).unwrap();
    engine.checkpoint().unwrap();

    assert_eq!(engine.len(), 0);

    cleanup(&dir);
}

#[test]
fn test_engine_large_vector_dimension() {
    let dir = test_dir("large_dim");
    cleanup(&dir);

    let dim = 1536; // OpenAI ada-002 dimension
    let config = StorageConfig {
        dim,
        initial_capacity: 10,
        checkpoint_threshold: 100,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    let vector = test_vector(dim, 0.5);
    engine
        .insert(1, &vector, test_payload("large.md", 0, 1), "large dim test")
        .unwrap();

    let retrieved = engine.vector_store().get(1).unwrap();
    assert_eq!(retrieved.len(), dim);
    assert!((retrieved[0] - 0.5).abs() < 0.001);

    cleanup(&dir);
}

#[test]
fn test_engine_dimension_mismatch_fails() {
    let dir = test_dir("dim_mismatch");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 8,
        initial_capacity: 10,
        checkpoint_threshold: 100,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    // Try inserting a vector with wrong dimension
    let wrong_vector = test_vector(16, 1.0);
    let result = engine.insert(1, &wrong_vector, test_payload("a.md", 0, 1), "text");
    assert!(result.is_err());

    cleanup(&dir);
}

#[test]
fn test_engine_multiple_checkpoints_and_reopen() {
    let dir = test_dir("multi_ckpt_reopen");
    cleanup(&dir);

    {
        let config = StorageConfig {
            dim: 4,
            initial_capacity: 100,
            checkpoint_threshold: 1000,
        };

        let mut engine = StorageEngine::create(&dir, config).unwrap();

        // Checkpoint 1: insert 3
        for i in 0..3 {
            engine
                .insert(i, &test_vector(4, i as f32), test_payload("a.md", 0, 1), "a")
                .unwrap();
        }
        engine.checkpoint().unwrap();

        // Checkpoint 2: insert 3 more
        for i in 10..13 {
            engine
                .insert(i, &test_vector(4, i as f32), test_payload("b.md", 0, 1), "b")
                .unwrap();
        }
        engine.checkpoint().unwrap();

        // Un-checkpointed: insert 2 more
        engine
            .insert(100, &test_vector(4, 100.0), test_payload("c.md", 0, 1), "c")
            .unwrap();
        engine
            .insert(101, &test_vector(4, 101.0), test_payload("c.md", 0, 1), "c")
            .unwrap();
    }

    let engine = StorageEngine::open(&dir).unwrap();
    // 3 + 3 + 2 = 8
    assert_eq!(engine.len(), 8);

    cleanup(&dir);
}

// --- New tests ---

#[test]
fn test_vector_retrieval_after_checkpoint() {
    let dir = test_dir("vec_after_ckpt");
    cleanup(&dir);

    let dim = 16;
    let config = StorageConfig {
        dim,
        initial_capacity: 100,
        checkpoint_threshold: 1000,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    // Insert several vectors with distinct values
    for i in 0u64..10 {
        let vector = test_vector(dim, i as f32 * 0.1);
        let payload = test_payload(&format!("src{i}.md"), 0, 1);
        engine.insert(i, &vector, payload, &format!("text {i}")).unwrap();
    }

    engine.checkpoint().unwrap();

    // Verify every vector is still retrievable with correct values after checkpoint
    for i in 0u64..10 {
        let retrieved = engine.vector_store().get(i).unwrap();
        assert_eq!(retrieved.len(), dim);
        let expected_first = i as f32 * 0.1;
        assert!(
            (retrieved[0] - expected_first).abs() < 0.001,
            "vector {i}: expected first element {expected_first}, got {}",
            retrieved[0]
        );
    }

    // Reopen and verify vectors survive the round-trip
    drop(engine);
    let engine = StorageEngine::open(&dir).unwrap();
    assert_eq!(engine.len(), 10);

    for i in 0u64..10 {
        let retrieved = engine.vector_store().get(i).unwrap();
        let expected_first = i as f32 * 0.1;
        assert!(
            (retrieved[0] - expected_first).abs() < 0.001,
            "after reopen, vector {i}: expected {expected_first}, got {}",
            retrieved[0]
        );
    }

    cleanup(&dir);
}

#[test]
fn test_multiple_source_replacement() {
    let dir = test_dir("multi_src_replace");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 4,
        initial_capacity: 100,
        checkpoint_threshold: 1000,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    // Insert chunks from 3 different sources
    let sources = ["file_a.rs", "file_b.rs", "file_c.rs"];
    let mut id = 0u64;
    for src in &sources {
        for chunk in 0..4 {
            engine
                .insert(
                    id,
                    &test_vector(4, id as f32),
                    test_payload(src, chunk, 4),
                    &format!("{src} chunk {chunk}"),
                )
                .unwrap();
            id += 1;
        }
    }
    engine.checkpoint().unwrap();
    assert_eq!(engine.len(), 12);

    // Replace file_a.rs (fewer chunks)
    let new_a = vec![
        (200, test_vector(4, 200.0), test_payload("file_a.rs", 0, 2), "new a0".to_string()),
        (201, test_vector(4, 201.0), test_payload("file_a.rs", 1, 2), "new a1".to_string()),
    ];
    engine.replace_source("file_a.rs", new_a).unwrap();

    // Replace file_c.rs (more chunks)
    let new_c = vec![
        (300, test_vector(4, 300.0), test_payload("file_c.rs", 0, 6), "new c0".to_string()),
        (301, test_vector(4, 301.0), test_payload("file_c.rs", 1, 6), "new c1".to_string()),
        (302, test_vector(4, 302.0), test_payload("file_c.rs", 2, 6), "new c2".to_string()),
        (303, test_vector(4, 303.0), test_payload("file_c.rs", 3, 6), "new c3".to_string()),
        (304, test_vector(4, 304.0), test_payload("file_c.rs", 4, 6), "new c4".to_string()),
        (305, test_vector(4, 305.0), test_payload("file_c.rs", 5, 6), "new c5".to_string()),
    ];
    engine.replace_source("file_c.rs", new_c).unwrap();

    engine.checkpoint().unwrap();

    // file_b.rs should be untouched (4 chunks, ids 4..8)
    let b_ids = engine.payload_store().points_by_source("file_b.rs");
    assert_eq!(b_ids.len(), 4);
    for i in 4u64..8 {
        assert!(engine.vector_store().get(i).is_ok());
    }

    // file_a.rs: old ids 0..4 gone, new ids 200,201
    for i in 0u64..4 {
        assert!(engine.vector_store().get(i).is_err());
    }
    assert!(engine.vector_store().get(200).is_ok());
    assert!(engine.vector_store().get(201).is_ok());

    // file_c.rs: old ids 8..12 gone, new ids 300..306
    for i in 8u64..12 {
        assert!(engine.vector_store().get(i).is_err());
    }
    for i in 300u64..306 {
        assert!(engine.vector_store().get(i).is_ok());
    }

    // Total: 2 (a) + 4 (b) + 6 (c) = 12
    assert_eq!(engine.len(), 12);

    cleanup(&dir);
}

#[test]
fn test_batch_operations_with_mixed_insert_remove() {
    let dir = test_dir("mixed_batch");
    cleanup(&dir);

    let config = StorageConfig {
        dim: 8,
        initial_capacity: 100,
        checkpoint_threshold: 1000,
    };

    let mut engine = StorageEngine::create(&dir, config).unwrap();

    // Phase 1: batch insert
    let v1 = test_vector(8, 1.0);
    let v2 = test_vector(8, 2.0);
    let v3 = test_vector(8, 3.0);
    let v4 = test_vector(8, 4.0);
    let v5 = test_vector(8, 5.0);
    let batch: Vec<(u64, &[f32], Payload, &str)> = vec![
        (10, &v1, test_payload("a.md", 0, 1), "A"),
        (20, &v2, test_payload("b.md", 0, 1), "B"),
        (30, &v3, test_payload("c.md", 0, 1), "C"),
        (40, &v4, test_payload("d.md", 0, 1), "D"),
        (50, &v5, test_payload("e.md", 0, 1), "E"),
    ];
    engine.insert_batch(&batch).unwrap();
    assert_eq!(engine.len(), 5);

    // Phase 2: remove some individually
    engine.remove(20).unwrap();
    engine.remove(40).unwrap();
    assert_eq!(engine.len(), 3);

    // Phase 3: insert another batch
    let v6 = test_vector(8, 6.0);
    let v7 = test_vector(8, 7.0);
    let batch2: Vec<(u64, &[f32], Payload, &str)> = vec![
        (60, &v6, test_payload("f.md", 0, 1), "F"),
        (70, &v7, test_payload("g.md", 0, 1), "G"),
    ];
    engine.insert_batch(&batch2).unwrap();
    assert_eq!(engine.len(), 5);

    // Phase 4: checkpoint and verify
    engine.checkpoint().unwrap();

    // Removed points should be gone
    assert!(engine.vector_store().get(20).is_err());
    assert!(engine.vector_store().get(40).is_err());

    // Remaining points should have correct vectors
    let v = engine.vector_store().get(10).unwrap();
    assert!((v[0] - 1.0).abs() < 0.001);

    let v = engine.vector_store().get(30).unwrap();
    assert!((v[0] - 3.0).abs() < 0.001);

    let v = engine.vector_store().get(60).unwrap();
    assert!((v[0] - 6.0).abs() < 0.001);

    // Phase 5: reopen and verify persistence
    drop(engine);
    let engine = StorageEngine::open(&dir).unwrap();
    assert_eq!(engine.len(), 5);

    assert!(engine.vector_store().get(20).is_err());
    assert!(engine.vector_store().get(40).is_err());
    assert!(engine.vector_store().get(10).is_ok());
    assert!(engine.vector_store().get(50).is_ok());
    assert!(engine.vector_store().get(60).is_ok());
    assert!(engine.vector_store().get(70).is_ok());

    cleanup(&dir);
}
