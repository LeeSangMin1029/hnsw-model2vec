use crate::payload_store::*;
use std::collections::HashMap;
use v_hnsw_core::{Payload, PayloadStore, PayloadValue, Result};

fn make_payload(source: &str, tags: &[&str]) -> Payload {
    Payload {
        source: source.to_string(),
        tags: tags.iter().map(|s| s.to_string()).collect(),
        created_at: 0,
        source_modified_at: 0,
        chunk_index: 0,
        chunk_total: 1,
        custom: HashMap::new(),
    }
}

fn create_store() -> (tempfile::TempDir, FilePayloadStore) {
    let temp_dir = tempfile::tempdir().unwrap();
    let payload_path = temp_dir.path().join("payload.dat");
    let text_path = temp_dir.path().join("text.dat");
    let store = FilePayloadStore::create(&payload_path, &text_path).unwrap();
    (temp_dir, store)
}

#[test]
fn test_payload_store_create() -> Result<()> {
    let temp_dir = std::env::temp_dir().join("payload_test_create");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir)?;

    let payload_path = temp_dir.join("payload.dat");
    let text_path = temp_dir.join("text.dat");

    let mut store = FilePayloadStore::create(&payload_path, &text_path)?;

    let payload = Payload {
        source: "test.md".to_string(),
        tags: vec!["tag1".to_string()],
        created_at: 123456,
        source_modified_at: 123456,
        chunk_index: 0,
        chunk_total: 1,
        custom: HashMap::new(),
    };

    store.buffer_payload(1, payload.clone());
    store.buffer_text(1, "test text".to_string());

    store.flush_buffers()?;

    // Read back
    let retrieved = store.get_payload(1)?;
    assert!(retrieved.is_some());
    assert_eq!(retrieved.as_ref().map(|p| &p.source), Some(&"test.md".to_string()));

    let text = store.get_text(1)?;
    assert_eq!(text.as_deref(), Some("test text"));

    let _ = std::fs::remove_dir_all(&temp_dir);
    Ok(())
}

#[test]
fn test_payload_store_buffering() -> Result<()> {
    let temp_dir = std::env::temp_dir().join("payload_test_buffer");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir)?;

    let payload_path = temp_dir.join("payload.dat");
    let text_path = temp_dir.join("text.dat");

    let mut store = FilePayloadStore::create(&payload_path, &text_path)?;

    let payload = Payload {
        source: "test.md".to_string(),
        tags: vec![],
        created_at: 0,
        source_modified_at: 0,
        chunk_index: 0,
        chunk_total: 1,
        custom: HashMap::new(),
    };

    // Buffer without flushing
    store.buffer_payload(1, payload.clone());
    store.buffer_text(1, "buffered".to_string());

    // Should be readable from buffer
    assert!(store.get_payload(1)?.is_some());
    assert_eq!(store.get_text(1)?.as_deref(), Some("buffered"));

    // Flush and verify
    store.flush_buffers()?;
    assert!(store.get_payload(1)?.is_some());

    let _ = std::fs::remove_dir_all(&temp_dir);
    Ok(())
}

#[test]
fn test_payload_store_source_index() -> Result<()> {
    let temp_dir = std::env::temp_dir().join("payload_test_source");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir)?;

    let payload_path = temp_dir.join("payload.dat");
    let text_path = temp_dir.join("text.dat");

    let mut store = FilePayloadStore::create(&payload_path, &text_path)?;

    let payload1 = Payload {
        source: "doc1.md".to_string(),
        tags: vec![],
        created_at: 0,
        source_modified_at: 0,
        chunk_index: 0,
        chunk_total: 2,
        custom: HashMap::new(),
    };

    let payload2 = Payload {
        source: "doc1.md".to_string(),
        tags: vec![],
        created_at: 0,
        source_modified_at: 0,
        chunk_index: 1,
        chunk_total: 2,
        custom: HashMap::new(),
    };

    store.buffer_payload(1, payload1);
    store.buffer_payload(2, payload2);
    store.flush_buffers()?;

    let points = store.points_by_source("doc1.md");
    assert_eq!(points.len(), 2);
    assert!(points.contains(&1));
    assert!(points.contains(&2));

    let _ = std::fs::remove_dir_all(&temp_dir);
    Ok(())
}

// --- New tests ---

#[test]
fn test_payload_get_nonexistent_returns_none() -> Result<()> {
    let (_dir, store) = create_store();
    assert!(store.get_payload(999)?.is_none());
    assert!(store.get_text(999)?.is_none());
    Ok(())
}

#[test]
fn test_payload_mark_removed_clears_buffer_and_index() -> Result<()> {
    let (_dir, mut store) = create_store();

    store.buffer_payload(1, make_payload("a.md", &[]));
    store.buffer_text(1, "hello".into());

    // Should be readable from buffer
    assert!(store.get_payload(1)?.is_some());
    assert!(store.get_text(1)?.is_some());

    // Mark removed - should clear from buffer
    store.mark_removed(1);
    assert!(store.get_payload(1)?.is_none());
    assert!(store.get_text(1)?.is_none());
    Ok(())
}

#[test]
fn test_payload_mark_removed_after_flush() -> Result<()> {
    let (_dir, mut store) = create_store();

    store.buffer_payload(10, make_payload("b.md", &[]));
    store.buffer_text(10, "world".into());
    store.flush_buffers()?;

    // Now remove - should clear from file index
    store.mark_removed(10);
    assert!(store.get_payload(10)?.is_none());
    assert!(store.get_text(10)?.is_none());
    Ok(())
}

#[test]
fn test_payload_update_existing_in_buffer() -> Result<()> {
    let (_dir, mut store) = create_store();

    store.buffer_payload(5, make_payload("v1.md", &[]));
    store.buffer_text(5, "version 1".into());

    // Overwrite with new data before flush
    store.buffer_payload(5, make_payload("v2.md", &[]));
    store.buffer_text(5, "version 2".into());

    let p = store.get_payload(5)?.unwrap();
    assert_eq!(p.source, "v2.md");
    let t = store.get_text(5)?.unwrap();
    assert_eq!(t, "version 2");
    Ok(())
}

#[test]
fn test_payload_special_characters_in_text() -> Result<()> {
    let (_dir, mut store) = create_store();

    let special = "Hello\n\t\"world\" <>&\x00null\u{1F600}emoji 한글 日本語";
    store.buffer_text(1, special.to_string());
    store.buffer_payload(1, make_payload("special.md", &[]));

    // Read from buffer
    assert_eq!(store.get_text(1)?.as_deref(), Some(special));

    // Flush and read from file
    store.flush_buffers()?;
    assert_eq!(store.get_text(1)?.as_deref(), Some(special));
    Ok(())
}

#[test]
fn test_payload_empty_text() -> Result<()> {
    let (_dir, mut store) = create_store();

    store.buffer_text(1, String::new());
    store.buffer_payload(1, make_payload("empty.md", &[]));

    store.flush_buffers()?;
    assert_eq!(store.get_text(1)?.as_deref(), Some(""));
    Ok(())
}

#[test]
fn test_payload_large_text() -> Result<()> {
    let (_dir, mut store) = create_store();

    let big_text = "A".repeat(500_000);
    store.buffer_text(1, big_text.clone());
    store.buffer_payload(1, make_payload("big.md", &[]));

    store.flush_buffers()?;
    let retrieved = store.get_text(1)?.unwrap();
    assert_eq!(retrieved.len(), 500_000);
    assert_eq!(retrieved, big_text);
    Ok(())
}

#[test]
fn test_payload_tag_index_single_tag() -> Result<()> {
    let (_dir, mut store) = create_store();

    store.buffer_payload(1, make_payload("a.md", &["rust"]));
    store.buffer_payload(2, make_payload("b.md", &["rust", "hnsw"]));
    store.buffer_payload(3, make_payload("c.md", &["python"]));

    let rust_points = store.points_by_tag("rust");
    assert_eq!(rust_points.len(), 2);
    assert!(rust_points.contains(&1));
    assert!(rust_points.contains(&2));

    let python_points = store.points_by_tag("python");
    assert_eq!(python_points.len(), 1);
    assert!(python_points.contains(&3));

    let none_points = store.points_by_tag("java");
    assert!(none_points.is_empty());
    Ok(())
}

#[test]
fn test_payload_tags_and_logic() -> Result<()> {
    let (_dir, mut store) = create_store();

    store.buffer_payload(1, make_payload("a.md", &["code", "rust"]));
    store.buffer_payload(2, make_payload("b.md", &["code", "python"]));
    store.buffer_payload(3, make_payload("c.md", &["docs", "rust"]));

    // AND: code + rust => only id 1
    let result = store.points_by_tags(&["code".into(), "rust".into()]);
    assert_eq!(result.len(), 1);
    assert!(result.contains(&1));

    // AND: code + python => only id 2
    let result = store.points_by_tags(&["code".into(), "python".into()]);
    assert_eq!(result.len(), 1);
    assert!(result.contains(&2));

    // AND with non-existent tag => empty
    let result = store.points_by_tags(&["code".into(), "java".into()]);
    assert!(result.is_empty());

    // Empty tags => empty result
    let result = store.points_by_tags(&[]);
    assert!(result.is_empty());
    Ok(())
}

#[test]
fn test_payload_mark_removed_clears_tag_index() -> Result<()> {
    let (_dir, mut store) = create_store();

    store.buffer_payload(1, make_payload("a.md", &["tag_x"]));
    store.buffer_payload(2, make_payload("b.md", &["tag_x"]));

    assert_eq!(store.points_by_tag("tag_x").len(), 2);

    store.mark_removed(1);
    let remaining = store.points_by_tag("tag_x");
    assert_eq!(remaining.len(), 1);
    assert!(remaining.contains(&2));
    Ok(())
}

#[test]
fn test_payload_mark_removed_clears_source_index() -> Result<()> {
    let (_dir, mut store) = create_store();

    store.buffer_payload(1, make_payload("doc.md", &[]));
    store.buffer_payload(2, make_payload("doc.md", &[]));

    assert_eq!(store.points_by_source("doc.md").len(), 2);

    store.mark_removed(1);
    let remaining = store.points_by_source("doc.md");
    assert_eq!(remaining.len(), 1);
    assert!(remaining.contains(&2));
    Ok(())
}

#[test]
fn test_payload_set_and_remove_via_trait() -> Result<()> {
    let (_dir, mut store) = create_store();

    let p = make_payload("trait.md", &[]);
    store.set_payload(1, p)?;

    assert!(store.get_payload(1)?.is_some());

    store.remove_payload(1)?;
    assert!(store.get_payload(1)?.is_none());
    Ok(())
}

#[test]
fn test_payload_custom_fields_roundtrip() -> Result<()> {
    let (_dir, mut store) = create_store();

    let mut p = make_payload("custom.md", &[]);
    p.custom.insert("priority".into(), PayloadValue::Integer(42));
    p.custom.insert("score".into(), PayloadValue::Float(3.14));
    p.custom.insert("label".into(), PayloadValue::String("important".into()));

    store.buffer_payload(1, p);
    store.flush_buffers()?;

    let retrieved = store.get_payload(1)?.unwrap();
    // PayloadValue does not implement PartialEq, so match manually
    match retrieved.custom.get("priority") {
        Some(PayloadValue::Integer(v)) => assert_eq!(*v, 42),
        other => panic!("expected Integer(42), got {:?}", other),
    }
    match retrieved.custom.get("label") {
        Some(PayloadValue::String(v)) => assert_eq!(v, "important"),
        other => panic!("expected String(\"important\"), got {:?}", other),
    }
    Ok(())
}

#[test]
fn test_payload_save_and_load_indices() -> Result<()> {
    let temp_dir = tempfile::tempdir().unwrap();
    let payload_path = temp_dir.path().join("payload.dat");
    let text_path = temp_dir.path().join("text.dat");
    let payload_idx = temp_dir.path().join("payload.idx");
    let text_idx = temp_dir.path().join("text.idx");

    // Create, write, flush, save indices
    {
        let mut store = FilePayloadStore::create(&payload_path, &text_path)?;
        store.buffer_payload(1, make_payload("a.md", &["t1"]));
        store.buffer_text(1, "aaa".into());
        store.buffer_payload(2, make_payload("b.md", &["t2"]));
        store.buffer_text(2, "bbb".into());
        store.flush_buffers()?;
        store.save_indices(&payload_idx, &text_idx)?;
    }

    // Open fresh store, load indices, verify
    {
        let mut store = FilePayloadStore::open(&payload_path, &text_path)?;
        store.load_indices(&payload_idx, &text_idx)?;

        let p1 = store.get_payload(1)?.unwrap();
        assert_eq!(p1.source, "a.md");
        let t1 = store.get_text(1)?.unwrap();
        assert_eq!(t1, "aaa");

        let p2 = store.get_payload(2)?.unwrap();
        assert_eq!(p2.source, "b.md");
        let t2 = store.get_text(2)?.unwrap();
        assert_eq!(t2, "bbb");
    }
    Ok(())
}

#[test]
fn test_payload_points_by_source_nonexistent() {
    let (_dir, store) = create_store();
    assert!(store.points_by_source("nonexistent.md").is_empty());
}

#[test]
fn test_payload_all_text_bytes() -> Result<()> {
    let (_dir, mut store) = create_store();

    store.buffer_payload(1, make_payload("a.md", &[]));
    store.buffer_text(1, "hello".into());
    store.buffer_payload(2, make_payload("b.md", &[]));
    store.buffer_text(2, "world".into());
    store.flush_buffers()?;

    let all_texts = store.all_text_bytes()?;
    assert_eq!(all_texts.len(), 2);

    let texts_map: HashMap<u64, &[u8]> = all_texts.iter().map(|(id, d)| (*id, d.as_slice())).collect();
    assert_eq!(texts_map[&1], b"hello");
    assert_eq!(texts_map[&2], b"world");
    Ok(())
}
