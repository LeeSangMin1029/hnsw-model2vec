//! Tests for MmapVectorStore.

use v_hnsw_core::VectorStore;

use crate::mmap_store::{MmapVectorStore, HEADER_SIZE};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tmp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "v-hnsw-mmap-test-{}-{}-{}",
        name,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ))
}

fn cleanup(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}

// ---------------------------------------------------------------------------
// Create & basic properties
// ---------------------------------------------------------------------------

#[test]
fn test_create_new_store() {
    let path = tmp_path("create");
    let store = MmapVectorStore::create(&path, 4, 10).unwrap();
    assert_eq!(store.dim(), 4);
    assert_eq!(store.len(), 0);
    assert!(store.is_empty());
    assert_eq!(store.capacity(), 10);
    cleanup(&path);
}

#[test]
fn test_create_sets_correct_file_size() {
    let path = tmp_path("file_size");
    let dim = 8;
    let cap = 20u32;
    let _store = MmapVectorStore::create(&path, dim, cap).unwrap();
    let meta = std::fs::metadata(&path).unwrap();
    let expected = HEADER_SIZE + (cap as usize) * dim * std::mem::size_of::<f32>();
    assert_eq!(meta.len(), expected as u64);
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// Insert & Get (VectorStore trait)
// ---------------------------------------------------------------------------

#[test]
fn test_insert_and_get() {
    let path = tmp_path("insert_get");
    let mut store = MmapVectorStore::create(&path, 4, 10).unwrap();
    let vec = [1.0f32, 2.0, 3.0, 4.0];
    store.insert(1, &vec).unwrap();
    assert_eq!(store.len(), 1);

    let retrieved = store.get(1).unwrap();
    assert_eq!(retrieved, &vec);
    cleanup(&path);
}

#[test]
fn test_insert_multiple_vectors() {
    let path = tmp_path("multi_insert");
    let mut store = MmapVectorStore::create(&path, 3, 10).unwrap();

    store.insert(0, &[1.0, 0.0, 0.0]).unwrap();
    store.insert(1, &[0.0, 1.0, 0.0]).unwrap();
    store.insert(2, &[0.0, 0.0, 1.0]).unwrap();

    assert_eq!(store.len(), 3);
    assert_eq!(store.get(0).unwrap(), &[1.0, 0.0, 0.0]);
    assert_eq!(store.get(1).unwrap(), &[0.0, 1.0, 0.0]);
    assert_eq!(store.get(2).unwrap(), &[0.0, 0.0, 1.0]);
    cleanup(&path);
}

#[test]
fn test_insert_updates_existing() {
    let path = tmp_path("update");
    let mut store = MmapVectorStore::create(&path, 2, 5).unwrap();
    store.insert(1, &[1.0, 2.0]).unwrap();
    store.insert(1, &[3.0, 4.0]).unwrap();

    // Should still be 1 vector, not 2
    assert_eq!(store.len(), 1);
    assert_eq!(store.get(1).unwrap(), &[3.0, 4.0]);
    cleanup(&path);
}

#[test]
fn test_insert_dimension_mismatch() {
    let path = tmp_path("dim_mismatch");
    let mut store = MmapVectorStore::create(&path, 4, 10).unwrap();
    let result = store.insert(1, &[1.0, 2.0]); // dim=2 but store expects 4
    assert!(result.is_err());
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// Remove
// ---------------------------------------------------------------------------

#[test]
fn test_remove() {
    let path = tmp_path("remove");
    let mut store = MmapVectorStore::create(&path, 3, 10).unwrap();
    store.insert(1, &[1.0, 2.0, 3.0]).unwrap();
    store.insert(2, &[4.0, 5.0, 6.0]).unwrap();
    assert_eq!(store.len(), 2);

    store.remove(1).unwrap();
    assert_eq!(store.len(), 1);

    // Getting removed vector should fail
    assert!(store.get(1).is_err());
    // Other vector still accessible
    assert_eq!(store.get(2).unwrap(), &[4.0, 5.0, 6.0]);
    cleanup(&path);
}

#[test]
fn test_remove_nonexistent() {
    let path = tmp_path("remove_noexist");
    let mut store = MmapVectorStore::create(&path, 3, 10).unwrap();
    let result = store.remove(999);
    assert!(result.is_err());
    cleanup(&path);
}

#[test]
fn test_slot_reuse_after_remove() {
    let path = tmp_path("slot_reuse");
    let mut store = MmapVectorStore::create(&path, 2, 3).unwrap();

    store.insert(1, &[1.0, 1.0]).unwrap();
    store.insert(2, &[2.0, 2.0]).unwrap();
    store.insert(3, &[3.0, 3.0]).unwrap();
    // capacity is 3, all slots used

    store.remove(2).unwrap();
    // Now slot should be freed; inserting should reuse it instead of growing
    store.insert(4, &[4.0, 4.0]).unwrap();
    assert_eq!(store.len(), 3);
    assert_eq!(store.get(4).unwrap(), &[4.0, 4.0]);
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// insert_batch
// ---------------------------------------------------------------------------

#[test]
fn test_insert_batch_basic() {
    let path = tmp_path("batch");
    let mut store = MmapVectorStore::create(&path, 2, 10).unwrap();

    let batch: Vec<(u64, &[f32])> = vec![
        (1, &[1.0, 0.0]),
        (2, &[0.0, 1.0]),
        (3, &[1.0, 1.0]),
    ];
    store.insert_batch(&batch).unwrap();

    assert_eq!(store.len(), 3);
    assert_eq!(store.get(1).unwrap(), &[1.0, 0.0]);
    assert_eq!(store.get(2).unwrap(), &[0.0, 1.0]);
    assert_eq!(store.get(3).unwrap(), &[1.0, 1.0]);
    cleanup(&path);
}

#[test]
fn test_insert_batch_empty() {
    let path = tmp_path("batch_empty");
    let mut store = MmapVectorStore::create(&path, 2, 10).unwrap();
    let batch: Vec<(u64, &[f32])> = vec![];
    store.insert_batch(&batch).unwrap();
    assert_eq!(store.len(), 0);
    cleanup(&path);
}

#[test]
fn test_insert_batch_auto_grows() {
    let path = tmp_path("batch_grow");
    let mut store = MmapVectorStore::create(&path, 2, 2).unwrap();

    // Insert 5 items when capacity is only 2 — should auto-grow
    let vecs: Vec<[f32; 2]> = (0..5).map(|i| [i as f32, i as f32 + 0.5]).collect();
    let batch: Vec<(u64, &[f32])> = vecs.iter().enumerate()
        .map(|(i, v)| (i as u64, v.as_slice()))
        .collect();
    store.insert_batch(&batch).unwrap();
    assert_eq!(store.len(), 5);
    assert!(store.capacity() >= 5);
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// grow
// ---------------------------------------------------------------------------

#[test]
fn test_grow_increases_capacity() {
    let path = tmp_path("grow");
    let mut store = MmapVectorStore::create(&path, 2, 5).unwrap();
    store.insert(1, &[1.0, 2.0]).unwrap();

    store.grow(20).unwrap();
    assert_eq!(store.capacity(), 20);
    // Existing data should still be accessible
    assert_eq!(store.get(1).unwrap(), &[1.0, 2.0]);
    cleanup(&path);
}

#[test]
fn test_grow_rejects_smaller_capacity() {
    let path = tmp_path("grow_reject");
    let mut store = MmapVectorStore::create(&path, 2, 10).unwrap();
    let result = store.grow(5);
    assert!(result.is_err());
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// Auto-grow on insert
// ---------------------------------------------------------------------------

#[test]
fn test_auto_grow_on_insert() {
    let path = tmp_path("auto_grow");
    let mut store = MmapVectorStore::create(&path, 2, 2).unwrap();
    store.insert(1, &[1.0, 0.0]).unwrap();
    store.insert(2, &[0.0, 1.0]).unwrap();
    // capacity=2, now insert a 3rd — should auto-grow
    store.insert(3, &[1.0, 1.0]).unwrap();
    assert_eq!(store.len(), 3);
    assert!(store.capacity() > 2);
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// Open existing file
// ---------------------------------------------------------------------------

#[test]
fn test_open_existing_store() {
    let path = tmp_path("open");
    {
        let mut store = MmapVectorStore::create(&path, 4, 10).unwrap();
        store.insert(1, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        store.flush().unwrap();
    }
    // Reopen
    let store = MmapVectorStore::open(&path).unwrap();
    assert_eq!(store.dim(), 4);
    assert_eq!(store.capacity(), 10);
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// id_map / restore_id_map
// ---------------------------------------------------------------------------

#[test]
fn test_id_map_and_restore() {
    let path = tmp_path("id_map");
    let mut store = MmapVectorStore::create(&path, 2, 10).unwrap();
    store.insert(10, &[1.0, 0.0]).unwrap();
    store.insert(20, &[0.0, 1.0]).unwrap();

    let map = store.id_map().clone();
    assert_eq!(map.len(), 2);
    assert!(map.contains_key(&10));
    assert!(map.contains_key(&20));

    // Restore on a fresh open
    drop(store);
    let mut store2 = MmapVectorStore::open(&path).unwrap();
    store2.restore_id_map(map);
    assert_eq!(store2.len(), 2);
    assert_eq!(store2.get(10).unwrap(), &[1.0, 0.0]);
    assert_eq!(store2.get(20).unwrap(), &[0.0, 1.0]);
    cleanup(&path);
}

#[test]
fn test_restore_id_map_rebuilds_free_slots() {
    let path = tmp_path("free_slots");
    let mut store = MmapVectorStore::create(&path, 2, 10).unwrap();
    // Insert into slots 0, 1, 2
    store.insert(10, &[1.0, 0.0]).unwrap();
    store.insert(20, &[0.0, 1.0]).unwrap();
    store.insert(30, &[1.0, 1.0]).unwrap();
    // Remove middle one (slot 1 freed)
    store.remove(20).unwrap();

    let map = store.id_map().clone();
    drop(store);

    let mut store2 = MmapVectorStore::open(&path).unwrap();
    store2.restore_id_map(map);
    // Insert should reuse freed slot
    store2.insert(40, &[2.0, 2.0]).unwrap();
    assert_eq!(store2.len(), 3);
    assert_eq!(store2.get(40).unwrap(), &[2.0, 2.0]);
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// is_empty
// ---------------------------------------------------------------------------

#[test]
fn test_is_empty() {
    let path = tmp_path("is_empty");
    let store = MmapVectorStore::create(&path, 2, 5).unwrap();
    assert!(store.is_empty());
    cleanup(&path);
}
