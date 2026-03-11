//! Tests for SQ8 vector store.

use crate::sq8_store::Sq8VectorStore;

#[test]
fn create_and_insert() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("sq8.bin");

    let mut store = Sq8VectorStore::create(&path, 4, 10).unwrap();
    store.insert(1, &[0, 128, 255, 64]).unwrap();

    let codes = store.get(1).unwrap();
    assert_eq!(codes, &[0, 128, 255, 64]);
}

#[test]
fn insert_dimension_mismatch() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("sq8.bin");

    let mut store = Sq8VectorStore::create(&path, 4, 10).unwrap();
    let result = store.insert(1, &[0, 128]);
    assert!(result.is_err());
}

#[test]
fn get_nonexistent() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("sq8.bin");

    let store = Sq8VectorStore::create(&path, 4, 10).unwrap();
    let result = store.get(999);
    assert!(result.is_err());
}

#[test]
fn overwrite_existing() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("sq8.bin");

    let mut store = Sq8VectorStore::create(&path, 3, 10).unwrap();
    store.insert(1, &[10, 20, 30]).unwrap();
    store.insert(1, &[40, 50, 60]).unwrap();

    assert_eq!(store.get(1).unwrap(), &[40, 50, 60]);
    assert_eq!(store.len(), 1); // No duplicate
}

#[test]
fn insert_batch_basic() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("sq8.bin");

    let mut store = Sq8VectorStore::create(&path, 2, 5).unwrap();
    let batch: Vec<(u64, &[u8])> = vec![(1, &[10, 20]), (2, &[30, 40]), (3, &[50, 60])];
    store.insert_batch(&batch).unwrap();

    assert_eq!(store.len(), 3);
    assert_eq!(store.get(1).unwrap(), &[10, 20]);
    assert_eq!(store.get(3).unwrap(), &[50, 60]);
}

#[test]
fn auto_grow() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("sq8.bin");

    let mut store = Sq8VectorStore::create(&path, 2, 2).unwrap();
    // Insert more than initial capacity
    for i in 0..10 {
        store.insert(i, &[i as u8, (i * 2) as u8]).unwrap();
    }

    assert_eq!(store.len(), 10);
    assert_eq!(store.get(5).unwrap(), &[5, 10]);
}

#[test]
fn restore_id_map() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("sq8.bin");

    let mut store = Sq8VectorStore::create(&path, 2, 10).unwrap();
    store.insert(100, &[10, 20]).unwrap();
    store.insert(200, &[30, 40]).unwrap();

    let id_map = std::collections::HashMap::from([(100, 0u32), (200, 1u32)]);

    // Simulate reopening
    let mut store2 = Sq8VectorStore::open(&path).unwrap();
    store2.restore_id_map(&id_map);

    assert_eq!(store2.get(100).unwrap(), &[10, 20]);
    assert_eq!(store2.get(200).unwrap(), &[30, 40]);
}

#[test]
fn is_empty_and_dim() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("sq8.bin");

    let store = Sq8VectorStore::create(&path, 8, 10).unwrap();
    assert!(store.is_empty());
    assert_eq!(store.dim(), 8);
}
