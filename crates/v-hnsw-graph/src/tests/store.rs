use crate::store::InMemoryVectorStore;
use v_hnsw_core::VectorStore;

#[test]
fn test_insert_and_get() -> v_hnsw_core::Result<()> {
    let mut store = InMemoryVectorStore::new(3);
    store.insert(1, &[1.0, 2.0, 3.0])?;
    let vec = store.get(1)?;
    assert_eq!(vec, &[1.0, 2.0, 3.0]);
    Ok(())
}

#[test]
fn test_dimension_mismatch() {
    let mut store = InMemoryVectorStore::new(3);
    let result = store.insert(1, &[1.0, 2.0]);
    assert!(result.is_err());
}

#[test]
fn test_get_missing() {
    let store = InMemoryVectorStore::new(3);
    let result = store.get(999);
    assert!(result.is_err());
}

#[test]
fn test_remove() -> v_hnsw_core::Result<()> {
    let mut store = InMemoryVectorStore::new(2);
    store.insert(1, &[1.0, 2.0])?;
    assert_eq!(store.len(), 1);
    store.remove(1)?;
    assert_eq!(store.len(), 0);
    assert!(store.get(1).is_err());
    Ok(())
}

#[test]
fn test_remove_missing() {
    let mut store = InMemoryVectorStore::new(2);
    let result = store.remove(999);
    assert!(result.is_err());
}

#[test]
fn test_is_empty() -> v_hnsw_core::Result<()> {
    let mut store = InMemoryVectorStore::new(2);
    assert!(store.is_empty());
    store.insert(1, &[1.0, 2.0])?;
    assert!(!store.is_empty());
    Ok(())
}

#[test]
fn test_overwrite_existing() -> v_hnsw_core::Result<()> {
    let mut store = InMemoryVectorStore::new(2);
    store.insert(1, &[1.0, 2.0])?;
    store.insert(1, &[3.0, 4.0])?; // overwrite
    let vec = store.get(1)?;
    assert_eq!(vec, &[3.0, 4.0]);
    assert_eq!(store.len(), 1); // still 1 entry
    Ok(())
}

#[test]
fn test_with_capacity() -> v_hnsw_core::Result<()> {
    let mut store = InMemoryVectorStore::with_capacity(3, 100);
    store.insert(1, &[1.0, 2.0, 3.0])?;
    assert_eq!(store.get(1)?, &[1.0, 2.0, 3.0]);
    assert_eq!(store.dim(), 3);
    Ok(())
}

#[test]
fn test_dim() {
    let store = InMemoryVectorStore::new(5);
    assert_eq!(store.dim(), 5);
}

#[test]
fn test_multiple_inserts_and_removes() -> v_hnsw_core::Result<()> {
    let mut store = InMemoryVectorStore::new(2);
    for i in 0..10 {
        store.insert(i, &[i as f32, i as f32 * 2.0])?;
    }
    assert_eq!(store.len(), 10);

    for i in 0..5 {
        store.remove(i)?;
    }
    assert_eq!(store.len(), 5);

    // Removed IDs should error
    assert!(store.get(0).is_err());
    assert!(store.get(4).is_err());

    // Remaining IDs still accessible
    assert_eq!(store.get(5)?, &[5.0, 10.0]);
    assert_eq!(store.get(9)?, &[9.0, 18.0]);
    Ok(())
}

#[test]
fn test_insert_empty_vector_dim_zero_store() -> v_hnsw_core::Result<()> {
    // A store with dim=0 shouldn't normally be created but let's test gracefully
    let mut store = InMemoryVectorStore::new(0);
    store.insert(1, &[])?;
    let v = store.get(1)?;
    assert!(v.is_empty());
    Ok(())
}

#[test]
fn test_dim_mismatch_longer() {
    let mut store = InMemoryVectorStore::new(2);
    let result = store.insert(1, &[1.0, 2.0, 3.0]); // too long
    assert!(result.is_err());
}
