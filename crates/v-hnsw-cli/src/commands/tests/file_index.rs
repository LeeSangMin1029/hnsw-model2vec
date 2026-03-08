use crate::commands::file_index::*;

use anyhow::Result;

#[test]
fn test_file_index_new() {
    let index = FileIndex::new();
    assert_eq!(index.version, FileIndex::VERSION);
    assert!(index.files.is_empty());
}

#[test]
fn test_file_index_update() {
    let mut index = FileIndex::new();
    index.update_file("test.md".to_string(), 123456, 1024, vec![1, 2, 3]);

    let meta = index.get_file("test.md").unwrap();
    assert_eq!(meta.path, "test.md");
    assert_eq!(meta.mtime, 123456);
    assert_eq!(meta.size, 1024);
    assert_eq!(meta.chunk_ids, vec![1, 2, 3]);
}

#[test]
fn test_file_index_is_modified() {
    let mut index = FileIndex::new();

    // New file is always modified
    assert!(index.is_modified("test.md", 123456, 1024));

    // Add file
    index.update_file("test.md".to_string(), 123456, 1024, vec![1]);

    // Same mtime and size = not modified
    assert!(!index.is_modified("test.md", 123456, 1024));

    // Different mtime = modified
    assert!(index.is_modified("test.md", 123457, 1024));

    // Different size = modified
    assert!(index.is_modified("test.md", 123456, 2048));
}

#[test]
fn test_save_load_file_index() -> Result<()> {
    let temp_dir = std::env::temp_dir().join("file_index_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir)?;

    let mut index = FileIndex::new();
    index.update_file("doc1.md".to_string(), 111111, 512, vec![1, 2]);
    index.update_file("doc2.md".to_string(), 222222, 1024, vec![3, 4, 5]);

    save_file_index(&temp_dir, &index)?;

    let loaded = load_file_index(&temp_dir)?;
    assert_eq!(loaded.version, FileIndex::VERSION);
    assert_eq!(loaded.files.len(), 2);

    let doc1 = loaded.get_file("doc1.md").unwrap();
    assert_eq!(doc1.mtime, 111111);
    assert_eq!(doc1.chunk_ids, vec![1, 2]);

    let _ = std::fs::remove_dir_all(&temp_dir);
    Ok(())
}

// ── Edge case tests ─────────────────────────────────────────────────

#[test]
fn test_file_index_default() {
    let index = FileIndex::default();
    assert_eq!(index.version, FileIndex::VERSION);
    assert!(index.files.is_empty());
}

#[test]
fn test_get_file_missing() {
    let index = FileIndex::new();
    assert!(index.get_file("nonexistent.md").is_none());
}

#[test]
fn test_update_overwrites_existing() {
    let mut index = FileIndex::new();
    index.update_file("test.md".to_string(), 100, 500, vec![1, 2]);
    index.update_file("test.md".to_string(), 200, 600, vec![3, 4, 5]);

    let meta = index.get_file("test.md").unwrap();
    assert_eq!(meta.mtime, 200, "mtime should be overwritten");
    assert_eq!(meta.size, 600, "size should be overwritten");
    assert_eq!(meta.chunk_ids, vec![3, 4, 5], "chunk_ids should be overwritten");
}

#[test]
fn test_update_file_with_hash() {
    let mut index = FileIndex::new();
    index.update_file_with_hash("hashed.rs".to_string(), 123, 456, vec![10], 0xDEADBEEF);

    let meta = index.get_file("hashed.rs").unwrap();
    assert_eq!(meta.content_hash, Some(0xDEADBEEF));
    assert_eq!(meta.mtime, 123);
    assert_eq!(meta.size, 456);
    assert_eq!(meta.chunk_ids, vec![10]);
}

#[test]
fn test_is_modified_new_file() {
    let index = FileIndex::new();
    assert!(
        index.is_modified("brand_new.md", 999, 100),
        "new file should always be considered modified"
    );
}

#[test]
fn test_is_modified_both_changed() {
    let mut index = FileIndex::new();
    index.update_file("test.md".to_string(), 100, 500, vec![]);
    assert!(
        index.is_modified("test.md", 200, 600),
        "both mtime and size changed => modified"
    );
}

#[test]
fn test_is_modified_zero_values() {
    let mut index = FileIndex::new();
    index.update_file("zero.md".to_string(), 0, 0, vec![]);
    assert!(
        !index.is_modified("zero.md", 0, 0),
        "same zero values should not be modified"
    );
}

#[test]
fn test_empty_chunk_ids() {
    let mut index = FileIndex::new();
    index.update_file("empty.md".to_string(), 100, 0, vec![]);
    let meta = index.get_file("empty.md").unwrap();
    assert!(meta.chunk_ids.is_empty());
}

#[test]
fn test_special_filename_characters() {
    let mut index = FileIndex::new();
    let path = "path/to/file with spaces.md".to_string();
    index.update_file(path.clone(), 100, 200, vec![1]);
    assert!(index.get_file(&path).is_some());
}

#[test]
fn test_unicode_filename() {
    let mut index = FileIndex::new();
    let path = "docs/한글문서.md".to_string();
    index.update_file(path.clone(), 100, 200, vec![1]);
    let meta = index.get_file(&path).unwrap();
    assert_eq!(meta.path, path);
}

#[test]
fn test_load_nonexistent_dir_returns_empty() -> Result<()> {
    let tmp = std::env::temp_dir().join("file_index_test_nonexistent_8374");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)?;
    // No file_index.json exists
    let index = load_file_index(&tmp)?;
    assert!(index.files.is_empty());
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}

#[test]
fn test_save_and_load_preserves_content_hash() -> Result<()> {
    let tmp = std::env::temp_dir().join("file_index_test_hash_preserve");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)?;

    let mut index = FileIndex::new();
    index.update_file_with_hash("src/main.rs".to_string(), 1000, 2000, vec![5, 6], 0xCAFE);

    save_file_index(&tmp, &index)?;
    let loaded = load_file_index(&tmp)?;

    let meta = loaded.get_file("src/main.rs").unwrap();
    assert_eq!(meta.content_hash, Some(0xCAFE));

    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}

#[test]
fn test_many_files_in_index() {
    let mut index = FileIndex::new();
    for i in 0..100 {
        index.update_file(format!("file_{i}.md"), i as u64, i as u64 * 10, vec![i as u64]);
    }
    assert_eq!(index.files.len(), 100);
    assert!(index.get_file("file_50.md").is_some());
    assert!(index.is_modified("file_50.md", 999, 999));
    assert!(!index.is_modified("file_50.md", 50, 500));
}
