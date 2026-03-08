use crate::fsst_text::*;
use v_hnsw_core::{PointId, Result};

/// Generate a corpus large enough for zstd dict training (>4KB).
fn generate_corpus(n: usize) -> Vec<(PointId, Vec<u8>)> {
    (0..n)
        .map(|i| {
            let text = format!(
                "Document {} contains text about vector search and HNSW indexing. \
                 This is chunk {} of the test corpus for zstd dictionary compression. \
                 The quick brown fox jumps over the lazy dog. Repeated patterns help compression.",
                i, i
            );
            (i as PointId, text.into_bytes())
        })
        .collect()
}

#[test]
fn test_compress_and_read() -> Result<()> {
    let temp_dir = std::env::temp_dir().join("zstd_text_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir)?;

    let texts = generate_corpus(100);
    compress_texts(&texts, &temp_dir)?;

    let reader = CompressedTextReader::load(&temp_dir)?.expect("compressed files should exist");

    for (id, original) in &texts {
        let decompressed = reader.get_text(*id)?.expect("text should exist");
        assert_eq!(decompressed.as_bytes(), original.as_slice());
    }

    // Non-existent ID
    assert!(reader.get_text(999)?.is_none());

    let _ = std::fs::remove_dir_all(&temp_dir);
    Ok(())
}

#[test]
fn test_empty_texts() -> Result<()> {
    let temp_dir = std::env::temp_dir().join("zstd_empty_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir)?;

    compress_texts(&[], &temp_dir)?;

    // No files should be created
    assert!(!temp_dir.join("text.zst").exists());

    let _ = std::fs::remove_dir_all(&temp_dir);
    Ok(())
}

// --- New tests ---

#[test]
fn test_compress_small_corpus_skipped() -> Result<()> {
    let temp = tempfile::tempdir().unwrap();

    // Total bytes < 4096 should skip compression
    let texts: Vec<(PointId, Vec<u8>)> = vec![
        (0, b"short".to_vec()),
        (1, b"tiny".to_vec()),
    ];

    compress_texts(&texts, temp.path())?;

    // No compressed files should be created
    assert!(!temp.path().join("text.zst").exists());
    Ok(())
}

#[test]
fn test_compress_unicode_text() -> Result<()> {
    let temp = tempfile::tempdir().unwrap();

    // Generate corpus with Unicode (must be >4KB total)
    let mut texts: Vec<(PointId, Vec<u8>)> = Vec::new();
    for i in 0..100 {
        let text = format!(
            "문서 {} 한글 텍스트와 日本語 テキスト, emojis: \u{1F600}\u{1F680}\u{2764}, \
             Ünïcödé characters: àéîõü, repeated: {}{}{}",
            i,
            "한글".repeat(10),
            "emoji\u{1F600}".repeat(5),
            "日本語テスト".repeat(5),
        );
        texts.push((i as PointId, text.into_bytes()));
    }

    compress_texts(&texts, temp.path())?;

    let reader = CompressedTextReader::load(temp.path())?.expect("compressed files should exist");

    for (id, original) in &texts {
        let decompressed = reader.get_text(*id)?.expect("text should exist");
        assert_eq!(decompressed.as_bytes(), original.as_slice());
    }
    Ok(())
}

#[test]
fn test_compress_very_long_strings() -> Result<()> {
    let temp = tempfile::tempdir().unwrap();

    // Each document is 10KB+
    let mut texts: Vec<(PointId, Vec<u8>)> = Vec::new();
    for i in 0..20 {
        let text = format!("Document {}: {}", i, "x".repeat(10_000));
        texts.push((i as PointId, text.into_bytes()));
    }

    compress_texts(&texts, temp.path())?;

    let reader = CompressedTextReader::load(temp.path())?.expect("compressed files");

    for (id, original) in &texts {
        let decompressed = reader.get_text(*id)?.unwrap();
        assert_eq!(decompressed.len(), original.len());
        assert_eq!(decompressed.as_bytes(), original.as_slice());
    }
    Ok(())
}

#[test]
fn test_compressed_reader_load_missing_files() -> Result<()> {
    let temp = tempfile::tempdir().unwrap();

    // No compressed files at all -> should return None
    let reader = CompressedTextReader::load(temp.path())?;
    assert!(reader.is_none());
    Ok(())
}

#[test]
fn test_compress_mixed_empty_and_nonempty() -> Result<()> {
    let temp = tempfile::tempdir().unwrap();

    // Mix of empty strings and normal strings, must be >4KB total
    let mut texts: Vec<(PointId, Vec<u8>)> = Vec::new();
    for i in 0..100 {
        let text = if i % 3 == 0 {
            String::new()
        } else {
            format!(
                "Non-empty document {} with enough padding to make the corpus large {}",
                i,
                "padding ".repeat(20)
            )
        };
        texts.push((i as PointId, text.into_bytes()));
    }

    compress_texts(&texts, temp.path())?;

    let reader = CompressedTextReader::load(temp.path())?.expect("compressed files");
    for (id, original) in &texts {
        let decompressed = reader.get_text(*id)?.unwrap();
        assert_eq!(decompressed.as_bytes(), original.as_slice());
    }
    Ok(())
}

#[test]
fn test_compress_overwrites_old_fsst_files() -> Result<()> {
    let temp = tempfile::tempdir().unwrap();

    // Create dummy old FSST files
    std::fs::write(temp.path().join("text.fsst"), b"old")?;
    std::fs::write(temp.path().join("text_symbols.fsst"), b"old")?;
    std::fs::write(temp.path().join("text_fsst.idx"), b"old")?;

    let texts = generate_corpus(100);
    compress_texts(&texts, temp.path())?;

    // Old FSST files should be removed
    assert!(!temp.path().join("text.fsst").exists());
    assert!(!temp.path().join("text_symbols.fsst").exists());
    assert!(!temp.path().join("text_fsst.idx").exists());

    // New zstd files should exist
    assert!(temp.path().join("text.zst").exists());
    assert!(temp.path().join("text_dict.zst").exists());
    assert!(temp.path().join("text_zst.idx").exists());
    Ok(())
}
