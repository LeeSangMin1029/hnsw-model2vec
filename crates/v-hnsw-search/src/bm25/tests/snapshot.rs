use crate::bm25::snapshot::Bm25Snapshot;
use crate::{Bm25Index, WhitespaceTokenizer};

fn make_temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(name);
    let _ = std::fs::create_dir_all(&dir);
    dir
}

#[test]
fn test_bm25_snapshot_roundtrip() {
    let dir = make_temp_dir("bm25_snapshot_test");

    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "the quick brown fox");
    index.add_document(2, "the lazy dog");
    index.add_document(3, "quick quick fox fox");

    // Save (creates bm25.bin + FST files)
    index.save(dir.join("bm25.bin")).expect("save failed");
    // Save snapshot
    index.save_snapshot(&dir).expect("snapshot save failed");

    // Open snapshot
    let snap = Bm25Snapshot::open(&dir).expect("snapshot open failed");
    assert_eq!(snap.total_docs(), 3);

    // Compare search results
    let tok = WhitespaceTokenizer;
    let snap_results = snap.search(&tok, "quick fox", 10);
    let index_results = index.search("quick fox", 10);

    assert_eq!(snap_results.len(), index_results.len());
    assert_eq!(snap_results[0].0, index_results[0].0); // same top doc
    // Scores should be close (both use same BM25 formula)
    for (s, i) in snap_results.iter().zip(index_results.iter()) {
        assert_eq!(s.0, i.0);
        assert!((s.1 - i.1).abs() < 0.01, "score mismatch: {} vs {}", s.1, i.1);
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_bm25_snapshot_score_documents() {
    let dir = make_temp_dir("bm25_snapshot_score_test");

    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "rust programming language");
    index.add_document(2, "python programming language");
    index.add_document(3, "rust systems programming");

    index.save(dir.join("bm25.bin")).expect("save");
    index.save_snapshot(&dir).expect("snapshot save");

    let snap = Bm25Snapshot::open(&dir).expect("open");
    let tok = WhitespaceTokenizer;

    // Score only docs 1 and 3
    let scores = snap.score_documents(&tok, "rust programming", &[1, 3]);
    assert_eq!(scores.len(), 2);
    // Both docs should have scores > 0
    for (_, score) in &scores {
        assert!(*score > 0.0);
    }

    let _ = std::fs::remove_dir_all(&dir);
}
