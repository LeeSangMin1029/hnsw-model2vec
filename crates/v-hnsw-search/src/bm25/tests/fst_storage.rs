//! Tests for FST-based BM25 storage round-trip (save ↔ load).

use std::collections::HashMap;

use v_hnsw_core::PointId;

use crate::bm25::fst_storage::{fst_exists, load_fst, save_fst, FstStorage};
use crate::bm25::index::PostingList;
use crate::bm25::scorer::Bm25Params;
use crate::WhitespaceTokenizer;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tmp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "v-hnsw-fst-test-{}-{}-{}",
        name,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn cleanup(dir: &std::path::Path) {
    let _ = std::fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------
// fst_exists
// ---------------------------------------------------------------------------

#[test]
fn test_fst_exists_false_on_empty_dir() {
    let dir = tmp_dir("exists_empty");
    assert!(!fst_exists(&dir));
    cleanup(&dir);
}

#[test]
fn test_fst_exists_true_after_save() {
    let dir = tmp_dir("exists_after_save");
    let postings = HashMap::new();
    let doc_lengths: HashMap<PointId, u32> = HashMap::new();
    let tokenizer = WhitespaceTokenizer;
    save_fst(&dir, &tokenizer, &postings, &doc_lengths, 0, 0, Bm25Params::default()).unwrap();
    assert!(fst_exists(&dir));
    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Round-trip: save → load
// ---------------------------------------------------------------------------

#[test]
fn test_round_trip_empty_index() {
    let dir = tmp_dir("rt_empty");
    let tokenizer = WhitespaceTokenizer;
    let postings = HashMap::new();
    let doc_lengths: HashMap<PointId, u32> = HashMap::new();
    let params = Bm25Params::default();

    save_fst(&dir, &tokenizer, &postings, &doc_lengths, 0, 0, params).unwrap();
    let loaded: FstStorage<WhitespaceTokenizer> = load_fst(&dir).unwrap();

    assert_eq!(loaded.postings.len(), 0);
    assert_eq!(loaded.doc_lengths.len(), 0);
    assert_eq!(loaded.total_length, 0);
    assert_eq!(loaded.total_docs, 0);
    cleanup(&dir);
}

#[test]
fn test_round_trip_single_term() {
    let dir = tmp_dir("rt_single");
    let tokenizer = WhitespaceTokenizer;

    let mut pl = PostingList::new();
    pl.add(1, 3);
    pl.add(5, 1);
    let mut postings = HashMap::new();
    postings.insert("hello".to_string(), pl);

    let mut doc_lengths = HashMap::new();
    doc_lengths.insert(1u64, 10u32);
    doc_lengths.insert(5u64, 20u32);

    let params = Bm25Params::new(1.5, 0.8);

    save_fst(&dir, &tokenizer, &postings, &doc_lengths, 30, 2, params).unwrap();
    let loaded: FstStorage<WhitespaceTokenizer> = load_fst(&dir).unwrap();

    assert_eq!(loaded.postings.len(), 1);
    assert_eq!(loaded.postings[0].postings.len(), 2);
    assert_eq!(loaded.total_length, 30);
    assert_eq!(loaded.total_docs, 2);
    assert_eq!(loaded.doc_lengths.len(), 2);
    assert_eq!(loaded.max_doc_id, 5);
    assert!((loaded.params.k1 - 1.5).abs() < 1e-6);
    assert!((loaded.params.b - 0.8).abs() < 1e-6);

    cleanup(&dir);
}

#[test]
fn test_round_trip_multiple_terms() {
    let dir = tmp_dir("rt_multi");
    let tokenizer = WhitespaceTokenizer;

    let terms = vec!["apple", "banana", "cherry"];
    let mut postings = HashMap::new();
    for (i, term) in terms.iter().enumerate() {
        let mut pl = PostingList::new();
        pl.add(i as u64, (i + 1) as u32);
        postings.insert(term.to_string(), pl);
    }

    let mut doc_lengths = HashMap::new();
    for i in 0..3u64 {
        doc_lengths.insert(i, 10);
    }

    save_fst(&dir, &tokenizer, &postings, &doc_lengths, 30, 3, Bm25Params::default()).unwrap();
    let loaded: FstStorage<WhitespaceTokenizer> = load_fst(&dir).unwrap();

    assert_eq!(loaded.postings.len(), 3);
    assert_eq!(loaded.total_docs, 3);

    // FST map should contain all terms
    assert!(loaded.fst_map.get("apple").is_some());
    assert!(loaded.fst_map.get("banana").is_some());
    assert!(loaded.fst_map.get("cherry").is_some());
    assert!(loaded.fst_map.get("nonexistent").is_none());

    cleanup(&dir);
}

#[test]
fn test_round_trip_postings_sorted_by_doc_id() {
    let dir = tmp_dir("rt_sorted");
    let tokenizer = WhitespaceTokenizer;

    let mut pl = PostingList::new();
    // Add in non-sorted order
    pl.add(10, 1);
    pl.add(2, 3);
    pl.add(7, 2);

    let mut postings = HashMap::new();
    postings.insert("term".to_string(), pl);

    let mut doc_lengths = HashMap::new();
    doc_lengths.insert(2u64, 5u32);
    doc_lengths.insert(7u64, 5u32);
    doc_lengths.insert(10u64, 5u32);

    save_fst(&dir, &tokenizer, &postings, &doc_lengths, 15, 3, Bm25Params::default()).unwrap();
    let loaded: FstStorage<WhitespaceTokenizer> = load_fst(&dir).unwrap();

    // Postings should be sorted by doc_id after save
    let loaded_postings = &loaded.postings[0].postings;
    assert_eq!(loaded_postings.len(), 3);
    assert_eq!(loaded_postings[0].doc_id, 2);
    assert_eq!(loaded_postings[1].doc_id, 7);
    assert_eq!(loaded_postings[2].doc_id, 10);

    // TFs should be preserved
    assert_eq!(loaded_postings[0].tf, 3);
    assert_eq!(loaded_postings[1].tf, 2);
    assert_eq!(loaded_postings[2].tf, 1);

    cleanup(&dir);
}

#[test]
fn test_round_trip_preserves_max_doc_id() {
    let dir = tmp_dir("rt_maxid");
    let tokenizer = WhitespaceTokenizer;
    let postings = HashMap::new();
    let mut doc_lengths = HashMap::new();
    doc_lengths.insert(100u64, 5u32);
    doc_lengths.insert(200u64, 10u32);

    save_fst(&dir, &tokenizer, &postings, &doc_lengths, 15, 2, Bm25Params::default()).unwrap();
    let loaded: FstStorage<WhitespaceTokenizer> = load_fst(&dir).unwrap();

    assert_eq!(loaded.max_doc_id, 200);
    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

#[test]
fn test_load_fst_missing_files() {
    let dir = tmp_dir("load_missing");
    let result: Result<FstStorage<WhitespaceTokenizer>, _> = load_fst(&dir);
    assert!(result.is_err());
    cleanup(&dir);
}
