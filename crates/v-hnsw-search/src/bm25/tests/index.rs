use serde::{Deserialize, Serialize};

use crate::bm25::index::{Bm25Index, PostingList};
use crate::Tokenizer;

/// Simple whitespace tokenizer for testing.
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
struct WhitespaceTokenizer;

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| s.to_lowercase())
            .collect()
    }
}

#[test]
fn test_empty_index() {
    let index = Bm25Index::new(WhitespaceTokenizer);
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert!((index.avg_doc_length() - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_add_document() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "hello world");
    assert_eq!(index.len(), 1);
    assert_eq!(index.document_frequency("hello"), 1);
    assert_eq!(index.document_frequency("world"), 1);
}

#[test]
fn test_remove_document() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "hello world");
    index.add_document(2, "hello there");

    assert!(index.remove_document(1));
    assert_eq!(index.len(), 1);
    assert_eq!(index.document_frequency("hello"), 1);
    assert_eq!(index.document_frequency("world"), 0);
}

#[test]
fn test_search_empty_index() {
    let index = Bm25Index::new(WhitespaceTokenizer);
    let results = index.search("hello", 10);
    assert!(results.is_empty());
}

#[test]
fn test_search_basic() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "the quick brown fox");
    index.add_document(2, "the lazy dog");
    index.add_document(3, "quick quick fox fox");

    let results = index.search("quick fox", 10);
    assert!(!results.is_empty());
    // Document 3 should rank highest (more term occurrences)
    assert_eq!(results[0].0, 3);
}

#[test]
fn test_search_no_match() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "hello world");
    let results = index.search("goodbye", 10);
    assert!(results.is_empty());
}

#[test]
fn test_update_document() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "hello world");
    index.add_document(1, "goodbye world"); // Replace

    assert_eq!(index.len(), 1);
    assert_eq!(index.document_frequency("hello"), 0);
    assert_eq!(index.document_frequency("goodbye"), 1);
}

#[test]
fn test_posting_list() {
    let mut pl = PostingList::new();
    pl.add(1, 3);
    pl.add(2, 1);
    assert_eq!(pl.df(), 2);

    pl.remove(1);
    assert_eq!(pl.df(), 1);
}

fn make_temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(name);
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn cleanup_dir(dir: &std::path::Path) {
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn test_save_load() {
    let dir = make_temp_dir("bm25_test_save_load");

    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "hello world");
    index.add_document(2, "hello rust");
    index.add_document(3, "rust programming");

    let bm25_path = dir.join("bm25.bin");
    index.save(&bm25_path).expect("Failed to save index");

    let loaded: Bm25Index<WhitespaceTokenizer> =
        Bm25Index::load(&bm25_path).expect("Failed to load index");

    assert_eq!(loaded.len(), 3);
    assert_eq!(loaded.document_frequency("hello"), 2);
    assert_eq!(loaded.document_frequency("rust"), 2);
    assert_eq!(loaded.document_frequency("world"), 1);
    assert_eq!(loaded.document_frequency("programming"), 1);

    let results = loaded.search("hello", 10);
    assert_eq!(results.len(), 2);

    cleanup_dir(&dir);
}

#[test]
fn test_fst_load_search() {
    let dir = make_temp_dir("bm25_test_fst_search");

    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "the quick brown fox");
    index.add_document(2, "the lazy dog");
    index.add_document(3, "quick quick fox fox");

    let bm25_path = dir.join("bm25.bin");
    index.save(&bm25_path).expect("Failed to save");

    // Load prefers FST
    let loaded: Bm25Index<WhitespaceTokenizer> =
        Bm25Index::load(&bm25_path).expect("Failed to load FST");

    assert_eq!(loaded.len(), 3);

    // FST search should match HashMap search
    let fst_results = loaded.search("quick fox", 10);
    let hash_results = index.search("quick fox", 10);
    assert_eq!(fst_results.len(), hash_results.len());
    assert_eq!(fst_results[0].0, hash_results[0].0);

    cleanup_dir(&dir);
}

// ============================================================================
// Additional index tests: edge cases, boundary conditions
// ============================================================================

#[test]
fn test_empty_corpus_search() {
    let index = Bm25Index::new(WhitespaceTokenizer);
    assert!(index.search("anything", 10).is_empty());
    assert!(index.search("", 10).is_empty());
    assert_eq!(index.avg_doc_length(), 0.0);
}

#[test]
fn test_single_doc_corpus() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "only document here");
    assert_eq!(index.len(), 1);
    assert_eq!(index.avg_doc_length(), 3.0); // 3 tokens

    let results = index.search("document", 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
    assert!(results[0].1 > 0.0);
}

#[test]
fn test_empty_document_text() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "");
    // Empty doc should be added (len increases)
    assert_eq!(index.len(), 1);
    // But no terms are indexed
    let results = index.search("anything", 10);
    assert!(results.is_empty());
}

#[test]
fn test_search_limit_zero() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "hello world");
    let results = index.search("hello", 0);
    assert!(results.is_empty());
}

#[test]
fn test_search_limit_one() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "hello world");
    index.add_document(2, "hello there");
    let results = index.search("hello", 1);
    assert_eq!(results.len(), 1);
}

#[test]
fn test_search_limit_exceeds_matches() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "hello world");
    let results = index.search("hello", 100);
    assert_eq!(results.len(), 1);
}

#[test]
fn test_document_frequency() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "alpha beta");
    index.add_document(2, "beta gamma");
    index.add_document(3, "gamma delta");

    assert_eq!(index.document_frequency("alpha"), 1);
    assert_eq!(index.document_frequency("beta"), 2);
    assert_eq!(index.document_frequency("gamma"), 2);
    assert_eq!(index.document_frequency("delta"), 1);
    assert_eq!(index.document_frequency("missing"), 0);
}

#[test]
fn test_remove_nonexistent_document() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "hello");
    assert!(!index.remove_document(999));
    assert_eq!(index.len(), 1);
}

#[test]
fn test_remove_then_readd_document() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "first version");
    assert!(index.remove_document(1));
    assert_eq!(index.len(), 0);

    index.add_document(1, "second version");
    assert_eq!(index.len(), 1);
    let results = index.search("second", 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
    // Old content should not be searchable
    assert!(index.search("first", 10).is_empty());
}

#[test]
fn test_remove_all_documents() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "hello world");
    index.add_document(2, "hello there");
    assert!(index.remove_document(1));
    assert!(index.remove_document(2));
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    assert!(index.search("hello", 10).is_empty());
}

#[test]
fn test_avg_doc_length_updates() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "a b c"); // 3 tokens
    assert_eq!(index.avg_doc_length(), 3.0);

    index.add_document(2, "x"); // 1 token
    assert_eq!(index.avg_doc_length(), 2.0); // (3+1)/2

    index.remove_document(1);
    assert_eq!(index.avg_doc_length(), 1.0);
}

#[test]
fn test_score_documents_specific_ids() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "rust programming");
    index.add_document(2, "python programming");
    index.add_document(3, "rust systems");

    // Score only docs 1 and 3
    let scores = index.score_documents("rust", &[1, 3]);
    assert_eq!(scores.len(), 2);
    for (id, score) in &scores {
        assert!(*id == 1 || *id == 3);
        assert!(*score > 0.0);
    }

    // Score non-matching doc
    let scores = index.score_documents("rust", &[2]);
    assert!(scores.is_empty());
}

#[test]
fn test_score_documents_empty_query() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "hello world");
    let scores = index.score_documents("", &[1]);
    assert!(scores.is_empty());
}

#[test]
fn test_score_documents_empty_ids() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "hello world");
    let scores = index.score_documents("hello", &[]);
    assert!(scores.is_empty());
}

#[test]
fn test_results_sorted_by_score_descending() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "rust");
    index.add_document(2, "rust rust");
    index.add_document(3, "rust rust rust");

    let results = index.search("rust", 10);
    assert!(results.len() >= 2);
    for w in results.windows(2) {
        assert!(w[0].1 >= w[1].1, "Results not sorted: {:?}", results);
    }
}

#[test]
fn test_posting_list_remove_nonexistent() {
    let mut pl = PostingList::new();
    pl.add(1, 3);
    assert!(!pl.remove(999));
    assert_eq!(pl.df(), 1);
}

#[test]
fn test_posting_list_empty() {
    let mut pl = PostingList::new();
    assert_eq!(pl.df(), 0);
    assert!(!pl.remove(1));
}

#[test]
fn test_bigram_boosting_adjacent_terms() {
    // Documents with adjacent query terms should score higher than those with
    // the same terms spread apart
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    // Adjacent "hello world"
    index.add_document(1, "hello world foo bar");
    // Same terms but not adjacent
    index.add_document(2, "hello foo bar world");

    let results = index.search("hello world", 10);
    assert!(results.len() >= 2);
    // Doc 1 should score higher because bigram "hello\x01world" matches
    assert_eq!(results[0].0, 1, "Adjacent terms should score higher");
}

#[test]
fn test_with_custom_params() {
    let mut index = Bm25Index::with_params(WhitespaceTokenizer, 2.0, 0.5);
    assert_eq!(index.params().k1, 2.0);
    assert_eq!(index.params().b, 0.5);

    index.add_document(1, "test document");
    let results = index.search("test", 10);
    assert_eq!(results.len(), 1);
}
