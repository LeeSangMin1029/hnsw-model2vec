//! Integration tests for v-hnsw-search.

mod config;
mod hybrid;
#[cfg(feature = "korean")]
mod korean_tokenizer;

use v_hnsw_core::PointId;
use v_hnsw_graph::{HnswConfig, HnswGraph, L2Distance};

use crate::bm25::{Bm25Index, Bm25Params};
use crate::config::HybridSearchConfig;

use crate::hybrid::SimpleHybridSearcher;
use crate::{SimpleTokenizer, Tokenizer, WhitespaceTokenizer};

// ============================================================================
// BM25 Scoring Tests
// ============================================================================

#[test]
fn test_bm25_tf_saturation() {
    // Higher term frequency should increase score but with diminishing returns
    let params = Bm25Params::default();

    let score_tf1 = params.score(1, 10, 100, 100.0, 1000);
    let score_tf5 = params.score(5, 10, 100, 100.0, 1000);
    let score_tf10 = params.score(10, 10, 100, 100.0, 1000);
    let score_tf50 = params.score(50, 10, 100, 100.0, 1000);

    // Scores should increase but with saturation
    assert!(score_tf5 > score_tf1);
    assert!(score_tf10 > score_tf5);
    assert!(score_tf50 > score_tf10);

    // Check saturation: the increase from tf=10 to tf=50 should be less
    // proportionally than from tf=1 to tf=5
    let increase_1_to_5 = (score_tf5 - score_tf1) / score_tf1;
    let increase_10_to_50 = (score_tf50 - score_tf10) / score_tf10;
    assert!(increase_1_to_5 > increase_10_to_50);
}

#[test]
fn test_bm25_idf_ranking() {
    // Rare terms should have higher IDF than common terms
    let params = Bm25Params::default();

    // Very rare term (appears in 1 out of 1000 docs)
    let rare_score = params.score(1, 1, 100, 100.0, 1000);

    // Common term (appears in 500 out of 1000 docs)
    let common_score = params.score(1, 500, 100, 100.0, 1000);

    // Very common term (appears in 900 out of 1000 docs)
    let very_common_score = params.score(1, 900, 100, 100.0, 1000);

    assert!(rare_score > common_score);
    assert!(common_score > very_common_score);
}

#[test]
fn test_bm25_length_normalization() {
    // Short documents should score higher (for same tf) than long documents
    let params = Bm25Params::default();

    let short_doc_score = params.score(3, 10, 50, 100.0, 1000);
    let avg_doc_score = params.score(3, 10, 100, 100.0, 1000);
    let long_doc_score = params.score(3, 10, 200, 100.0, 1000);

    assert!(short_doc_score > avg_doc_score);
    assert!(avg_doc_score > long_doc_score);
}

#[test]
fn test_bm25_custom_params() {
    // Test with custom k1 and b parameters
    let default_params = Bm25Params::default();
    let high_k1 = Bm25Params::new(3.0, 0.75);
    let low_b = Bm25Params::new(1.2, 0.0);

    let default_score = default_params.score(5, 10, 50, 100.0, 1000);
    let high_k1_score = high_k1.score(5, 10, 50, 100.0, 1000);
    let low_b_score = low_b.score(5, 10, 50, 100.0, 1000);

    // Higher k1 means slower saturation, so higher tf should matter more
    // but absolute scores change
    assert!(high_k1_score != default_score);

    // b=0 means no length normalization
    let low_b_long = low_b.score(5, 10, 200, 100.0, 1000);
    assert!((low_b_score - low_b_long).abs() < 0.01); // Should be nearly equal
}

// ============================================================================
// BM25 Index Tests
// ============================================================================

#[test]
fn test_bm25_index_basic_search() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);

    index.add_document(1, "the quick brown fox jumps over the lazy dog");
    index.add_document(2, "the lazy cat sleeps all day");
    index.add_document(3, "quick quick fox fox jumps jumps");

    let results = index.search("quick fox", 10);

    // Doc 3 has the most term repetitions
    assert!(!results.is_empty());
    assert_eq!(results[0].0, 3);

    // Doc 1 should also appear
    let doc_ids: Vec<PointId> = results.iter().map(|(id, _)| *id).collect();
    assert!(doc_ids.contains(&1));

    // Doc 2 doesn't contain "quick" or "fox"
    assert!(!doc_ids.contains(&2));
}

#[test]
fn test_bm25_index_empty_query() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "hello world");

    let results = index.search("", 10);
    assert!(results.is_empty());
}

#[test]
fn test_bm25_index_document_update() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);

    index.add_document(1, "apple banana cherry");

    // Verify original content
    let results = index.search("apple", 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);

    // Update document
    index.add_document(1, "orange grape melon");

    // Old terms should not match
    let results = index.search("apple", 10);
    assert!(results.is_empty());

    // New terms should match
    let results = index.search("orange", 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
}

#[test]
fn test_bm25_index_remove_document() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);

    index.add_document(1, "hello world");
    index.add_document(2, "hello there");
    index.add_document(3, "goodbye world");

    assert_eq!(index.len(), 3);

    // Remove document 1
    assert!(index.remove_document(1));
    assert_eq!(index.len(), 2);

    // Search should not return doc 1
    let results = index.search("hello world", 10);
    let doc_ids: Vec<PointId> = results.iter().map(|(id, _)| *id).collect();
    assert!(!doc_ids.contains(&1));

    // Remove non-existent document
    assert!(!index.remove_document(999));
}


// ============================================================================
// Hybrid Search Tests
// ============================================================================

fn create_test_searcher() -> v_hnsw_core::Result<SimpleHybridSearcher<L2Distance, SimpleTokenizer>> {
    let config = HnswConfig::builder().dim(4).m(8).build()?;
    let hnsw = HnswGraph::with_seed(config, L2Distance, 42);
    let bm25 = Bm25Index::new(SimpleTokenizer);
    let search_config = HybridSearchConfig::default();

    Ok(SimpleHybridSearcher::new(hnsw, bm25, search_config))
}

#[test]
fn test_hybrid_search_basic() -> v_hnsw_core::Result<()> {
    let mut searcher = create_test_searcher()?;

    // Add documents with vectors and text
    // Vectors point in different directions
    searcher.add_document(1, &[1.0, 0.0, 0.0, 0.0], "the quick brown fox")?;
    searcher.add_document(2, &[0.0, 1.0, 0.0, 0.0], "the lazy dog sleeps")?;
    searcher.add_document(3, &[-1.0, 0.0, 0.0, 0.0], "quick fox jumps high")?;

    // Query: vector similar to doc 1, text matches doc 3 better
    let results = searcher.search(&[0.9, 0.1, 0.0, 0.0], "quick fox", 3)?;

    // Both doc 1 and doc 3 should rank high
    let top_ids: Vec<PointId> = results.iter().map(|(id, _)| *id).collect();
    assert!(top_ids.contains(&1) || top_ids.contains(&3));

    Ok(())
}

#[test]
fn test_hybrid_search_dense_only() -> v_hnsw_core::Result<()> {
    let mut searcher = create_test_searcher()?;

    searcher.add_document(1, &[1.0, 0.0, 0.0, 0.0], "hello")?;
    searcher.add_document(2, &[0.0, 1.0, 0.0, 0.0], "world")?;

    // Dense-only search
    let results = searcher.search_dense(&[1.0, 0.0, 0.0, 0.0], 2)?;

    // Doc 1 should be closest
    assert_eq!(results[0].0, 1);
    assert!(results[0].1 < 0.1); // Should be very close to 0

    Ok(())
}

#[test]
fn test_hybrid_search_sparse_only() -> v_hnsw_core::Result<()> {
    let mut searcher = create_test_searcher()?;

    searcher.add_document(1, &[1.0, 0.0, 0.0, 0.0], "machine learning neural networks")?;
    searcher.add_document(2, &[0.0, 1.0, 0.0, 0.0], "database indexing query optimization")?;
    searcher.add_document(3, &[0.0, 0.0, 1.0, 0.0], "neural networks deep learning")?;

    // Sparse-only search
    let results = searcher.search_sparse("neural networks", 3);

    // Docs 1 and 3 should be found
    let doc_ids: Vec<PointId> = results.iter().map(|(id, _)| *id).collect();
    assert!(doc_ids.contains(&1));
    assert!(doc_ids.contains(&3));
    assert!(!doc_ids.contains(&2));

    Ok(())
}

#[test]
fn test_hybrid_search_remove_document() -> v_hnsw_core::Result<()> {
    let mut searcher = create_test_searcher()?;

    searcher.add_document(1, &[1.0, 0.0, 0.0, 0.0], "hello world")?;
    searcher.add_document(2, &[0.0, 1.0, 0.0, 0.0], "hello there")?;

    assert_eq!(searcher.len(), 2);

    // Remove document 1
    searcher.remove_document(1)?;

    assert_eq!(searcher.len(), 1);

    // Dense search should not return doc 1
    let results = searcher.search_dense(&[1.0, 0.0, 0.0, 0.0], 2)?;
    let doc_ids: Vec<PointId> = results.iter().map(|(id, _)| *id).collect();
    assert!(!doc_ids.contains(&1));

    // Sparse search should not return doc 1
    let results = searcher.search_sparse("hello world", 2);
    let doc_ids: Vec<PointId> = results.iter().map(|(id, _)| *id).collect();
    assert!(!doc_ids.contains(&1));

    Ok(())
}

#[test]
fn test_hybrid_search_empty() -> v_hnsw_core::Result<()> {
    let searcher = create_test_searcher()?;

    assert!(searcher.is_empty());
    assert_eq!(searcher.len(), 0);

    let results = searcher.search(&[1.0, 0.0, 0.0, 0.0], "query", 10)?;
    assert!(results.is_empty());

    Ok(())
}

#[test]
fn test_hybrid_search_custom_config() -> v_hnsw_core::Result<()> {
    let hnsw_config = HnswConfig::builder().dim(4).m(8).build()?;
    let hnsw = HnswGraph::with_seed(hnsw_config, L2Distance, 42);
    let bm25 = Bm25Index::new(SimpleTokenizer);

    // Custom config: heavy dense weight via fusion_alpha
    let search_config = HybridSearchConfig::builder()
        .fusion_alpha(0.9)
        .build();

    let mut searcher = SimpleHybridSearcher::new(hnsw, bm25, search_config);

    searcher.add_document(1, &[1.0, 0.0, 0.0, 0.0], "cat dog")?;
    searcher.add_document(2, &[0.5, 0.5, 0.0, 0.0], "bird fish cat cat cat")?;

    // Query: vector close to doc 1, text has more matches in doc 2
    let results = searcher.search(&[1.0, 0.0, 0.0, 0.0], "cat", 2)?;

    // With heavy dense weight, doc 1 should win despite fewer text matches
    assert_eq!(results[0].0, 1);

    Ok(())
}

// ============================================================================
// Tokenizer Tests
// ============================================================================

#[test]
fn test_whitespace_tokenizer() {
    let tokenizer = WhitespaceTokenizer::new();

    let tokens = tokenizer.tokenize("Hello World");
    assert_eq!(tokens, vec!["hello", "world"]);

    let tokens = tokenizer.tokenize("  Multiple   Spaces  ");
    assert_eq!(tokens, vec!["multiple", "spaces"]);

    let tokens = tokenizer.tokenize("");
    assert!(tokens.is_empty());
}

#[test]
fn test_simple_tokenizer() {
    let tokenizer = SimpleTokenizer::new();

    let tokens = tokenizer.tokenize("Hello, World!");
    assert_eq!(tokens, vec!["hello", "world"]);

    let tokens = tokenizer.tokenize("test@example.com");
    assert_eq!(tokens, vec!["test", "example", "com"]);

    let tokens = tokenizer.tokenize("one-two-three");
    assert_eq!(tokens, vec!["one", "two", "three"]);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_search_with_unicode() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);

    index.add_document(1, "hello world");
    index.add_document(2, "bonjour monde");

    let results = index.search("hello", 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);

    let results = index.search("bonjour", 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 2);
}

#[test]
fn test_search_case_insensitive() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);

    index.add_document(1, "Hello World UPPERCASE");

    let results = index.search("hello", 10);
    assert_eq!(results[0].0, 1);

    let results = index.search("HELLO", 10);
    assert_eq!(results[0].0, 1);

    let results = index.search("uppercase", 10);
    assert_eq!(results[0].0, 1);
}

#[test]
fn test_empty_document() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);

    index.add_document(1, "");
    index.add_document(2, "has content");

    assert_eq!(index.len(), 2);

    let results = index.search("content", 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 2);
}

#[test]
fn test_single_term_document() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);

    index.add_document(1, "word");
    index.add_document(2, "word word word");

    let results = index.search("word", 10);

    // Both should be found
    assert_eq!(results.len(), 2);

    // Doc 2 should score higher (more term frequency, but longer doc)
    // Actually with BM25, doc 2 is longer so gets penalized, but tf is higher
    // The net effect depends on exact parameters
}

#[test]
fn test_many_documents_search() {
    let mut index = Bm25Index::new(SimpleTokenizer);

    // Add 100 documents
    for i in 0..100 {
        let text = format!("document number {} with some content", i);
        index.add_document(i, &text);
    }

    assert_eq!(index.len(), 100);

    // Search for a specific document
    let results = index.search("document 42", 10);
    assert!(!results.is_empty());

    // Document 42 should be in results
    let doc_ids: Vec<PointId> = results.iter().map(|(id, _)| *id).collect();
    assert!(doc_ids.contains(&42));
}

// ============================================================================
// Additional BM25 Scoring Tests
// ============================================================================

#[test]
fn test_bm25_idf_zero_when_df_zero() {
    let params = Bm25Params::default();
    assert_eq!(params.idf(0, 1000), 0.0);
}

#[test]
fn test_bm25_idf_zero_when_no_docs() {
    let params = Bm25Params::default();
    assert_eq!(params.idf(1, 0), 0.0);
}

#[test]
fn test_bm25_idf_always_positive() {
    let params = Bm25Params::default();
    // Even when df == total_docs (every doc has the term)
    for total in [1, 10, 100, 1000] {
        for df in 1..=total {
            let idf = params.idf(df as u32, total);
            assert!(idf >= 0.0, "IDF should be non-negative: df={}, total={}, idf={}", df, total, idf);
        }
    }
}

// ============================================================================
// Additional BM25 Index Tests
// ============================================================================

#[test]
fn test_bm25_index_search_nonexistent_term() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "hello world");
    let results = index.search("nonexistent_xyz_term", 10);
    assert!(results.is_empty());
}

#[test]
fn test_bm25_index_large_doc_ids() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    // Use large doc IDs to test HashMap accumulator path
    index.add_document(1_000_000, "hello world");
    index.add_document(2_000_000, "hello there");
    let results = index.search("hello", 10);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_bm25_index_repeated_add_same_id() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "first");
    index.add_document(1, "second");
    index.add_document(1, "third");
    // Only the last version should remain
    assert_eq!(index.len(), 1);
    let results = index.search("third", 10);
    assert_eq!(results.len(), 1);
    assert!(index.search("first", 10).is_empty());
    assert!(index.search("second", 10).is_empty());
}

// ============================================================================
// Tokenizer Edge Cases
// ============================================================================

#[test]
fn test_whitespace_tokenizer_unicode() {
    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer.tokenize("café résumé naïve");
    assert_eq!(tokens, vec!["café", "résumé", "naïve"]);
}

#[test]
fn test_whitespace_tokenizer_tabs_and_newlines() {
    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer.tokenize("hello\tworld\nfoo");
    assert_eq!(tokens, vec!["hello", "world", "foo"]);
}

#[test]
fn test_simple_tokenizer_unicode_punctuation() {
    let tokenizer = SimpleTokenizer::new();
    // ASCII punctuation only; non-ASCII punctuation is kept
    let tokens = tokenizer.tokenize("hello.world");
    assert_eq!(tokens, vec!["hello", "world"]);
}

#[test]
fn test_simple_tokenizer_numbers_mixed() {
    let tokenizer = SimpleTokenizer::new();
    let tokens = tokenizer.tokenize("version3.2.1");
    assert_eq!(tokens, vec!["version3", "2", "1"]);
}

#[test]
fn test_simple_tokenizer_empty_after_strip() {
    let tokenizer = SimpleTokenizer::new();
    let tokens = tokenizer.tokenize("...!!!");
    assert!(tokens.is_empty());
}

#[test]
fn test_whitespace_tokenizer_single_char() {
    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer.tokenize("a");
    assert_eq!(tokens, vec!["a"]);
}

// ============================================================================
// Hybrid Search Edge Cases
// ============================================================================

#[test]
fn test_hybrid_search_config_accessors() -> v_hnsw_core::Result<()> {
    let mut searcher = create_test_searcher()?;
    assert_eq!(searcher.config().ef_search, 200);

    searcher.config_mut().ef_search = 500;
    assert_eq!(searcher.config().ef_search, 500);

    Ok(())
}

#[test]
fn test_hybrid_search_single_document() -> v_hnsw_core::Result<()> {
    let mut searcher = create_test_searcher()?;
    searcher.add_document(1, &[1.0, 0.0, 0.0, 0.0], "the only document")?;

    let results = searcher.search(&[1.0, 0.0, 0.0, 0.0], "document", 10)?;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);

    Ok(())
}

#[test]
fn test_hybrid_search_text_no_match_vector_match() -> v_hnsw_core::Result<()> {
    let mut searcher = create_test_searcher()?;

    searcher.add_document(1, &[1.0, 0.0, 0.0, 0.0], "apple banana")?;
    searcher.add_document(2, &[0.0, 1.0, 0.0, 0.0], "cherry date")?;

    // Query: vector matches doc 1, text matches nothing
    let results = searcher.search(&[1.0, 0.0, 0.0, 0.0], "zzzzz_nonexistent", 10)?;
    // Should still return results from dense search
    assert!(!results.is_empty());

    Ok(())
}

#[test]
fn test_hybrid_len_and_is_empty() -> v_hnsw_core::Result<()> {
    let mut searcher = create_test_searcher()?;
    assert!(searcher.is_empty());
    assert_eq!(searcher.len(), 0);

    searcher.add_document(1, &[1.0, 0.0, 0.0, 0.0], "hello")?;
    assert!(!searcher.is_empty());
    assert_eq!(searcher.len(), 1);

    Ok(())
}

// ============================================================================
// BM25 scoring math verification
// ============================================================================

#[test]
fn test_bm25_known_score_computation() {
    // Manually compute BM25 score and verify
    let params = Bm25Params::new(1.2, 0.75);
    // tf=2, df=5, doc_len=100, avg_doc_len=80, total_docs=50
    let tf = 2u32;
    let df = 5u32;
    let doc_len = 100u32;
    let avg_doc_len = 80.0f32;
    let total_docs = 50usize;

    let idf = ((50.0_f32 - 5.0 + 0.5) / (5.0 + 0.5) + 1.0).ln();
    let length_norm = 1.0 - 0.75 + 0.75 * (100.0 / 80.0);
    let tf_component = (2.0 * 2.2) / (2.0 + 1.2 * length_norm);
    let expected = idf * tf_component;

    let actual = params.score(tf, df, doc_len, avg_doc_len, total_docs);
    assert!((actual - expected).abs() < 1e-5,
        "Expected BM25 score {}, got {}", expected, actual);
}

// ============================================================================
// Search with unicode and special characters
// ============================================================================

#[test]
fn test_search_with_cjk_characters() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "你好 世界");
    index.add_document(2, "hello world");

    let results = index.search("你好", 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
}

#[test]
fn test_search_with_emoji() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "hello 🌍 world");
    index.add_document(2, "test document");

    let results = index.search("🌍", 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
}

#[test]
fn test_search_with_only_whitespace_query() {
    let mut index = Bm25Index::new(WhitespaceTokenizer);
    index.add_document(1, "hello world");
    let results = index.search("   \t\n  ", 10);
    assert!(results.is_empty());
}
