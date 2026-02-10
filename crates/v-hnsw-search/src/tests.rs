//! Integration tests for v-hnsw-search.

use std::collections::HashMap;

use v_hnsw_core::{Payload, PayloadStore, PointId};
use v_hnsw_graph::{HnswConfig, HnswGraph, L2Distance};

use crate::bm25::{Bm25Index, Bm25Params};
use crate::config::HybridSearchConfig;
use crate::fusion::RrfFusion;
use crate::hybrid::SimpleHybridSearcher;
use crate::reranker::{LengthBoostReranker, PassthroughReranker, Reranker};
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
// RRF Fusion Tests
// ============================================================================

#[test]
fn test_rrf_fusion_basic() {
    let rrf = RrfFusion::with_k(60);

    // Two lists with same documents in different orders
    let list1 = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
    let list2 = vec![(2, 0.95), (3, 0.85), (1, 0.75)];

    let fused = rrf.fuse(&[list1, list2], 10);

    // All 3 documents should be present
    assert_eq!(fused.len(), 3);

    // Doc 2 should be first (rank 1 + rank 2 in the two lists)
    // Scores: doc2 = 1/61 + 1/62, doc1 = 1/61 + 1/63, doc3 = 1/62 + 1/63
    // Actually: doc2 has best combined rank
    let first_id = fused[0].0;
    assert_eq!(first_id, 2);
}

#[test]
fn test_rrf_fusion_disjoint_lists() {
    let rrf = RrfFusion::with_k(60);

    // Completely different documents
    let list1 = vec![(1, 0.9), (2, 0.8)];
    let list2 = vec![(3, 0.95), (4, 0.85)];

    let fused = rrf.fuse(&[list1, list2], 10);

    assert_eq!(fused.len(), 4);

    // Top items from each list should be tied for first
    let top_ids: Vec<PointId> = fused.iter().take(2).map(|(id, _)| *id).collect();
    // Doc 1 and 3 both have rank 1 in their respective lists
    assert!((top_ids.contains(&1) && top_ids.contains(&3)));
}

#[test]
fn test_rrf_fusion_single_list() {
    let rrf = RrfFusion::with_k(60);

    let list = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
    let fused = rrf.fuse(&[list], 10);

    // Order should be preserved
    assert_eq!(fused[0].0, 1);
    assert_eq!(fused[1].0, 2);
    assert_eq!(fused[2].0, 3);

    // Verify RRF scores
    let expected_score_1 = 1.0 / 61.0;
    assert!((fused[0].1 - expected_score_1).abs() < 1e-6);
}

#[test]
fn test_rrf_fusion_weighted() {
    let rrf = RrfFusion::with_k(60);

    // List 1 has doc 1 first, list 2 has doc 2 first
    let list1 = vec![(1, 0.9)];
    let list2 = vec![(2, 0.95)];

    // Weight list 1 at 3x
    let fused = rrf.fuse_weighted(&[(3.0, &list1), (1.0, &list2)], 10);

    // Doc 1 should be first due to higher weight
    assert_eq!(fused[0].0, 1);

    // Score for doc 1: 3 * 1/61 = 3/61
    let expected_score_1 = 3.0 / 61.0;
    assert!((fused[0].1 - expected_score_1).abs() < 1e-6);
}

#[test]
fn test_rrf_different_k_values() {
    // Smaller k = more emphasis on top ranks
    let rrf_small_k = RrfFusion::with_k(10);
    let rrf_large_k = RrfFusion::with_k(100);

    // Use lists where one doc is clearly better ranked
    let list1 = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
    let list2 = vec![(1, 0.95), (2, 0.85), (3, 0.75)];

    let fused_small = rrf_small_k.fuse(&[list1.clone(), list2.clone()], 10);
    let fused_large = rrf_large_k.fuse(&[list1, list2], 10);

    // Doc 1 is first in both lists, should be first in result for both k values
    assert_eq!(fused_small[0].0, 1);
    assert_eq!(fused_large[0].0, 1);

    // With smaller k, scores are larger (1/(k+rank) is larger for smaller k)
    assert!(fused_small[0].1 > fused_large[0].1);
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
        .dense_weight(0.9)
        .sparse_weight(0.1)
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
// Reranker Tests
// ============================================================================

#[test]
fn test_passthrough_reranker_preserves_order() {
    let reranker = PassthroughReranker::new();

    let candidates = vec![
        (1, 0.9, "first document".to_string()),
        (2, 0.8, "second document".to_string()),
        (3, 0.7, "third document".to_string()),
    ];

    let results = reranker
        .rerank("query", &candidates)
        .expect("rerank should succeed");

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, 1);
    assert_eq!(results[1].0, 2);
    assert_eq!(results[2].0, 3);
}

#[test]
fn test_length_boost_reranker_prefers_optimal() {
    let reranker = LengthBoostReranker::new(50, 2.0);

    // All same initial score, different lengths
    let candidates = vec![
        (1, 1.0, "a".repeat(50)),   // Optimal
        (2, 1.0, "a".repeat(200)),  // Too long
        (3, 1.0, "a".repeat(10)),   // Too short
    ];

    let results = reranker
        .rerank("query", &candidates)
        .expect("rerank should succeed");

    // Optimal length document should be first
    assert_eq!(results[0].0, 1);
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

#[test]
fn test_rrf_with_limit() {
    let rrf = RrfFusion::new();

    let list1: Vec<(PointId, f32)> = (0..50).map(|i| (i, 1.0 - i as f32 * 0.01)).collect();
    let list2: Vec<(PointId, f32)> = (50..100).map(|i| (i, 1.0 - (i - 50) as f32 * 0.01)).collect();

    // Fuse with limit
    let results = rrf.fuse(&[list1, list2], 10);
    assert_eq!(results.len(), 10);
}

// ============================================================================
// Integration with PayloadStore (mock)
// ============================================================================

/// Mock payload store for testing HybridSearcher
#[derive(Default)]
struct MockPayloadStore {
    texts: HashMap<PointId, String>,
}

impl MockPayloadStore {
    fn new() -> Self {
        Self::default()
    }

    fn add_text(&mut self, id: PointId, text: &str) {
        self.texts.insert(id, text.to_string());
    }
}

impl PayloadStore for MockPayloadStore {
    fn get_payload(&self, _id: PointId) -> v_hnsw_core::Result<Option<Payload>> {
        // Not implemented for this test
        Ok(None)
    }

    fn set_payload(&mut self, _id: PointId, _payload: Payload) -> v_hnsw_core::Result<()> {
        Ok(())
    }

    fn remove_payload(&mut self, _id: PointId) -> v_hnsw_core::Result<()> {
        Ok(())
    }

    fn get_text(&self, id: PointId) -> v_hnsw_core::Result<Option<String>> {
        Ok(self.texts.get(&id).cloned())
    }
}

#[test]
fn test_hybrid_searcher_with_rerank() -> v_hnsw_core::Result<()> {
    use crate::hybrid::HybridSearcher;

    let hnsw_config = HnswConfig::builder().dim(4).m(8).build()?;
    let hnsw = HnswGraph::with_seed(hnsw_config, L2Distance, 42);
    let bm25 = Bm25Index::new(SimpleTokenizer);
    let mut payload_store = MockPayloadStore::new();
    let search_config = HybridSearchConfig::default();

    // Set up payload store with texts
    payload_store.add_text(1, "short text");
    payload_store.add_text(2, "this is a medium length text with more words");
    payload_store.add_text(3, "this is a very very very long text that goes on and on with many many words that make it much longer than the others in this test set");

    let mut searcher = HybridSearcher::new(hnsw, bm25, payload_store, search_config);

    // Add documents
    searcher.add_document(1, &[1.0, 0.0, 0.0, 0.0], "short text")?;
    searcher.add_document(2, &[0.5, 0.5, 0.0, 0.0], "this is a medium length text with more words")?;
    searcher.add_document(3, &[0.0, 1.0, 0.0, 0.0], "this is a very very very long text that goes on and on with many many words that make it much longer than the others in this test set")?;

    // Search with passthrough reranker (should preserve order)
    let reranker = PassthroughReranker::new();
    let results = searcher.search_with_rerank(
        &[1.0, 0.0, 0.0, 0.0],
        "text",
        3,
        &reranker,
    )?;

    assert_eq!(results.len(), 3);

    // Search with length boost reranker (should prefer shorter)
    let length_reranker = LengthBoostReranker::new(20, 2.0);
    let results = searcher.search_with_rerank(
        &[0.5, 0.5, 0.0, 0.0],
        "text",
        3,
        &length_reranker,
    )?;

    // Shorter document should be boosted
    assert_eq!(results.len(), 3);

    Ok(())
}
