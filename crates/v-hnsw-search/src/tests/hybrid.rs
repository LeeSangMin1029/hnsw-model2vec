//! Unit tests for the hybrid search module.

use v_hnsw_core::{PointId, VectorIndex};
use v_hnsw_graph::{HnswConfig, HnswGraph, L2Distance};

use crate::bm25::Bm25Index;
use crate::config::HybridSearchConfig;
use crate::hybrid::SimpleHybridSearcher;
use crate::{SimpleTokenizer, WhitespaceTokenizer};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn make_searcher(dim: usize) -> SimpleHybridSearcher<L2Distance, SimpleTokenizer> {
    let config = HnswConfig::builder().dim(dim).m(8).build().unwrap();
    let hnsw = HnswGraph::with_seed(config, L2Distance, 42);
    let bm25 = Bm25Index::new(SimpleTokenizer);
    let search_config = HybridSearchConfig::default();
    SimpleHybridSearcher::new(hnsw, bm25, search_config)
}

// ---------------------------------------------------------------------------
// Constructor & accessor tests
// ---------------------------------------------------------------------------

#[test]
fn test_new_searcher_is_empty() {
    let s = make_searcher(4);
    assert!(s.is_empty());
    assert_eq!(s.len(), 0);
}

#[test]
fn test_config_accessor() {
    let s = make_searcher(4);
    assert_eq!(s.config().fusion_alpha, 0.5);
    assert_eq!(s.config().ef_search, 200);
}

#[test]
fn test_config_mut_accessor() {
    let mut s = make_searcher(4);
    s.config_mut().ef_search = 64;
    assert_eq!(s.config().ef_search, 64);
}

#[test]
fn test_dense_index_accessor() {
    let s = make_searcher(4);
    assert!(s.dense_index().is_empty());
}

#[test]
fn test_sparse_index_accessor() {
    let s = make_searcher(4);
    assert!(s.sparse_index().is_empty());
}

// ---------------------------------------------------------------------------
// add_document / remove_document
// ---------------------------------------------------------------------------

#[test]
fn test_add_document_updates_both_indexes() {
    let mut s = make_searcher(4);
    s.add_document(1, &[1.0, 0.0, 0.0, 0.0], "hello world").unwrap();
    assert_eq!(s.len(), 1);
    assert!(!s.sparse_index().is_empty());
}

#[test]
fn test_add_multiple_documents() {
    let mut s = make_searcher(4);
    for i in 0..5 {
        let v = [i as f32; 4];
        s.add_document(i, &v, &format!("doc {i}")).unwrap();
    }
    assert_eq!(s.len(), 5);
}

#[test]
fn test_remove_document_updates_both_indexes() {
    let mut s = make_searcher(4);
    s.add_document(1, &[1.0, 0.0, 0.0, 0.0], "hello world").unwrap();
    s.remove_document(1).unwrap();
    assert_eq!(s.len(), 0);

    // Sparse search should return nothing for this doc
    let results = s.search_sparse("hello world", 10);
    let ids: Vec<PointId> = results.iter().map(|&(id, _)| id).collect();
    assert!(!ids.contains(&1));
}

// ---------------------------------------------------------------------------
// search_dense / search_sparse
// ---------------------------------------------------------------------------

#[test]
fn test_search_dense_returns_nearest() {
    let mut s = make_searcher(4);
    s.add_document(1, &[1.0, 0.0, 0.0, 0.0], "a").unwrap();
    s.add_document(2, &[0.0, 1.0, 0.0, 0.0], "b").unwrap();
    s.add_document(3, &[0.0, 0.0, 1.0, 0.0], "c").unwrap();

    let results = s.search_dense(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(results[0].0, 1);
    assert!(results[0].1 < 0.01);
}

#[test]
fn test_search_sparse_returns_matching_docs() {
    let mut s = make_searcher(4);
    s.add_document(1, &[0.0; 4], "rust programming language").unwrap();
    s.add_document(2, &[0.0; 4], "python scripting language").unwrap();
    s.add_document(3, &[0.0; 4], "rust compiler optimization").unwrap();

    let results = s.search_sparse("rust", 10);
    let ids: Vec<PointId> = results.iter().map(|&(id, _)| id).collect();
    assert!(ids.contains(&1));
    assert!(ids.contains(&3));
    assert!(!ids.contains(&2));
}

// ---------------------------------------------------------------------------
// hybrid search
// ---------------------------------------------------------------------------

#[test]
fn test_hybrid_search_combines_signals() {
    let mut s = make_searcher(4);
    // doc 1: close vector, weak text match
    s.add_document(1, &[1.0, 0.0, 0.0, 0.0], "alpha beta").unwrap();
    // doc 2: far vector, strong text match
    s.add_document(2, &[-1.0, 0.0, 0.0, 0.0], "gamma gamma gamma").unwrap();

    let results = s.search(&[0.9, 0.1, 0.0, 0.0], "gamma", 2).unwrap();
    // Both docs should appear
    assert_eq!(results.len(), 2);
}

#[test]
fn test_hybrid_search_empty_index_returns_empty() {
    let s = make_searcher(4);
    let results = s.search(&[1.0, 0.0, 0.0, 0.0], "hello", 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_hybrid_search_k_limits_results() {
    let mut s = make_searcher(4);
    for i in 0..10 {
        s.add_document(i, &[i as f32, 0.0, 0.0, 0.0], "common word").unwrap();
    }
    let results = s.search(&[0.0, 0.0, 0.0, 0.0], "common", 3).unwrap();
    assert!(results.len() <= 3);
}

// ---------------------------------------------------------------------------
// fusion alpha extremes
// ---------------------------------------------------------------------------

#[test]
fn test_fusion_alpha_dense_only() {
    let config = HnswConfig::builder().dim(4).m(8).build().unwrap();
    let hnsw = HnswGraph::with_seed(config, L2Distance, 42);
    let bm25 = Bm25Index::new(SimpleTokenizer);
    let search_config = HybridSearchConfig::builder().fusion_alpha(1.0).build();
    let mut s = SimpleHybridSearcher::new(hnsw, bm25, search_config);

    s.add_document(1, &[1.0, 0.0, 0.0, 0.0], "no match text").unwrap();
    s.add_document(2, &[0.0, 1.0, 0.0, 0.0], "query match text").unwrap();

    // alpha=1.0 means dense-only; vector-nearest should win
    let results = s.search(&[1.0, 0.0, 0.0, 0.0], "query match text", 2).unwrap();
    assert_eq!(results[0].0, 1);
}

#[test]
fn test_fusion_alpha_sparse_only() {
    let config = HnswConfig::builder().dim(4).m(8).build().unwrap();
    let hnsw = HnswGraph::with_seed(config, L2Distance, 42);
    let bm25 = Bm25Index::new(WhitespaceTokenizer);
    let search_config = HybridSearchConfig::builder().fusion_alpha(0.0).build();
    let mut s = SimpleHybridSearcher::new(hnsw, bm25, search_config);

    s.add_document(1, &[1.0, 0.0, 0.0, 0.0], "unrelated words").unwrap();
    s.add_document(2, &[0.0, 1.0, 0.0, 0.0], "target keyword keyword").unwrap();

    // alpha=0.0 means sparse-only; text match should win
    let results = s.search(&[1.0, 0.0, 0.0, 0.0], "target keyword", 2).unwrap();
    assert_eq!(results[0].0, 2);
}

// ---------------------------------------------------------------------------
// len / is_empty boundary
// ---------------------------------------------------------------------------

#[test]
fn test_len_after_add_and_remove() {
    let mut s = make_searcher(4);
    s.add_document(10, &[1.0; 4], "a").unwrap();
    s.add_document(20, &[2.0; 4], "b").unwrap();
    assert_eq!(s.len(), 2);
    assert!(!s.is_empty());

    s.remove_document(10).unwrap();
    assert_eq!(s.len(), 1);

    s.remove_document(20).unwrap();
    assert_eq!(s.len(), 0);
    assert!(s.is_empty());
}
