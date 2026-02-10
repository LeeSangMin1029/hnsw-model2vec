//! Hybrid search combining dense and sparse retrieval.

use std::collections::HashSet;

use v_hnsw_core::{DistanceMetric, PayloadStore, PointId, VectorIndex};
use v_hnsw_graph::HnswGraph;

use crate::bm25::Bm25Index;
use crate::config::HybridSearchConfig;
use crate::fusion::ConvexFusion;
use crate::reranker::Reranker;
use crate::Tokenizer;

/// A hybrid searcher combining dense vector search with sparse BM25 search.
///
/// Uses Convex Combination fusion to combine results from both methods.
/// Stores document texts internally in the BM25 index.
pub struct SimpleHybridSearcher<D, T>
where
    D: DistanceMetric,
    T: Tokenizer,
{
    dense_index: HnswGraph<D>,
    sparse_index: Bm25Index<T>,
    config: HybridSearchConfig,
    fusion: ConvexFusion,
}

impl<D, T> SimpleHybridSearcher<D, T>
where
    D: DistanceMetric,
    T: Tokenizer,
{
    /// Create a new simple hybrid searcher.
    pub fn new(
        dense_index: HnswGraph<D>,
        sparse_index: Bm25Index<T>,
        config: HybridSearchConfig,
    ) -> Self {
        let fusion = ConvexFusion::with_alpha(config.fusion_alpha);
        Self { dense_index, sparse_index, config, fusion }
    }

    pub fn config(&self) -> &HybridSearchConfig { &self.config }
    pub fn config_mut(&mut self) -> &mut HybridSearchConfig { &mut self.config }
    pub fn dense_index(&self) -> &HnswGraph<D> { &self.dense_index }
    pub fn dense_index_mut(&mut self) -> &mut HnswGraph<D> { &mut self.dense_index }
    pub fn sparse_index(&self) -> &Bm25Index<T> { &self.sparse_index }
    pub fn sparse_index_mut(&mut self) -> &mut Bm25Index<T> { &mut self.sparse_index }

    /// Add a document to both indexes.
    pub fn add_document(&mut self, id: PointId, vector: &[f32], text: &str) -> v_hnsw_core::Result<()> {
        self.dense_index.insert(id, vector)?;
        self.sparse_index.add_document(id, text);
        Ok(())
    }

    /// Remove a document from both indexes.
    pub fn remove_document(&mut self, id: PointId) -> v_hnsw_core::Result<()> {
        self.dense_index.delete(id)?;
        self.sparse_index.remove_document(id);
        Ok(())
    }

    /// Perform hybrid search combining dense and sparse results.
    pub fn search(
        &self,
        query_vector: &[f32],
        query_text: &str,
        k: usize,
    ) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        let dense = self.dense_index.search(query_vector, self.config.dense_limit, self.config.ef_search)?;
        let sparse = self.sparse_index.search(query_text, self.config.sparse_limit);
        let all_sparse = enrich_sparse(&self.sparse_index, query_text, &dense, sparse);
        Ok(self.fusion.fuse(&dense, &all_sparse, k))
    }

    /// Perform hybrid search reading vectors from an external store.
    pub fn search_ext(
        &self,
        store: &dyn v_hnsw_core::VectorStore,
        query_vector: &[f32],
        query_text: &str,
        k: usize,
    ) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        let dense = self.dense_index.search_ext(store, query_vector, self.config.dense_limit, self.config.ef_search)?;
        let sparse = self.sparse_index.search(query_text, self.config.sparse_limit);
        let all_sparse = enrich_sparse(&self.sparse_index, query_text, &dense, sparse);
        Ok(self.fusion.fuse(&dense, &all_sparse, k))
    }

    /// Perform dense-only search.
    pub fn search_dense(&self, query_vector: &[f32], k: usize) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        self.dense_index.search(query_vector, k, self.config.ef_search)
    }

    /// Perform sparse-only search.
    pub fn search_sparse(&self, query_text: &str, k: usize) -> Vec<(PointId, f32)> {
        self.sparse_index.search(query_text, k)
    }

    pub fn len(&self) -> usize { self.dense_index.len() }
    pub fn is_empty(&self) -> bool { self.dense_index.is_empty() }
}

/// Extended hybrid searcher with a payload store for reranking.
///
/// Wraps [`SimpleHybridSearcher`] and adds `search_with_rerank` using document text.
pub struct HybridSearcher<D, T, P>
where
    D: DistanceMetric,
    T: Tokenizer,
    P: PayloadStore,
{
    inner: SimpleHybridSearcher<D, T>,
    payload_store: P,
}

impl<D, T, P> HybridSearcher<D, T, P>
where
    D: DistanceMetric,
    T: Tokenizer,
    P: PayloadStore,
{
    pub fn new(
        dense_index: HnswGraph<D>,
        sparse_index: Bm25Index<T>,
        payload_store: P,
        config: HybridSearchConfig,
    ) -> Self {
        Self {
            inner: SimpleHybridSearcher::new(dense_index, sparse_index, config),
            payload_store,
        }
    }

    // -- Delegate common methods --
    pub fn config(&self) -> &HybridSearchConfig { self.inner.config() }
    pub fn config_mut(&mut self) -> &mut HybridSearchConfig { self.inner.config_mut() }
    pub fn dense_index(&self) -> &HnswGraph<D> { self.inner.dense_index() }
    pub fn dense_index_mut(&mut self) -> &mut HnswGraph<D> { self.inner.dense_index_mut() }
    pub fn sparse_index(&self) -> &Bm25Index<T> { self.inner.sparse_index() }
    pub fn sparse_index_mut(&mut self) -> &mut Bm25Index<T> { self.inner.sparse_index_mut() }

    pub fn add_document(&mut self, id: PointId, vector: &[f32], text: &str) -> v_hnsw_core::Result<()> {
        self.inner.add_document(id, vector, text)
    }
    pub fn remove_document(&mut self, id: PointId) -> v_hnsw_core::Result<()> {
        self.inner.remove_document(id)
    }
    pub fn search(&self, query_vector: &[f32], query_text: &str, k: usize) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        self.inner.search(query_vector, query_text, k)
    }
    pub fn search_dense(&self, query_vector: &[f32], k: usize) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        self.inner.search_dense(query_vector, k)
    }
    pub fn search_sparse(&self, query_text: &str, k: usize) -> Vec<(PointId, f32)> {
        self.inner.search_sparse(query_text, k)
    }
    pub fn len(&self) -> usize { self.inner.len() }
    pub fn is_empty(&self) -> bool { self.inner.is_empty() }

    // -- Own methods --
    pub fn payload_store(&self) -> &P { &self.payload_store }
    pub fn payload_store_mut(&mut self) -> &mut P { &mut self.payload_store }

    /// Perform hybrid search with reranking.
    pub fn search_with_rerank<R: Reranker>(
        &self,
        query_vector: &[f32],
        query_text: &str,
        k: usize,
        reranker: &R,
    ) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        let candidates = self.search(query_vector, query_text, k * 3)?;

        let mut docs_with_text: Vec<(PointId, f32, String)> = Vec::with_capacity(candidates.len());
        for (id, score) in candidates {
            let text = self.payload_store.get_text(id)?.unwrap_or_default();
            docs_with_text.push((id, score, text));
        }

        let mut results = reranker.rerank(query_text, &docs_with_text)?;
        results.truncate(k);
        Ok(results)
    }
}

/// Enrich BM25 results with scores for dense candidates not in BM25 top-k.
fn enrich_sparse<T: Tokenizer>(
    sparse_index: &Bm25Index<T>,
    query_text: &str,
    dense_results: &[(PointId, f32)],
    mut sparse_results: Vec<(PointId, f32)>,
) -> Vec<(PointId, f32)> {
    if dense_results.is_empty() {
        return sparse_results;
    }

    let sparse_ids: HashSet<PointId> = sparse_results.iter().map(|&(id, _)| id).collect();
    let missing: Vec<PointId> = dense_results
        .iter()
        .filter(|&&(id, _)| !sparse_ids.contains(&id))
        .map(|&(id, _)| id)
        .collect();

    if !missing.is_empty() {
        let extra = sparse_index.score_documents(query_text, &missing);
        sparse_results.extend(extra);
    }

    sparse_results
}
