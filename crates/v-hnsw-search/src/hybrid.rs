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
/// Uses Reciprocal Rank Fusion (RRF) to combine results from both methods.
///
/// # Type Parameters
/// - `D`: Distance metric for the HNSW graph
/// - `T`: Tokenizer for BM25 indexing
/// - `P`: Payload store for retrieving document text
pub struct HybridSearcher<D, T, P>
where
    D: DistanceMetric,
    T: Tokenizer,
    P: PayloadStore,
{
    /// Dense vector index (HNSW).
    dense_index: HnswGraph<D>,
    /// Sparse text index (BM25).
    sparse_index: Bm25Index<T>,
    /// Payload store for document text.
    payload_store: P,
    /// Search configuration.
    config: HybridSearchConfig,
    /// Convex fusion combiner.
    fusion: ConvexFusion,
}

impl<D, T, P> HybridSearcher<D, T, P>
where
    D: DistanceMetric,
    T: Tokenizer,
    P: PayloadStore,
{
    /// Create a new hybrid searcher.
    pub fn new(
        dense_index: HnswGraph<D>,
        sparse_index: Bm25Index<T>,
        payload_store: P,
        config: HybridSearchConfig,
    ) -> Self {
        let fusion = ConvexFusion::with_alpha(config.fusion_alpha);
        Self {
            dense_index,
            sparse_index,
            payload_store,
            config,
            fusion,
        }
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &HybridSearchConfig {
        &self.config
    }

    /// Get a mutable reference to the configuration.
    pub fn config_mut(&mut self) -> &mut HybridSearchConfig {
        &mut self.config
    }

    /// Get a reference to the dense index.
    pub fn dense_index(&self) -> &HnswGraph<D> {
        &self.dense_index
    }

    /// Get a mutable reference to the dense index.
    pub fn dense_index_mut(&mut self) -> &mut HnswGraph<D> {
        &mut self.dense_index
    }

    /// Get a reference to the sparse index.
    pub fn sparse_index(&self) -> &Bm25Index<T> {
        &self.sparse_index
    }

    /// Get a mutable reference to the sparse index.
    pub fn sparse_index_mut(&mut self) -> &mut Bm25Index<T> {
        &mut self.sparse_index
    }

    /// Get a reference to the payload store.
    pub fn payload_store(&self) -> &P {
        &self.payload_store
    }

    /// Get a mutable reference to the payload store.
    pub fn payload_store_mut(&mut self) -> &mut P {
        &mut self.payload_store
    }

    /// Add a document to both dense and sparse indexes.
    ///
    /// # Parameters
    /// - `id`: Unique document identifier
    /// - `vector`: Dense embedding vector
    /// - `text`: Document text for BM25 indexing
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
    ///
    /// # Parameters
    /// - `query_vector`: Dense query vector
    /// - `query_text`: Query text for BM25
    /// - `k`: Number of results to return
    ///
    /// # Returns
    /// Results sorted by fused RRF score descending.
    pub fn search(
        &self,
        query_vector: &[f32],
        query_text: &str,
        k: usize,
    ) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        let dense_results = self.dense_index.search(
            query_vector,
            self.config.dense_limit,
            self.config.ef_search,
        )?;

        let sparse_results = self
            .sparse_index
            .search(query_text, self.config.sparse_limit);

        // Dense-Guided: score dense candidates missing from BM25 top-k
        let all_sparse = enrich_sparse(
            &self.sparse_index,
            query_text,
            &dense_results,
            sparse_results,
        );

        let results = self.fusion.fuse(&dense_results, &all_sparse, k);
        Ok(results)
    }

    /// Perform dense-only search (vector similarity).
    ///
    /// # Parameters
    /// - `query_vector`: Dense query vector
    /// - `k`: Number of results to return
    pub fn search_dense(
        &self,
        query_vector: &[f32],
        k: usize,
    ) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        self.dense_index
            .search(query_vector, k, self.config.ef_search)
    }

    /// Perform sparse-only search (BM25 text matching).
    ///
    /// # Parameters
    /// - `query_text`: Query text
    /// - `k`: Number of results to return
    pub fn search_sparse(&self, query_text: &str, k: usize) -> Vec<(PointId, f32)> {
        self.sparse_index.search(query_text, k)
    }

    /// Perform hybrid search with reranking.
    ///
    /// # Parameters
    /// - `query_vector`: Dense query vector
    /// - `query_text`: Query text for BM25
    /// - `k`: Number of results to return
    /// - `reranker`: Reranker implementation
    ///
    /// # Returns
    /// Results after reranking.
    pub fn search_with_rerank<R: Reranker>(
        &self,
        query_vector: &[f32],
        query_text: &str,
        k: usize,
        reranker: &R,
    ) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        // Get more results than needed for reranking
        let candidate_count = k * 3;
        let candidates = self.search(query_vector, query_text, candidate_count)?;

        // Collect document texts for reranking
        let mut docs_with_text: Vec<(PointId, f32, String)> = Vec::with_capacity(candidates.len());
        for (id, score) in candidates {
            let text = self
                .payload_store
                .get_text(id)?
                .unwrap_or_default();
            docs_with_text.push((id, score, text));
        }

        // Rerank
        let reranked = reranker.rerank(query_text, &docs_with_text)?;

        // Truncate to k
        let mut results = reranked;
        results.truncate(k);
        Ok(results)
    }

    /// Total number of documents in the searcher.
    ///
    /// Returns the count from the dense index.
    pub fn len(&self) -> usize {
        self.dense_index.len()
    }

    /// Check if the searcher is empty.
    pub fn is_empty(&self) -> bool {
        self.dense_index.is_empty()
    }
}

/// A simpler hybrid searcher that doesn't require a separate payload store.
///
/// Stores document texts internally in the BM25 index.
/// Use this when you don't have a separate payload store.
pub struct SimpleHybridSearcher<D, T>
where
    D: DistanceMetric,
    T: Tokenizer,
{
    /// Dense vector index (HNSW).
    dense_index: HnswGraph<D>,
    /// Sparse text index (BM25).
    sparse_index: Bm25Index<T>,
    /// Search configuration.
    config: HybridSearchConfig,
    /// Convex fusion combiner.
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
        Self {
            dense_index,
            sparse_index,
            config,
            fusion,
        }
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &HybridSearchConfig {
        &self.config
    }

    /// Get a reference to the dense index.
    pub fn dense_index(&self) -> &HnswGraph<D> {
        &self.dense_index
    }

    /// Get a mutable reference to the dense index.
    pub fn dense_index_mut(&mut self) -> &mut HnswGraph<D> {
        &mut self.dense_index
    }

    /// Get a reference to the sparse index.
    pub fn sparse_index(&self) -> &Bm25Index<T> {
        &self.sparse_index
    }

    /// Get a mutable reference to the sparse index.
    pub fn sparse_index_mut(&mut self) -> &mut Bm25Index<T> {
        &mut self.sparse_index
    }

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

    /// Perform hybrid search using the graph's internal vector store.
    pub fn search(
        &self,
        query_vector: &[f32],
        query_text: &str,
        k: usize,
    ) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        let dense_results = self.dense_index.search(
            query_vector,
            self.config.dense_limit,
            self.config.ef_search,
        )?;

        let sparse_results = self
            .sparse_index
            .search(query_text, self.config.sparse_limit);

        let all_sparse = enrich_sparse(
            &self.sparse_index,
            query_text,
            &dense_results,
            sparse_results,
        );

        let results = self.fusion.fuse(&dense_results, &all_sparse, k);
        Ok(results)
    }

    /// Perform hybrid search reading vectors from an external store.
    ///
    /// Avoids `populate_store()` by reading directly from mmap or other
    /// external vector storage.
    pub fn search_ext(
        &self,
        store: &dyn v_hnsw_core::VectorStore,
        query_vector: &[f32],
        query_text: &str,
        k: usize,
    ) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        let dense_results = self.dense_index.search_ext(
            store,
            query_vector,
            self.config.dense_limit,
            self.config.ef_search,
        )?;

        let sparse_results = self
            .sparse_index
            .search(query_text, self.config.sparse_limit);

        let all_sparse = enrich_sparse(
            &self.sparse_index,
            query_text,
            &dense_results,
            sparse_results,
        );

        let results = self.fusion.fuse(&dense_results, &all_sparse, k);
        Ok(results)
    }

    /// Perform dense-only search.
    pub fn search_dense(
        &self,
        query_vector: &[f32],
        k: usize,
    ) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        self.dense_index
            .search(query_vector, k, self.config.ef_search)
    }

    /// Perform sparse-only search.
    pub fn search_sparse(&self, query_text: &str, k: usize) -> Vec<(PointId, f32)> {
        self.sparse_index.search(query_text, k)
    }

    /// Total number of documents.
    pub fn len(&self) -> usize {
        self.dense_index.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.dense_index.is_empty()
    }
}

/// Enrich BM25 results with scores for dense candidates not in BM25 top-k.
///
/// Ensures all dense candidates get a BM25 score for proper fusion,
/// without running a second full BM25 scan. Uses binary search on sorted
/// posting lists — O(|missing| * |terms| * log(df)) per candidate.
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
