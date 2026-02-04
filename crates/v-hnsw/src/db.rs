//! VectorDb facade and builder.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::RwLock;
use v_hnsw_core::{Dim, Payload, PointId, Result, VhnswError, VectorIndex};
use v_hnsw_distance::{CosineDistance, DotProductDistance, L2Distance};
use v_hnsw_graph::{HnswConfig, HnswGraph};
use v_hnsw_search::{Bm25Index, HybridSearchConfig, RrfFusion, SimpleTokenizer, Tokenizer as SearchTokenizer};

#[cfg(feature = "korean")]
use v_hnsw_tokenizer::{KoreanTokenizer, Tokenizer as KoreanTokenizerTrait};

use crate::config::{Metric, Quantization};
use crate::result::SearchResult;

/// Builder for constructing a VectorDb instance.
#[derive(Debug)]
pub struct VectorDbBuilder {
    dim: Option<Dim>,
    metric: Metric,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    quantization: Quantization,
    korean: bool,
    storage_path: Option<PathBuf>,
    max_elements: usize,
}

impl Default for VectorDbBuilder {
    fn default() -> Self {
        Self {
            dim: None,
            metric: Metric::Cosine,
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            quantization: Quantization::None,
            korean: false,
            storage_path: None,
            max_elements: 100_000,
        }
    }
}

impl VectorDbBuilder {
    /// Set the vector dimension (required).
    pub fn dim(mut self, dim: Dim) -> Self {
        self.dim = Some(dim);
        self
    }

    /// Set the distance metric (default: Cosine).
    pub fn metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Set HNSW max connections per node (default: 16).
    pub fn m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Set HNSW construction beam width (default: 200).
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set HNSW search beam width (default: 100).
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Set quantization strategy (default: None).
    pub fn quantization(mut self, q: Quantization) -> Self {
        self.quantization = q;
        self
    }

    /// Enable Korean tokenizer for hybrid search (default: false).
    /// Requires the `korean` feature.
    pub fn korean(mut self, enabled: bool) -> Self {
        self.korean = enabled;
        self
    }

    /// Set storage path for persistence (default: in-memory only).
    pub fn storage_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.storage_path = Some(path.into());
        self
    }

    /// Set maximum number of elements (default: 100,000).
    pub fn max_elements(mut self, max: usize) -> Self {
        self.max_elements = max;
        self
    }

    /// Build the VectorDb.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `dim` is not set or is 0
    /// - Korean tokenizer is requested but feature is not enabled
    pub fn build(self) -> Result<VectorDb> {
        let dim = self.dim.ok_or(VhnswError::DimensionMismatch {
            expected: 1,
            got: 0,
        })?;

        if dim == 0 {
            return Err(VhnswError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }

        let hnsw_config = HnswConfig::builder()
            .dim(dim)
            .m(self.m)
            .ef_construction(self.ef_construction)
            .max_elements(self.max_elements)
            .build()?;

        // Create the appropriate distance metric and graph
        let graph = match self.metric {
            Metric::L2 => {
                let graph = HnswGraph::new(hnsw_config, L2Distance);
                GraphWrapper::L2(graph)
            }
            Metric::Cosine => {
                let graph = HnswGraph::new(hnsw_config, CosineDistance);
                GraphWrapper::Cosine(graph)
            }
            Metric::DotProduct => {
                let graph = HnswGraph::new(hnsw_config, DotProductDistance);
                GraphWrapper::DotProduct(graph)
            }
        };

        // Create tokenizer wrapper (Korean or Simple)
        let tokenizer = if self.korean {
            #[cfg(feature = "korean")]
            {
                let kt = KoreanTokenizer::new()
                    .map_err(|e| VhnswError::Tokenizer(e.to_string()))?;
                TokenizerWrapper::Korean(Box::new(kt))
            }
            #[cfg(not(feature = "korean"))]
            {
                return Err(VhnswError::Tokenizer(
                    "Korean tokenizer requires the 'korean' feature".to_string(),
                ));
            }
        } else {
            TokenizerWrapper::Simple(SimpleTokenizer::new())
        };

        // BM25 index always uses SimpleTokenizer for serialization compatibility.
        // The VectorDb wraps tokenization to preprocess with Korean if enabled.
        let bm25 = Bm25Index::new(SimpleTokenizer::new());

        let hybrid_config = HybridSearchConfig::builder()
            .ef_search(self.ef_search)
            .build();

        Ok(VectorDb {
            inner: RwLock::new(VectorDbInner {
                graph,
                bm25,
                texts: HashMap::new(),
                vectors: HashMap::new(),
            }),
            tokenizer: Arc::new(tokenizer),
            dim,
            metric: self.metric,
            ef_search: self.ef_search,
            quantization: self.quantization,
            storage_path: self.storage_path,
            hybrid_config,
        })
    }
}

/// Internal graph wrapper to handle different distance metrics.
enum GraphWrapper {
    L2(HnswGraph<L2Distance>),
    Cosine(HnswGraph<CosineDistance>),
    DotProduct(HnswGraph<DotProductDistance>),
}

impl GraphWrapper {
    fn insert(&mut self, id: PointId, vector: &[f32]) -> Result<()> {
        match self {
            GraphWrapper::L2(g) => g.insert(id, vector),
            GraphWrapper::Cosine(g) => g.insert(id, vector),
            GraphWrapper::DotProduct(g) => g.insert(id, vector),
        }
    }

    fn delete(&mut self, id: PointId) -> Result<()> {
        match self {
            GraphWrapper::L2(g) => g.delete(id),
            GraphWrapper::Cosine(g) => g.delete(id),
            GraphWrapper::DotProduct(g) => g.delete(id),
        }
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<(PointId, f32)>> {
        match self {
            GraphWrapper::L2(g) => g.search(query, k, ef),
            GraphWrapper::Cosine(g) => g.search(query, k, ef),
            GraphWrapper::DotProduct(g) => g.search(query, k, ef),
        }
    }

    fn len(&self) -> usize {
        match self {
            GraphWrapper::L2(g) => g.len(),
            GraphWrapper::Cosine(g) => g.len(),
            GraphWrapper::DotProduct(g) => g.len(),
        }
    }
}

/// Internal tokenizer wrapper.
enum TokenizerWrapper {
    Simple(SimpleTokenizer),
    #[cfg(feature = "korean")]
    Korean(Box<KoreanTokenizer>),
}

impl TokenizerWrapper {
    /// Tokenize text and return space-joined string for BM25 indexing.
    fn preprocess_for_bm25(&self, text: &str) -> String {
        match self {
            TokenizerWrapper::Simple(t) => {
                // Already space-separated tokens
                t.tokenize(text).join(" ")
            }
            #[cfg(feature = "korean")]
            TokenizerWrapper::Korean(t) => {
                // Use Korean tokenizer and join tokens with spaces
                match t.tokenize(text) {
                    Ok(tokens) => tokens.into_iter().map(|tok| tok.text).collect::<Vec<_>>().join(" "),
                    Err(_) => text.to_string(),
                }
            }
        }
    }
}

// SAFETY: TokenizerWrapper contains only Send + Sync types:
// - SimpleTokenizer: derives Send + Sync
// - KoreanTokenizer: Lindera tokenizer is thread-safe
#[allow(unsafe_code)]
unsafe impl Send for TokenizerWrapper {}
#[allow(unsafe_code)]
unsafe impl Sync for TokenizerWrapper {}

/// Internal state protected by RwLock.
struct VectorDbInner {
    graph: GraphWrapper,
    bm25: Bm25Index<SimpleTokenizer>,
    texts: HashMap<PointId, String>,
    vectors: HashMap<PointId, Vec<f32>>,
}

/// A unified vector database combining HNSW graph and BM25 index.
///
/// Supports dense vector search, sparse text search, and hybrid search
/// combining both methods using Reciprocal Rank Fusion (RRF).
///
/// # Example
///
/// ```rust,ignore
/// use v_hnsw::{VectorDb, Metric};
///
/// let db = VectorDb::builder()
///     .dim(384)
///     .metric(Metric::Cosine)
///     .korean(true)
///     .build()?;
///
/// // Insert with text for hybrid search
/// db.insert_with_text(1, &embedding, "안녕하세요")?;
///
/// // Hybrid search
/// let results = db.hybrid_search(&query_vec, "검색어", 10)?;
/// ```
pub struct VectorDb {
    inner: RwLock<VectorDbInner>,
    tokenizer: Arc<TokenizerWrapper>,
    dim: Dim,
    metric: Metric,
    ef_search: usize,
    quantization: Quantization,
    storage_path: Option<PathBuf>,
    hybrid_config: HybridSearchConfig,
}

impl VectorDb {
    /// Create a new builder for VectorDb.
    pub fn builder() -> VectorDbBuilder {
        VectorDbBuilder::default()
    }

    /// Insert a vector (dense only, no text).
    pub fn insert(&self, id: PointId, vector: &[f32]) -> Result<()> {
        self.validate_dim(vector.len())?;
        let mut inner = self.inner.write();
        inner.graph.insert(id, vector)?;
        inner.vectors.insert(id, vector.to_vec());
        Ok(())
    }

    /// Insert a vector with associated text for hybrid search.
    pub fn insert_with_text(&self, id: PointId, vector: &[f32], text: &str) -> Result<()> {
        self.validate_dim(vector.len())?;

        // Preprocess text with tokenizer (Korean or Simple)
        let preprocessed = self.tokenizer.preprocess_for_bm25(text);

        let mut inner = self.inner.write();
        inner.graph.insert(id, vector)?;
        inner.bm25.add_document(id, &preprocessed);
        inner.texts.insert(id, text.to_string());
        inner.vectors.insert(id, vector.to_vec());
        Ok(())
    }

    /// Insert a vector with payload and text.
    pub fn insert_full(
        &self,
        id: PointId,
        vector: &[f32],
        _payload: Payload,
        text: &str,
    ) -> Result<()> {
        // Note: Payload storage requires StorageEngine integration.
        // For now, we store the text but not the payload in the in-memory implementation.
        self.insert_with_text(id, vector, text)
    }

    /// Delete a point from the database.
    pub fn delete(&self, id: PointId) -> Result<()> {
        let mut inner = self.inner.write();
        inner.graph.delete(id)?;
        inner.bm25.remove_document(id);
        inner.texts.remove(&id);
        inner.vectors.remove(&id);
        Ok(())
    }

    /// Search for nearest neighbors (dense vector search only).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.validate_dim(query.len())?;
        let inner = self.inner.read();
        let results = inner.graph.search(query, k, self.ef_search)?;
        Ok(results.into_iter().map(SearchResult::from).collect())
    }

    /// Hybrid search combining dense vector and sparse text search.
    ///
    /// Uses Reciprocal Rank Fusion (RRF) to combine results.
    pub fn hybrid_search(
        &self,
        query_vector: &[f32],
        query_text: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        self.validate_dim(query_vector.len())?;

        // Preprocess query text with tokenizer
        let preprocessed_query = self.tokenizer.preprocess_for_bm25(query_text);

        let inner = self.inner.read();

        // Dense search
        let dense_results = inner.graph.search(
            query_vector,
            self.hybrid_config.dense_limit,
            self.ef_search,
        )?;

        // Sparse search
        let sparse_results = inner.bm25.search(&preprocessed_query, self.hybrid_config.sparse_limit);

        // Fuse with RRF
        let fusion = RrfFusion::with_k(self.hybrid_config.rrf_k);
        let weighted_lists = vec![
            (self.hybrid_config.dense_weight, dense_results),
            (self.hybrid_config.sparse_weight, sparse_results),
        ];
        let fused = fusion.fuse_weighted(&weighted_lists, k);

        Ok(fused.into_iter().map(SearchResult::from).collect())
    }

    /// Get the vector for a point.
    pub fn get(&self, id: PointId) -> Result<Option<Vec<f32>>> {
        let inner = self.inner.read();
        Ok(inner.vectors.get(&id).cloned())
    }

    /// Get the text associated with a point.
    pub fn get_text(&self, id: PointId) -> Result<Option<String>> {
        let inner = self.inner.read();
        Ok(inner.texts.get(&id).cloned())
    }

    /// Number of points in the database.
    pub fn len(&self) -> usize {
        self.inner.read().graph.len()
    }

    /// Whether the database is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the vector dimension.
    pub fn dim(&self) -> Dim {
        self.dim
    }

    /// Get the distance metric.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Get the quantization strategy.
    pub fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// Get the storage path (if persistent).
    pub fn storage_path(&self) -> Option<&PathBuf> {
        self.storage_path.as_ref()
    }

    /// Save the database to disk.
    ///
    /// Requires `storage_path` to be set during construction.
    pub fn save(&self) -> Result<()> {
        let _path = self.storage_path.as_ref().ok_or_else(|| {
            VhnswError::Storage(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "storage_path not set",
            ))
        })?;
        // TODO: Implement full persistence with StorageEngine
        // For now, this is a placeholder for the API.
        Ok(())
    }

    /// Create a checkpoint (flush to disk).
    ///
    /// Requires `storage_path` to be set during construction.
    pub fn checkpoint(&self) -> Result<()> {
        self.save()
    }

    /// Validate that a vector has the expected dimension.
    fn validate_dim(&self, len: usize) -> Result<()> {
        if len != self.dim {
            return Err(VhnswError::DimensionMismatch {
                expected: self.dim,
                got: len,
            });
        }
        Ok(())
    }
}

// SAFETY: VectorDb is Send + Sync because:
// - VectorDbInner is protected by RwLock (which is Send + Sync when T is)
// - TokenizerWrapper is wrapped in Arc (which is Send + Sync when T is)
// - All contained types (HnswGraph, Bm25Index, HashMap) are Send + Sync
#[allow(unsafe_code)]
unsafe impl Send for VectorDb {}
#[allow(unsafe_code)]
unsafe impl Sync for VectorDb {}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vector(id: u64, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|j| (id as f32 * 0.1 + j as f32 * 0.3).sin())
            .collect()
    }

    #[test]
    fn test_builder_no_dim() {
        let result = VectorDb::builder().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_zero_dim() {
        let result = VectorDb::builder().dim(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_defaults() {
        let db = VectorDb::builder().dim(128).build();
        assert!(db.is_ok());
        let db = db.ok();
        assert!(db.is_some());
        let db = db.as_ref();
        assert!(db.is_some());
        let db = db.map(|d| {
            assert_eq!(d.dim(), 128);
            assert_eq!(d.metric(), Metric::Cosine);
            assert_eq!(d.quantization(), Quantization::None);
        });
        assert!(db.is_some());
    }

    #[test]
    fn test_builder_custom() {
        let db = VectorDb::builder()
            .dim(384)
            .metric(Metric::L2)
            .m(32)
            .ef_construction(400)
            .ef_search(200)
            .quantization(Quantization::SQ8)
            .max_elements(50_000)
            .build();
        assert!(db.is_ok());
    }

    #[test]
    fn test_insert_and_search() {
        let db = VectorDb::builder().dim(16).build();
        assert!(db.is_ok());
        let db = db.ok();
        assert!(db.is_some());
        let db = db.map(|db| {
            for i in 0..10 {
                let vec = test_vector(i, 16);
                let result = db.insert(i, &vec);
                assert!(result.is_ok());
            }

            assert_eq!(db.len(), 10);

            let query = test_vector(5, 16);
            let results = db.search(&query, 3);
            assert!(results.is_ok());
            let results = results.ok();
            assert!(results.is_some());
            results.map(|results| {
                assert_eq!(results.len(), 3);
                assert_eq!(results[0].id, 5);
            });
        });
        assert!(db.is_some());
    }

    #[test]
    fn test_insert_with_text() {
        let db = VectorDb::builder().dim(4).build();
        assert!(db.is_ok());
        let db = db.ok();
        assert!(db.is_some());
        let db = db.map(|db| {
            let result = db.insert_with_text(1, &[1.0, 2.0, 3.0, 4.0], "hello world");
            assert!(result.is_ok());

            let text = db.get_text(1);
            assert!(text.is_ok());
            let text = text.ok();
            assert!(text.is_some());
            let text = text.as_ref();
            assert!(text.is_some());
            text.map(|t| {
                assert!(t.is_some());
                let t = t.as_ref();
                assert!(t.is_some());
                t.map(|s| assert_eq!(s, "hello world"));
            });
        });
        assert!(db.is_some());
    }

    #[test]
    fn test_hybrid_search() {
        let db = VectorDb::builder().dim(4).build();
        assert!(db.is_ok());
        let db = db.ok();
        assert!(db.is_some());
        let db = db.map(|db| {
            let _ = db.insert_with_text(1, &[1.0, 0.0, 0.0, 0.0], "the quick brown fox");
            let _ = db.insert_with_text(2, &[0.0, 1.0, 0.0, 0.0], "the lazy dog");
            let _ = db.insert_with_text(3, &[0.0, 0.0, 1.0, 0.0], "quick quick fox fox");

            // Search for "fox" with a vector near point 3
            let query_vec = [0.1, 0.1, 0.9, 0.0];
            let results = db.hybrid_search(&query_vec, "fox", 3);
            assert!(results.is_ok());
            let results = results.ok();
            assert!(results.is_some());
            results.map(|results| {
                assert!(!results.is_empty());
                // Point 3 should rank high (matches both vector and text)
            });
        });
        assert!(db.is_some());
    }

    #[test]
    fn test_delete() {
        let db = VectorDb::builder().dim(4).build();
        assert!(db.is_ok());
        let db = db.ok();
        assert!(db.is_some());
        let db = db.map(|db| {
            let _ = db.insert(1, &[1.0, 2.0, 3.0, 4.0]);
            let _ = db.insert(2, &[5.0, 6.0, 7.0, 8.0]);
            assert_eq!(db.len(), 2);

            let result = db.delete(1);
            assert!(result.is_ok());
            assert_eq!(db.len(), 1);

            let vec = db.get(1);
            assert!(vec.is_ok());
            let vec = vec.ok();
            assert!(vec.is_some());
            vec.map(|v| assert!(v.is_none()));
        });
        assert!(db.is_some());
    }

    #[test]
    fn test_dimension_mismatch_insert() {
        let db = VectorDb::builder().dim(4).build();
        assert!(db.is_ok());
        let db = db.ok();
        assert!(db.is_some());
        let db = db.map(|db| {
            let result = db.insert(1, &[1.0, 2.0]); // Wrong dimension
            assert!(result.is_err());
        });
        assert!(db.is_some());
    }

    #[test]
    fn test_dimension_mismatch_search() {
        let db = VectorDb::builder().dim(4).build();
        assert!(db.is_ok());
        let db = db.ok();
        assert!(db.is_some());
        let db = db.map(|db| {
            let _ = db.insert(1, &[1.0, 2.0, 3.0, 4.0]);
            let result = db.search(&[1.0, 2.0], 1); // Wrong dimension
            assert!(result.is_err());
        });
        assert!(db.is_some());
    }

    #[test]
    fn test_get_vector() {
        let db = VectorDb::builder().dim(4).build();
        assert!(db.is_ok());
        let db = db.ok();
        assert!(db.is_some());
        let db = db.map(|db| {
            let vec = vec![1.0, 2.0, 3.0, 4.0];
            let _ = db.insert(42, &vec);

            let retrieved = db.get(42);
            assert!(retrieved.is_ok());
            let retrieved = retrieved.ok();
            assert!(retrieved.is_some());
            retrieved.map(|r| assert_eq!(r, Some(vec)));
        });
        assert!(db.is_some());
    }

    #[test]
    fn test_different_metrics() {
        // L2
        let db_l2 = VectorDb::builder().dim(4).metric(Metric::L2).build();
        assert!(db_l2.is_ok());

        // DotProduct
        let db_dot = VectorDb::builder()
            .dim(4)
            .metric(Metric::DotProduct)
            .build();
        assert!(db_dot.is_ok());
    }

    #[test]
    fn test_empty_db() {
        let db = VectorDb::builder().dim(4).build();
        assert!(db.is_ok());
        let db = db.ok();
        assert!(db.is_some());
        let db = db.map(|db| {
            assert!(db.is_empty());
            assert_eq!(db.len(), 0);

            let results = db.search(&[1.0, 2.0, 3.0, 4.0], 10);
            assert!(results.is_ok());
            let results = results.ok();
            assert!(results.is_some());
            results.map(|r| assert!(r.is_empty()));
        });
        assert!(db.is_some());
    }
}
