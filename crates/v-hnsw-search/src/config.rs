//! Configuration for hybrid search.

/// Configuration for hybrid search combining dense and sparse retrieval.
#[derive(Debug, Clone)]
pub struct HybridSearchConfig {
    /// Weight for dense (vector) search results in fusion (default: 0.5).
    pub dense_weight: f32,
    /// Weight for sparse (BM25) search results in fusion (default: 0.5).
    pub sparse_weight: f32,
    /// HNSW ef parameter for search quality (default: 200).
    pub ef_search: usize,
    /// Maximum number of dense results to retrieve (default: 100).
    pub dense_limit: usize,
    /// Maximum number of sparse results to retrieve (default: 100).
    pub sparse_limit: usize,
    /// RRF k parameter (default: 60).
    pub rrf_k: u32,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            dense_weight: 0.5,
            sparse_weight: 0.5,
            ef_search: 200,
            dense_limit: 100,
            sparse_limit: 100,
            rrf_k: 60,
        }
    }
}

impl HybridSearchConfig {
    /// Create a new builder for HybridSearchConfig.
    pub fn builder() -> HybridSearchConfigBuilder {
        HybridSearchConfigBuilder::default()
    }
}

/// Builder for [`HybridSearchConfig`].
#[derive(Debug, Clone, Default)]
pub struct HybridSearchConfigBuilder {
    dense_weight: Option<f32>,
    sparse_weight: Option<f32>,
    ef_search: Option<usize>,
    dense_limit: Option<usize>,
    sparse_limit: Option<usize>,
    rrf_k: Option<u32>,
}

impl HybridSearchConfigBuilder {
    /// Set the dense search weight (default: 0.5).
    pub fn dense_weight(mut self, weight: f32) -> Self {
        self.dense_weight = Some(weight);
        self
    }

    /// Set the sparse search weight (default: 0.5).
    pub fn sparse_weight(mut self, weight: f32) -> Self {
        self.sparse_weight = Some(weight);
        self
    }

    /// Set the HNSW ef_search parameter (default: 200).
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        self
    }

    /// Set the maximum dense results (default: 100).
    pub fn dense_limit(mut self, limit: usize) -> Self {
        self.dense_limit = Some(limit);
        self
    }

    /// Set the maximum sparse results (default: 100).
    pub fn sparse_limit(mut self, limit: usize) -> Self {
        self.sparse_limit = Some(limit);
        self
    }

    /// Set the RRF k parameter (default: 60).
    pub fn rrf_k(mut self, k: u32) -> Self {
        self.rrf_k = Some(k);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> HybridSearchConfig {
        let defaults = HybridSearchConfig::default();
        HybridSearchConfig {
            dense_weight: self.dense_weight.unwrap_or(defaults.dense_weight),
            sparse_weight: self.sparse_weight.unwrap_or(defaults.sparse_weight),
            ef_search: self.ef_search.unwrap_or(defaults.ef_search),
            dense_limit: self.dense_limit.unwrap_or(defaults.dense_limit),
            sparse_limit: self.sparse_limit.unwrap_or(defaults.sparse_limit),
            rrf_k: self.rrf_k.unwrap_or(defaults.rrf_k),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HybridSearchConfig::default();
        assert!((config.dense_weight - 0.5).abs() < f32::EPSILON);
        assert!((config.sparse_weight - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.ef_search, 200);
        assert_eq!(config.dense_limit, 100);
        assert_eq!(config.sparse_limit, 100);
        assert_eq!(config.rrf_k, 60);
    }

    #[test]
    fn test_builder() {
        let config = HybridSearchConfig::builder()
            .dense_weight(0.7)
            .sparse_weight(0.3)
            .ef_search(300)
            .dense_limit(50)
            .sparse_limit(200)
            .rrf_k(100)
            .build();

        assert!((config.dense_weight - 0.7).abs() < f32::EPSILON);
        assert!((config.sparse_weight - 0.3).abs() < f32::EPSILON);
        assert_eq!(config.ef_search, 300);
        assert_eq!(config.dense_limit, 50);
        assert_eq!(config.sparse_limit, 200);
        assert_eq!(config.rrf_k, 100);
    }

    #[test]
    fn test_builder_partial() {
        let config = HybridSearchConfig::builder().dense_weight(0.8).build();

        assert!((config.dense_weight - 0.8).abs() < f32::EPSILON);
        // Other values should be defaults
        assert!((config.sparse_weight - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.ef_search, 200);
    }
}
