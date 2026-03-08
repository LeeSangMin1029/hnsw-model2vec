//! Configuration for hybrid search.

/// Configuration for hybrid search combining dense and sparse retrieval.
#[derive(Debug, Clone)]
pub struct HybridSearchConfig {
    /// HNSW ef parameter for search quality (default: 200).
    pub ef_search: usize,
    /// Maximum number of dense results to retrieve (default: 100).
    pub dense_limit: usize,
    /// Maximum number of sparse results to retrieve (default: 100).
    pub sparse_limit: usize,
    /// Convex fusion alpha (default: 0.5). 0=sparse only, 1=dense only.
    pub fusion_alpha: f32,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            ef_search: 200,
            dense_limit: 100,
            sparse_limit: 100,
            fusion_alpha: 0.5,
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
    ef_search: Option<usize>,
    dense_limit: Option<usize>,
    sparse_limit: Option<usize>,
    fusion_alpha: Option<f32>,
}

impl HybridSearchConfigBuilder {
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

    /// Set the convex fusion alpha (default: 0.5).
    /// 0.0 = sparse only, 1.0 = dense only.
    pub fn fusion_alpha(mut self, alpha: f32) -> Self {
        self.fusion_alpha = Some(alpha);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> HybridSearchConfig {
        let defaults = HybridSearchConfig::default();
        HybridSearchConfig {
            ef_search: self.ef_search.unwrap_or(defaults.ef_search),
            dense_limit: self.dense_limit.unwrap_or(defaults.dense_limit),
            sparse_limit: self.sparse_limit.unwrap_or(defaults.sparse_limit),
            fusion_alpha: self.fusion_alpha.unwrap_or(defaults.fusion_alpha),
        }
    }
}