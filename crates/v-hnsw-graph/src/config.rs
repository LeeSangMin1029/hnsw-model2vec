//! HNSW configuration with builder pattern.

use v_hnsw_core::{Dim, VhnswError};

/// Configuration for an HNSW graph.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct HnswConfig {
    /// Vector dimensionality.
    pub dim: Dim,
    /// Maximum number of neighbors per node on layers >= 1.
    pub m: usize,
    /// Maximum number of neighbors per node on layer 0 (typically `2 * m`).
    pub m0: usize,
    /// Beam width during construction (higher = better quality, slower build).
    pub ef_construction: usize,
    /// Level generation multiplier: `1 / ln(m)`.
    pub ml: f64,
    /// Maximum number of elements the graph can hold.
    pub max_elements: usize,
}

/// Builder for [`HnswConfig`].
pub struct HnswConfigBuilder {
    dim: Option<Dim>,
    m: usize,
    m0: Option<usize>,
    ef_construction: usize,
    max_elements: usize,
}

impl HnswConfig {
    /// Create a new builder for `HnswConfig`.
    pub fn builder() -> HnswConfigBuilder {
        HnswConfigBuilder {
            dim: None,
            m: 16,
            m0: None,
            ef_construction: 200,
            max_elements: 100_000,
        }
    }
}

impl HnswConfigBuilder {
    /// Set the vector dimensionality (required).
    pub fn dim(mut self, dim: Dim) -> Self {
        self.dim = Some(dim);
        self
    }

    /// Set the max neighbors per layer (default: 16). Must be >= 2.
    pub fn m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Override the max neighbors at layer 0 (default: `2 * m`).
    pub fn m0(mut self, m0: usize) -> Self {
        self.m0 = Some(m0);
        self
    }

    /// Set the construction beam width (default: 200).
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set the maximum capacity (default: 100,000).
    pub fn max_elements(mut self, max: usize) -> Self {
        self.max_elements = max;
        self
    }

    /// Build the configuration, validating all parameters.
    ///
    /// # Errors
    ///
    /// Returns `VhnswError::DimensionMismatch` if `dim` is 0 or not set.
    /// Returns `VhnswError::IndexFull` if `m < 2` or `max_elements == 0`.
    pub fn build(self) -> v_hnsw_core::Result<HnswConfig> {
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

        if self.m < 2 {
            return Err(VhnswError::IndexFull {
                capacity: self.m,
            });
        }

        if self.max_elements == 0 {
            return Err(VhnswError::IndexFull { capacity: 0 });
        }

        let m0 = self.m0.unwrap_or_default();
        let m0 = if m0 == 0 { 2 * self.m } else { m0 };
        let ml = 1.0 / (self.m as f64).ln();

        Ok(HnswConfig {
            dim,
            m: self.m,
            m0,
            ef_construction: self.ef_construction,
            ml,
            max_elements: self.max_elements,
        })
    }
}
