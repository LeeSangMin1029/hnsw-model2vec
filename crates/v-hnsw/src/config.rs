//! Configuration enums for VectorDb.

/// Distance metric for vector comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Metric {
    /// L2 (Euclidean) squared distance.
    L2,
    /// Cosine distance (1 - cosine similarity).
    #[default]
    Cosine,
    /// Negative dot product (higher similarity = lower distance).
    DotProduct,
}

impl Metric {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Metric::L2 => "l2",
            Metric::Cosine => "cosine",
            Metric::DotProduct => "dot_product",
        }
    }
}

/// Vector quantization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quantization {
    /// No quantization (full precision f32).
    #[default]
    None,
    /// Scalar quantization (8-bit).
    SQ8,
    /// Product quantization with configurable subvectors.
    PQ {
        /// Number of subvectors for PQ.
        num_subvectors: usize,
    },
}

impl Quantization {
    /// Check if quantization is enabled.
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Quantization::None)
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Quantization::None => "none",
            Quantization::SQ8 => "sq8",
            Quantization::PQ { .. } => "pq",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_defaults() {
        assert_eq!(Metric::default(), Metric::Cosine);
    }

    #[test]
    fn test_metric_names() {
        assert_eq!(Metric::L2.name(), "l2");
        assert_eq!(Metric::Cosine.name(), "cosine");
        assert_eq!(Metric::DotProduct.name(), "dot_product");
    }

    #[test]
    fn test_quantization_defaults() {
        assert_eq!(Quantization::default(), Quantization::None);
    }

    #[test]
    fn test_quantization_is_enabled() {
        assert!(!Quantization::None.is_enabled());
        assert!(Quantization::SQ8.is_enabled());
        assert!(Quantization::PQ { num_subvectors: 8 }.is_enabled());
    }

    #[test]
    fn test_quantization_names() {
        assert_eq!(Quantization::None.name(), "none");
        assert_eq!(Quantization::SQ8.name(), "sq8");
        assert_eq!(Quantization::PQ { num_subvectors: 8 }.name(), "pq");
    }
}
