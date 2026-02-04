//! BM25 scoring implementation.
//!
//! Implements the Okapi BM25 ranking function for term-document scoring.

use serde::{Deserialize, Serialize};

/// BM25 scoring parameters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Bm25Params {
    /// Term frequency saturation parameter (default: 1.2).
    pub k1: f32,
    /// Document length normalization parameter (default: 0.75).
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

impl Bm25Params {
    /// Create new BM25 parameters with custom values.
    pub fn new(k1: f32, b: f32) -> Self {
        Self { k1, b }
    }

    /// Compute the BM25 score for a single term in a document.
    ///
    /// # Parameters
    /// - `tf`: Term frequency in the document
    /// - `df`: Document frequency (number of documents containing the term)
    /// - `doc_len`: Length of the document (number of tokens)
    /// - `avg_doc_len`: Average document length in the corpus
    /// - `total_docs`: Total number of documents in the corpus
    ///
    /// # Returns
    /// The BM25 score contribution for this term.
    #[inline]
    pub fn score(
        &self,
        tf: u32,
        df: u32,
        doc_len: u32,
        avg_doc_len: f32,
        total_docs: usize,
    ) -> f32 {
        if df == 0 || total_docs == 0 {
            return 0.0;
        }

        // IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        let n = total_docs as f32;
        let df_f = df as f32;
        let idf = ((n - df_f + 0.5) / (df_f + 0.5) + 1.0).ln();

        // TF normalization
        let tf_f = tf as f32;
        let doc_len_f = doc_len as f32;
        let length_norm = 1.0 - self.b + self.b * (doc_len_f / avg_doc_len);
        let tf_norm = (tf_f * (self.k1 + 1.0)) / (tf_f + self.k1 * length_norm);

        idf * tf_norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = Bm25Params::default();
        assert!((params.k1 - 1.2).abs() < f32::EPSILON);
        assert!((params.b - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_score_basic() {
        let params = Bm25Params::default();
        // Term appears once in a doc of avg length
        let score = params.score(1, 1, 100, 100.0, 100);
        assert!(score > 0.0);
    }

    #[test]
    fn test_score_zero_df() {
        let params = Bm25Params::default();
        let score = params.score(1, 0, 100, 100.0, 100);
        assert!((score - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_score_zero_docs() {
        let params = Bm25Params::default();
        let score = params.score(1, 1, 100, 100.0, 0);
        assert!((score - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_higher_tf_higher_score() {
        let params = Bm25Params::default();
        let score1 = params.score(1, 10, 100, 100.0, 100);
        let score2 = params.score(5, 10, 100, 100.0, 100);
        assert!(score2 > score1);
    }

    #[test]
    fn test_rarer_term_higher_idf() {
        let params = Bm25Params::default();
        // Rare term (appears in 1 doc)
        let rare_score = params.score(1, 1, 100, 100.0, 100);
        // Common term (appears in 50 docs)
        let common_score = params.score(1, 50, 100, 100.0, 100);
        assert!(rare_score > common_score);
    }

    #[test]
    fn test_shorter_doc_higher_score() {
        let params = Bm25Params::default();
        // Short document
        let short_score = params.score(1, 10, 50, 100.0, 100);
        // Long document
        let long_score = params.score(1, 10, 200, 100.0, 100);
        assert!(short_score > long_score);
    }
}
