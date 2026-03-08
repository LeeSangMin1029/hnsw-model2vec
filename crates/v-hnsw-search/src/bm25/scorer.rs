//! BM25 scoring implementation.
//!
//! Implements the Okapi BM25 ranking function for term-document scoring.

use serde::{Deserialize, Serialize};

/// BM25 scoring parameters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
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

    /// Compute the IDF (inverse document frequency) component.
    ///
    /// Call once per query term, then reuse for all documents.
    #[inline]
    pub fn idf(&self, df: u32, total_docs: usize) -> f32 {
        if df == 0 || total_docs == 0 {
            return 0.0;
        }
        let n = total_docs as f32;
        let df_f = df as f32;
        ((n - df_f + 0.5) / (df_f + 0.5) + 1.0).ln()
    }

    /// Compute the TF normalization component (without IDF).
    ///
    /// Multiply by `idf()` to get the full BM25 score.
    #[inline]
    pub fn tf_norm(&self, tf: u32, doc_len: u32, avg_doc_len: f32) -> f32 {
        let tf_f = tf as f32;
        let doc_len_f = doc_len as f32;
        let length_norm = 1.0 - self.b + self.b * (doc_len_f / avg_doc_len);
        (tf_f * (self.k1 + 1.0)) / (tf_f + self.k1 * length_norm)
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
        self.idf(df, total_docs) * self.tf_norm(tf, doc_len, avg_doc_len)
    }
}