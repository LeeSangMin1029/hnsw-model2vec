//! FieldNorm: 256-entry LUT for BM25 length normalization.
//!
//! Replaces per-scoring `doc_len / avg_doc_len` float division with a single
//! byte lookup. Quantization follows Tantivy/Lucene's log-scale approach.
//!
//! Encode: `doc_len -> u8 code` (log2 scale, 0–255)
//! Decode: `code -> approximate doc_len`
//! LUT:    `code -> 1 / (1 - b + b * decoded_len / avg_dl)`

/// Encode a document length to a single-byte field norm code.
///
/// Uses log2 scaling to map any doc_len into 0..=255.
/// Zero-length documents map to code 0.
#[inline]
pub fn encode(doc_len: u32) -> u8 {
    if doc_len == 0 {
        return 0;
    }
    // log2(doc_len) * (256 / 32) = log2(doc_len) * 8
    // This maps doc_len 1 -> 0, doc_len ~4B -> 255
    let val = (doc_len as f32).log2() * 8.0;
    val.round().clamp(0.0, 255.0) as u8
}

/// Decode a field norm code back to an approximate document length.
#[inline]
pub fn decode(code: u8) -> f32 {
    if code == 0 {
        return 1.0; // minimum doc length
    }
    // Inverse of encode: 2^(code / 8)
    2.0f32.powf(code as f32 / 8.0)
}

/// 256-entry lookup table for BM25 length normalization.
///
/// Each entry stores the **reciprocal** of the length norm factor:
/// `lut[code] = 1.0 / (1.0 - b + b * decode(code) / avg_dl)`
///
/// Usage in scoring: `score = idf * tf_f * (k1 + 1) / (tf_f + k1 / lut[code])`
/// which is equivalent to: `score = idf * tf_f * (k1 + 1) / (tf_f + k1 * length_norm)`
#[derive(Debug)]
pub struct FieldNormLut {
    /// `inv_length_norm[code]`: reciprocal of length normalization factor.
    inv_length_norm: [f32; 256],
}

impl FieldNormLut {
    /// Build the LUT for given BM25 `b` parameter and average document length.
    pub fn build(b: f32, avg_doc_len: f32) -> Self {
        let mut inv_length_norm = [0.0f32; 256];
        for (code, entry) in inv_length_norm.iter_mut().enumerate() {
            let decoded_len = decode(code as u8);
            let length_norm = 1.0 - b + b * (decoded_len / avg_doc_len);
            *entry = if length_norm > f32::EPSILON {
                1.0 / length_norm
            } else {
                1.0
            };
        }
        Self { inv_length_norm }
    }

    /// Compute TF normalization using the cached LUT.
    ///
    /// Equivalent to `Bm25Params::tf_norm()` but avoids float division.
    #[inline]
    pub fn tf_norm(&self, k1: f32, tf: u32, code: u8) -> f32 {
        let tf_f = tf as f32;
        let inv_ln = self.inv_length_norm[code as usize];
        // tf_norm = tf * (k1 + 1) / (tf + k1 * length_norm)
        //         = tf * (k1 + 1) / (tf + k1 / inv_length_norm)
        //         = tf * (k1 + 1) * inv_ln / (tf * inv_ln + k1)
        (tf_f * (k1 + 1.0) * inv_ln) / (tf_f * inv_ln + k1)
    }
}
