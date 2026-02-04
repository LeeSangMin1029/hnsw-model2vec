//! Scalar Quantization (SQ8): maps each `f32` dimension to `u8` [0, 255].
//!
//! Per-dimension min/max values are learned during training. Encoding scales
//! linearly into the `[0, 255]` range. Decoding reverses the mapping.

use v_hnsw_core::{Quantizer, Result, VhnswError};

/// Compressed representation produced by [`Sq8Quantizer`].
#[derive(Debug, Clone)]
pub struct Sq8Encoded {
    /// One byte per dimension.
    pub data: Vec<u8>,
}

/// Scalar quantizer that compresses `f32` vectors to `u8` per dimension.
///
/// # Training
///
/// Call [`Quantizer::train`] with representative vectors to learn per-dimension
/// min/max bounds. Encoding before training returns an error.
pub struct Sq8Quantizer {
    dim: usize,
    /// Per-dimension minimum value observed during training.
    mins: Vec<f32>,
    /// Per-dimension range `(max - min)`. Zero for constant dimensions.
    ranges: Vec<f32>,
    trained: bool,
}

impl Sq8Quantizer {
    /// Create a new untrained SQ8 quantizer for vectors of the given dimensionality.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            mins: Vec::new(),
            ranges: Vec::new(),
            trained: false,
        }
    }

    /// Decode an SQ8-encoded vector back to approximate `f32` values.
    ///
    /// # Errors
    ///
    /// Returns an error if the quantizer has not been trained.
    pub fn decode(&self, encoded: &Sq8Encoded) -> Result<Vec<f32>> {
        if !self.trained {
            return Err(VhnswError::Quantization(
                "quantizer not trained".to_string(),
            ));
        }
        Ok(encoded
            .data
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                let scale = self.ranges[i] / 255.0;
                b as f32 * scale + self.mins[i]
            })
            .collect())
    }

    /// Fast integer-only L2-squared distance (uniform scaling assumption).
    ///
    /// Returns the raw integer sum of squared differences. This is useful when
    /// all dimensions have roughly equal range and you only need a ranking, not
    /// a calibrated distance value.
    #[must_use]
    pub fn distance_encoded_fast(a: &Sq8Encoded, b: &Sq8Encoded) -> u32 {
        a.data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| {
                let d = x as i16 - y as i16;
                (d * d) as u32
            })
            .sum()
    }
}

impl Quantizer for Sq8Quantizer {
    type Encoded = Sq8Encoded;

    fn train(&mut self, vectors: &[&[f32]]) -> Result<()> {
        if vectors.is_empty() {
            return Err(VhnswError::Quantization(
                "training set is empty".to_string(),
            ));
        }

        // Validate dimensions.
        for v in vectors {
            if v.len() != self.dim {
                return Err(VhnswError::DimensionMismatch {
                    expected: self.dim,
                    got: v.len(),
                });
            }
        }

        let mut mins = vec![f32::INFINITY; self.dim];
        let mut maxs = vec![f32::NEG_INFINITY; self.dim];

        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                if val < mins[i] {
                    mins[i] = val;
                }
                if val > maxs[i] {
                    maxs[i] = val;
                }
            }
        }

        let ranges: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&mn, &mx)| {
                let r = mx - mn;
                if r == 0.0 { 0.0 } else { r }
            })
            .collect();

        self.mins = mins;
        self.ranges = ranges;
        self.trained = true;
        Ok(())
    }

    fn encode(&self, vector: &[f32]) -> Result<Sq8Encoded> {
        if !self.trained {
            return Err(VhnswError::Quantization(
                "quantizer not trained".to_string(),
            ));
        }
        if vector.len() != self.dim {
            return Err(VhnswError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }

        let data: Vec<u8> = vector
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                if self.ranges[i] == 0.0 {
                    // Constant dimension -- encode at midpoint.
                    128
                } else {
                    let normalized = ((val - self.mins[i]) / self.ranges[i]).clamp(0.0, 1.0);
                    // Round to nearest u8.
                    (normalized * 255.0 + 0.5) as u8
                }
            })
            .collect();

        Ok(Sq8Encoded { data })
    }

    fn distance_encoded(&self, a: &Sq8Encoded, b: &Sq8Encoded) -> f32 {
        let mut sum = 0.0_f32;
        for i in 0..self.dim {
            let scale = self.ranges[i] / 255.0;
            let diff = (a.data[i] as f32 - b.data[i] as f32) * scale;
            sum += diff * diff;
        }
        sum
    }

    fn compression_ratio(&self) -> f32 {
        // f32 = 4 bytes per dimension, u8 = 1 byte per dimension.
        4.0
    }
}
