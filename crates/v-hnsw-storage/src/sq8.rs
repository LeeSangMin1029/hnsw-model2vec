//! SQ8 scalar quantization: f32 → u8 with per-dimension min/max linear mapping.
//!
//! Each dimension is independently mapped from [min, max] to [0, 255].
//! Training requires a single pass over the vectors to collect per-dimension bounds.
//! Asymmetric distance: query stays f32, database vectors are u8.

use std::path::Path;

use v_hnsw_core::Result;

/// Per-dimension quantization parameters.
///
/// For dimension `i`:
/// - quantize: `q[i] = round((v[i] - mins[i]) * inv_ranges[i] * 255)`
/// - dequantize: `v[i] ≈ mins[i] + q[i] * ranges[i] / 255`
#[derive(Debug, Clone)]
pub struct Sq8Params {
    /// Per-dimension minimum values.
    pub mins: Vec<f32>,
    /// Per-dimension range (max - min). Zero range → constant dimension.
    pub ranges: Vec<f32>,
    /// Pre-computed `1.0 / range` for each dimension (0.0 if range is zero).
    inv_ranges: Vec<f32>,
    /// Pre-computed lookup table for asymmetric distance.
    /// For each dimension `d` and quantized value `q` (0..256):
    /// `lut[d * 256 + q]` = dequantized value of `q` in dimension `d`.
    dequant_lut: Vec<f32>,
}

impl Sq8Params {
    /// Train SQ8 parameters from a set of vectors.
    ///
    /// Performs a single pass to find per-dimension min/max bounds.
    /// Requires at least one vector; returns error if `vectors` is empty.
    pub fn train(dim: usize, vectors: &[&[f32]]) -> Result<Self> {
        if vectors.is_empty() {
            return Err(v_hnsw_core::VhnswError::InvalidArgument(
                "SQ8 training requires at least one vector".into(),
            ));
        }

        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];

        for vec in vectors {
            debug_assert_eq!(vec.len(), dim);
            for (i, &v) in vec.iter().enumerate() {
                if v < mins[i] {
                    mins[i] = v;
                }
                if v > maxs[i] {
                    maxs[i] = v;
                }
            }
        }

        let ranges: Vec<f32> = mins.iter().zip(maxs.iter()).map(|(&mn, &mx)| mx - mn).collect();
        let inv_ranges: Vec<f32> = ranges
            .iter()
            .map(|&r| if r > f32::EPSILON { 1.0 / r } else { 0.0 })
            .collect();

        let dequant_lut = build_dequant_lut(&mins, &ranges);

        Ok(Self {
            mins,
            ranges,
            inv_ranges,
            dequant_lut,
        })
    }

    /// Quantize a single f32 vector to u8.
    #[inline]
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        debug_assert_eq!(vector.len(), self.mins.len());
        vector
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let normalized = (v - self.mins[i]) * self.inv_ranges[i];
                // Clamp to [0, 1] then scale to [0, 255]
                (normalized.clamp(0.0, 1.0) * 255.0).round() as u8
            })
            .collect()
    }

    /// Quantize directly into a pre-allocated buffer.
    #[inline]
    pub fn quantize_into(&self, vector: &[f32], out: &mut [u8]) {
        debug_assert_eq!(vector.len(), self.mins.len());
        debug_assert_eq!(out.len(), self.mins.len());
        for (i, &v) in vector.iter().enumerate() {
            let normalized = (v - self.mins[i]) * self.inv_ranges[i];
            out[i] = (normalized.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }

    /// Dequantize a u8 vector back to f32 (approximate).
    #[inline]
    pub fn dequantize(&self, codes: &[u8]) -> Vec<f32> {
        debug_assert_eq!(codes.len(), self.mins.len());
        codes
            .iter()
            .enumerate()
            .map(|(i, &q)| self.dequant_lut[i * 256 + q as usize])
            .collect()
    }

    /// Asymmetric dot-product distance for normalized cosine:
    /// `1 - dot(query_f32, dequant(code_u8))`.
    ///
    /// Uses pre-computed lookup table for fast dequantization.
    #[inline]
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        debug_assert_eq!(query.len(), codes.len());
        let dim = query.len();
        let mut dot = 0.0_f32;
        for i in 0..dim {
            let dequant = self.dequant_lut[i * 256 + codes[i] as usize];
            dot += query[i] * dequant;
        }
        1.0 - dot.clamp(-1.0, 1.0)
    }

    /// Dimension count.
    pub fn dim(&self) -> usize {
        self.mins.len()
    }

    /// Save parameters to a binary file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let dim = self.mins.len() as u32;
        let mut buf = Vec::with_capacity(4 + self.mins.len() * 8);
        buf.extend_from_slice(&dim.to_le_bytes());
        for &m in &self.mins {
            buf.extend_from_slice(&m.to_le_bytes());
        }
        for &r in &self.ranges {
            buf.extend_from_slice(&r.to_le_bytes());
        }
        std::fs::write(path, &buf)?;
        Ok(())
    }

    /// Load parameters from a binary file.
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        if data.len() < 4 {
            return Err(v_hnsw_core::VhnswError::InvalidArgument(
                "SQ8 params file too short".into(),
            ));
        }
        let dim = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let expected = 4 + dim * 8;
        if data.len() < expected {
            return Err(v_hnsw_core::VhnswError::InvalidArgument(format!(
                "SQ8 params file truncated: expected {expected}, got {}",
                data.len()
            )));
        }

        let mut mins = Vec::with_capacity(dim);
        let mut ranges = Vec::with_capacity(dim);
        for i in 0..dim {
            let offset = 4 + i * 4;
            mins.push(f32::from_le_bytes(
                data[offset..offset + 4].try_into().unwrap(),
            ));
        }
        for i in 0..dim {
            let offset = 4 + dim * 4 + i * 4;
            ranges.push(f32::from_le_bytes(
                data[offset..offset + 4].try_into().unwrap(),
            ));
        }

        let inv_ranges: Vec<f32> = ranges
            .iter()
            .map(|&r| if r > f32::EPSILON { 1.0 / r } else { 0.0 })
            .collect();
        let dequant_lut = build_dequant_lut(&mins, &ranges);

        Ok(Self {
            mins,
            ranges,
            inv_ranges,
            dequant_lut,
        })
    }
}

/// Build the dequantization lookup table: `lut[d * 256 + q] = min[d] + q * range[d] / 255`.
fn build_dequant_lut(mins: &[f32], ranges: &[f32]) -> Vec<f32> {
    let dim = mins.len();
    let mut lut = vec![0.0_f32; dim * 256];
    for d in 0..dim {
        let scale = ranges[d] / 255.0;
        for q in 0..256 {
            lut[d * 256 + q] = mins[d] + q as f32 * scale;
        }
    }
    lut
}
