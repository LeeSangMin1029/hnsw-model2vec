//! Product Quantization (PQ): codebook-based vector compression.
//!
//! Each vector is split into `n_sub` subvectors. For every subspace a codebook
//! of `n_codes` centroids is learned via k-means. Encoding replaces each
//! subvector with the index of its nearest centroid (`u8` when `n_codes <= 256`).

use rayon::prelude::*;
use v_hnsw_core::{Quantizer, Result, VhnswError};

/// Compressed representation produced by [`PqQuantizer`].
#[derive(Debug, Clone)]
pub struct PqEncoded {
    /// One code per subspace (`n_sub` bytes).
    pub codes: Vec<u8>,
}

/// Product quantizer that splits vectors into subspaces and encodes via codebooks.
///
/// # Construction
///
/// Use [`PqQuantizer::new`] which validates that `dim` is divisible by `n_sub`
/// and that `n_codes <= 256`.
pub struct PqQuantizer {
    dim: usize,
    /// Number of subspaces.
    n_sub: usize,
    /// Number of centroids per subspace (max 256).
    n_codes: usize,
    /// Dimension of each subvector (`dim / n_sub`).
    sub_dim: usize,
    /// `codebooks[sub][code_idx]` is a centroid vector of length `sub_dim`.
    codebooks: Vec<Vec<Vec<f32>>>,
    trained: bool,
}

impl PqQuantizer {
    /// Create a new untrained PQ quantizer.
    ///
    /// # Errors
    ///
    /// Returns an error if `dim` is not divisible by `n_sub` or if `n_codes > 256`.
    pub fn new(dim: usize, n_sub: usize, n_codes: usize) -> Result<Self> {
        if !dim.is_multiple_of(n_sub) {
            return Err(VhnswError::Quantization(format!(
                "dim ({dim}) must be divisible by n_sub ({n_sub})"
            )));
        }
        if n_codes > 256 {
            return Err(VhnswError::Quantization(format!(
                "n_codes ({n_codes}) must be <= 256 to fit in u8"
            )));
        }
        if n_codes == 0 {
            return Err(VhnswError::Quantization(
                "n_codes must be > 0".to_string(),
            ));
        }
        Ok(Self {
            dim,
            n_sub,
            n_codes,
            sub_dim: dim / n_sub,
            codebooks: Vec::new(),
            trained: false,
        })
    }

    /// Precompute an asymmetric distance table for `query`.
    ///
    /// `table[sub][code]` gives the L2-squared distance between the query's
    /// subvector in subspace `sub` and centroid `code`.
    ///
    /// # Errors
    ///
    /// Returns an error if the quantizer is untrained or `query` has the wrong dimension.
    pub fn distance_table(&self, query: &[f32]) -> Result<Vec<Vec<f32>>> {
        if !self.trained {
            return Err(VhnswError::Quantization(
                "quantizer not trained".to_string(),
            ));
        }
        if query.len() != self.dim {
            return Err(VhnswError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        let table: Vec<Vec<f32>> = (0..self.n_sub)
            .map(|sub| {
                let offset = sub * self.sub_dim;
                let q_sub = &query[offset..offset + self.sub_dim];
                self.codebooks[sub]
                    .iter()
                    .map(|centroid| {
                        q_sub
                            .iter()
                            .zip(centroid.iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum()
                    })
                    .collect()
            })
            .collect();

        Ok(table)
    }

    /// Compute asymmetric distance using a precomputed table.
    #[must_use]
    pub fn distance_with_table(&self, table: &[Vec<f32>], encoded: &PqEncoded) -> f32 {
        encoded
            .codes
            .iter()
            .enumerate()
            .map(|(sub, &code)| table[sub][code as usize])
            .sum()
    }
}

// ---------------------------------------------------------------------------
// K-means helper
// ---------------------------------------------------------------------------

/// Simple k-means clustering.
///
/// Returns `k` centroid vectors. Uses at most `max_iters` iterations.
/// Empty clusters are reinitialized from the farthest point.
fn kmeans(data: &[Vec<f32>], k: usize, sub_dim: usize, max_iters: usize) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return Vec::new();
    }

    let actual_k = k.min(data.len());

    // -- Initialization: spread out using max-distance heuristic.
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(actual_k);
    centroids.push(data[0].clone());

    for _ in 1..actual_k {
        // Pick the data point farthest from the nearest existing centroid.
        let best = data
            .iter()
            .map(|pt| {
                centroids
                    .iter()
                    .map(|c| l2_sq(pt, c))
                    .fold(f32::INFINITY, f32::min)
            })
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        match best {
            Some((idx, _)) => centroids.push(data[idx].clone()),
            None => break,
        }
    }

    // Pad with zeros if we somehow have fewer than actual_k centroids.
    while centroids.len() < actual_k {
        centroids.push(vec![0.0; sub_dim]);
    }

    let mut assignments = vec![0usize; data.len()];

    for _iter in 0..max_iters {
        // -- Assign step (parallel).
        let new_assignments: Vec<usize> = data
            .par_iter()
            .map(|pt| {
                centroids
                    .iter()
                    .enumerate()
                    .map(|(ci, c)| (ci, l2_sq(pt, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(ci, _)| ci)
            })
            .collect();

        let converged = new_assignments == assignments;
        assignments = new_assignments;

        // -- Update step: recompute centroids.
        let mut sums = vec![vec![0.0_f64; sub_dim]; actual_k];
        let mut counts = vec![0usize; actual_k];

        for (pt, &a) in data.iter().zip(assignments.iter()) {
            counts[a] += 1;
            for (j, &v) in pt.iter().enumerate() {
                sums[a][j] += v as f64;
            }
        }

        for ci in 0..actual_k {
            if counts[ci] == 0 {
                // Empty cluster: reinitialize from the point farthest from its centroid.
                let farthest = data
                    .iter()
                    .enumerate()
                    .map(|(idx, pt)| (idx, l2_sq(pt, &centroids[assignments[idx]])))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                if let Some((idx, _)) = farthest {
                    centroids[ci] = data[idx].clone();
                }
            } else {
                let cnt = counts[ci] as f64;
                centroids[ci] = sums[ci].iter().map(|&s| (s / cnt) as f32).collect();
            }
        }

        if converged {
            break;
        }
    }

    centroids
}

/// L2-squared distance between two slices.
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

// ---------------------------------------------------------------------------
// Quantizer trait impl
// ---------------------------------------------------------------------------

const KMEANS_MAX_ITERS: usize = 20;

impl Quantizer for PqQuantizer {
    type Encoded = PqEncoded;

    fn train(&mut self, vectors: &[&[f32]]) -> Result<()> {
        if vectors.is_empty() {
            return Err(VhnswError::Quantization(
                "training set is empty".to_string(),
            ));
        }
        for v in vectors {
            if v.len() != self.dim {
                return Err(VhnswError::DimensionMismatch {
                    expected: self.dim,
                    got: v.len(),
                });
            }
        }

        // Train one codebook per subspace.
        let codebooks: Vec<Vec<Vec<f32>>> = (0..self.n_sub)
            .map(|sub| {
                let offset = sub * self.sub_dim;

                // Extract subvectors for this subspace.
                let sub_vecs: Vec<Vec<f32>> = vectors
                    .iter()
                    .map(|v| v[offset..offset + self.sub_dim].to_vec())
                    .collect();

                kmeans(&sub_vecs, self.n_codes, self.sub_dim, KMEANS_MAX_ITERS)
            })
            .collect();

        self.codebooks = codebooks;
        self.trained = true;
        Ok(())
    }

    fn encode(&self, vector: &[f32]) -> Result<PqEncoded> {
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

        let codes: Vec<u8> = (0..self.n_sub)
            .map(|sub| {
                let offset = sub * self.sub_dim;
                let sub_vec = &vector[offset..offset + self.sub_dim];

                self.codebooks[sub]
                    .iter()
                    .enumerate()
                    .map(|(ci, c)| (ci, l2_sq(sub_vec, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(ci, _)| ci as u8)
            })
            .collect();

        Ok(PqEncoded { codes })
    }

    fn distance_encoded(&self, a: &PqEncoded, b: &PqEncoded) -> f32 {
        a.codes
            .iter()
            .zip(b.codes.iter())
            .enumerate()
            .map(|(sub, (&ca, &cb))| {
                let centroid_a = &self.codebooks[sub][ca as usize];
                let centroid_b = &self.codebooks[sub][cb as usize];
                l2_sq(centroid_a, centroid_b)
            })
            .sum()
    }

    fn compression_ratio(&self) -> f32 {
        // Original: dim * 4 bytes. Encoded: n_sub * 1 byte.
        (self.dim as f32 * 4.0) / self.n_sub as f32
    }
}
