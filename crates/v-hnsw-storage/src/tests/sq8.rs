//! Tests for SQ8 scalar quantization.

use crate::sq8::Sq8Params;

fn make_random_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    // Simple deterministic pseudo-random using xorshift
    let mut state = seed;
    let mut vectors = Vec::with_capacity(count);
    for _ in 0..count {
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to [-1, 1] range
            let f = (state as f64 / u64::MAX as f64) * 2.0 - 1.0;
            v.push(f as f32);
        }
        vectors.push(v);
    }
    vectors
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[test]
fn train_basic() {
    let vectors = vec![vec![0.0, 1.0, -1.0], vec![1.0, 0.0, 0.5]];
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let params = Sq8Params::train(3, &refs).unwrap();

    assert_eq!(params.dim(), 3);
    // min: [0, 0, -1], max: [1, 1, 0.5], range: [1, 1, 1.5]
    assert!((params.mins[0] - 0.0).abs() < 1e-6);
    assert!((params.mins[2] - (-1.0)).abs() < 1e-6);
    assert!((params.ranges[0] - 1.0).abs() < 1e-6);
    assert!((params.ranges[2] - 1.5).abs() < 1e-6);
}

#[test]
fn train_empty_returns_error() {
    let result = Sq8Params::train(3, &[]);
    assert!(result.is_err());
}

#[test]
fn quantize_boundary_values() {
    let vectors = vec![vec![0.0, 10.0], vec![1.0, 20.0]];
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let params = Sq8Params::train(2, &refs).unwrap();

    // Min value → 0
    let q = params.quantize(&[0.0, 10.0]);
    assert_eq!(q[0], 0);
    assert_eq!(q[1], 0);

    // Max value → 255
    let q = params.quantize(&[1.0, 20.0]);
    assert_eq!(q[0], 255);
    assert_eq!(q[1], 255);

    // Mid value → ~128
    let q = params.quantize(&[0.5, 15.0]);
    assert!((q[0] as i16 - 128).unsigned_abs() <= 1);
    assert!((q[1] as i16 - 128).unsigned_abs() <= 1);
}

#[test]
fn quantize_clamps_out_of_range() {
    let vectors = vec![vec![0.0], vec![1.0]];
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let params = Sq8Params::train(1, &refs).unwrap();

    // Below min → 0
    assert_eq!(params.quantize(&[-5.0])[0], 0);
    // Above max → 255
    assert_eq!(params.quantize(&[10.0])[0], 255);
}

#[test]
fn dequantize_roundtrip_accuracy() {
    let dim = 128;
    let mut vectors = make_random_vectors(100, dim, 42);
    for v in &mut vectors {
        normalize(v);
    }
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let params = Sq8Params::train(dim, &refs).unwrap();

    // Check each vector's roundtrip error
    for v in &vectors {
        let quantized = params.quantize(v);
        let dequantized = params.dequantize(&quantized);
        let max_error: f32 = v
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        // For normalized vectors in [-1, 1], max quantization error ≈ 2/255 ≈ 0.0078
        assert!(
            max_error < 0.01,
            "Max roundtrip error too large: {max_error}"
        );
    }
}

#[test]
fn asymmetric_distance_close_to_exact() {
    let dim = 256;
    let mut vectors = make_random_vectors(200, dim, 123);
    for v in &mut vectors {
        normalize(v);
    }
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let params = Sq8Params::train(dim, &refs).unwrap();

    // Compare asymmetric SQ8 distance vs exact cosine distance
    let query = &vectors[0];
    let mut max_abs_error = 0.0_f32;

    for target in &vectors[1..] {
        let exact: f32 = {
            let dot: f32 = query.iter().zip(target.iter()).map(|(a, b)| a * b).sum();
            1.0 - dot.clamp(-1.0, 1.0)
        };

        let quantized = params.quantize(target);
        let approx = params.asymmetric_distance(query, &quantized);

        let error = (exact - approx).abs();
        if error > max_abs_error {
            max_abs_error = error;
        }
    }

    // SQ8 with normalized cosine should have very small error
    assert!(
        max_abs_error < 0.02,
        "Max asymmetric distance error too large: {max_abs_error}"
    );
}

#[test]
fn asymmetric_distance_preserves_ranking() {
    let dim = 256;
    let mut vectors = make_random_vectors(50, dim, 77);
    for v in &mut vectors {
        normalize(v);
    }
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let params = Sq8Params::train(dim, &refs).unwrap();

    let query = &vectors[0];

    // Compute exact distances
    let mut exact_dists: Vec<(usize, f32)> = vectors[1..]
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            (i, 1.0 - dot.clamp(-1.0, 1.0))
        })
        .collect();
    exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Compute SQ8 distances
    let mut sq8_dists: Vec<(usize, f32)> = vectors[1..]
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let q = params.quantize(v);
            (i, params.asymmetric_distance(query, &q))
        })
        .collect();
    sq8_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Check recall@10: at least 9 out of 10 exact top-10 should appear in SQ8 top-10
    let exact_top10: std::collections::HashSet<usize> =
        exact_dists.iter().take(10).map(|(i, _)| *i).collect();
    let sq8_top10: std::collections::HashSet<usize> =
        sq8_dists.iter().take(10).map(|(i, _)| *i).collect();

    let overlap = exact_top10.intersection(&sq8_top10).count();
    assert!(
        overlap >= 9,
        "Recall@10 too low: {overlap}/10 (exact top-10 vs SQ8 top-10)"
    );
}

#[test]
fn quantize_into_matches_quantize() {
    let vectors = vec![vec![0.1, 0.5, -0.3], vec![0.9, -0.2, 0.7]];
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let params = Sq8Params::train(3, &refs).unwrap();

    let input = [0.5, 0.1, 0.2];
    let expected = params.quantize(&input);
    let mut actual = vec![0u8; 3];
    params.quantize_into(&input, &mut actual);
    assert_eq!(expected, actual);
}

#[test]
fn save_and_load_roundtrip() {
    let dim = 64;
    let vectors = make_random_vectors(50, dim, 999);
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let params = Sq8Params::train(dim, &refs).unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("sq8_params.bin");

    params.save(&path).unwrap();
    let loaded = Sq8Params::load(&path).unwrap();

    assert_eq!(params.dim(), loaded.dim());
    for i in 0..dim {
        assert!((params.mins[i] - loaded.mins[i]).abs() < 1e-6);
        assert!((params.ranges[i] - loaded.ranges[i]).abs() < 1e-6);
    }

    // Verify loaded params produce same quantization
    let test_vec = &vectors[0];
    let q1 = params.quantize(test_vec);
    let q2 = loaded.quantize(test_vec);
    assert_eq!(q1, q2);
}

#[test]
fn constant_dimension_handled() {
    // All vectors have same value in dim 0 → range = 0
    let vectors = vec![vec![5.0, 1.0], vec![5.0, 2.0]];
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let params = Sq8Params::train(2, &refs).unwrap();

    // Constant dimension should quantize to 0 (since (v - min) * 0 = 0)
    let q = params.quantize(&[5.0, 1.5]);
    assert_eq!(q[0], 0);
    // Non-constant dimension should work normally
    assert!((q[1] as i16 - 128).unsigned_abs() <= 1);
}

#[test]
fn query_lut_matches_asymmetric_distance() {
    let dim = 256;
    let mut vectors = make_random_vectors(50, dim, 42);
    for v in &mut vectors {
        normalize(v);
    }
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let params = Sq8Params::train(dim, &refs).unwrap();

    let query = &vectors[0];
    let lut = params.build_query_lut(query);

    for target in &vectors[1..] {
        let quantized = params.quantize(target);
        let d_direct = params.asymmetric_distance(query, &quantized);
        let d_lut = Sq8Params::distance_with_lut(&lut, &quantized);
        assert!(
            (d_direct - d_lut).abs() < 1e-6,
            "LUT distance {d_lut} != direct {d_direct}"
        );
    }
}
