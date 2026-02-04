//! Integration tests for v-hnsw-quantize.

use v_hnsw_core::Quantizer;
use v_hnsw_distance::L2Distance;

use crate::pq::PqQuantizer;
use crate::rescore::rescore;
use crate::sq8::Sq8Quantizer;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Build a simple dataset: `count` vectors of `dim` dimensions with predictable
/// values so that tests are deterministic.
fn make_vectors(dim: usize, count: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            (0..dim)
                .map(|d| (i as f32 * 0.1) + (d as f32 * 0.01))
                .collect()
        })
        .collect()
}

fn as_refs(vecs: &[Vec<f32>]) -> Vec<&[f32]> {
    vecs.iter().map(|v| v.as_slice()).collect()
}

// =======================================================================
// SQ8 Tests
// =======================================================================

#[test]
fn test_sq8_train_encode_decode() {
    let dim = 4;
    let data = make_vectors(dim, 50);
    let refs = as_refs(&data);

    let mut q = Sq8Quantizer::new(dim);
    q.train(&refs).ok();

    let original = &data[25];
    let encoded = q.encode(original).ok();
    let encoded = encoded.as_ref();
    let decoded = encoded.and_then(|e| q.decode(e).ok());
    let decoded = decoded.as_ref();

    if let (Some(decoded), Some(_enc)) = (decoded, encoded) {
        // Each decoded dimension should be close to the original.
        for (i, (&orig, &dec)) in original.iter().zip(decoded.iter()).enumerate() {
            let err = (orig - dec).abs();
            // Maximum quantization error per dim is range / 255.
            // Be generous with the tolerance.
            assert!(
                err < 0.1,
                "dim {i}: original={orig}, decoded={dec}, error={err}"
            );
        }
    }
}

#[test]
fn test_sq8_distance_accuracy() {
    let dim = 8;
    let data = make_vectors(dim, 100);
    let refs = as_refs(&data);

    let mut q = Sq8Quantizer::new(dim);
    q.train(&refs).ok();

    let a = &data[10];
    let b = &data[90];

    // True L2-squared distance.
    let true_dist: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();

    let ea = q.encode(a);
    let eb = q.encode(b);

    if let (Ok(ref ea), Ok(ref eb)) = (ea, eb) {
        let q_dist = q.distance_encoded(ea, eb);

        // Relative error should be reasonable (< 10%).
        let rel_err = (q_dist - true_dist).abs() / true_dist.max(1e-9);
        assert!(
            rel_err < 0.10,
            "relative error {rel_err:.4} too large (true={true_dist}, quantized={q_dist})"
        );
    }
}

#[test]
fn test_sq8_compression_ratio() {
    let q = Sq8Quantizer::new(16);
    assert!((q.compression_ratio() - 4.0).abs() < f32::EPSILON);
}

#[test]
fn test_sq8_untrained_error() {
    let q = Sq8Quantizer::new(4);
    let result = q.encode(&[1.0, 2.0, 3.0, 4.0]);
    assert!(result.is_err());
}

#[test]
fn test_sq8_dimension_mismatch() {
    let dim = 4;
    let data = make_vectors(dim, 10);
    let refs = as_refs(&data);

    let mut q = Sq8Quantizer::new(dim);
    q.train(&refs).ok();

    // Wrong dimension.
    let result = q.encode(&[1.0, 2.0, 3.0]);
    assert!(result.is_err());
}

#[test]
fn test_sq8_constant_dimension() {
    // All values in one dimension are the same.
    let data: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, 5.0, i as f32]).collect();
    let refs = as_refs(&data);

    let mut q = Sq8Quantizer::new(3);
    q.train(&refs).ok();

    let enc = q.encode(&[5.0, 5.0, 5.0]);
    assert!(enc.is_ok());
    if let Ok(ref enc) = enc {
        // Constant dimension should encode as 128.
        assert_eq!(enc.data[1], 128);
    }
}

// =======================================================================
// PQ Tests
// =======================================================================

#[test]
fn test_pq_train_encode() {
    let dim = 8;
    let n_sub = 4;
    let n_codes = 16;

    let data = make_vectors(dim, 100);
    let refs = as_refs(&data);

    let mut q = PqQuantizer::new(dim, n_sub, n_codes);
    assert!(q.is_ok());

    if let Ok(ref mut q) = q {
        q.train(&refs).ok();

        let encoded = q.encode(&data[0]);
        assert!(encoded.is_ok());
        if let Ok(ref enc) = encoded {
            assert_eq!(enc.codes.len(), n_sub);
            // All codes should be < n_codes.
            for &c in &enc.codes {
                assert!((c as usize) < n_codes);
            }
        }
    }
}

#[test]
fn test_pq_distance_table() {
    let dim = 8;
    let n_sub = 4;
    let n_codes = 16;

    let data = make_vectors(dim, 100);
    let refs = as_refs(&data);

    let mut q = PqQuantizer::new(dim, n_sub, n_codes);
    assert!(q.is_ok());

    if let Ok(ref mut q) = q {
        q.train(&refs).ok();

        let table = q.distance_table(&data[0]);
        assert!(table.is_ok());
        if let Ok(ref table) = table {
            assert_eq!(table.len(), n_sub);
            for sub_table in table {
                assert_eq!(sub_table.len(), n_codes);
                // All distances should be non-negative.
                for &d in sub_table {
                    assert!(d >= 0.0, "negative distance in table: {d}");
                }
            }
        }
    }
}

#[test]
fn test_pq_compression_ratio() {
    let dim = 32;
    let n_sub = 8;
    let q = PqQuantizer::new(dim, n_sub, 256);
    assert!(q.is_ok());
    if let Ok(ref q) = q {
        // (32 * 4) / 8 = 16.0
        let expected = (dim as f32 * 4.0) / n_sub as f32;
        assert!((q.compression_ratio() - expected).abs() < f32::EPSILON);
    }
}

#[test]
fn test_pq_symmetric_distance() {
    let dim = 8;
    let n_sub = 4;
    let n_codes = 16;

    let data = make_vectors(dim, 100);
    let refs = as_refs(&data);

    let mut q = PqQuantizer::new(dim, n_sub, n_codes);
    assert!(q.is_ok());

    if let Ok(ref mut q) = q {
        q.train(&refs).ok();

        let ea = q.encode(&data[0]);
        let eb = q.encode(&data[50]);

        if let (Ok(ref ea), Ok(ref eb)) = (ea, eb) {
            let d = q.distance_encoded(ea, eb);
            assert!(d >= 0.0, "symmetric distance must be non-negative: {d}");

            // Symmetric: d(a,b) == d(b,a).
            let d_rev = q.distance_encoded(eb, ea);
            assert!(
                (d - d_rev).abs() < 1e-6,
                "distance not symmetric: {d} vs {d_rev}"
            );
        }
    }
}

#[test]
fn test_pq_dim_not_divisible() {
    let result = PqQuantizer::new(7, 4, 256);
    assert!(result.is_err());
}

#[test]
fn test_pq_untrained_error() {
    let q = PqQuantizer::new(8, 4, 256);
    assert!(q.is_ok());
    if let Ok(ref q) = q {
        let result = q.encode(&[0.0; 8]);
        assert!(result.is_err());
    }
}

// =======================================================================
// Rescore Tests
// =======================================================================

#[test]
fn test_rescore_basic() {
    let l2 = L2Distance;

    // Three stored vectors with distinct distances from the query.
    let vectors: Vec<(u64, Vec<f32>)> = vec![
        (1, vec![1.0, 0.0]),  // dist to (0,0) = 1.0
        (2, vec![2.0, 0.0]),  // dist to (0,0) = 4.0
        (3, vec![10.0, 10.0]), // dist to (0,0) = 200.0
    ];

    let get_vector = |id: u64| -> Option<Vec<f32>> {
        vectors.iter().find(|(vid, _)| *vid == id).map(|(_, v)| v.clone())
    };

    let query = vec![0.0, 0.0];

    // Candidates with intentionally wrong approximate distances
    // (id=3 has a small approx distance but is actually far).
    let candidates: Vec<(u64, f32)> = vec![(3, 0.1), (2, 5.0), (1, 10.0)];

    let results = rescore(&candidates, &query, &get_vector, &l2, 2);

    assert_eq!(results.len(), 2);
    // After rescoring, id=1 (dist=1) and id=2 (dist=4) should be closest.
    assert_eq!(results[0].0, 1);
    assert_eq!(results[1].0, 2);
}

#[test]
fn test_rescore_truncates() {
    let l2 = L2Distance;

    let vectors: Vec<(u64, Vec<f32>)> = vec![
        (1, vec![0.0]),
        (2, vec![1.0]),
        (3, vec![2.0]),
        (4, vec![3.0]),
        (5, vec![4.0]),
    ];

    let get_vector = |id: u64| -> Option<Vec<f32>> {
        vectors.iter().find(|(vid, _)| *vid == id).map(|(_, v)| v.clone())
    };

    let query = vec![0.0];
    let candidates: Vec<(u64, f32)> = vec![(1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0)];

    let results = rescore(&candidates, &query, &get_vector, &l2, 3);
    assert_eq!(results.len(), 3);
}

#[test]
fn test_rescore_missing_vectors() {
    let l2 = L2Distance;

    // Only id=1 exists.
    let get_vector = |id: u64| -> Option<Vec<f32>> {
        if id == 1 {
            Some(vec![0.0, 0.0])
        } else {
            None
        }
    };

    let query = vec![0.0, 0.0];
    let candidates: Vec<(u64, f32)> = vec![(1, 0.0), (2, 0.0), (3, 0.0)];

    let results = rescore(&candidates, &query, &get_vector, &l2, 5);
    // Only 1 result because other vectors are missing.
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
}

// =======================================================================
// Proptest
// =======================================================================

mod proptests {
    use proptest::prelude::*;
    use v_hnsw_core::Quantizer;

    use crate::sq8::Sq8Quantizer;

    const DIM: usize = 4;

    /// Generate a random vector with values in [-100, 100].
    fn arb_vector() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-100.0_f32..100.0_f32, DIM)
    }

    /// Generate a set of training vectors.
    fn arb_training_set() -> impl Strategy<Value = Vec<Vec<f32>>> {
        proptest::collection::vec(arb_vector(), 10..50)
    }

    proptest! {
        #[test]
        fn sq8_encode_decode_bounded_error(
            training in arb_training_set(),
        ) {
            let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
            let mut q = Sq8Quantizer::new(DIM);

            if q.train(&refs).is_ok() {
                // Pick a sample from within the training set so it is within
                // the learned min/max range. Out-of-range values are clamped
                // and can have arbitrarily large error.
                let sample = &training[0];
                if let Ok(enc) = q.encode(sample) {
                    if let Ok(dec) = q.decode(&enc) {
                        for (i, (&orig, &d)) in sample.iter().zip(dec.iter()).enumerate() {
                            // Maximum quantization error per dim = range / 255.
                            // With range up to 200 (-100..100), that is ~0.79.
                            // Use 1.0 as tolerance with a small epsilon for rounding.
                            let err = (orig - d).abs();
                            prop_assert!(
                                err < 1.0,
                                "dim {}: orig={}, dec={}, err={}",
                                i, orig, d, err,
                            );
                        }
                    }
                }
            }
        }

        #[test]
        fn sq8_distance_non_negative(
            training in arb_training_set(),
            a in arb_vector(),
            b in arb_vector(),
        ) {
            let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
            let mut q = Sq8Quantizer::new(DIM);

            if q.train(&refs).is_ok() {
                if let (Ok(ea), Ok(eb)) = (q.encode(&a), q.encode(&b)) {
                    let d = q.distance_encoded(&ea, &eb);
                    prop_assert!(d >= 0.0, "distance must be non-negative: {}", d);
                }
            }
        }
    }
}
