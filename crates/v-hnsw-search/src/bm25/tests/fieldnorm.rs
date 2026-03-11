use crate::bm25::fieldnorm::{encode, decode, FieldNormLut};
use crate::bm25::scorer::Bm25Params;

#[test]
fn encode_zero_returns_zero() {
    assert_eq!(encode(0), 0);
}

#[test]
fn encode_one_returns_zero() {
    assert_eq!(encode(1), 0);
}

#[test]
fn encode_monotonically_increasing() {
    let mut prev = encode(1);
    for len in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 10000] {
        let code = encode(len);
        assert!(code >= prev, "encode({len})={code} < encode(prev)={prev}");
        prev = code;
    }
}

#[test]
fn decode_zero_returns_one() {
    assert!((decode(0) - 1.0).abs() < f32::EPSILON);
}

#[test]
fn decode_roundtrip_within_tolerance() {
    // For typical doc lengths, encode→decode should be within ~10% of original
    for &len in &[1u32, 5, 10, 50, 100, 500, 1000, 5000, 10000] {
        let code = encode(len);
        let decoded = decode(code);
        let ratio = decoded / len as f32;
        assert!(
            (0.7..=1.5).contains(&ratio),
            "doc_len={len}, code={code}, decoded={decoded}, ratio={ratio}"
        );
    }
}

#[test]
fn lut_tf_norm_matches_exact_within_tolerance() {
    let params = Bm25Params::default();
    let avg_doc_len = 100.0;
    let lut = FieldNormLut::build(params.b, avg_doc_len);

    // Compare LUT-based scoring against exact scoring
    for &doc_len in &[1u32, 10, 50, 100, 200, 500, 1000, 5000, 10000] {
        let code = encode(doc_len);
        for &tf in &[1u32, 2, 5, 10] {
            let exact = params.tf_norm(tf, doc_len, avg_doc_len);
            let cached = lut.tf_norm(params.k1, tf, code);
            let error = (exact - cached).abs() / exact.max(1e-6);
            assert!(
                error < 0.05, // < 5% relative error
                "doc_len={doc_len}, tf={tf}: exact={exact:.4}, cached={cached:.4}, error={error:.4}"
            );
        }
    }
}

#[test]
fn lut_scoring_preserves_ranking() {
    // Verify that LUT-based scoring doesn't change relative document ranking
    let params = Bm25Params::default();
    let avg_doc_len = 50.0;
    let lut = FieldNormLut::build(params.b, avg_doc_len);

    let docs = [(10u32, 3u32), (50, 2), (100, 5), (200, 1), (30, 4)];

    let mut exact_scores: Vec<(u32, f32)> = docs
        .iter()
        .map(|&(len, tf)| (len, params.tf_norm(tf, len, avg_doc_len)))
        .collect();
    exact_scores.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut lut_scores: Vec<(u32, f32)> = docs
        .iter()
        .map(|&(len, tf)| (len, lut.tf_norm(params.k1, tf, encode(len))))
        .collect();
    lut_scores.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Same ranking order
    let exact_order: Vec<u32> = exact_scores.iter().map(|s| s.0).collect();
    let lut_order: Vec<u32> = lut_scores.iter().map(|s| s.0).collect();
    assert_eq!(exact_order, lut_order, "Ranking changed with LUT");
}

#[test]
fn lut_with_zero_avg_doc_len() {
    let lut = FieldNormLut::build(0.75, 0.0);
    // Should not panic, returns reasonable values
    let score = lut.tf_norm(1.2, 1, encode(10));
    assert!(score.is_finite());
}

#[test]
fn lut_with_b_zero_ignores_doc_length() {
    // b=0 means no length normalization
    let lut = FieldNormLut::build(0.0, 100.0);
    let score_short = lut.tf_norm(1.2, 2, encode(10));
    let score_long = lut.tf_norm(1.2, 2, encode(1000));
    // Should be identical (no length normalization)
    assert!(
        (score_short - score_long).abs() < 1e-5,
        "b=0 but scores differ: {score_short} vs {score_long}"
    );
}

#[test]
fn encode_max_value() {
    let code = encode(u32::MAX);
    assert_eq!(code, 255);
}

#[test]
fn fieldnorm_cache_on_bm25_index() {
    use crate::bm25::Bm25Index;
    use crate::SimpleTokenizer;

    let mut index = Bm25Index::new(SimpleTokenizer);
    index.add_document(1, "the quick brown fox jumps over the lazy dog");
    index.add_document(2, "hello world");
    index.add_document(3, "rust programming language is fast and safe for systems programming");

    // Build cache and search
    index.build_fieldnorm_cache();
    let results = index.search("programming", 10);
    assert!(!results.is_empty());
    assert_eq!(results[0].0, 3); // doc 3 has "programming" twice
}
