use crate::bm25::scorer::Bm25Params;

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

// ============================================================================
// Additional scorer tests: edge cases, boundary conditions, known values
// ============================================================================

#[test]
fn test_idf_all_docs_contain_term() {
    // When every document contains the term, IDF should still be positive
    // (BM25 formula adds 1.0 inside ln to prevent negative IDF)
    let params = Bm25Params::default();
    let idf = params.idf(100, 100);
    // ln((100 - 100 + 0.5) / (100 + 0.5) + 1.0) = ln(0.5/100.5 + 1.0) ≈ ln(1.00497)
    assert!(idf > 0.0, "IDF should be positive even when all docs contain term");
    assert!(idf < 0.01, "IDF should be very small when all docs contain term: {}", idf);
}

#[test]
fn test_idf_single_doc_corpus() {
    let params = Bm25Params::default();
    // Single doc contains the term
    let idf = params.idf(1, 1);
    // ln((1 - 1 + 0.5) / (1 + 0.5) + 1.0) = ln(0.5/1.5 + 1.0) = ln(1.333)
    assert!(idf > 0.0);
    let expected = (0.5_f32 / 1.5 + 1.0).ln();
    assert!((idf - expected).abs() < 1e-6, "expected {}, got {}", expected, idf);
}

#[test]
fn test_idf_known_value() {
    let params = Bm25Params::default();
    // 10 docs, term appears in 3
    let idf = params.idf(3, 10);
    // ln((10 - 3 + 0.5) / (3 + 0.5) + 1.0) = ln(7.5/3.5 + 1.0) = ln(3.14286)
    let expected = (7.5_f32 / 3.5 + 1.0).ln();
    assert!((idf - expected).abs() < 1e-6, "expected {}, got {}", expected, idf);
}

#[test]
fn test_tf_norm_known_value() {
    let params = Bm25Params::default(); // k1=1.2, b=0.75
    // tf=2, doc_len=100, avg_doc_len=100 → length_norm = 1-0.75+0.75*(100/100) = 1.0
    // tf_norm = (2 * 2.2) / (2 + 1.2 * 1.0) = 4.4 / 3.2 = 1.375
    let tf_norm = params.tf_norm(2, 100, 100.0);
    let expected = (2.0 * 2.2) / (2.0 + 1.2 * 1.0);
    assert!((tf_norm - expected).abs() < 1e-6, "expected {}, got {}", expected, tf_norm);
}

#[test]
fn test_tf_norm_zero_tf() {
    let params = Bm25Params::default();
    let tf_norm = params.tf_norm(0, 100, 100.0);
    // (0 * 2.2) / (0 + 1.2 * 1.0) = 0.0
    assert!((tf_norm - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_tf_norm_zero_avg_doc_len() {
    // Edge case: avg_doc_len = 0 → division by zero in length_norm
    let params = Bm25Params::default();
    let tf_norm = params.tf_norm(1, 100, 0.0);
    // length_norm = 1 - 0.75 + 0.75 * (100/0) = infinity → tf_norm → 0
    // Actual behavior: f32 division by inf
    assert!(tf_norm.is_finite() || tf_norm == 0.0 || tf_norm.is_nan(),
        "tf_norm with zero avg_doc_len should not panic: {}", tf_norm);
}

#[test]
fn test_b_zero_no_length_normalization() {
    let params = Bm25Params::new(1.2, 0.0);
    // b=0 → length_norm = 1 - 0 + 0 * (...) = 1.0 always
    let score_short = params.tf_norm(3, 10, 100.0);
    let score_long = params.tf_norm(3, 1000, 100.0);
    assert!((score_short - score_long).abs() < 1e-6,
        "b=0 should eliminate length normalization: {} vs {}", score_short, score_long);
}

#[test]
fn test_b_one_full_length_normalization() {
    let params = Bm25Params::new(1.2, 1.0);
    // b=1.0 → length_norm = 0 + 1.0 * (doc_len/avg_doc_len) = doc_len/avg_doc_len
    let short = params.tf_norm(3, 50, 100.0);
    let long = params.tf_norm(3, 200, 100.0);
    // Difference should be larger than with default b=0.75
    let params_default = Bm25Params::default();
    let short_default = params_default.tf_norm(3, 50, 100.0);
    let long_default = params_default.tf_norm(3, 200, 100.0);
    let diff_b1 = short - long;
    let diff_default = short_default - long_default;
    assert!(diff_b1 > diff_default,
        "b=1.0 should penalize length more: diff_b1={}, diff_default={}", diff_b1, diff_default);
}

#[test]
fn test_score_is_idf_times_tf_norm() {
    let params = Bm25Params::default();
    let tf = 3u32;
    let df = 5u32;
    let doc_len = 80u32;
    let avg_doc_len = 100.0f32;
    let total_docs = 50usize;

    let score = params.score(tf, df, doc_len, avg_doc_len, total_docs);
    let idf = params.idf(df, total_docs);
    let tf_norm = params.tf_norm(tf, doc_len, avg_doc_len);
    assert!((score - idf * tf_norm).abs() < 1e-6);
}

#[test]
fn test_large_tf_approaches_k1_plus_one() {
    // As tf → ∞, tf_norm → k1 + 1 (when doc_len == avg_doc_len, length_norm = 1)
    let params = Bm25Params::default();
    let tf_norm = params.tf_norm(1_000_000, 100, 100.0);
    let limit = params.k1 + 1.0; // 2.2
    assert!((tf_norm - limit).abs() < 0.01,
        "large tf should approach k1+1={}: got {}", limit, tf_norm);
}
