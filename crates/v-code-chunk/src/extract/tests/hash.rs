use crate::extract::hash::{body_hash, jaccard_from_minhash, minhash_from_hex, minhash_to_hex};

// ---------------------------------------------------------------------------
// body_hash
// ---------------------------------------------------------------------------

#[test]
fn body_hash_deterministic() {
    let code = "fn foo() {\n    let x = 1;\n}\n";
    assert_eq!(body_hash(code), body_hash(code));
}

#[test]
fn body_hash_ignores_whitespace_differences() {
    let a = "fn foo() {\n    let x = 1;\n}\n";
    let b = "fn foo() {\n        let x = 1;\n}\n";
    assert_eq!(body_hash(a), body_hash(b));
}

#[test]
fn body_hash_ignores_blank_lines() {
    let a = "fn foo() {\nlet x = 1;\n}\n";
    let b = "fn foo() {\n\n\nlet x = 1;\n\n}\n";
    assert_eq!(body_hash(a), body_hash(b));
}

#[test]
fn body_hash_ignores_comments() {
    let a = "fn foo() {\nlet x = 1;\n}\n";
    let b = "// comment\nfn foo() {\n/// doc comment\nlet x = 1; // inline\n# hash comment\n}\n";
    assert_eq!(body_hash(a), body_hash(b));
}

#[test]
fn body_hash_differs_for_different_code() {
    let a = "fn foo() {\nlet x = 1;\n}\n";
    let b = "fn bar() {\nlet y = 2;\n}\n";
    assert_ne!(body_hash(a), body_hash(b));
}

// ---------------------------------------------------------------------------
// jaccard_from_minhash
// ---------------------------------------------------------------------------

#[test]
fn jaccard_identical_signatures() {
    let sig = vec![1u64, 2, 3, 4];
    let j = jaccard_from_minhash(&sig, &sig);
    assert!((j - 1.0).abs() < f64::EPSILON);
}

#[test]
fn jaccard_completely_different() {
    let a = vec![1u64, 2, 3, 4];
    let b = vec![5u64, 6, 7, 8];
    let j = jaccard_from_minhash(&a, &b);
    assert!((j - 0.0).abs() < f64::EPSILON);
}

#[test]
fn jaccard_partial_overlap() {
    let a = vec![1u64, 2, 3, 4];
    let b = vec![1u64, 2, 7, 8];
    let j = jaccard_from_minhash(&a, &b);
    assert!((j - 0.5).abs() < f64::EPSILON);
}

#[test]
fn jaccard_empty_returns_zero() {
    let empty: Vec<u64> = vec![];
    assert!((jaccard_from_minhash(&empty, &empty) - 0.0).abs() < f64::EPSILON);
}

#[test]
fn jaccard_different_lengths_returns_zero() {
    let a = vec![1u64, 2, 3];
    let b = vec![1u64, 2];
    assert!((jaccard_from_minhash(&a, &b) - 0.0).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// minhash_from_hex
// ---------------------------------------------------------------------------

#[test]
fn minhash_hex_roundtrip() {
    let original = vec![0u64, 1, 255, u64::MAX];
    let hex = minhash_to_hex(&original);
    let decoded = minhash_from_hex(&hex).expect("valid hex should decode");
    assert_eq!(original, decoded);
}

#[test]
fn minhash_from_hex_invalid_length() {
    // 15 chars — not a multiple of 16
    assert!(minhash_from_hex("abcdef012345678").is_none());
}

#[test]
fn minhash_from_hex_invalid_chars() {
    // 16 chars but contains 'g' which is not hex
    assert!(minhash_from_hex("000000000000000g").is_none());
}

#[test]
fn minhash_from_hex_empty_string() {
    // Empty string has length 0 which is a multiple of 16 → Some(empty vec)
    let result = minhash_from_hex("").expect("empty string should give empty vec");
    assert!(result.is_empty());
}
