//! Unit tests for `extract::hash` — body_hash, code_tokens, minhash, hex encoding.

use crate::extract;

// ═══════════════════════════════════════════════════════════════════════════
// body_hash
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn body_hash_identical_code_same_hash() {
    let a = "fn foo() {\n    let x = 1;\n    x + 2\n}";
    let b = "fn foo() {\n    let x = 1;\n    x + 2\n}";
    assert_eq!(extract::body_hash(a), extract::body_hash(b));
}

#[test]
fn body_hash_different_code_different_hash() {
    let a = "fn foo() {\n    let x = 1;\n}";
    let b = "fn bar() {\n    let y = 2;\n}";
    assert_ne!(extract::body_hash(a), extract::body_hash(b));
}

#[test]
fn body_hash_ignores_comments() {
    let a = "fn foo() {\n    // this is a comment\n    let x = 1;\n}";
    let b = "fn foo() {\n    // different comment\n    let x = 1;\n}";
    assert_eq!(extract::body_hash(a), extract::body_hash(b));
}

#[test]
fn body_hash_ignores_doc_comments() {
    let a = "/// doc comment\nfn foo() { let x = 1; }";
    let b = "/// different doc\nfn foo() { let x = 1; }";
    assert_eq!(extract::body_hash(a), extract::body_hash(b));
}

#[test]
fn body_hash_ignores_hash_comments() {
    let a = "# python comment\ndef foo():\n    x = 1";
    let b = "# different\ndef foo():\n    x = 1";
    assert_eq!(extract::body_hash(a), extract::body_hash(b));
}

#[test]
fn body_hash_strips_inline_comments() {
    let a = "let x = 1; // inline";
    let b = "let x = 1; // other";
    assert_eq!(extract::body_hash(a), extract::body_hash(b));
}

#[test]
fn body_hash_empty_string() {
    // Should not panic
    let _ = extract::body_hash("");
}

#[test]
fn body_hash_ignores_blank_lines() {
    let a = "fn foo() {\n\n    let x = 1;\n\n}";
    let b = "fn foo() {\n    let x = 1;\n}";
    assert_eq!(extract::body_hash(a), extract::body_hash(b));
}

// ═══════════════════════════════════════════════════════════════════════════
// code_tokens
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn code_tokens_basic() {
    let tokens = extract::code_tokens("let x = 42;");
    assert!(tokens.contains(&"let".to_owned()));
    assert!(tokens.contains(&"x".to_owned()));
    assert!(tokens.contains(&"$N".to_owned()), "numbers should be normalized: {tokens:?}");
}

#[test]
fn code_tokens_skips_comments() {
    let tokens = extract::code_tokens("// this is a comment\nlet x = 1;");
    assert!(!tokens.iter().any(|t| t.contains("comment")));
    assert!(tokens.contains(&"let".to_owned()));
}

#[test]
fn code_tokens_skips_hash_comments() {
    let tokens = extract::code_tokens("# python comment\nx = 1");
    assert!(!tokens.iter().any(|t| t.contains("python")));
    assert!(tokens.contains(&"x".to_owned()));
}

#[test]
fn code_tokens_block_comment() {
    let tokens = extract::code_tokens("/* block\ncomment */\nlet y = 2;");
    assert!(!tokens.iter().any(|t| t.contains("block")));
    assert!(tokens.contains(&"let".to_owned()));
}

#[test]
fn code_tokens_number_normalization() {
    let a = extract::code_tokens("let x = 42;");
    let b = extract::code_tokens("let x = 99;");
    // Both should normalize numbers to $N
    assert_eq!(a, b);
}

#[test]
fn code_tokens_empty() {
    let tokens = extract::code_tokens("");
    assert!(tokens.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// minhash_signature + jaccard
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn minhash_identical_tokens_jaccard_one() {
    let tokens: Vec<String> = vec!["let", "x", "foo", "bar"]
        .into_iter().map(String::from).collect();
    let sig_a = extract::minhash_signature(&tokens, 32);
    let sig_b = extract::minhash_signature(&tokens, 32);
    let j = extract::jaccard_from_minhash(&sig_a, &sig_b);
    assert!((j - 1.0).abs() < f64::EPSILON, "identical tokens should have jaccard=1.0, got {j}");
}

#[test]
fn minhash_different_tokens_low_jaccard() {
    let a: Vec<String> = vec!["alpha", "beta", "gamma", "delta"]
        .into_iter().map(String::from).collect();
    let b: Vec<String> = vec!["one", "two", "three", "four"]
        .into_iter().map(String::from).collect();
    let sig_a = extract::minhash_signature(&a, 64);
    let sig_b = extract::minhash_signature(&b, 64);
    let j = extract::jaccard_from_minhash(&sig_a, &sig_b);
    assert!(j < 0.5, "different tokens should have low jaccard, got {j}");
}

#[test]
fn minhash_empty_tokens() {
    let tokens: Vec<String> = vec![];
    let sig = extract::minhash_signature(&tokens, 16);
    // All values should be u64::MAX (no features)
    assert!(sig.iter().all(|&v| v == u64::MAX));
}

#[test]
fn jaccard_mismatched_lengths() {
    let a = vec![1u64, 2, 3];
    let b = vec![1u64, 2];
    assert!((extract::jaccard_from_minhash(&a, &b) - 0.0).abs() < f64::EPSILON);
}

#[test]
fn jaccard_empty_sigs() {
    let a: Vec<u64> = vec![];
    let b: Vec<u64> = vec![];
    assert!((extract::jaccard_from_minhash(&a, &b) - 0.0).abs() < f64::EPSILON);
}

// ═══════════════════════════════════════════════════════════════════════════
// minhash hex encoding roundtrip
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn minhash_hex_roundtrip() {
    let original = vec![0u64, 1, 255, u64::MAX, 12345678901234];
    let hex = extract::minhash_to_hex(&original);
    let decoded = extract::minhash_from_hex(&hex);
    assert_eq!(decoded, Some(original));
}

#[test]
fn minhash_from_hex_invalid_length() {
    // Not a multiple of 16
    assert!(extract::minhash_from_hex("abc").is_none());
}

#[test]
fn minhash_from_hex_invalid_chars() {
    // Valid length but invalid hex chars
    assert!(extract::minhash_from_hex("gggggggggggggggg").is_none());
}

#[test]
fn minhash_from_hex_empty() {
    assert_eq!(extract::minhash_from_hex(""), Some(vec![]));
}
