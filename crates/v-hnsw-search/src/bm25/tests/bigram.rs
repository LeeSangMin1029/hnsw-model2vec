use crate::bm25::bigram::generate;

#[test]
fn test_empty() {
    assert!(generate(&[]).is_empty());
}

#[test]
fn test_single_token() {
    let tokens = vec!["hello".to_string()];
    assert!(generate(&tokens).is_empty());
}

#[test]
fn test_two_tokens() {
    let tokens = vec!["hello".to_string(), "world".to_string()];
    let bigrams = generate(&tokens);
    assert_eq!(bigrams.len(), 1);
    assert_eq!(bigrams[0], "hello\x01world");
}

#[test]
fn test_three_tokens() {
    let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let bigrams = generate(&tokens);
    assert_eq!(bigrams, vec!["a\x01b", "b\x01c"]);
}

// ============================================================================
// Additional bigram tests: edge cases
// ============================================================================

#[test]
fn test_four_tokens_produces_three_bigrams() {
    let tokens: Vec<String> = vec!["a", "b", "c", "d"].into_iter().map(String::from).collect();
    let bigrams = generate(&tokens);
    assert_eq!(bigrams.len(), 3);
    assert_eq!(bigrams[0], "a\x01b");
    assert_eq!(bigrams[1], "b\x01c");
    assert_eq!(bigrams[2], "c\x01d");
}

#[test]
fn test_bigram_with_unicode_tokens() {
    let tokens: Vec<String> = vec!["한글", "테스트"].into_iter().map(String::from).collect();
    let bigrams = generate(&tokens);
    assert_eq!(bigrams.len(), 1);
    assert_eq!(bigrams[0], "한글\x01테스트");
}

#[test]
fn test_bigram_separator_is_not_printable() {
    let tokens: Vec<String> = vec!["foo", "bar"].into_iter().map(String::from).collect();
    let bigrams = generate(&tokens);
    // The separator \x01 should not match any normal text token
    assert!(!bigrams[0].contains(' '));
    assert!(bigrams[0].contains('\x01'));
}

#[test]
fn test_bigram_preserves_token_order() {
    let tokens: Vec<String> = vec!["x", "y"].into_iter().map(String::from).collect();
    let bigrams = generate(&tokens);
    assert_eq!(bigrams[0], "x\x01y");
    // Reversed order should produce different bigram
    let tokens_rev: Vec<String> = vec!["y", "x"].into_iter().map(String::from).collect();
    let bigrams_rev = generate(&tokens_rev);
    assert_eq!(bigrams_rev[0], "y\x01x");
    assert_ne!(bigrams[0], bigrams_rev[0]);
}

#[test]
fn test_bigram_with_empty_string_tokens() {
    // Even empty strings produce bigrams (separator separates them)
    let tokens: Vec<String> = vec!["", ""].into_iter().map(String::from).collect();
    let bigrams = generate(&tokens);
    assert_eq!(bigrams.len(), 1);
    assert_eq!(bigrams[0], "\x01");
}

#[test]
fn test_bigram_duplicate_tokens() {
    let tokens: Vec<String> = vec!["a", "a", "a"].into_iter().map(String::from).collect();
    let bigrams = generate(&tokens);
    assert_eq!(bigrams.len(), 2);
    assert_eq!(bigrams[0], "a\x01a");
    assert_eq!(bigrams[1], "a\x01a");
}
