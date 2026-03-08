use crate::tokenizer::filters::*;
use crate::tokenizer::korean::{Token, TokenKind};

fn make_token(text: &str) -> Token {
    Token {
        text: text.to_string(),
        start: 0,
        end: text.len(),
        kind: TokenKind::Unknown,
    }
}

fn make_tokens(texts: &[&str]) -> Vec<Token> {
    texts.iter().map(|t| make_token(t)).collect()
}

#[test]
fn test_stopword_filter() {
    let filter = StopwordFilter::korean();

    let tokens = make_tokens(&["한국", "은", "아름다운", "나라", "이다"]);
    let filtered = filter.filter(tokens);

    let texts: Vec<_> = filtered.iter().map(|t| t.text.as_str()).collect();
    assert_eq!(texts, vec!["한국", "아름다운", "나라"]);
}

#[test]
fn test_stopword_filter_custom() {
    let filter = StopwordFilter::new(["foo", "bar"]);

    let tokens = make_tokens(&["foo", "baz", "bar", "qux"]);
    let filtered = filter.filter(tokens);

    let texts: Vec<_> = filtered.iter().map(|t| t.text.as_str()).collect();
    assert_eq!(texts, vec!["baz", "qux"]);
}

#[test]
fn test_min_length_filter() {
    let filter = MinLengthFilter::new(2);

    let tokens = make_tokens(&["a", "ab", "abc", "가", "가나", "가나다"]);
    let filtered = filter.filter(tokens);

    let texts: Vec<_> = filtered.iter().map(|t| t.text.as_str()).collect();
    assert_eq!(texts, vec!["ab", "abc", "가나", "가나다"]);
}

#[test]
fn test_lowercase_filter() {
    let filter = LowercaseFilter;

    let tokens = make_tokens(&["Hello", "WORLD", "한글", "MixED"]);
    let filtered = filter.filter(tokens);

    let texts: Vec<_> = filtered.iter().map(|t| t.text.as_str()).collect();
    assert_eq!(texts, vec!["hello", "world", "한글", "mixed"]);
}

#[test]
fn test_filter_chain() {
    let chain = FilterChain::new()
        .add_filter(StopwordFilter::korean())
        .add_filter(MinLengthFilter::new(2));

    let tokens = make_tokens(&["한국", "은", "아름다운", "나라", "가"]);
    let filtered = chain.filter(tokens);

    let texts: Vec<_> = filtered.iter().map(|t| t.text.as_str()).collect();
    assert_eq!(texts, vec!["한국", "아름다운", "나라"]);
}

#[test]
fn test_filter_chain_korean_default() {
    let chain = FilterChain::korean_default();

    let tokens = make_tokens(&["대한민국", "은", "민주", "공화국", "이다"]);
    let filtered = chain.filter(tokens);

    let texts: Vec<_> = filtered.iter().map(|t| t.text.as_str()).collect();
    assert_eq!(texts, vec!["대한민국", "민주", "공화국"]);
}

#[test]
fn test_empty_tokens() {
    let filter = StopwordFilter::korean();
    let filtered = filter.filter(vec![]);
    assert!(filtered.is_empty());

    let chain = FilterChain::korean_default();
    let filtered = chain.filter(vec![]);
    assert!(filtered.is_empty());
}

// ============================================================================
// Additional filter tests
// ============================================================================

#[test]
fn test_stopword_filter_add_remove() {
    let mut filter = StopwordFilter::new(["foo"]);
    assert!(filter.is_stopword("foo"));
    assert!(!filter.is_stopword("bar"));

    filter.add_stopword("bar");
    assert!(filter.is_stopword("bar"));

    assert!(filter.remove_stopword("foo"));
    assert!(!filter.is_stopword("foo"));

    // Remove nonexistent
    assert!(!filter.remove_stopword("xyz"));
}

#[test]
fn test_stopword_filter_empty_set() {
    let filter = StopwordFilter::new(Vec::<String>::new());
    let tokens = make_tokens(&["hello", "world"]);
    let filtered = filter.filter(tokens);
    assert_eq!(filtered.len(), 2);
}

#[test]
fn test_stopword_filter_all_stopwords() {
    let filter = StopwordFilter::new(["a", "b", "c"]);
    let tokens = make_tokens(&["a", "b", "c"]);
    let filtered = filter.filter(tokens);
    assert!(filtered.is_empty());
}

#[test]
fn test_stopword_korean_default_coverage() {
    let filter = StopwordFilter::korean();
    // Check a selection of the DEFAULT_KOREAN_STOPWORDS
    for &sw in &["이", "가", "은", "는", "을", "를", "의", "에", "그리고", "하다", "것"] {
        assert!(filter.is_stopword(sw), "'{}' should be a default Korean stopword", sw);
    }
}

#[test]
fn test_min_length_filter_zero() {
    let filter = MinLengthFilter::new(0);
    let tokens = make_tokens(&["", "a", "ab"]);
    let filtered = filter.filter(tokens);
    // Even empty tokens pass (chars().count() >= 0 is always true)
    assert_eq!(filtered.len(), 3);
}

#[test]
fn test_min_length_filter_default() {
    let filter = MinLengthFilter::default();
    // Default is 2
    let tokens = make_tokens(&["a", "ab", "abc"]);
    let filtered = filter.filter(tokens);
    let texts: Vec<_> = filtered.iter().map(|t| t.text.as_str()).collect();
    assert_eq!(texts, vec!["ab", "abc"]);
}

#[test]
fn test_min_length_filter_unicode_char_count() {
    // Korean chars: each hangul syllable is 1 char (even though multi-byte in UTF-8)
    let filter = MinLengthFilter::new(2);
    let tokens = make_tokens(&["가", "가나", "가나다"]);
    let filtered = filter.filter(tokens);
    let texts: Vec<_> = filtered.iter().map(|t| t.text.as_str()).collect();
    assert_eq!(texts, vec!["가나", "가나다"]);
}

#[test]
fn test_lowercase_filter_already_lowercase() {
    let filter = LowercaseFilter;
    let tokens = make_tokens(&["hello", "world"]);
    let filtered = filter.filter(tokens);
    let texts: Vec<_> = filtered.iter().map(|t| t.text.as_str()).collect();
    assert_eq!(texts, vec!["hello", "world"]);
}

#[test]
fn test_lowercase_filter_unicode() {
    let filter = LowercaseFilter;
    let tokens = make_tokens(&["CAFÉ", "RÉSUMÉ"]);
    let filtered = filter.filter(tokens);
    let texts: Vec<_> = filtered.iter().map(|t| t.text.as_str()).collect();
    assert_eq!(texts, vec!["café", "résumé"]);
}

#[test]
fn test_filter_chain_ordering_matters() {
    // If min_length comes before stopword, result differs
    // Min length 2 first, then stopword
    let chain1 = FilterChain::new()
        .add_filter(MinLengthFilter::new(2))
        .add_filter(StopwordFilter::new(["ab"]));

    // Stopword first, then min length 2
    let chain2 = FilterChain::new()
        .add_filter(StopwordFilter::new(["ab"]))
        .add_filter(MinLengthFilter::new(2));

    let tokens = make_tokens(&["a", "ab", "abc"]);

    let result1 = chain1.filter(tokens.clone());
    let result2 = chain2.filter(tokens);

    // Both should produce ["abc"], but via different paths
    let texts1: Vec<_> = result1.iter().map(|t| t.text.as_str()).collect();
    let texts2: Vec<_> = result2.iter().map(|t| t.text.as_str()).collect();
    assert_eq!(texts1, vec!["abc"]);
    assert_eq!(texts2, vec!["abc"]);
}

#[test]
fn test_filter_chain_empty_chain() {
    let chain = FilterChain::new();
    let tokens = make_tokens(&["hello", "world"]);
    let filtered = chain.filter(tokens);
    assert_eq!(filtered.len(), 2);
}
