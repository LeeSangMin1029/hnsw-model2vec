use crate::tokenizer::korean::{
    KoreanTokenizer, KoreanTokenizerConfig, Token, TokenKind, Tokenizer, TokenizerMode,
    WhitespaceTokenizer,
};

use super::{test_dict_path, shared_korean_tokenizer};

#[test]
fn test_token_kind_from_pos() {
    assert_eq!(TokenKind::from_pos("NNG"), TokenKind::Noun);
    assert_eq!(TokenKind::from_pos("NNP"), TokenKind::Noun);
    assert_eq!(TokenKind::from_pos("VV"), TokenKind::Verb);
    assert_eq!(TokenKind::from_pos("JKS"), TokenKind::Particle);
    assert_eq!(TokenKind::from_pos("SF"), TokenKind::Punctuation);
    assert_eq!(TokenKind::from_pos("SN"), TokenKind::Number);
    assert_eq!(TokenKind::from_pos("SL"), TokenKind::Foreign);
    assert_eq!(TokenKind::from_pos(""), TokenKind::Unknown);
}

#[test]
fn test_token_kind_is_searchable() {
    assert!(TokenKind::Noun.is_searchable());
    assert!(TokenKind::Verb.is_searchable());
    assert!(TokenKind::Adjective.is_searchable());
    assert!(!TokenKind::Particle.is_searchable());
    assert!(!TokenKind::Punctuation.is_searchable());
}

#[test]
fn test_whitespace_tokenizer() {
    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer.tokenize("Hello World Test").expect("tokenize");

    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].text, "hello");
    assert_eq!(tokens[1].text, "world");
    assert_eq!(tokens[2].text, "test");
}

#[test]
fn test_whitespace_tokenizer_no_lowercase() {
    let tokenizer = WhitespaceTokenizer::with_lowercase(false);
    let tokens = tokenizer.tokenize("Hello World").expect("tokenize");

    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0].text, "Hello");
    assert_eq!(tokens[1].text, "World");
}

#[test]
fn test_whitespace_tokenizer_empty() {
    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer.tokenize("").expect("tokenize");
    assert!(tokens.is_empty());
}

#[test]
fn test_korean_tokenizer_creation() {
    let result = KoreanTokenizer::new(&test_dict_path());
    assert!(result.is_ok());
}

#[test]
fn test_korean_tokenizer_basic() {
    let tokenizer = shared_korean_tokenizer();
    let tokens = tokenizer.tokenize("안녕하세요").expect("tokenize");

    assert!(!tokens.is_empty());
    // The exact tokenization depends on ko-dic
}

#[test]
fn test_korean_tokenizer_mixed() {
    let tokenizer = shared_korean_tokenizer();
    let tokens = tokenizer.tokenize("Hello 세계").expect("tokenize");

    assert!(!tokens.is_empty());
}

#[test]
fn test_korean_tokenizer_empty() {
    let tokenizer = shared_korean_tokenizer();
    let tokens = tokenizer.tokenize("").expect("tokenize");
    assert!(tokens.is_empty());
}

#[test]
fn test_korean_tokenizer_for_index() {
    let tokenizer = shared_korean_tokenizer();
    let tokens = tokenizer.tokenize_for_index("한국은 아름다운 나라입니다").expect("tokenize");

    // Should filter out stopwords and short tokens
    for token in &tokens {
        // Stopwords like "은" should be filtered
        assert_ne!(token.text, "은");
    }
}

#[test]
fn test_korean_tokenizer_unicode_normalization() {
    let tokenizer = shared_korean_tokenizer();

    // Full-width characters should be normalized
    let tokens = tokenizer.tokenize("１２３").expect("tokenize");
    // After NFKC normalization, full-width digits become ASCII
    assert!(!tokens.is_empty());
}

// ============================================================================
// Additional korean tokenizer tests
// ============================================================================

#[test]
fn test_token_kind_from_pos_all_categories() {
    // Exhaustive coverage of first-char matching
    assert_eq!(TokenKind::from_pos("NNG"), TokenKind::Noun);
    assert_eq!(TokenKind::from_pos("NNP"), TokenKind::Noun);
    assert_eq!(TokenKind::from_pos("NNB"), TokenKind::Noun);
    assert_eq!(TokenKind::from_pos("NR"), TokenKind::Noun);
    assert_eq!(TokenKind::from_pos("NP"), TokenKind::Noun);

    assert_eq!(TokenKind::from_pos("VV"), TokenKind::Verb);
    assert_eq!(TokenKind::from_pos("VA"), TokenKind::Verb);
    assert_eq!(TokenKind::from_pos("VX"), TokenKind::Verb);
    assert_eq!(TokenKind::from_pos("VCP"), TokenKind::Verb);
    assert_eq!(TokenKind::from_pos("VCN"), TokenKind::Verb);

    assert_eq!(TokenKind::from_pos("MM"), TokenKind::Adjective);

    assert_eq!(TokenKind::from_pos("JKS"), TokenKind::Particle);
    assert_eq!(TokenKind::from_pos("JKC"), TokenKind::Particle);
    assert_eq!(TokenKind::from_pos("JKG"), TokenKind::Particle);

    assert_eq!(TokenKind::from_pos("EC"), TokenKind::Particle); // Ending
    assert_eq!(TokenKind::from_pos("EF"), TokenKind::Particle);

    assert_eq!(TokenKind::from_pos("SF"), TokenKind::Punctuation);
    assert_eq!(TokenKind::from_pos("SP"), TokenKind::Punctuation);
    assert_eq!(TokenKind::from_pos("SS"), TokenKind::Punctuation);

    assert_eq!(TokenKind::from_pos("SN"), TokenKind::Number);
    assert_eq!(TokenKind::from_pos("SL"), TokenKind::Foreign);
    assert_eq!(TokenKind::from_pos("SH"), TokenKind::Foreign);

    assert_eq!(TokenKind::from_pos("XPN"), TokenKind::Unknown);
    assert_eq!(TokenKind::from_pos("XSN"), TokenKind::Unknown);

    // Unknown first character
    assert_eq!(TokenKind::from_pos("ZZZ"), TokenKind::Unknown);
    assert_eq!(TokenKind::from_pos("*"), TokenKind::Unknown);
}

#[test]
fn test_token_kind_is_searchable_all() {
    assert!(TokenKind::Noun.is_searchable());
    assert!(TokenKind::Verb.is_searchable());
    assert!(TokenKind::Adjective.is_searchable());
    assert!(TokenKind::Adverb.is_searchable());
    assert!(TokenKind::Foreign.is_searchable());

    assert!(!TokenKind::Particle.is_searchable());
    assert!(!TokenKind::Punctuation.is_searchable());
    assert!(!TokenKind::Number.is_searchable());
    assert!(!TokenKind::Unknown.is_searchable());
}

#[test]
fn test_token_creation() {
    let token = Token::new("hello", 0, 5, TokenKind::Foreign);
    assert_eq!(token.text, "hello");
    assert_eq!(token.start, 0);
    assert_eq!(token.end, 5);
    assert_eq!(token.kind, TokenKind::Foreign);
}

#[test]
fn test_whitespace_tokenizer_multiple_spaces() {
    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer.tokenize("  a   b   c  ").expect("tokenize");
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].text, "a");
    assert_eq!(tokens[1].text, "b");
    assert_eq!(tokens[2].text, "c");
}

#[test]
fn test_whitespace_tokenizer_single_token() {
    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer.tokenize("hello").expect("tokenize");
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].text, "hello");
    assert_eq!(tokens[0].start, 0);
    assert_eq!(tokens[0].end, 5);
}

#[test]
fn test_whitespace_tokenizer_positions() {
    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer.tokenize("aa bb cc").expect("tokenize");
    assert_eq!(tokens.len(), 3);
    // Check that positions are reasonable (start < end, non-overlapping)
    for t in &tokens {
        assert!(t.start < t.end);
    }
}

#[test]
fn test_korean_tokenizer_long_sentence() {
    let tokenizer = shared_korean_tokenizer();
    let text = "인공지능은 인간의 학습능력과 추론능력 지각능력 자연언어의 이해능력 등을 컴퓨터 프로그램으로 실현한 기술이다";
    let tokens = tokenizer.tokenize(text).expect("tokenize");
    assert!(tokens.len() >= 5, "Long sentence should produce many tokens: {}", tokens.len());
}

#[test]
fn test_korean_tokenizer_repeated_text() {
    let tokenizer = shared_korean_tokenizer();
    let text = "가가가가가";
    let tokens = tokenizer.tokenize(text).expect("tokenize");
    assert!(!tokens.is_empty());
}

#[test]
fn test_korean_tokenizer_config_defaults() {
    let config = KoreanTokenizerConfig::default();
    assert!(config.normalize_unicode);
    assert!(config.lowercase);
    assert_eq!(config.mode, TokenizerMode::Normal);
}
