//! Integration tests for the tokenizer crate.

mod filters;
mod jamo;
#[cfg(feature = "korean")]
mod korean;
mod user_dict;

use std::path::PathBuf;
use std::sync::OnceLock;

use super::filters::{FilterChain, MinLengthFilter, StopwordFilter, TokenFilter};
use super::jamo::{decompose_hangul, extract_choseong, matches_choseong};
use super::korean::{KoreanTokenizer, Tokenizer, WhitespaceTokenizer};
use super::user_dict::{DictionaryEntry, UserDictionary};

/// Resolve ko-dic path for tests.
#[cfg(feature = "korean")]
pub(super) fn test_dict_path() -> PathBuf {
    if let Ok(path) = std::env::var("LINDERA_KO_DIC_PATH") {
        return PathBuf::from(path);
    }
    v_hnsw_core::ko_dic_dir()
}

/// Shared KoreanTokenizer — loaded once, reused across all tests.
/// Lindera dictionary load takes ~8-10 seconds, so sharing is essential.
#[cfg(feature = "korean")]
pub(super) fn shared_korean_tokenizer() -> &'static KoreanTokenizer {
    static TOKENIZER: OnceLock<KoreanTokenizer> = OnceLock::new();
    TOKENIZER.get_or_init(|| {
        KoreanTokenizer::new(&test_dict_path())
            .unwrap_or_else(|e| panic!("failed to create shared KoreanTokenizer: {e}"))
    })
}

#[cfg(feature = "korean")]
#[test]
fn test_korean_tokenization_sentence() {
    let tokenizer = shared_korean_tokenizer();

    let text = "대한민국은 민주공화국이다";
    let tokens = tokenizer.tokenize(text).expect("tokenize");

    assert!(!tokens.is_empty());

    // Check that we got meaningful tokens
    let texts: Vec<_> = tokens.iter().map(|t| t.text.as_str()).collect();
    // Should contain "대한민국" or parts of it
    assert!(
        texts.iter().any(|t| t.contains("대한") || t.contains("민국")),
        "should contain parts of 대한민국: {:?}",
        texts
    );
}

#[cfg(feature = "korean")]
#[test]
fn test_korean_tokenization_with_filters() {
    let tokenizer = shared_korean_tokenizer();

    let text = "서울은 한국의 수도이다";
    let tokens = tokenizer.tokenize_for_index(text).expect("tokenize");

    // Stopwords should be filtered out
    let texts: Vec<_> = tokens.iter().map(|t| t.text.as_str()).collect();

    // "은", "의" are stopwords and should be filtered
    assert!(
        !texts.contains(&"은"),
        "stopword '은' should be filtered: {:?}",
        texts
    );
    assert!(
        !texts.contains(&"의"),
        "stopword '의' should be filtered: {:?}",
        texts
    );
}

#[test]
fn test_jamo_decomposition_correctness() {
    // Test basic decomposition
    assert_eq!(decompose_hangul("가"), "ㄱㅏ");
    assert_eq!(decompose_hangul("한"), "ㅎㅏㄴ");
    assert_eq!(decompose_hangul("글"), "ㄱㅡㄹ");
    assert_eq!(decompose_hangul("한글"), "ㅎㅏㄴㄱㅡㄹ");

    // Test with no final consonant
    assert_eq!(decompose_hangul("아"), "ㅇㅏ");
    assert_eq!(decompose_hangul("이"), "ㅇㅣ");

    // Test mixed content
    assert_eq!(decompose_hangul("한A글B"), "ㅎㅏㄴAㄱㅡㄹB");

    // Test choseong extraction
    assert_eq!(extract_choseong("한글"), "ㅎㄱ");
    assert_eq!(extract_choseong("대한민국"), "ㄷㅎㅁㄱ");
    assert_eq!(extract_choseong("프로그래밍"), "ㅍㄹㄱㄹㅁ");
}

#[cfg(feature = "korean")]
#[test]
fn test_choseong_search() {
    // Basic matching
    assert!(matches_choseong("한글", "ㅎㄱ"));
    assert!(matches_choseong("프로그래밍", "ㅍㄹㄱ"));

    // Partial matching
    assert!(matches_choseong("한글 프로그래밍", "ㅎㄱ"));
    assert!(matches_choseong("한글 프로그래밍", "ㅍㄹㄱ"));

    // Non-matching
    assert!(!matches_choseong("한글", "ㄱㅎ"));
    assert!(!matches_choseong("영어", "ㅎㄱ"));
}

#[test]
fn test_stopword_filtering() {
    let filter = StopwordFilter::korean();

    // Common particles should be stopwords
    assert!(filter.is_stopword("은"));
    assert!(filter.is_stopword("는"));
    assert!(filter.is_stopword("이"));
    assert!(filter.is_stopword("가"));
    assert!(filter.is_stopword("을"));
    assert!(filter.is_stopword("를"));
    assert!(filter.is_stopword("의"));

    // Content words should not be stopwords
    assert!(!filter.is_stopword("한국"));
    assert!(!filter.is_stopword("사랑"));
    assert!(!filter.is_stopword("컴퓨터"));
}

#[test]
fn test_filter_chain() {
    let chain = FilterChain::new()
        .add_filter(StopwordFilter::korean())
        .add_filter(MinLengthFilter::new(2));

    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer.tokenize("a ab abc 가 나다 라마바").expect("tokenize");

    let filtered = chain.filter(tokens);
    let texts: Vec<_> = filtered.iter().map(|t| t.text.as_str()).collect();

    // Single chars should be filtered by MinLengthFilter
    assert!(!texts.contains(&"a"));
    assert!(!texts.contains(&"가"));

    // Longer tokens should remain (unless stopword)
    assert!(texts.contains(&"ab"));
    assert!(texts.contains(&"abc"));
    assert!(texts.contains(&"나다"));
    assert!(texts.contains(&"라마바"));
}

#[cfg(feature = "korean")]
#[test]
fn test_edge_case_empty_string() {
    let tokenizer = shared_korean_tokenizer();
    let tokens = tokenizer.tokenize("").expect("tokenize");
    assert!(tokens.is_empty());

    let tokens = tokenizer.tokenize_for_index("").expect("tokenize");
    assert!(tokens.is_empty());

    let tokens = tokenizer.tokenize_for_query("").expect("tokenize");
    assert!(tokens.is_empty());
}

#[cfg(feature = "korean")]
#[test]
fn test_edge_case_ascii_only() {
    let tokenizer = shared_korean_tokenizer();
    let tokens = tokenizer.tokenize("hello world").expect("tokenize");

    assert!(!tokens.is_empty());
    let texts: Vec<_> = tokens.iter().map(|t| t.text.as_str()).collect();
    assert!(texts.contains(&"hello"));
    assert!(texts.contains(&"world"));
}

#[cfg(feature = "korean")]
#[test]
fn test_edge_case_mixed_content() {
    let tokenizer = shared_korean_tokenizer();
    let tokens = tokenizer.tokenize("Hello 세계 World 123").expect("tokenize");

    assert!(!tokens.is_empty());

    // Should handle both Korean and ASCII
    let texts: Vec<_> = tokens.iter().map(|t| t.text.as_str()).collect();
    assert!(
        texts.iter().any(|t| t.contains("hello") || t.contains("Hello")),
        "should contain 'hello': {:?}",
        texts
    );
}

#[cfg(feature = "korean")]
#[test]
fn test_edge_case_whitespace_only() {
    let tokenizer = shared_korean_tokenizer();
    let tokens = tokenizer.tokenize("   \t\n  ").expect("tokenize");
    // Whitespace-only input should produce minimal/no tokens
    // The exact behavior depends on Lindera
    assert!(tokens.is_empty() || tokens.iter().all(|t| t.text.trim().is_empty()));
}

#[cfg(feature = "korean")]
#[test]
fn test_edge_case_special_characters() {
    let tokenizer = shared_korean_tokenizer();
    let tokens = tokenizer.tokenize("한글!@#$%^&*()테스트").expect("tokenize");

    // Should handle special characters gracefully
    assert!(!tokens.is_empty());
}

#[test]
fn test_user_dictionary_parsing() {
    let csv = r#"
삼성전자,NNP,삼성전자
카카오,NNP
# 주석
네이버,NNP,네이버
"#;

    let dict = UserDictionary::load_from_str(csv).expect("parse");
    assert_eq!(dict.len(), 3);

    let entries = dict.entries();
    assert_eq!(entries[0].term, "삼성전자");
    assert_eq!(entries[0].pos, "NNP");
    assert_eq!(entries[1].term, "카카오");
    assert_eq!(entries[2].term, "네이버");
}

#[test]
fn test_user_dictionary_entry_creation() {
    let entry1 = DictionaryEntry::new("테스트", "NNG", "테스트");
    assert_eq!(entry1.term, "테스트");
    assert_eq!(entry1.pos, "NNG");
    assert_eq!(entry1.reading, "테스트");

    let entry2 = DictionaryEntry::simple("간단한", "NNG");
    assert_eq!(entry2.term, "간단한");
    assert_eq!(entry2.reading, "간단한"); // Should be same as term
}

#[test]
fn test_jamo_edge_cases() {
    // Already decomposed jamo should pass through
    assert_eq!(extract_choseong("ㅎㄱ"), "ㅎㄱ");

    // Numbers and symbols
    assert_eq!(extract_choseong("123"), "123");
    assert_eq!(extract_choseong("!@#"), "!@#");

    // Mixed jamo and syllables
    assert_eq!(extract_choseong("ㅎ한ㄱ글"), "ㅎㅎㄱㄱ");
}

#[cfg(feature = "korean")]
#[test]
fn test_tokenizer_consistency() {
    let tokenizer = shared_korean_tokenizer();
    let text = "동일한 텍스트를 여러 번 토큰화";

    let tokens1 = tokenizer.tokenize(text).expect("first tokenize");
    let tokens2 = tokenizer.tokenize(text).expect("second tokenize");

    // Same input should produce same output
    assert_eq!(tokens1.len(), tokens2.len());
    for (t1, t2) in tokens1.iter().zip(tokens2.iter()) {
        assert_eq!(t1.text, t2.text);
        assert_eq!(t1.start, t2.start);
        assert_eq!(t1.end, t2.end);
    }
}

#[cfg(feature = "korean")]
#[test]
fn test_tokenize_for_index_vs_query() {
    let tokenizer = shared_korean_tokenizer();
    let text = "검색 테스트 문장입니다";

    let index_tokens = tokenizer.tokenize_for_index(text).expect("tokenize for index");
    let query_tokens = tokenizer.tokenize_for_query(text).expect("tokenize for query");

    // With default filters, should produce similar results
    // But the interface allows for different filtering strategies
    assert!(!index_tokens.is_empty());
    assert!(!query_tokens.is_empty());
}
