//! Korean BM25 tokenizer adapter.
//!
//! Wraps `v_hnsw_tokenizer::KoreanTokenizer` (Lindera ko-dic) for use with
//! the BM25 `Tokenizer` trait. The Lindera tokenizer is initialized lazily
//! via a global `LazyLock` and shared across all instances.

use std::sync::LazyLock;

use v_hnsw_tokenizer::KoreanTokenizer;
use v_hnsw_tokenizer::Tokenizer as KorTokenizer;

#[allow(clippy::expect_used)]
static KOREAN_TOKENIZER: LazyLock<KoreanTokenizer> = LazyLock::new(|| {
    KoreanTokenizer::new().expect("Failed to create KoreanTokenizer (ko-dic)")
});

/// Korean morphological tokenizer for BM25 indexing.
///
/// Uses Lindera with ko-dic dictionary for Korean text segmentation.
/// Applies stopword removal and minimum length filtering.
/// Also handles English and mixed-language text.
///
/// This is a zero-size marker type. The actual tokenizer state is
/// held in a global `LazyLock` and shared across all instances.
#[derive(
    Debug, Clone, Default, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode,
)]
pub struct KoreanBm25Tokenizer;

impl KoreanBm25Tokenizer {
    /// Create a new Korean BM25 tokenizer.
    pub fn new() -> Self {
        Self
    }
}

impl crate::Tokenizer for KoreanBm25Tokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        match KOREAN_TOKENIZER.tokenize_for_index(text) {
            Ok(tokens) => tokens.into_iter().map(|t| t.text).collect(),
            Err(_) => {
                // Fallback: whitespace tokenization if morphological analysis fails
                text.split_whitespace()
                    .map(|s| s.to_lowercase())
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tokenizer;

    #[test]
    fn test_korean_tokenizer_basic() {
        let tokenizer = KoreanBm25Tokenizer::new();
        let tokens = tokenizer.tokenize("안녕하세요 세계");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_korean_tokenizer_morphological() {
        let tokenizer = KoreanBm25Tokenizer::new();
        // "한국은 아름다운 나라입니다" → should split morphemes and remove particles
        let tokens = tokenizer.tokenize("한국은 아름다운 나라입니다");
        // "은" (particle) and short tokens should be filtered out
        assert!(!tokens.contains(&"은".to_string()));
    }

    #[test]
    fn test_korean_tokenizer_english() {
        let tokenizer = KoreanBm25Tokenizer::new();
        let tokens = tokenizer.tokenize("hello world test");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_korean_tokenizer_mixed() {
        let tokenizer = KoreanBm25Tokenizer::new();
        let tokens = tokenizer.tokenize("벡터 데이터베이스 vector database");
        assert!(tokens.len() >= 2);
    }

    #[test]
    fn test_korean_tokenizer_empty() {
        let tokenizer = KoreanBm25Tokenizer::new();
        let tokens = tokenizer.tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_korean_tokenizer_clone() {
        let t1 = KoreanBm25Tokenizer::new();
        let t2 = t1.clone();
        let tokens1 = t1.tokenize("테스트");
        let tokens2 = t2.tokenize("테스트");
        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn test_korean_tokenizer_serialization() {
        let tokenizer = KoreanBm25Tokenizer::new();

        // serde roundtrip
        let json = serde_json::to_string(&tokenizer).expect("serialize");
        let _: KoreanBm25Tokenizer = serde_json::from_str(&json).expect("deserialize");

        // bincode roundtrip
        let bytes =
            bincode::encode_to_vec(&tokenizer, bincode::config::standard()).expect("encode");
        let (decoded, _): (KoreanBm25Tokenizer, usize) =
            bincode::decode_from_slice(&bytes, bincode::config::standard()).expect("decode");

        // Verify it still works after deserialization
        let tokens = decoded.tokenize("테스트 문장");
        assert!(!tokens.is_empty());
    }
}
