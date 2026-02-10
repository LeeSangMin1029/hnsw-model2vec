//! Korean BM25 tokenizer adapter.
//!
//! Wraps `crate::tokenizer::KoreanTokenizer` (Lindera ko-dic) for use with
//! the BM25 `Tokenizer` trait. The Lindera tokenizer must be initialized
//! explicitly via `init_korean_tokenizer()` before first use.

use std::path::Path;
use std::sync::OnceLock;

use v_hnsw_core::VhnswError;

use crate::tokenizer::KoreanTokenizer;
use crate::tokenizer::Tokenizer as KorTokenizer;

static KOREAN_TOKENIZER: OnceLock<KoreanTokenizer> = OnceLock::new();

/// Initialize the Korean tokenizer with a compiled dictionary path.
///
/// Must be called before any tokenization. Thread-safe; first call wins.
/// Subsequent calls return `Ok(())` if already initialized.
///
/// # Arguments
///
/// * `dict_path` - Path to a compiled lindera ko-dic directory
///   (e.g., `~/.v-hnsw/dict/ko-dic/`)
pub fn init_korean_tokenizer(dict_path: &Path) -> Result<(), VhnswError> {
    if KOREAN_TOKENIZER.get().is_some() {
        return Ok(());
    }
    let tokenizer = KoreanTokenizer::new(dict_path)?;
    // Ignore set error (race: another thread initialized first)
    let _ = KOREAN_TOKENIZER.set(tokenizer);
    Ok(())
}

fn get_tokenizer() -> &'static KoreanTokenizer {
    #[allow(clippy::expect_used)]
    KOREAN_TOKENIZER
        .get()
        .expect("Korean tokenizer not initialized. Call init_korean_tokenizer() first.")
}

/// Korean morphological tokenizer for BM25 indexing.
///
/// Uses Lindera with ko-dic dictionary for Korean text segmentation.
/// Applies stopword removal and minimum length filtering.
/// Also handles English and mixed-language text.
///
/// This is a zero-size marker type. The actual tokenizer state is
/// held in a global `OnceLock` and shared across all instances.
/// Call `init_korean_tokenizer()` before using this type.
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
        match get_tokenizer().tokenize_for_index(text) {
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
    use std::path::PathBuf;

    use super::*;
    use crate::Tokenizer;

    fn test_dict_path() -> PathBuf {
        if let Ok(path) = std::env::var("LINDERA_KO_DIC_PATH") {
            return PathBuf::from(path);
        }
        v_hnsw_core::ko_dic_dir()
    }

    fn ensure_init() {
        let _ = init_korean_tokenizer(&test_dict_path());
    }

    #[test]
    fn test_korean_tokenizer_basic() {
        ensure_init();
        let tokenizer = KoreanBm25Tokenizer::new();
        let tokens = tokenizer.tokenize("안녕하세요 세계");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_korean_tokenizer_morphological() {
        ensure_init();
        let tokenizer = KoreanBm25Tokenizer::new();
        let tokens = tokenizer.tokenize("한국은 아름다운 나라입니다");
        assert!(!tokens.contains(&"은".to_string()));
    }

    #[test]
    fn test_korean_tokenizer_english() {
        ensure_init();
        let tokenizer = KoreanBm25Tokenizer::new();
        let tokens = tokenizer.tokenize("hello world test");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_korean_tokenizer_mixed() {
        ensure_init();
        let tokenizer = KoreanBm25Tokenizer::new();
        let tokens = tokenizer.tokenize("벡터 데이터베이스 vector database");
        assert!(tokens.len() >= 2);
    }

    #[test]
    fn test_korean_tokenizer_empty() {
        ensure_init();
        let tokenizer = KoreanBm25Tokenizer::new();
        let tokens = tokenizer.tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_korean_tokenizer_clone() {
        ensure_init();
        let t1 = KoreanBm25Tokenizer::new();
        let t2 = t1.clone();
        let tokens1 = t1.tokenize("테스트");
        let tokens2 = t2.tokenize("테스트");
        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn test_korean_tokenizer_serialization() {
        ensure_init();
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
