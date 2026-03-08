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