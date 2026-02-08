//! Tokenizer trait and Korean tokenizer implementation.
//!
//! Provides morphological analysis for Korean text using Lindera with ko-dic.

use std::path::Path;

use lindera::dictionary::load_fs_dictionary;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;
use lindera::tokenizer::Tokenizer as LinderaTokenizer;
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;
use v_hnsw_core::VhnswError;

use super::filters::{FilterChain, TokenFilter};

/// Token kind based on part-of-speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TokenKind {
    /// Noun (명사)
    Noun,
    /// Verb (동사)
    Verb,
    /// Adjective (형용사)
    Adjective,
    /// Adverb (부사)
    Adverb,
    /// Particle (조사)
    Particle,
    /// Punctuation
    Punctuation,
    /// Number
    Number,
    /// Foreign word / Symbol
    Foreign,
    /// Unknown or other
    Unknown,
}

impl TokenKind {
    /// Parse from Korean POS tag (e.g., "NNG", "VV", "JKS").
    fn from_pos(pos: &str) -> Self {
        if pos.is_empty() {
            return Self::Unknown;
        }

        // Korean POS tags (Sejong tagset used by ko-dic)
        match &pos[..1] {
            "N" => Self::Noun,      // NNG, NNP, NNB, NR, NP
            "V" => Self::Verb,      // VV, VA, VX, VCP, VCN
            "M" => Self::Adjective, // MM (관형사)
            "J" => Self::Particle,  // JKS, JKC, JKG, etc.
            "E" => Self::Particle,  // Endings (어미)
            "S" => {
                // SF, SP, SS, SE, SO, SW
                if pos.starts_with("SN") {
                    Self::Number
                } else if pos.starts_with("SL") || pos.starts_with("SH") {
                    Self::Foreign
                } else {
                    Self::Punctuation
                }
            }
            "X" => Self::Unknown, // XPN, XSN, XSV, XSA, XR
            _ => Self::Unknown,
        }
    }

    /// Check if this token kind is typically important for search.
    pub fn is_searchable(&self) -> bool {
        matches!(
            self,
            Self::Noun | Self::Verb | Self::Adjective | Self::Adverb | Self::Foreign
        )
    }
}

/// A single token from tokenization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Token {
    /// The token text.
    pub text: String,
    /// Start byte position in the original text.
    pub start: usize,
    /// End byte position in the original text.
    pub end: usize,
    /// Token kind based on POS.
    pub kind: TokenKind,
}

impl Token {
    /// Create a new token.
    pub fn new(text: impl Into<String>, start: usize, end: usize, kind: TokenKind) -> Self {
        Self {
            text: text.into(),
            start,
            end,
            kind,
        }
    }
}

/// Trait for text tokenization.
pub trait Tokenizer: Send + Sync {
    /// Tokenize text into tokens.
    fn tokenize(&self, text: &str) -> Result<Vec<Token>, VhnswError>;

    /// Tokenize text for indexing (may apply different filters).
    fn tokenize_for_index(&self, text: &str) -> Result<Vec<Token>, VhnswError> {
        self.tokenize(text)
    }

    /// Tokenize text for query (may apply different filters).
    fn tokenize_for_query(&self, text: &str) -> Result<Vec<Token>, VhnswError> {
        self.tokenize(text)
    }
}

/// Configuration for KoreanTokenizer.
#[derive(Debug, Clone)]
pub struct KoreanTokenizerConfig {
    /// Whether to normalize Unicode (NFKC).
    pub normalize_unicode: bool,
    /// Whether to convert to lowercase.
    pub lowercase: bool,
    /// Mode for tokenization (normal or search).
    pub mode: TokenizerMode,
}

impl Default for KoreanTokenizerConfig {
    fn default() -> Self {
        Self {
            normalize_unicode: true,
            lowercase: true,
            mode: TokenizerMode::Normal,
        }
    }
}

/// Tokenization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerMode {
    /// Normal tokenization.
    Normal,
    /// Search mode (may split compound words).
    Search,
}

/// Korean tokenizer using Lindera with ko-dic.
pub struct KoreanTokenizer {
    tokenizer: LinderaTokenizer,
    config: KoreanTokenizerConfig,
    index_filters: FilterChain,
    query_filters: FilterChain,
}

impl KoreanTokenizer {
    /// Create a new Korean tokenizer loading the dictionary from disk.
    ///
    /// `dict_path` should point to a compiled lindera dictionary directory
    /// (e.g., `~/.v-hnsw/dict/ko-dic/`).
    pub fn new(dict_path: &Path) -> Result<Self, VhnswError> {
        Self::with_config(dict_path, KoreanTokenizerConfig::default())
    }

    /// Create a new Korean tokenizer with custom configuration.
    pub fn with_config(
        dict_path: &Path,
        config: KoreanTokenizerConfig,
    ) -> Result<Self, VhnswError> {
        let mode = match config.mode {
            TokenizerMode::Normal => Mode::Normal,
            TokenizerMode::Search => Mode::Normal, // Lindera doesn't have search mode for ko-dic
        };

        // Load the ko-dic dictionary from filesystem
        let dictionary = load_fs_dictionary(dict_path).map_err(|e| {
            VhnswError::Tokenizer(format!(
                "failed to load ko-dic dictionary from {}: {}",
                dict_path.display(),
                e
            ))
        })?;

        // Create segmenter with the dictionary
        let segmenter = Segmenter::new(mode, dictionary, None);

        // Create tokenizer from segmenter
        let tokenizer = LinderaTokenizer::new(segmenter);

        Ok(Self {
            tokenizer,
            config,
            index_filters: FilterChain::korean_default(),
            query_filters: FilterChain::korean_default(),
        })
    }

    /// Set the filter chain for indexing.
    pub fn set_index_filters(&mut self, filters: FilterChain) {
        self.index_filters = filters;
    }

    /// Set the filter chain for queries.
    pub fn set_query_filters(&mut self, filters: FilterChain) {
        self.query_filters = filters;
    }

    /// Preprocess text before tokenization.
    fn preprocess(&self, text: &str) -> String {
        let mut result = text.to_string();

        if self.config.normalize_unicode {
            result = result.nfkc().collect();
        }

        if self.config.lowercase {
            result = result.to_lowercase();
        }

        result
    }

    /// Internal tokenization.
    fn tokenize_internal(&self, text: &str) -> Result<Vec<Token>, VhnswError> {
        let preprocessed = self.preprocess(text);

        let lindera_tokens = self.tokenizer.tokenize(&preprocessed).map_err(|e| {
            VhnswError::Tokenizer(format!("tokenization failed: {}", e))
        })?;

        let mut tokens = Vec::with_capacity(lindera_tokens.len());

        for lt in lindera_tokens {
            let text_str = lt.surface.to_string();

            // Get POS tag - Lindera returns details as Option<Vec<Cow<str>>>
            let pos = lt
                .details
                .as_ref()
                .and_then(|d| d.first())
                .map(|s| s.as_ref())
                .unwrap_or("*");

            let kind = TokenKind::from_pos(pos);

            tokens.push(Token {
                text: text_str,
                start: lt.byte_start,
                end: lt.byte_end,
                kind,
            });
        }

        Ok(tokens)
    }
}

impl Tokenizer for KoreanTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<Token>, VhnswError> {
        self.tokenize_internal(text)
    }

    fn tokenize_for_index(&self, text: &str) -> Result<Vec<Token>, VhnswError> {
        let tokens = self.tokenize_internal(text)?;
        Ok(self.index_filters.filter(tokens))
    }

    fn tokenize_for_query(&self, text: &str) -> Result<Vec<Token>, VhnswError> {
        let tokens = self.tokenize_internal(text)?;
        Ok(self.query_filters.filter(tokens))
    }
}

/// A simple whitespace tokenizer for non-Korean text.
#[derive(Debug, Clone, Default)]
pub struct WhitespaceTokenizer {
    lowercase: bool,
}

impl WhitespaceTokenizer {
    /// Create a new whitespace tokenizer.
    pub fn new() -> Self {
        Self { lowercase: true }
    }

    /// Create a whitespace tokenizer with custom settings.
    pub fn with_lowercase(lowercase: bool) -> Self {
        Self { lowercase }
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<Token>, VhnswError> {
        let text = if self.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        let mut tokens = Vec::new();
        let mut start = 0;

        for word in text.split_whitespace() {
            // Find the actual position in the original text
            if let Some(pos) = text[start..].find(word) {
                let word_start = start + pos;
                let word_end = word_start + word.len();

                tokens.push(Token {
                    text: word.to_string(),
                    start: word_start,
                    end: word_end,
                    kind: TokenKind::Unknown,
                });

                start = word_end;
            }
        }

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    /// Resolve ko-dic path for tests.
    /// Uses LINDERA_KO_DIC_PATH env var, falls back to ~/.v-hnsw/dict/ko-dic/
    fn test_dict_path() -> PathBuf {
        if let Ok(path) = std::env::var("LINDERA_KO_DIC_PATH") {
            return PathBuf::from(path);
        }
        let home = std::env::var("USERPROFILE")
            .or_else(|_| std::env::var("HOME"))
            .expect("HOME or USERPROFILE env var");
        PathBuf::from(home).join(".v-hnsw/dict/ko-dic")
    }

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
        let tokenizer = KoreanTokenizer::new(&test_dict_path()).expect("create tokenizer");
        let tokens = tokenizer.tokenize("안녕하세요").expect("tokenize");

        assert!(!tokens.is_empty());
        // The exact tokenization depends on ko-dic
    }

    #[test]
    fn test_korean_tokenizer_mixed() {
        let tokenizer = KoreanTokenizer::new(&test_dict_path()).expect("create tokenizer");
        let tokens = tokenizer.tokenize("Hello 세계").expect("tokenize");

        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_korean_tokenizer_empty() {
        let tokenizer = KoreanTokenizer::new(&test_dict_path()).expect("create tokenizer");
        let tokens = tokenizer.tokenize("").expect("tokenize");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_korean_tokenizer_for_index() {
        let tokenizer = KoreanTokenizer::new(&test_dict_path()).expect("create tokenizer");
        let tokens = tokenizer.tokenize_for_index("한국은 아름다운 나라입니다").expect("tokenize");

        // Should filter out stopwords and short tokens
        for token in &tokens {
            // Stopwords like "은" should be filtered
            assert_ne!(token.text, "은");
        }
    }

    #[test]
    fn test_korean_tokenizer_unicode_normalization() {
        let tokenizer = KoreanTokenizer::new(&test_dict_path()).expect("create tokenizer");

        // Full-width characters should be normalized
        let tokens = tokenizer.tokenize("１２３").expect("tokenize");
        // After NFKC normalization, full-width digits become ASCII
        assert!(!tokens.is_empty());
    }
}
