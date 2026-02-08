//! Korean tokenizer for v-hnsw.
//!
//! Morphological analysis via Lindera ko-dic for BM25 indexing.
//! Supports user dictionaries and Hangul Jamo decomposition.
//!
//! # Features
//!
//! - **Korean Tokenization**: Morphological analysis using Lindera with ko-dic dictionary
//! - **Hangul Jamo Decomposition**: Break syllables into consonants and vowels
//! - **Choseong Search**: Enable search by initial consonants (초성 검색)
//! - **Token Filters**: Stopwords, minimum length, and custom filter chains
//! - **User Dictionary**: Add domain-specific terms for better tokenization
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use v_hnsw_search::tokenizer::{KoreanTokenizer, Tokenizer};
//! use v_hnsw_search::tokenizer::jamo::{extract_choseong, matches_choseong};
//!
//! // Create a Korean tokenizer with compiled dictionary path
//! let dict_path = Path::new("/path/to/ko-dic");
//! let tokenizer = KoreanTokenizer::new(dict_path).unwrap();
//!
//! // Tokenize Korean text
//! let tokens = tokenizer.tokenize("안녕하세요 세계").unwrap();
//! for token in &tokens {
//!     println!("{}: {:?}", token.text, token.kind);
//! }
//!
//! // Choseong (initial consonant) search
//! assert_eq!(extract_choseong("한글"), "ㅎㄱ");
//! assert!(matches_choseong("한글 프로그래밍", "ㅎㄱ"));
//! ```

pub mod filters;
pub mod jamo;
pub mod korean;
pub mod user_dict;

#[cfg(test)]
mod tests;

// Re-export main types at crate root
pub use filters::{FilterChain, LowercaseFilter, MinLengthFilter, StopwordFilter, TokenFilter};
pub use korean::{
    KoreanTokenizer, KoreanTokenizerConfig, Token, TokenKind, Tokenizer, TokenizerMode,
    WhitespaceTokenizer,
};
pub use user_dict::{DictionaryEntry, UserDictionary};
