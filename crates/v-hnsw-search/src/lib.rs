//! Hybrid search for v-hnsw.
//!
//! Combines dense vector search (HNSW) with sparse keyword search (BM25)
//! using Reciprocal Rank Fusion (RRF) for optimal recall.
//!
//! # Features
//!
//! - **BM25 Index**: Okapi BM25 ranking for keyword-based retrieval
//! - **Convex Fusion**: Combines dense and sparse ranked lists
//! - **Hybrid Search**: Seamlessly combines dense and sparse results
//!
//! # Example
//!
//! ```ignore
//! use v_hnsw_search::{HybridSearcher, HybridSearchConfig, Bm25Index};
//! use v_hnsw_graph::{HnswGraph, HnswConfig};
//!
//! // Create dense and sparse indexes
//! let hnsw_config = HnswConfig::builder().dim(384).build()?;
//! let hnsw = HnswGraph::new(hnsw_config, distance);
//! let bm25 = Bm25Index::new(tokenizer);
//!
//! // Create hybrid searcher
//! let config = HybridSearchConfig::default();
//! let mut searcher = SimpleHybridSearcher::new(hnsw, bm25, config);
//!
//! // Add documents
//! searcher.add_document(1, &embedding, "document text")?;
//!
//! // Search
//! let results = searcher.search(&query_vector, "query text", 10)?;
//! ```

pub mod bm25;
pub mod config;
pub mod fusion;
pub mod hybrid;
#[cfg(feature = "korean")]
pub mod korean_tokenizer;
#[cfg(feature = "korean")]
pub mod tokenizer;

#[cfg(test)]
mod tests;

// Re-exports
pub use bm25::{Bm25Index, Bm25Params, Bm25Snapshot, Posting, PostingList};
pub use config::{HybridSearchConfig, HybridSearchConfigBuilder};
pub use fusion::ConvexFusion;
pub use hybrid::SimpleHybridSearcher;
#[cfg(feature = "korean")]
pub use korean_tokenizer::{KoreanBm25Tokenizer, init_korean_tokenizer};

/// Tokenizer trait for text processing.
///
/// Implementations convert text into a sequence of tokens for BM25 indexing.
pub trait Tokenizer:
    Clone
    + Send
    + Sync
    + serde::Serialize
    + for<'de> serde::Deserialize<'de>
    + bincode::Encode
    + bincode::Decode<()>
{
    /// Tokenize text into a list of tokens.
    ///
    /// Tokens should be normalized (e.g., lowercased, stemmed) for best results.
    fn tokenize(&self, text: &str) -> Vec<String>;
}

/// A simple whitespace tokenizer.
///
/// Splits on whitespace and lowercases tokens. Suitable for basic use cases.
#[derive(
    Debug, Clone, Default, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode,
)]
pub struct WhitespaceTokenizer;

impl WhitespaceTokenizer {
    /// Create a new whitespace tokenizer.
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| s.to_lowercase())
            .collect()
    }
}

/// A simple tokenizer that splits on whitespace and punctuation.
///
/// More thorough than `WhitespaceTokenizer` as it also removes punctuation.
#[derive(
    Debug, Clone, Default, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode,
)]
pub struct SimpleTokenizer;

impl SimpleTokenizer {
    /// Create a new simple tokenizer.
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for SimpleTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect()
    }
}

/// Code-aware tokenizer for BM25 indexing of source code.
///
/// Splits identifiers at camelCase, snake_case, and SCREAMING_CASE boundaries,
/// strips language keywords and short tokens (<2 chars). Designed for code DB
/// search where Korean morphological analysis is inappropriate.
#[derive(
    Debug, Clone, Default, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode,
)]
pub struct CodeTokenizer;

impl CodeTokenizer {
    /// Create a new code tokenizer.
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for CodeTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();

        // Split on whitespace, punctuation, operators, brackets
        for word in text.split(|c: char| {
            c.is_whitespace()
                || matches!(
                    c,
                    '(' | ')' | '{' | '}' | '[' | ']' | '<' | '>' | ','
                        | ';' | ':' | '.' | '=' | '+' | '-' | '*' | '/'
                        | '&' | '|' | '!' | '?' | '#' | '@' | '"' | '\''
                        | '`' | '~' | '^' | '%' | '\\'
                )
        }) {
            if word.is_empty() {
                continue;
            }
            // Split camelCase / snake_case / SCREAMING_CASE
            for sub in split_identifier(word) {
                let lower = sub.to_lowercase();
                if lower.len() >= 2 && !is_code_stopword(&lower) {
                    tokens.push(lower);
                }
            }
        }
        tokens
    }
}

/// Split an identifier at camelCase, snake_case, and digit boundaries.
fn split_identifier(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();

    // First split on underscores
    for segment in s.split('_') {
        if segment.is_empty() {
            continue;
        }
        split_camel_case(segment, &mut parts);
    }

    if parts.is_empty() && !s.is_empty() {
        parts.push(s);
    }
    parts
}

/// Split a single segment (no underscores) at camelCase boundaries.
fn split_camel_case<'a>(segment: &'a str, parts: &mut Vec<&'a str>) {
    let bytes = segment.as_bytes();
    let len = bytes.len();
    if len == 0 {
        return;
    }

    let mut start = 0;
    let mut i = 1;

    while i < len {
        let prev = bytes[i - 1];
        let curr = bytes[i];

        let split_here =
            // lowercase → uppercase: "camelCase" → "camel" | "Case"
            (prev.is_ascii_lowercase() && curr.is_ascii_uppercase())
            // letter → digit: "vec3" → "vec" | "3"
            || (prev.is_ascii_alphabetic() && curr.is_ascii_digit())
            // digit → letter: "3d" → "3" | "d"
            || (prev.is_ascii_digit() && curr.is_ascii_alphabetic())
            // UPPER run end: "HTMLParser" → "HTML" | "Parser"
            || (i + 1 < len
                && prev.is_ascii_uppercase()
                && curr.is_ascii_uppercase()
                && bytes[i + 1].is_ascii_lowercase());

        if split_here {
            if i > start {
                parts.push(&segment[start..i]);
            }
            start = i;
        }
        i += 1;
    }

    if start < len {
        parts.push(&segment[start..]);
    }
}

/// Common programming language keywords that add no search value.
fn is_code_stopword(token: &str) -> bool {
    matches!(
        token,
        "fn" | "if" | "else" | "for" | "while" | "let" | "mut" | "pub" | "use"
            | "mod" | "impl" | "self" | "super" | "crate" | "as" | "in" | "ref"
            | "return" | "match" | "true" | "false" | "none" | "some" | "ok" | "err"
            | "var" | "const" | "new" | "this" | "null" | "void" | "int" | "def"
            | "class" | "import" | "from" | "try" | "catch" | "throw" | "throws"
            | "static" | "final" | "public" | "private" | "protected" | "abstract"
            | "func" | "struct" | "enum" | "type" | "interface" | "package"
            | "do" | "end" | "then" | "begin" | "not" | "and" | "or"
            | "the" | "is" | "it" | "to" | "of" | "an" | "be" | "no"
    )
}
