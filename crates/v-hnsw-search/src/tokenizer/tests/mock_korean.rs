//! Mock Korean tokenizer for fast unit tests.
//!
//! Splits on whitespace + Unicode boundaries, assigns POS heuristically.
//! No lindera dictionary load — instant initialization.

use crate::tokenizer::korean::{Token, TokenKind, Tokenizer};
use crate::tokenizer::filters::{FilterChain, TokenFilter};
use v_hnsw_core::VhnswError;

/// Mock Korean tokenizer — whitespace split + heuristic POS tagging.
/// Replicates `KoreanTokenizer` interface without lindera.
pub struct MockKoreanTokenizer {
    index_filters: FilterChain,
    query_filters: FilterChain,
}

impl MockKoreanTokenizer {
    pub fn new() -> Self {
        Self {
            index_filters: FilterChain::korean_default(),
            query_filters: FilterChain::korean_default(),
        }
    }
}

impl Tokenizer for MockKoreanTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<Token>, VhnswError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let lower = text.to_lowercase();
        let mut tokens = Vec::new();
        let mut pos = 0;

        for word in lower.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation()) {
            let trimmed = word.trim();
            if trimmed.is_empty() { pos += word.len() + 1; continue; }

            let kind = if trimmed.chars().all(|c| c.is_ascii_alphabetic()) {
                TokenKind::Foreign
            } else if trimmed.chars().all(|c| c.is_ascii_digit()) {
                TokenKind::Number
            } else if trimmed.chars().any(|c| ('\u{AC00}'..='\u{D7AF}').contains(&c)) {
                TokenKind::Noun // Heuristic: Korean text → Noun
            } else {
                TokenKind::Unknown
            };

            let start = text.to_lowercase().find(trimmed).unwrap_or(pos);
            tokens.push(Token::new(trimmed, start, start + trimmed.len(), kind));
            pos = start + trimmed.len();
        }

        Ok(tokens)
    }

    fn tokenize_for_index(&self, text: &str) -> Result<Vec<Token>, VhnswError> {
        let tokens = self.tokenize(text)?;
        Ok(self.index_filters.filter(tokens))
    }

    fn tokenize_for_query(&self, text: &str) -> Result<Vec<Token>, VhnswError> {
        let tokens = self.tokenize(text)?;
        Ok(self.query_filters.filter(tokens))
    }
}
