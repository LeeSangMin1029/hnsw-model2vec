//! Token filters for text processing.
//!
//! Provides filters to remove or transform tokens during indexing and search.

use super::korean::Token;
use std::collections::HashSet;

/// Default Korean stopwords.
///
/// Common particles, conjunctions, and auxiliary verbs that
/// typically don't contribute to search relevance.
pub const DEFAULT_KOREAN_STOPWORDS: &[&str] = &[
    // Particles (조사)
    "이",
    "가",
    "은",
    "는",
    "을",
    "를",
    "의",
    "에",
    "에서",
    "로",
    "으로",
    "와",
    "과",
    "도",
    "만",
    "까지",
    "부터",
    "마다",
    "처럼",
    "같이",
    "보다",
    "에게",
    "한테",
    "께",
    "더러",
    // Conjunctions (접속사)
    "그리고",
    "그러나",
    "그래서",
    "그러면",
    "하지만",
    "또한",
    "또는",
    "즉",
    "및",
    // Auxiliary verbs/endings
    "있다",
    "없다",
    "하다",
    "되다",
    "이다",
    "아니다",
    "것",
    "수",
    "등",
    "때",
    "중",
    "후",
    "전",
];

/// A trait for filtering tokens.
pub trait TokenFilter: Send + Sync {
    /// Filter a list of tokens, returning only those that pass the filter.
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token>;
}

/// Removes stopwords from the token stream.
#[derive(Debug, Clone)]
pub struct StopwordFilter {
    stopwords: HashSet<String>,
}

impl StopwordFilter {
    /// Create a new stopword filter with the given stopwords.
    pub fn new<I, S>(stopwords: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            stopwords: stopwords.into_iter().map(Into::into).collect(),
        }
    }

    /// Create a stopword filter with default Korean stopwords.
    pub fn korean() -> Self {
        Self::new(DEFAULT_KOREAN_STOPWORDS.iter().copied())
    }

    /// Check if a word is a stopword.
    pub fn is_stopword(&self, word: &str) -> bool {
        self.stopwords.contains(word)
    }

    /// Add a stopword to the filter.
    pub fn add_stopword(&mut self, word: impl Into<String>) {
        self.stopwords.insert(word.into());
    }

    /// Remove a stopword from the filter.
    pub fn remove_stopword(&mut self, word: &str) -> bool {
        self.stopwords.remove(word)
    }
}

impl TokenFilter for StopwordFilter {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .filter(|t| !self.is_stopword(&t.text))
            .collect()
    }
}

impl Default for StopwordFilter {
    fn default() -> Self {
        Self::korean()
    }
}

/// Removes tokens shorter than a minimum length.
#[derive(Debug, Clone)]
pub struct MinLengthFilter {
    min_length: usize,
}

impl MinLengthFilter {
    /// Create a new minimum length filter.
    ///
    /// Tokens with fewer characters than `min_length` will be removed.
    pub fn new(min_length: usize) -> Self {
        Self { min_length }
    }
}

impl TokenFilter for MinLengthFilter {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .filter(|t| t.text.chars().count() >= self.min_length)
            .collect()
    }
}

impl Default for MinLengthFilter {
    fn default() -> Self {
        Self::new(2)
    }
}

/// Converts all token text to lowercase.
#[derive(Debug, Clone, Default)]
pub struct LowercaseFilter;

impl TokenFilter for LowercaseFilter {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                t.text = t.text.to_lowercase();
                t
            })
            .collect()
    }
}

/// Chains multiple filters together.
#[derive(Default)]
pub struct FilterChain {
    filters: Vec<Box<dyn TokenFilter>>,
}

impl FilterChain {
    /// Create an empty filter chain.
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Add a filter to the chain.
    pub fn add_filter<F: TokenFilter + 'static>(mut self, filter: F) -> Self {
        self.filters.push(Box::new(filter));
        self
    }

    /// Create a default filter chain for Korean text.
    ///
    /// Includes: stopword removal, minimum length (2 chars).
    pub fn korean_default() -> Self {
        Self::new()
            .add_filter(StopwordFilter::korean())
            .add_filter(MinLengthFilter::new(2))
    }
}

impl TokenFilter for FilterChain {
    fn filter(&self, mut tokens: Vec<Token>) -> Vec<Token> {
        for filter in &self.filters {
            tokens = filter.filter(tokens);
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::TokenKind;

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
}
