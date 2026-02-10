//! Bigram proximity tokens for BM25.
//!
//! Adjacent unigrams are joined with SEP to create bigram terms.
//! These terms naturally get high IDF in the inverted index,
//! boosting documents where query terms appear adjacent.

const SEP: char = '\x01';

/// Generate bigram tokens from a list of unigram tokens.
///
/// e.g. `["hello", "world", "foo"]` → `["hello\x01world", "world\x01foo"]`
pub fn generate(tokens: &[String]) -> Vec<String> {
    tokens
        .windows(2)
        .map(|w| format!("{}{SEP}{}", w[0], w[1]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert!(generate(&[]).is_empty());
    }

    #[test]
    fn test_single_token() {
        let tokens = vec!["hello".to_string()];
        assert!(generate(&tokens).is_empty());
    }

    #[test]
    fn test_two_tokens() {
        let tokens = vec!["hello".to_string(), "world".to_string()];
        let bigrams = generate(&tokens);
        assert_eq!(bigrams.len(), 1);
        assert_eq!(bigrams[0], "hello\x01world");
    }

    #[test]
    fn test_three_tokens() {
        let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let bigrams = generate(&tokens);
        assert_eq!(bigrams, vec!["a\x01b", "b\x01c"]);
    }
}
