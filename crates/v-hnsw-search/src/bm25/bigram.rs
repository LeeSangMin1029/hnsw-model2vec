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