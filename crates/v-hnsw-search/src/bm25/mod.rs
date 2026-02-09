//! BM25 sparse text search index.
//!
//! Implements Okapi BM25 for keyword-based document retrieval.
//! Used as the sparse component in hybrid search.

pub(crate) mod fst_storage;
mod index;
mod maxscore;
mod scorer;

pub use index::{Bm25Index, Posting, PostingList};
pub use scorer::Bm25Params;
