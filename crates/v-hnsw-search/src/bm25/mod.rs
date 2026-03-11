//! BM25 sparse text search index.
//!
//! Implements Okapi BM25 for keyword-based document retrieval.
//! Used as the sparse component in hybrid search.

pub(crate) mod bigram;
pub(crate) mod fieldnorm;
pub(crate) mod fst_storage;
mod index;
mod maxscore;
mod scorer;
pub(crate) mod snapshot;

#[cfg(test)]
mod tests;

pub use fieldnorm::{FieldNormLut, encode as encode_fieldnorm, decode as decode_fieldnorm};
pub use index::{Bm25Index, Posting, PostingList};
pub use scorer::Bm25Params;
pub use snapshot::Bm25Snapshot;
