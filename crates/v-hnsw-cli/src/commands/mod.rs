//! CLI command implementations.

pub mod common;
pub mod dict;
pub mod file_utils;
pub mod ingest;
pub mod search_result;
pub mod add;
pub mod bench;
pub mod buildindex;
pub mod collection;
pub mod create;
pub mod delete;
pub mod export;
pub mod file_index;
pub mod find;
pub mod get;
pub mod indexing;
pub mod info;
pub mod insert;
pub mod query_cache;
pub mod readers;
pub mod serve;
pub mod update;
pub mod code_intel;
pub mod dupes;

#[cfg(test)]
mod tests;
