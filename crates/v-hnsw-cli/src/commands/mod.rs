//! CLI command implementations.

// Shared modules (no feature gates)
pub mod db_config;
pub mod dict;
pub mod file_index;
pub mod file_utils;
pub mod ingest;
pub mod info;
pub mod readers;
pub mod pipeline;
pub mod search_context;
pub mod search_result;

// Doc-only modules
#[cfg(feature = "doc")]
pub mod add;
#[cfg(feature = "doc")]
pub mod bench;
#[cfg(feature = "doc")]
pub mod buildindex;
#[cfg(feature = "doc")]
pub mod collection;
#[cfg(feature = "doc")]
pub mod common;
#[cfg(feature = "doc")]
pub mod create;
#[cfg(feature = "doc")]
pub mod delete;
#[cfg(feature = "doc")]
pub mod export;
#[cfg(feature = "doc")]
pub mod find;
#[cfg(feature = "doc")]
pub mod get;
#[cfg(feature = "doc")]
pub mod indexing;
#[cfg(feature = "doc")]
pub mod insert;
#[cfg(feature = "doc")]
pub mod query_cache;
#[cfg(feature = "doc")]
pub mod serve;
#[cfg(feature = "doc")]
pub mod update;

#[cfg(test)]
mod tests;
