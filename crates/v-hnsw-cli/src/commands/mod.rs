//! CLI command implementations.
//!
//! Module compilation is managed centrally here.
//! Individual module files contain NO `#[cfg]` attributes.

// ── Shared infrastructure (both v-hnsw and v-code) ──────────────────
pub mod common;
pub mod db_config;
pub mod dict;
pub mod file_index;
pub mod file_utils;
pub mod indexing;
pub mod ingest;
pub mod info;
pub mod pipeline;
pub mod query_cache;
pub mod readers;
pub mod search_context;
pub mod search_result;
pub mod serve;

// ── Document commands (v-hnsw) ──────────────────────────────────────
pub mod add;
pub mod buildindex;
pub mod collection;
pub mod create;
pub mod delete;
pub mod export;
pub mod find;
pub mod get;
pub mod insert;
pub mod update;

// ── Benchmark / diagnostics ─────────────────────────────────────────
pub mod bench;

#[cfg(test)]
mod tests;
